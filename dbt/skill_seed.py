"""dbt/seeds/skill_dictionary_v1.csv: skill spellings by the hash of their lookup key, mapped to the names of
the skill dictionary v1 (configs/skills.toml), for the marts on extracted fields (models/extracted).

The extractor writes a skill as it read it, and extract.vacancy_fields keeps that raw answer. The marts run in
ClickHouse and cannot run radar.extract.skills.normalize_list (inflection patterns, version, vendor and
generic-word rules, glued lists), so it runs here, once per distinct spelling:

- origin dictionary: every name and alias of the dictionary, with the spelling and its key;
- origin corpus: every other distinct spelling the extraction model wrote (var extraction_model of
  dbt_project.yml, valid answers), resolved by normalize_list: known (one row per skill it maps to),
  non_skill (what the annotation guide does not count) or unknown (the dictionary lacks it). These rows
  carry only the hash of the key, never the spelling: the spellings come from Habr and Telegram texts,
  which are not published (DECISIONS 2).

The marts fold a spelling to its key (macro skill_key, the folding of radar.extract.skills.key) and look
up sha256 of the key, its first 16 hex digits. A spelling written after the seed was made is not in it:
int_extracted_fields counts it in skills_unchecked_n and assert_skill_seed_covers_extracted_spellings
warns. Rerun this as the corpus run grows, then `bash scripts/dbt.sh seed --select skill_dictionary_v1`.

    python dbt/skill_seed.py           # dictionary and corpus spellings (reads ClickHouse)
    python dbt/skill_seed.py --check   # exit 1 when the dictionary rows are stale (no ClickHouse; tests/dbt)
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections.abc import Iterable, Mapping
from typing import Any

import yaml

from radar.config import ROOT, load_dotenv
from radar.extract import skills

SEED = ROOT / "dbt" / "seeds" / "skill_dictionary_v1.csv"
COLUMNS = ("key_hash", "spelling_key", "spelling", "skill", "category", "parent", "status", "origin")
STATUSES = ("known", "non_skill", "unknown")


def key_hash(spelling_key: str) -> str:
    return hashlib.sha256(spelling_key.encode("utf-8")).hexdigest()[:16]


def dictionary_rows(d: skills.Dictionary) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for name, skill in sorted(d.skills.items()):
        for spelling in (name, *skill.aliases):
            k = skills.key(spelling)
            # a name and its alias may fold to one key; load() already refuses a key of two skills
            rows.setdefault(
                k,
                {
                    "key_hash": key_hash(k),
                    "spelling_key": k,
                    "spelling": spelling,
                    "skill": name,
                    "category": skill.category,
                    "parent": skill.parent or "",
                    "status": "known",
                    "origin": "dictionary",
                },
            )
    return [rows[k] for k in sorted(rows)]


def corpus_rows(d: skills.Dictionary, spellings: Iterable[str], taken: set[str]) -> tuple[list[dict[str, str]], int]:
    """Rows of spellings whose key the dictionary rows do not hold, most frequent first; the second value
    counts spellings skipped because another spelling of the same key resolved differently."""
    rows: list[dict[str, str]] = []
    resolved: dict[str, tuple[str, ...]] = {}
    conflicts = 0
    for raw in spellings:
        k = skills.key(raw)
        if not k:
            continue
        h = key_hash(k)
        names = tuple(s for s in skills.normalize_list([raw], d) if s in d.skills)
        status = "known" if names else ("non_skill" if skills.resolve(raw, d).status == "non_skill" else "unknown")
        outcome = (status, *names)
        if h in taken:
            conflicts += h in resolved and resolved[h] != outcome
            continue
        taken.add(h)
        resolved[h] = outcome
        base = {"key_hash": h, "spelling_key": "", "spelling": "", "status": status, "origin": "corpus"}
        if not names:
            rows.append(base | {"skill": "", "category": "", "parent": ""})
        for name in names:
            s = d.skills[name]
            rows.append(base | {"skill": name, "category": s.category, "parent": s.parent or ""})
    return rows, conflicts


def seed_rows(spellings: Iterable[str] = ()) -> tuple[list[dict[str, str]], int]:
    d = skills.load()
    dictionary = dictionary_rows(d)
    corpus, conflicts = corpus_rows(d, spellings, {r["key_hash"] for r in dictionary})
    corpus.sort(key=lambda r: (r["key_hash"], r["skill"]))
    return dictionary + corpus, conflicts


def render(rows: Iterable[Mapping[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def read_seed() -> list[dict[str, str]]:
    with SEED.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def extraction_model() -> str:
    project = yaml.safe_load((ROOT / "dbt" / "dbt_project.yml").read_text(encoding="utf-8"))
    return str(project["vars"]["extraction_model"])


def corpus_spellings(model_id: str) -> list[str]:
    """Distinct spellings of the model's valid answers, most frequent first."""
    from radar.stream.clickhouse import ClickHouse

    sql = (
        "SELECT s, count() AS n FROM extract.vacancy_fields FINAL ARRAY JOIN skills AS s "
        "WHERE model_id = {model:String} AND valid GROUP BY s ORDER BY n DESC, s FORMAT JSONEachRow"
    )
    with ClickHouse() as ch:
        lines: list[dict[str, Any]] = [json.loads(x) for x in ch.query(sql, model=model_id).splitlines() if x]
    return [str(r["s"]) for r in lines]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare the dictionary rows with configs/skills.toml")
    args = ap.parse_args(argv)
    if args.check:
        current = [r for r in read_seed() if r["origin"] == "dictionary"] if SEED.exists() else []
        if render(current) != render(dictionary_rows(skills.load())):
            print(f"{SEED.relative_to(ROOT).as_posix()} is stale: python dbt/skill_seed.py", file=sys.stderr)
            return 1
        return 0
    load_dotenv()
    model_id = extraction_model()
    if not model_id:
        raise SystemExit("var extraction_model is empty in dbt/dbt_project.yml")
    spellings = corpus_spellings(model_id)
    rows, conflicts = seed_rows(spellings)
    SEED.write_text(render(rows), encoding="utf-8", newline="")
    by_status = {
        s: len({r["key_hash"] for r in rows if r["origin"] == "corpus" and r["status"] == s}) for s in STATUSES
    }
    print(
        f"{SEED.relative_to(ROOT).as_posix()}: {len(rows)} rows; {len(spellings)} spellings of {model_id}; "
        f"corpus keys {by_status}; keys resolved differently by another spelling: {conflicts}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
