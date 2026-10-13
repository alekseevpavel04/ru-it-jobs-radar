"""dbt/seeds/skill_dictionary_v1.csv: every spelling of the skill dictionary v1 (configs/skills.toml) by its
lookup key, for the marts on extracted fields (models/extracted).

The extractor writes a skill as it read it; radar.extract.skills normalizes a spelling in steps (key lookup,
inflection patterns, version and vendor rules). The marts run in ClickHouse and take the first step only:
the key of a spelling (macro skill_key, the same folding as radar.extract.skills.key) looked up among the
keys of the names and aliases listed here. A spelling it misses is counted, not guessed
(int_extracted_fields skills_in_dictionary_n against skills_written_n).

    python dbt/skill_seed.py           # rewrite the seed after configs/skills.toml changed
    python dbt/skill_seed.py --check   # exit 1 when the seed is stale (tests/dbt runs the same check)
"""

import argparse
import csv
import io
import sys

from radar.config import ROOT
from radar.extract import skills

SEED = ROOT / "dbt" / "seeds" / "skill_dictionary_v1.csv"
COLUMNS = ("spelling_key", "spelling", "skill", "category", "parent")


def seed_rows() -> list[dict[str, str]]:
    d = skills.load()
    rows: dict[str, dict[str, str]] = {}
    for name, skill in sorted(d.skills.items()):
        for spelling in (name, *skill.aliases):
            k = skills.key(spelling)
            # a name and its alias may fold to one key; load() already refuses a key of two skills
            rows.setdefault(
                k,
                {
                    "spelling_key": k,
                    "spelling": spelling,
                    "skill": name,
                    "category": skill.category,
                    "parent": skill.parent or "",
                },
            )
    return [rows[k] for k in sorted(rows)]


def render(rows: list[dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the seed instead of writing it")
    args = ap.parse_args(argv)
    text = render(seed_rows())
    if args.check:
        current = SEED.read_text(encoding="utf-8") if SEED.exists() else ""
        if current != text:
            print(f"{SEED.relative_to(ROOT).as_posix()} is stale: python dbt/skill_seed.py", file=sys.stderr)
            return 1
        return 0
    SEED.write_text(text, encoding="utf-8", newline="")
    print(f"{SEED.relative_to(ROOT).as_posix()}: {text.count(chr(10)) - 1} spellings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
