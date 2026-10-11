"""Silver labels for the student: each schema field takes its label from the source that
configs/silver.toml names for it.

Sources of a field:
- "rules": the regex baseline (radar.extract.baseline) on the same input;
- "qwen", "yandex": that teacher's answer, data/teacher/<teacher>/<split>.jsonl;
- "agree": the teachers' value where both answered and agree (lists compared after ordering and
  casefolding, as radar.eval.agreement does), the rules' value otherwise.

Each source family compares its candidate compositions on its own dev split, trudvsem and Habr on
dev, Telegram on tg_dev, and chooses one. Their inputs differ: on trudvsem and Habr salary, city and
experience come from a header the rules read as well as the teachers (EXPERIMENTS E4, E5), a
Telegram post is free text only. A family without a chosen composition is not built.

`dev` writes data/predictions/<dev split>/silver_<name>.jsonl and, once the gold of that split
exists, scores it against regex with radar.eval. There a field whose teacher has no answer (a failed
request) takes the rules' value and the line lists it under `fallback`, so a technical failure is not
counted against the source. `build` drops such a record instead: every silver label comes from the
source named for it.

`build` reads, per family, the training-pool records both teachers labeled, keeps those still in the
pool after a corpus rebuild and whose input is the one the teachers saw, drops Telegram posts a
teacher marked `exclude` (a list of vacancies or not a vacancy, which the gold set drops too), and
writes data/silver/train.jsonl, one line {"id", "input", "output"} per record, `output` being the
label as a JSON string (the text the student learns to produce). Counts go to results/silver/build.json.

    python -m radar.extract.silver dev                                # trudvsem and Habr candidates
    python -m radar.extract.silver dev --family telegram --composition qwen_role_format
    python -m radar.extract.silver build                              # every family with a chosen composition
"""

import argparse
import hashlib
import json
import logging
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radar.config import ROOT, data_dir
from radar.eval.agreement import FIELDS, field_values
from radar.eval.cli import SYSTEM_NAME
from radar.eval.cli import load_config as load_eval_config
from radar.eval.cli import main as evaluate
from radar.extract.baseline import extract, read_inputs
from radar.extract.schema import VacancyFields, dump, try_parse
from radar.extract.teacher import (
    IDENTITY,
    PAIR,
    Item,
    asks_exclude,
    meta_path,
    output_path,
    read_rows,
    sha256_text,
    share,
    tg_train_pool_items,
    train_pool_items,
    write_rows,
)
from radar.results import write_result

log = logging.getLogger("radar.extract.silver")

CONFIG = ROOT / "configs" / "silver.toml"
SOURCES = ("rules", *PAIR, "agree")


@dataclass(frozen=True)
class Family:
    name: str
    pool: str
    dev: str

    def drop_reasons(self) -> tuple[str, ...]:
        excluded = ("excluded",) if asks_exclude(self.pool) else ()
        return ("not_in_train_pool", "input_changed", *excluded, *(f"{t}_invalid" for t in PAIR))

    def pool_items(self, seed: int) -> list[Item]:
        """Every record of the training pool: a labeled record stays while it is in the pool."""
        if self.name == "telegram":
            return tg_train_pool_items(data_dir() / "corpus" / "telegram", seed, one_per_group=False)
        return train_pool_items(data_dir() / "corpus", seed)


FAMILIES = {
    "trudvsem_habr": Family("trudvsem_habr", "train_pool", "dev"),
    "telegram": Family("telegram", "tg_train_pool", "tg_dev"),
}


@dataclass(frozen=True)
class Composition:
    name: str
    sources: Mapping[str, str]


MAX_CANDIDATES = 4


@dataclass(frozen=True)
class SilverConfig:
    compositions: dict[str, Composition]
    candidates: dict[str, list[str]]  # family -> compositions compared on its dev split
    chosen: dict[str, str]  # family -> composition; a family may have none yet


def load_config(path: Path = CONFIG) -> SilverConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    compositions: dict[str, Composition] = {}
    for name, table in raw.get("compositions", {}).items():
        if not SYSTEM_NAME.match(name):
            raise ValueError(f"composition {name!r}: use lowercase letters, digits, _ and -")
        unknown = sorted(set(table) - set(FIELDS))
        if unknown:
            raise ValueError(f"composition {name!r}: not schema fields: {unknown}")
        bad = sorted({s for s in table.values() if s not in SOURCES})
        if bad:
            raise ValueError(f"composition {name!r}: a source is one of {SOURCES}, not {bad}")
        compositions[name] = Composition(name, {f: table.get(f, "rules") for f in FIELDS})
    candidates = {f: list(names) for f, names in raw.get("candidates", {}).items()}
    chosen = dict(raw.get("chosen", {}))
    for family in sorted(set(candidates) | set(chosen)):
        if family not in FAMILIES:
            raise ValueError(f"{family!r} is not a source family, one of {sorted(FAMILIES)}")
    for family, names in candidates.items():
        unknown = sorted(set(names) - set(compositions))
        if unknown:
            raise ValueError(f"candidates.{family}: not in [compositions.*]: {unknown}")
        if len(names) > MAX_CANDIDATES:
            raise ValueError(f"candidates.{family}: {len(names)} compositions, at most {MAX_CANDIDATES} per family")
    for family, name in chosen.items():
        if name not in candidates.get(family, []):
            raise ValueError(f"chosen.{family} = {name!r} is not one of candidates.{family}")
    return SilverConfig(compositions, candidates, chosen)


@dataclass(frozen=True)
class Composed:
    output: VacancyFields | None
    missing: str | None = None
    agreed: frozenset[str] = frozenset()
    fallback: frozenset[str] = frozenset()


def compose(
    composition: Composition,
    rules: VacancyFields,
    answers: Mapping[str, VacancyFields | None],
    fallback: bool,
) -> Composed:
    """One record's label from the rules' answer and each teacher's valid answer (None if it has
    none). A field whose teacher has no answer takes the rules' value when `fallback`; otherwise the
    record is dropped and `missing` names the teacher."""
    rule_values = dump(rules)
    given = {t: a for t, a in answers.items() if a is not None}
    dumped = {t: dump(a) for t, a in given.items()}
    out: dict[str, Any] = {}
    agreed: set[str] = set()
    fell_back: set[str] = set()
    for field, source in composition.sources.items():
        if source == "rules":
            out[field] = rule_values[field]
        elif source == "agree":
            first, second = PAIR
            if (
                first in given
                and second in given
                and field_values(given[first])[field] == field_values(given[second])[field]
            ):
                out[field] = dumped[first][field]
                agreed.add(field)
            else:
                out[field] = rule_values[field]
        elif source in dumped:
            out[field] = dumped[source][field]
        elif fallback:
            out[field] = rule_values[field]
            fell_back.add(field)
        else:
            return Composed(None, missing=source)
    return Composed(VacancyFields.model_validate(out), agreed=frozenset(agreed), fallback=frozenset(fell_back))


@dataclass(frozen=True)
class Answer:
    fields: VacancyFields | None  # None for an invalid answer or a failed request
    input_sha256: str
    exclude: str | None = None


def teacher_answers(split: str) -> dict[str, dict[str, Answer]]:
    """Per teacher and record: the valid answer, the sha256 of the input it answered and, on
    Telegram, the teacher's `exclude`."""
    answers = {}
    for name in PAIR:
        rows = read_rows(output_path(name, split))
        if not rows:
            raise SystemExit(f"{output_path(name, split)} is missing or empty: label {split} with {name} first")
        answers[name] = {r["id"]: Answer(try_parse(r["output"]), r["input_sha256"], r.get("exclude")) for r in rows}
    return answers


def label_line(composed: Composed) -> str:
    assert composed.output is not None
    return json.dumps(dump(composed.output), ensure_ascii=False)


def cmd_dev(composition: Composition, family: Family) -> int:
    gold_dir = data_dir() / "gold"
    inputs = read_inputs(gold_dir / f"{family.dev}_inputs.jsonl")
    answers = teacher_answers(family.dev)
    rows = []
    fell_back: Counter[str] = Counter()
    for rid, text in inputs:
        current = sha256_text(text)
        per_teacher: dict[str, VacancyFields | None] = {}
        for name in PAIR:
            answer = answers[name].get(rid, Answer(None, current))
            if answer.input_sha256 != current:
                raise SystemExit(f"{name} answered another input for {rid}: label {family.dev} again")
            per_teacher[name] = answer.fields
        composed = compose(composition, extract(text), per_teacher, fallback=True)
        fell_back.update(composed.fallback)
        rows.append({"id": rid, "output": label_line(composed), "fallback": sorted(composed.fallback)})
    system = f"silver_{composition.name}"
    out = data_dir() / "predictions" / family.dev / f"{system}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, rows)
    log.info(
        "%s: %d records -> %s, rules stood in for a missing teacher answer: %s", system, len(rows), out, dict(fell_back)
    )
    if not (gold_dir / f"{family.dev}.jsonl").exists():
        log.info("no %s gold yet: predictions written, nothing scored", family.dev)
        return 0
    return evaluate(["--system", system, "--split", family.dev, "--compare", "regex"])


@dataclass(frozen=True)
class Built:
    rows: list[dict[str, str]]
    metrics: dict[str, dict[str, Any]]
    strata: dict[str, int]
    teachers: dict[str, dict[str, Any]]


def build_family(family: Family, composition: Composition, seed: int) -> Built:
    answers = teacher_answers(family.pool)
    labeled = set(answers[PAIR[0]]) & set(answers[PAIR[1]])
    pool = [i for i in family.pool_items(seed) if i.id in labeled]
    dropped: Counter[str] = Counter({"not_in_train_pool": len(labeled) - len(pool)})
    agreed: Counter[str] = Counter()
    excluded_by: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    rows = []
    for item in pool:
        current = sha256_text(item.input)
        if any(answers[name][item.id].input_sha256 != current for name in PAIR):
            dropped["input_changed"] += 1
            continue
        flagged = [name for name in PAIR if answers[name][item.id].exclude is not None]
        excluded_by.update(flagged)
        if flagged:
            dropped["excluded"] += 1
            continue
        per_teacher = {name: answers[name][item.id].fields for name in PAIR}
        composed = compose(composition, extract(item.input), per_teacher, fallback=False)
        if composed.output is None:
            dropped[f"{composed.missing}_invalid"] += 1
            continue
        agreed.update(composed.agreed)
        strata[item.stratum] += 1
        rows.append({"id": item.id, "input": item.input, "output": label_line(composed)})

    n = len(labeled)
    p = f"{family.name}."
    metrics: dict[str, dict[str, Any]] = {
        f"{p}records_labeled_by_both": {"value": n},
        f"{p}records_written": share(len(rows), n),
        **{f"{p}dropped_{reason}": share(dropped[reason], n) for reason in family.drop_reasons()},
        **{
            f"{p}agreed_{field}": share(agreed[field], len(rows))
            for field, source in composition.sources.items()
            if source == "agree"
        },
    }
    if asks_exclude(family.pool):
        for name in PAIR:
            metrics[f"{p}excluded_by_{name}"] = share(excluded_by[name], n)
    teachers = {}
    for name in PAIR:
        meta = json.loads(meta_path(name, family.pool).read_text(encoding="utf-8"))
        teachers[name] = {k: meta[k] for k in IDENTITY}
    log.info("%s: %d of %d written; dropped %s", family.name, len(rows), n, dict(dropped))
    return Built(rows, metrics, {s: strata[s] for s in sorted(strata)}, teachers)


def cmd_build(chosen: Mapping[str, str], compositions: Mapping[str, Composition], config: Path) -> int:
    seed = load_eval_config().seed
    built = {f: build_family(FAMILIES[f], compositions[chosen[f]], seed) for f in FAMILIES if f in chosen}
    if not built:
        raise SystemExit(f"no source family has a chosen composition in {config}")
    not_built = [f for f in FAMILIES if f not in chosen]
    for f in not_built:
        log.info("%s: no composition chosen in %s, not built", f, config)

    rows = [row for b in built.values() for row in b.rows]
    out = data_dir() / "silver" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, rows)
    metrics: dict[str, dict[str, Any]] = {"records_written": {"value": len(rows)}}
    for b in built.values():
        metrics |= b.metrics
    ids = sorted(r["id"] for r in rows)
    path = write_result(
        "silver/build.json",
        metrics,
        source_script="src/radar/extract/silver.py",
        compositions={f: chosen[f] for f in built},
        sources={f: dict(compositions[chosen[f]].sources) for f in built},
        families_not_built=not_built,
        config_file=config.resolve().relative_to(ROOT).as_posix()
        if config.resolve().is_relative_to(ROOT)
        else str(config),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        teachers={f: b.teachers for f, b in built.items()},
        output_file="data/silver/train.jsonl",
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        ids_sha256=sha256_text("\n".join(ids)),
        strata={f: b.strata for f, b in built.items()},
        method="per source family, records labeled by both teachers; a record leaves the set if "
        "it is no longer in its training pool, if its input is not the one the teachers saw, if a teacher "
        "marked a Telegram post exclude (multi_vacancy or not_vacancy), or if a teacher the composition "
        "names for a field has no valid answer; agreed_<field>: records where the teachers agreed on a field "
        "the composition takes from their agreement (the rest take the rules' value)",
    )
    log.info("%d records -> %s", len(rows), out)
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.extract.silver", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = sub.add_parser("dev", help="score compositions on the family's dev split against regex")
    dev.add_argument("--family", choices=sorted(FAMILIES), default="trudvsem_habr")
    dev.add_argument("--composition", action="append", help="default: the family's candidates")
    sub.add_parser("build", help="data/silver/train.jsonl from each family's chosen composition")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "build":
        return cmd_build(cfg.chosen, cfg.compositions, args.config)
    names = args.composition or cfg.candidates.get(args.family, [])
    if not names:
        raise SystemExit(f"{args.config} lists no candidates.{args.family}: name one with --composition")
    unknown = sorted(set(names) - set(cfg.compositions))
    if unknown:
        raise SystemExit(f"no such composition in {args.config}: {unknown}")
    status = 0
    for name in names:
        status = cmd_dev(cfg.compositions[name], FAMILIES[args.family]) or status
    return status


if __name__ == "__main__":
    sys.exit(main())
