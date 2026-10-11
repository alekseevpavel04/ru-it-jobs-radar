"""Silver labels for the student: each schema field takes its label from the source that
configs/silver.toml names for it.

Sources of a field:
- "rules": the regex baseline (radar.extract.baseline) on the same input;
- "qwen", "yandex": that teacher's answer, data/teacher/<teacher>/<split>.jsonl;
- "agree": the teachers' value where both answered and agree (lists compared after ordering and
  casefolding, as radar.eval.agreement does), the rules' value otherwise.

The composition is chosen on dev only. `dev` writes data/predictions/dev/silver_<name>.jsonl and scores
it against regex with radar.eval. There a field whose teacher has no answer (a failed request) takes
the rules' value and the line lists it under `fallback`, so a technical failure is not counted against
the source. `build` drops such a record instead: every silver label comes from the source named for it.

`build` reads the train_pool records both teachers labeled, keeps those still in train_pool after a
corpus rebuild and whose input is the one the teachers saw, and writes data/silver/train.jsonl, one line
{"id", "input", "output"} per record, `output` being the label as a JSON string (the text the student
learns to produce). Counts go to results/silver/build.json.

    python -m radar.extract.silver dev                                # every composition of the config
    python -m radar.extract.silver dev --composition qwen_role_format
    python -m radar.extract.silver build                              # the chosen composition
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
    meta_path,
    output_path,
    read_rows,
    sha256_text,
    share,
    train_pool_items,
    write_rows,
)
from radar.results import write_result

log = logging.getLogger("radar.extract.silver")

CONFIG = ROOT / "configs" / "silver.toml"
SOURCES = ("rules", *PAIR, "agree")
DROP_REASONS = ("not_in_train_pool", "input_changed", *(f"{t}_invalid" for t in PAIR))


@dataclass(frozen=True)
class Composition:
    name: str
    sources: Mapping[str, str]


def load_compositions(path: Path = CONFIG) -> tuple[str, dict[str, Composition]]:
    """The chosen composition's name and every composition, each with a source for every field."""
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
    chosen = raw.get("chosen")
    if chosen not in compositions:
        raise ValueError(f"chosen = {chosen!r} is not one of [compositions.*]: {sorted(compositions)}")
    return chosen, compositions


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


def teacher_answers(split: str) -> dict[str, dict[str, tuple[VacancyFields | None, str]]]:
    """Per teacher and record: the valid answer (None for an invalid answer or a failed request) and
    the sha256 of the input it answered."""
    answers = {}
    for name in PAIR:
        rows = read_rows(output_path(name, split))
        if not rows:
            raise SystemExit(f"{output_path(name, split)} is missing or empty: label {split} with {name} first")
        answers[name] = {r["id"]: (try_parse(r["output"]), r["input_sha256"]) for r in rows}
    return answers


def label_line(composed: Composed) -> str:
    assert composed.output is not None
    return json.dumps(dump(composed.output), ensure_ascii=False)


def cmd_dev(composition: Composition) -> int:
    inputs = read_inputs(data_dir() / "gold" / "dev_inputs.jsonl")
    answers = teacher_answers("dev")
    rows = []
    fell_back: Counter[str] = Counter()
    for rid, text in inputs:
        current = sha256_text(text)
        per_teacher: dict[str, VacancyFields | None] = {}
        for name in PAIR:
            answer, seen = answers[name].get(rid, (None, current))
            if seen != current:
                raise SystemExit(f"{name} answered another input for {rid}: label dev again")
            per_teacher[name] = answer
        composed = compose(composition, extract(text), per_teacher, fallback=True)
        fell_back.update(composed.fallback)
        rows.append({"id": rid, "output": label_line(composed), "fallback": sorted(composed.fallback)})
    system = f"silver_{composition.name}"
    out = data_dir() / "predictions" / "dev" / f"{system}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, rows)
    log.info(
        "%s: %d records -> %s, rules stood in for a missing teacher answer: %s", system, len(rows), out, dict(fell_back)
    )
    return evaluate(["--system", system, "--split", "dev", "--compare", "regex"])


def cmd_build(composition: Composition, config: Path) -> int:
    answers = teacher_answers("train_pool")
    labeled = set(answers[PAIR[0]]) & set(answers[PAIR[1]])
    pool = [i for i in train_pool_items(data_dir() / "corpus", load_eval_config().seed) if i.id in labeled]
    dropped: Counter[str] = Counter({"not_in_train_pool": len(labeled) - len(pool)})
    agreed: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    rows = []
    for item in pool:
        current = sha256_text(item.input)
        if any(answers[name][item.id][1] != current for name in PAIR):
            dropped["input_changed"] += 1
            continue
        per_teacher = {name: answers[name][item.id][0] for name in PAIR}
        composed = compose(composition, extract(item.input), per_teacher, fallback=False)
        if composed.output is None:
            dropped[f"{composed.missing}_invalid"] += 1
            continue
        agreed.update(composed.agreed)
        strata[item.stratum] += 1
        rows.append({"id": item.id, "input": item.input, "output": label_line(composed)})

    out = data_dir() / "silver" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, rows)
    n = len(labeled)
    metrics: dict[str, dict[str, Any]] = {
        "records_labeled_by_both": {"value": n},
        "records_written": share(len(rows), n),
        **{f"dropped_{reason}": share(dropped[reason], n) for reason in DROP_REASONS},
        **{
            f"agreed_{field}": share(agreed[field], len(rows))
            for field, source in composition.sources.items()
            if source == "agree"
        },
    }
    teachers = {}
    for name in PAIR:
        meta = json.loads(meta_path(name, "train_pool").read_text(encoding="utf-8"))
        teachers[name] = {k: meta[k] for k in IDENTITY}
    ids = sorted(r["id"] for r in rows)
    path = write_result(
        "silver/build.json",
        metrics,
        source_script="src/radar/extract/silver.py",
        composition=composition.name,
        sources=dict(composition.sources),
        config_file=config.resolve().relative_to(ROOT).as_posix()
        if config.resolve().is_relative_to(ROOT)
        else str(config),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        teachers=teachers,
        output_file="data/silver/train.jsonl",
        output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        ids_sha256=sha256_text("\n".join(ids)),
        strata={s: strata[s] for s in sorted(strata)},
        method="records labeled by both teachers; a record leaves the set if it is no longer in "
        "train_pool, if its input is not the one the teachers saw, or if a teacher the composition names "
        "for a field has no valid answer; agreed_<field>: records where the teachers agreed on a field the "
        "composition takes from their agreement (the rest take the rules' value)",
    )
    log.info("%d of %d written -> %s; dropped %s", len(rows), n, out, dict(dropped))
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="python -m radar.extract.silver", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = sub.add_parser("dev", help="score compositions on dev against regex")
    dev.add_argument("--composition", action="append", help="default: every composition of the config")
    sub.add_parser("build", help="data/silver/train.jsonl from the chosen composition")
    args = ap.parse_args(argv)
    chosen, compositions = load_compositions(args.config)
    if args.cmd == "build":
        return cmd_build(compositions[chosen], args.config)
    names = args.composition or list(compositions)
    unknown = sorted(set(names) - set(compositions))
    if unknown:
        raise SystemExit(f"no such composition in {args.config}: {unknown}")
    status = 0
    for name in names:
        status = cmd_dev(compositions[name]) or status
    return status


if __name__ == "__main__":
    sys.exit(main())
