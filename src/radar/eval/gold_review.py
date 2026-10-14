"""Full review of the gold set by the author: every record of a split as an editable Markdown file.

The 50-record check (radar.eval.human_check) judged records next to their labels and gave no share of
correct fields. Here the author reads every record of a split against docs/annotation_guide.md and
edits the JSON block in place where a field is wrong; an untouched block means the record was read
and agreed with. Texts are not published, so the files live in data/review/gold/ (not in git); only
counts, ids and field names go to results/gold/full_review.json.

    python -m radar.eval.gold_review export             # data/review/gold/<split>.md, existing files kept
    python -m radar.eval.gold_review compare            # parse the edited files, write the result

A record counts as reviewed when its block carries the mark `[x] прочитано`; the author ticks it after
reading, so a file read halfway is reported as such rather than as full agreement. Edited blocks are
validated against the extraction schema; a broken block stops the comparison with its record number.
Corrected labels go to data/review/gold/<split>.corrected.jsonl in the gold format, so a re-score on
the corrected gold can sit next to the original numbers.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radar.config import data_dir
from radar.eval.data import load_gold
from radar.extract.schema import VacancyFields, dump
from radar.results import write_result

FIELDS_ORDER: tuple[str, ...] = tuple(VacancyFields.model_fields)
SPLITS = ("tg_test", "test", "tg_dev", "dev")
UNREAD, READ = "[ ] прочитано", "[x] прочитано"
RECORD = re.compile(
    r"^## (?P<no>\d+)\. (?P<id>\S+)\n.*?^```json\n(?P<json>.*?)^```\n(?P<rest>.*?)(?=^## \d+\. |\Z)",
    re.S | re.M,
)
HEADER = """# Эталон {split}: {n} записей

Правила полей: docs/annotation_guide.md. Читаешь текст, сверяешь JSON под ним.
Поле неверно: исправь значение прямо в JSON (null, если в тексте этого нет).
Прочитал запись: поставь x в строке «[ ] прочитано». Запись без x считается непрочитанной.
Комментарий пиши в строке «Комментарий:», если хочешь объяснить правку.

"""


def inputs(gold_dir: Path, split: str) -> dict[str, str]:
    out = {}
    for line in (gold_dir / f"{split}_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            out[row["id"]] = row["input"]
    return out


def block(fields: VacancyFields) -> str:
    data = dump(fields)
    lines = [f'  "{k}": {json.dumps(data[k], ensure_ascii=False)}' for k in FIELDS_ORDER]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def export(gold_dir: Path, out_dir: Path, split: str) -> Path | None:
    path = out_dir / f"{split}.md"
    if path.exists():
        return None
    texts = inputs(gold_dir, split)
    records = load_gold(gold_dir / f"{split}.jsonl")
    parts = [HEADER.format(split=split, n=len(records))]
    for no, rec in enumerate(records, 1):
        text = texts[rec.id].replace("~~~", "~ ~ ~")
        parts.append(
            f"## {no}. {rec.id}\n\n~~~text\n{text}\n~~~\n\n```json\n{block(rec.fields)}```\n\n{UNREAD}\n"
            "Комментарий:\n\n"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8", newline="\n")
    return path


def compare(gold_dir: Path, out_dir: Path, split: str) -> dict[str, Any] | None:
    path = out_dir / f"{split}.md"
    if not path.exists():
        return None
    gold = {r.id: r for r in load_gold(gold_dir / f"{split}.jsonl")}
    read, changed_records = 0, 0
    changed_fields: Counter[str] = Counter()
    corrected = []
    for m in RECORD.finditer(path.read_text(encoding="utf-8")):
        rid = m["id"]
        try:
            edited = VacancyFields.model_validate(json.loads(m["json"]))
        except (json.JSONDecodeError, ValidationError) as e:
            raise SystemExit(f"{path}: record {m['no']} ({rid}): {e}") from e
        original = dump(gold[rid].fields)
        new = dump(edited)
        diff = [k for k in FIELDS_ORDER if original[k] != new[k]]
        if READ in m["rest"]:
            read += 1
        if diff:
            changed_records += 1
            changed_fields.update(diff)
        corrected.append({"id": rid, "source": gold[rid].source, "fields": new, "provenance": "author_review"})
    if len(corrected) != len(gold):
        raise SystemExit(f"{path}: found {len(corrected)} records of {len(gold)}; a header line was broken")
    with (out_dir / f"{split}.corrected.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in corrected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(gold)
    return {
        "records_read": {"value": round(read / n, 4), "n": read, "denominator": n},
        "records_changed": {"value": round(changed_records / n, 4), "n": changed_records, "denominator": n},
        **{
            f"field_changed.{k}": {"value": round(changed_fields[k] / n, 4), "n": changed_fields[k], "denominator": n}
            for k in FIELDS_ORDER
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cmd", choices=("export", "compare"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args(argv)
    gold_dir, out_dir = data_dir() / "gold", data_dir() / "review" / "gold"
    if args.cmd == "export":
        for split in args.splits:
            path = export(gold_dir, out_dir, split)
            print(f"{split}: {path or 'kept the existing file'}")
        return 0
    metrics: dict[str, Any] = {}
    for split in args.splits:
        summary = compare(gold_dir, out_dir, split)
        if summary is None:
            continue
        metrics.update({f"{split}.{k}": v for k, v in summary.items()})
        print(f"{split}: read {summary['records_read']['n']}, changed {summary['records_changed']['n']}")
    if not metrics:
        print("no review files", file=sys.stderr)
        return 1
    write_result("gold/full_review.json", metrics, source_script="src/radar/eval/gold_review.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
