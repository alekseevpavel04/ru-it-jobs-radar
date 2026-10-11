"""Agreement between two independent labeling passes of the gold set.

Each pass is a directory of `<split>_NN.jsonl` batches with lines {"id", "output", "notes"}. Per
split and field it reports exact agreement (n of denominator) and the field F1 of pass B against
pass A, which is symmetric for lists and categorical slots. Records where the passes differ in any
field are written to `data/gold/disagreements_<split>.jsonl` for the adjudication pass.

Both passes are the same model following the same guide, so agreement measures how consistently the
guide is applied, not how correct the labels are.

    python -m radar.eval.agreement [--a data/gold/pass_a] [--b data/gold/pass_b]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radar.config import data_dir
from radar.eval.cli import load_config
from radar.eval.metrics import score
from radar.extract.schema import VacancyFields
from radar.results import write_result

FIELDS = tuple(VacancyFields.model_fields)
SPLITS = ("test", "dev", "tg_test", "tg_dev")
EXCLUDE = "exclude:"


def load_pass(directory: Path, split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for batch in sorted(directory.glob(f"{split}_*.jsonl")):
        for line in batch.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def parse(output: Any) -> VacancyFields | None:
    try:
        return VacancyFields.model_validate(output)
    except ValidationError:
        return None


def excluded(row: dict[str, Any]) -> str | None:
    """The reason a pass left a Telegram post out of the gold set (rule 6 of the guide), from its notes."""
    notes = (row.get("notes") or "").strip()
    if not notes.startswith(EXCLUDE):
        return None
    return notes[len(EXCLUDE) :].split()[0].strip(" ,;.") if notes[len(EXCLUDE) :].strip() else "unspecified"


def canonical(value: Any) -> Any:
    """A field value in a form where order of list items and case do not matter."""
    if isinstance(value, list):
        return sorted(str(v).strip().lower() for v in value)
    return value


def field_values(fields: VacancyFields) -> dict[str, Any]:
    dumped = fields.model_dump(mode="json")
    return {k: canonical(dumped[k]) for k in FIELDS}


def compare(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]], salary_rel_tol: float) -> dict[str, Any]:
    """Posts a pass left out by rule 6 are compared by their exclusion only: their fields are empty
    and need not be valid. A post one pass left out and the other labeled is a disagreement."""
    ids = sorted(set(a) & set(b))
    disagreements = []
    labeled = []
    excluded_by_both = 0
    for i in ids:
        ea, eb = excluded(a[i]), excluded(b[i])
        if ea is None and eb is None:
            labeled.append(i)
        elif ea == eb:
            excluded_by_both += 1
        else:
            disagreements.append({"id": i, "fields": ["exclude"]})
    parsed_a = [parse(a[i]["output"]) for i in labeled]
    parsed_b = [parse(b[i]["output"]) for i in labeled]
    both = [(i, x, y) for i, x, y in zip(labeled, parsed_a, parsed_b, strict=True) if x is not None and y is not None]
    exact = dict.fromkeys(FIELDS, 0)
    for record_id, x, y in both:
        va, vb = field_values(x), field_values(y)
        differing = [f for f in FIELDS if va[f] != vb[f]]
        for f in FIELDS:
            exact[f] += f not in differing
        if differing:
            disagreements.append({"id": record_id, "fields": differing})
    stats = score([x for _, x, _ in both], [y for _, _, y in both], salary_rel_tol)
    f1 = {k: round(s.value(), 4) for k, s in stats.items() if k.endswith("_f1") or k == "macro_f1"}
    n = len(both)
    exclusion_differs = sum(1 for d in disagreements if d["fields"] == ["exclude"])
    return {
        "records": {"a": len(a), "b": len(b), "both": len(ids)},
        "invalid": {"a": parsed_a.count(None), "b": parsed_b.count(None)},
        "exact": {f: {"n": exact[f], "denominator": n} for f in FIELDS},
        "all_fields_equal": {"n": n - (len(disagreements) - exclusion_differs), "denominator": n},
        "excluded_by_both": {"n": excluded_by_both, "denominator": len(ids)},
        "exclusion_differs": {"n": exclusion_differs, "denominator": len(ids)},
        "f1_b_against_a": f1,
        "disagreements": disagreements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    gold = data_dir() / "gold"
    parser.add_argument("--a", type=Path, default=gold / "pass_a")
    parser.add_argument("--b", type=Path, default=gold / "pass_b")
    args = parser.parse_args(argv)
    cfg = load_config()
    splits: dict[str, Any] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        pass_a, pass_b = load_pass(args.a, split), load_pass(args.b, split)
        if not pass_a or not pass_b:
            print(f"{split}: no labeling passes yet, skipped")
            continue
        result = compare(pass_a, pass_b, cfg.salary_rel_tol)
        disagreements = result.pop("disagreements")
        out = gold / f"disagreements_{split}.jsonl"
        out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in disagreements), encoding="utf-8")
        splits[split] = result
        equal = result["all_fields_equal"]
        if equal["denominator"]:
            metrics[f"{split}_all_fields_equal"] = {
                "value": round(equal["n"] / equal["denominator"], 4),
                "n": equal["n"],
                "denominator": equal["denominator"],
            }
        for field, share in result["exact"].items():
            if share["denominator"]:
                metrics[f"{split}_{field}_exact"] = {
                    "value": round(share["n"] / share["denominator"], 4),
                    **share,
                }
        print(f"{split}: all fields equal {equal['n']} of {equal['denominator']}, {len(disagreements)} to adjudicate")
    path = write_result(
        "gold/agreement.json",
        metrics,
        source_script="src/radar/eval/agreement.py",
        method="two independent passes of the same model by docs/annotation_guide.md; exact agreement per field "
        "after ordering lists, F1 of pass B against pass A by radar.eval.metrics",
        splits=splits,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
