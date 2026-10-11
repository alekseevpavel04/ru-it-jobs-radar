"""Agreement of the two labeling passes of the suspicious-vacancy sample and the final labels.

Pass files are data/suspicious/pass_{a,b}/<split>.jsonl, one {"item_id", "label", "categories",
"reason"} per card in input order (docs/suspicious_guide.md, section 8). A card both passes label the
same (label and set of categories) keeps that label; a card they differ on takes the line of
data/suspicious/adjudicated.jsonl, which must exist for every such card. The result is
data/suspicious/labels_<split>.jsonl, read by `python -m radar.models.suspicious run`, and the counts
go to results/models/suspicious_agreement.json.

Both passes are the same model following the same guide: agreement measures how consistently the
guide is applied, not how correct the labels are.

    python -m radar.models.suspicious_agreement
"""

import json
import sys
from pathlib import Path
from typing import Any

from radar.models.suspicious import work_dir
from radar.results import write_result

SPLITS = ("dev", "test")


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(a["label"] == b["label"] and sorted(a["categories"]) == sorted(b["categories"]))


def merge(
    a: list[dict[str, Any]], b: list[dict[str, Any]], adjudicated: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if [r["item_id"] for r in a] != [r["item_id"] for r in b]:
        raise ValueError("passes list different cards or a different order")
    labels, differ = [], []
    label_equal = 0
    for x, y in zip(a, b, strict=True):
        label_equal += x["label"] == y["label"]
        if same(x, y):
            labels.append({**{k: x[k] for k in ("item_id", "label", "categories")}, "provenance": "agreed"})
            continue
        differ.append({"item_id": x["item_id"], "a": [x["label"], x["categories"]], "b": [y["label"], y["categories"]]})
        if x["item_id"] not in adjudicated:
            raise ValueError(f"{x['item_id']}: passes differ and there is no adjudicated label")
        z = adjudicated[x["item_id"]]
        labels.append(
            {"item_id": z["item_id"], "label": z["label"], "categories": z["categories"], "provenance": "adjudicated"}
        )
    n = len(a)
    counts = {
        "cards": n,
        "label_equal": {"n": label_equal, "denominator": n},
        "label_and_categories_equal": {"n": n - len(differ), "denominator": n},
        "suspicious_final": {"n": sum(r["label"] == "suspicious" for r in labels), "denominator": n},
        "unsure_final": {"n": sum(r["label"] == "unsure" for r in labels), "denominator": n},
        "disagreements": differ,
    }
    return labels, counts


def main(argv: list[str] | None = None) -> int:
    root = work_dir()
    adjudicated = {r["item_id"]: r for r in read_lines(root / "adjudicated.jsonl")}
    metrics: dict[str, dict[str, Any]] = {}
    splits: dict[str, Any] = {}
    for split in SPLITS:
        labels, counts = merge(
            read_lines(root / "pass_a" / f"{split}.jsonl"), read_lines(root / "pass_b" / f"{split}.jsonl"), adjudicated
        )
        (root / f"labels_{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8"
        )
        splits[split] = counts
        for key in ("label_equal", "label_and_categories_equal", "suspicious_final", "unsure_final"):
            share = counts[key]
            metrics[f"{split}_{key}"] = {"value": round(share["n"] / share["denominator"], 4), **share}
    path = write_result(
        "models/suspicious_agreement.json",
        metrics,
        source_script="src/radar/models/suspicious_agreement.py",
        method="two independent passes of the same model by docs/suspicious_guide.md; equal means the same label "
        "and the same set of categories; differing cards take the adjudicated label",
        splits=splits,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
