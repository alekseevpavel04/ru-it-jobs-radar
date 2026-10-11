"""Build the gold set `data/gold/<split>.jsonl` from two labeling passes and the adjudication pass.

A record where both passes agree takes their labels; a record they disagree on takes the
adjudicated labels. Then the skill rules of docs/annotation_guide.md that need no reading of the
text (cases 48, 55, 57, 58, 61, 62) are applied to every record, because the guide gained them while
labeling was under way and agreed records were labeled before some of them existed. Every change is
counted in `results/gold/build.json`.

    python -m radar.eval.gold
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from radar.config import data_dir
from radar.eval.agreement import SPLITS, excluded, field_values, load_pass, parse
from radar.extract.schema import VacancyFields
from radar.results import write_result

# Areas, classes of systems and models, network technology classes, prompting tricks, document types
# and numbered standards: not skills by cases 35, 48, 55, 57.
NOT_SKILLS = frozenset(
    s.casefold()
    for s in (
        "LLM",
        "NLP",
        "CV",
        "Computer Vision",
        "Machine Learning",
        "ML",
        "Deep Learning",
        "DL",
        "AI",
        "ИИ",
        "Data Science",
        "NoSQL",
        "ERP",
        "CRM",
        "SCADA",
        "SIEM",
        "DLP",
        "MES",
        "BRD",
        "FSD",
        "SRS",
        "VPN",
        "VLAN",
        "VoIP",
        "QoS",
        "Wi-Fi",
        "OSI",
        "zero-shot",
        "few-shot",
        "CoT",
        "Chain-of-Thought",
        "квантизация",
        "Quantization",
        "CVSS",
        "CWE",
    )
)
NOT_SKILL_PREFIXES = (
    "гост",
    "iso ",
    "iso/",
    "152-фз",
    "44-фз",
    "223-фз",
    "ccna",
    "ccnp",
    "rhcsa",
    "rhce",
    "aws certified",
)
OFFICE = frozenset({"Excel", "Word", "PowerPoint", "Visio", "Project", "Access", "Outlook", "Teams"})


def normalize_skills(skills: list[str], changes: Counter[str]) -> list[str]:
    out: list[str] = []
    for skill in skills:
        name = skill.strip()
        key = name.casefold()
        if key in NOT_SKILLS or key.startswith(NOT_SKILL_PREFIXES):
            changes["removed"] += 1
            continue
        if name.startswith("Apache ") and len(name) > len("Apache "):
            name = name[len("Apache ") :]
            changes["prefix_dropped"] += 1
        elif name.startswith("Microsoft ") and name[len("Microsoft ") :] in OFFICE:
            name = name[len("Microsoft ") :]
            changes["prefix_dropped"] += 1
        if name not in out:
            out.append(name)
    if any(s.startswith("1С:") for s in out) and "1С" not in out:
        out.insert(0, "1С")
        changes["1c_added"] += 1
    return out


def load_adjudicated(directory: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for batch in sorted(directory.glob("*.jsonl")):
        for line in batch.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def build(gold: Path) -> dict[str, Any]:
    adjudicated = load_adjudicated(gold / "adjudicated")
    report: dict[str, Any] = {}
    for split in SPLITS:
        a, b = load_pass(gold / "pass_a", split), load_pass(gold / "pass_b", split)
        if not a or not b:
            continue
        inputs = [json.loads(x) for x in (gold / f"{split}_inputs.jsonl").read_text(encoding="utf-8").splitlines() if x]
        counts: Counter[str] = Counter()
        changes: Counter[str] = Counter()
        lines = []
        for record in inputs:
            rid = record["id"]
            if rid in adjudicated:
                row, provenance = adjudicated[rid], "adjudicated"
            elif excluded(a[rid]) is not None and excluded(a[rid]) == excluded(b[rid]):
                row, provenance = a[rid], "agreed"
            else:
                fa, fb = parse(a[rid]["output"]), parse(b[rid]["output"])
                if (
                    fa is None
                    or fb is None
                    or field_values(fa) != field_values(fb)
                    or excluded(a[rid]) != excluded(b[rid])
                ):
                    raise ValueError(f"{split}/{rid}: passes disagree and there is no adjudicated label")
                row, provenance = a[rid], "agreed"
            reason = excluded(row)
            if reason is not None:
                # Rule 6: a Telegram post that is not one vacancy leaves the gold set and is counted.
                counts[f"excluded_{reason}"] += 1
                continue
            fields = parse(row["output"])
            if fields is None:
                raise ValueError(f"{split}/{rid}: adjudicated output is not valid against the schema")
            before = list(fields.skills)
            skills = normalize_skills(before, changes)
            if skills != before:
                counts["skills_changed_by_rules"] += 1
            final = VacancyFields.model_validate({**fields.model_dump(mode="json"), "skills": skills})
            counts[provenance] += 1
            lines.append(
                json.dumps(
                    {
                        "id": rid,
                        "source": record.get("source"),
                        "fields": final.model_dump(mode="json"),
                        "provenance": provenance,
                    },
                    ensure_ascii=False,
                )
            )
        (gold / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report[split] = {"records": len(lines), "sampled": len(inputs), **counts, "skill_changes": dict(changes)}
    return report


def main(argv: list[str] | None = None) -> int:
    report = build(data_dir() / "gold")
    metrics: dict[str, dict[str, Any]] = {}
    for split, r in report.items():
        metrics[f"{split}_records"] = {"value": r["records"]}
        metrics[f"{split}_adjudicated"] = {
            "value": round(r.get("adjudicated", 0) / r["records"], 4),
            "n": r.get("adjudicated", 0),
            "denominator": r["records"],
        }
        excluded_n = sum(v for k, v in r.items() if k.startswith("excluded_"))
        metrics[f"{split}_excluded"] = {
            "value": round(excluded_n / r["sampled"], 4),
            "n": excluded_n,
            "denominator": r["sampled"],
        }
        metrics[f"{split}_skills_changed_by_rules"] = {
            "value": round(r.get("skills_changed_by_rules", 0) / r["records"], 4),
            "n": r.get("skills_changed_by_rules", 0),
            "denominator": r["records"],
        }
    path = write_result(
        "gold/build.json",
        metrics,
        source_script="src/radar/eval/gold.py",
        method="agreed labels of two passes, adjudicated labels where they disagree, skill rules applied to all",
        splits=report,
    )
    print(json.dumps(report, ensure_ascii=False))
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
