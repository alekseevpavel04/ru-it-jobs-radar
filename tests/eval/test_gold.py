import json
from collections import Counter
from pathlib import Path
from typing import Any

from radar.eval.agreement import FIELDS, compare
from radar.eval.gold import build, normalize_skills


def test_skill_rules_drop_classes_prefixes_and_add_platform() -> None:
    changes: Counter[str] = Counter()
    skills = ["Apache Kafka", "VPN", "LLM", "Microsoft Excel", "1С:ERP", "ГОСТ 34", "Python", "Kafka"]
    assert normalize_skills(skills, changes) == ["1С", "Kafka", "Excel", "1С:ERP", "Python"]
    assert changes == Counter({"removed": 3, "prefix_dropped": 2, "1c_added": 1})


def test_skills_without_rules_stay_as_labeled() -> None:
    changes: Counter[str] = Counter()
    assert normalize_skills(["SQL", "Power BI"], changes) == ["SQL", "Power BI"]
    assert not changes


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_posts_excluded_by_both_passes_or_by_adjudication_leave_the_gold_set(tmp_path: Path) -> None:
    empty = {f: None for f in FIELDS} | {"skills": [], "cities": []}
    vacancy = empty | {"role": "data_scientist", "skills": ["Python"], "lang": "ru"}
    _write(
        tmp_path / "tg_dev_inputs.jsonl", [{"id": i, "source": "telegram", "input": "-"} for i in ("p1", "p2", "p3")]
    )
    a = [
        {"id": "p1", "output": vacancy, "notes": None},
        {"id": "p2", "output": empty, "notes": "exclude: multi_vacancy"},
        {"id": "p3", "output": vacancy, "notes": None},
    ]
    b = [a[0], a[1], {"id": "p3", "output": empty, "notes": "exclude: not_vacancy, курс"}]
    _write(tmp_path / "pass_a" / "tg_dev_01.jsonl", a)
    _write(tmp_path / "pass_b" / "tg_dev_01.jsonl", b)
    _write(tmp_path / "adjudicated" / "tg_dev_01.jsonl", [b[2]])

    report = build(tmp_path)
    assert list(report) == ["tg_dev"]
    assert report["tg_dev"]["records"] == 1
    assert report["tg_dev"]["sampled"] == 3
    assert report["tg_dev"]["excluded_multi_vacancy"] == 1
    assert report["tg_dev"]["excluded_not_vacancy"] == 1
    assert [json.loads(x)["id"] for x in (tmp_path / "tg_dev.jsonl").read_text(encoding="utf-8").splitlines()] == ["p1"]


def test_a_pass_that_excludes_a_post_the_other_labels_is_a_disagreement() -> None:
    empty = {f: None for f in FIELDS} | {"skills": [], "cities": []}
    a = {"p1": {"id": "p1", "output": empty, "notes": "exclude: not_vacancy"}}
    b = {"p1": {"id": "p1", "output": empty, "notes": None}}
    assert compare(a, b, 0.01)["disagreements"] == [{"id": "p1", "fields": ["exclude"]}]
