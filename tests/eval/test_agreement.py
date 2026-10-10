import json
from pathlib import Path

from radar.eval.agreement import compare, load_pass

FIELDS = {
    "role": "data_analyst",
    "grade": None,
    "skills": ["SQL", "Python"],
    "salary": None,
    "work_format": None,
    "cities": ["Москва"],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}


def _write(directory: Path, name: str, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_list_order_and_case_do_not_count_as_disagreement(tmp_path: Path) -> None:
    _write(tmp_path / "a", "test_01.jsonl", [{"id": "x", "output": FIELDS, "notes": None}])
    swapped = {**FIELDS, "skills": ["python", "SQL"]}
    _write(tmp_path / "b", "test_01.jsonl", [{"id": "x", "output": swapped, "notes": None}])
    result = compare(load_pass(tmp_path / "a", "test"), load_pass(tmp_path / "b", "test"), 0.01)
    assert result["all_fields_equal"] == {"n": 1, "denominator": 1}
    assert result["disagreements"] == []


def test_differing_fields_and_invalid_outputs_are_reported(tmp_path: Path) -> None:
    other = {**FIELDS, "role": "data_engineer"}
    _write(tmp_path / "a", "test_01.jsonl", [{"id": "x", "output": FIELDS}, {"id": "y", "output": FIELDS}])
    _write(tmp_path / "b", "test_01.jsonl", [{"id": "x", "output": other}, {"id": "y", "output": {"role": "?"}}])
    result = compare(load_pass(tmp_path / "a", "test"), load_pass(tmp_path / "b", "test"), 0.01)
    assert result["invalid"] == {"a": 0, "b": 1}
    assert result["disagreements"] == [{"id": "x", "fields": ["role"]}]
    assert result["exact"]["role"] == {"n": 0, "denominator": 1}
