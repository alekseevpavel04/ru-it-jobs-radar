import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from radar.config import ROOT
from radar.extract import schema
from radar.extract.schema import ROLES, VacancyFields, dump, json_schema, parse, try_parse


def full() -> dict[str, Any]:
    return {
        "role": "data_analyst",
        "grade": "senior",
        "skills": ["SQL", "Power BI"],
        "salary": {"from": 140000, "to": 180000, "currency": "RUB", "gross_net": "net", "period": "month"},
        "work_format": "hybrid",
        "cities": ["Москва"],
        "experience_years": {"min": 3, "max": None},
        "employment_type": "labor_contract",
        "lang": "ru",
    }


def nothing() -> dict[str, Any]:
    return {
        "role": None,
        "grade": None,
        "skills": [],
        "salary": None,
        "work_format": None,
        "cities": [],
        "experience_years": None,
        "employment_type": None,
        "lang": "en",
    }


def test_full_answer_parses_and_round_trips() -> None:
    fields = parse(json.dumps(full(), ensure_ascii=False))
    assert fields.salary is not None and fields.salary.from_ == 140000
    assert dump(fields) == full()


def test_all_null_answer_is_valid() -> None:
    assert parse(json.dumps(nothing())).role is None


def test_whitespace_in_items_is_stripped() -> None:
    doc = full() | {"skills": ["  Python "]}
    assert parse(json.dumps(doc)).skills == ["Python"]


def _with(path: str, value: Any) -> dict[str, Any]:
    doc = full()
    *parents, last = path.split(".")
    node = doc
    for p in parents:
        node = node[p]
    if value is _DROP:
        del node[last]
    else:
        node[last] = value
    return doc


_DROP = object()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("role", _DROP),  # every key is required, null must be explicit
        ("lang", None),
        ("role", "prompt_engineer"),
        ("grade", "principal"),
        ("extra_field", 1),
        ("skills", "Python, SQL"),
        ("skills", [""]),
        ("skills", ["x" * 81]),
        ("skills", [f"s{i}" for i in range(61)]),
        ("salary.from", 0),  # zero is not a salary, null is
        ("salary.from", 200000),  # from > to
        ("salary.currency", "rub"),
        ("salary.gross_net", "unknown"),
        ("salary.period", "week"),
        ("salary.bonus", 1),
        ("experience_years.min", -1),
        ("experience_years.max", 1),  # min 3 > max 1
        ("work_format", "anywhere"),
        ("employment_type", "contract"),
    ],
)
def test_invalid_answers_are_rejected(path: str, value: Any) -> None:
    text = json.dumps(_with(path, value), ensure_ascii=False)
    with pytest.raises(ValidationError):
        parse(text)
    assert try_parse(text) is None


def test_empty_salary_and_experience_objects_are_rejected() -> None:
    no_amount = full() | {"salary": {"from": None, "to": None, "currency": "RUB", "gross_net": None, "period": None}}
    no_years = full() | {"experience_years": {"min": None, "max": None}}
    assert try_parse(json.dumps(no_amount)) is None
    assert try_parse(json.dumps(no_years)) is None


@pytest.mark.parametrize("text", ["", "not json", '{"role": "backend"', "[]", "null", None])
def test_broken_json_is_invalid(text: str | None) -> None:
    assert try_parse(text) is None


def test_json_schema_requires_every_key_and_forbids_extra() -> None:
    s = json_schema()
    assert set(s["required"]) == set(VacancyFields.model_fields) - {"from_"} == set(s["properties"])
    assert s["additionalProperties"] is False
    assert s["$defs"]["Salary"]["required"] == ["from", "to", "currency", "gross_net", "period"]
    role_enum = next(option["enum"] for option in s["properties"]["role"]["anyOf"] if "enum" in option)
    assert tuple(role_enum) == ROLES


def test_committed_schema_file_matches_code() -> None:
    committed = (ROOT / "configs" / "extraction_schema.json").read_text(encoding="utf-8")
    assert committed == schema.schema_text(), "run: python -m radar.extract.schema --json-schema --out ..."


def test_cli_writes_schema(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    assert schema.main(["--json-schema", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json_schema()
