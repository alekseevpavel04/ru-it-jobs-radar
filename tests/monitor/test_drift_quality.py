import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from radar.eval.cli import EvalConfig
from radar.extract.schema import VacancyFields
from radar.monitor import drift_quality
from radar.monitor.drift_quality import (
    IDENTITIES,
    REPORTED,
    cache_key,
    cache_row,
    check_identities,
    difference,
    draw,
    evaluate,
    read_cache,
    student_fields,
    teacher_client,
    window_stats,
)
from radar.results import Metric

CFG = EvalConfig(
    n_resamples=1000,
    seed=1,
    confidence=0.95,
    salary_rel_tol=0.01,
    macro_fields=("role", "grade", "skills", "salary", "work_format", "cities", "experience_years", "employment_type"),
    skill_normalization="casefold",
)


def student_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "valid": True,
        "role": "backend",
        "grade": "middle",
        "skills": ["Python", "PostgreSQL"],
        "salary_from": 150000,
        "salary_to": None,
        "salary_currency": "RUB",
        "salary_gross_net": None,
        "salary_period": "month",
        "work_format": "remote",
        "cities": ["Москва"],
        "experience_min": 3.0,
        "experience_max": None,
        "employment_type": None,
        "lang": "ru",
    }
    return row | overrides


def teacher_row(fields: VacancyFields | None, **overrides: Any) -> dict[str, Any]:
    output = None if fields is None else fields.model_dump_json(by_alias=True)
    return {"output": output, "valid": fields is not None, "exclude": None, "mismatch": False} | overrides


def test_draw_is_fixed_by_the_seed_and_not_by_the_order() -> None:
    keys = [("habr", str(i), "h") for i in range(500)]
    first = draw(keys, 20)
    assert len(first) == 20
    assert draw(list(reversed(keys)), 20) == first
    assert draw(keys, 20, seed=2) != first
    # The order is the ranking, so a smaller sample is the head of a larger one.
    assert draw(keys, 50)[:20] == first
    assert draw(keys[:5], 20) == draw(list(reversed(keys[:5])), 20)
    assert len(draw(keys[:5], 20)) == 5


def test_student_row_becomes_the_schema_object() -> None:
    fields = student_fields(student_row())
    assert fields is not None
    assert fields.salary is not None and fields.salary.from_ == 150000 and fields.salary.to is None
    assert fields.experience_years is not None and fields.experience_years.min == 3.0
    bare = student_fields(student_row(salary_from=None, salary_currency=None, experience_min=None))
    assert bare is not None and bare.salary is None and bare.experience_years is None
    assert student_fields(student_row(valid=False)) is None


def test_agreement_counts_what_it_leaves_out() -> None:
    same = student_fields(student_row())
    other_role = student_fields(student_row(role="frontend"))
    assert same is not None and other_role is not None
    pairs = [
        (teacher_row(same), same),
        (teacher_row(same), other_role),
        (teacher_row(same), None),  # the student's answer was invalid: scored as empty
        (teacher_row(None), same),  # the teacher's answer was invalid: left out
        (teacher_row(same, exclude="multi_vacancy"), same),
        (teacher_row(None, mismatch=True), same),
    ]
    stats, counts = window_stats(pairs, CFG)

    assert counts == {
        "sample": 6,
        "input_mismatch": 1,
        "teacher_excluded_multi_vacancy": 1,
        "teacher_invalid": 1,
        "scored": 3,
    }
    assert stats is not None
    assert stats["role_accuracy"].numerator_denominator() == (1, 3)
    assert stats["valid_json"].numerator_denominator() == (2, 3)
    assert stats["work_format_accuracy"].numerator_denominator() == (2, 3)
    assert stats["skills_f1"].value() == pytest.approx(2 * 4 / (4 + 6))


def test_identical_windows_differ_by_zero() -> None:
    fields = student_fields(student_row())
    other = student_fields(student_row(role="qa", grade=None))
    pairs = [(teacher_row(fields), fields)] * 30 + [(teacher_row(fields), other)] * 10
    stats, _ = window_stats(pairs, CFG)
    assert stats is not None
    diff = difference(stats, stats, CFG)
    assert diff["role_accuracy"]["value"] == 0.0
    assert diff["role_accuracy"]["ci_low"] < 0 < diff["role_accuracy"]["ci_high"]


def drawn_and_students() -> tuple[dict[str, dict[str, list[Any]]], dict[Any, dict[str, Any]]]:
    keys = {w: [("telegram", f"{w}{i}", "h") for i in range(4)] for w in ("reference", "current")}
    students = {k: student_row(input_sha256=f"sha-{k[1]}") for ks in keys.values() for k in ks}
    return {"portals": {"reference": [], "current": []}, "telegram": keys}, students


def full_cache(students: Mapping[Any, Mapping[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    identity = IDENTITIES["telegram"]
    out = {}
    for row in students.values():
        fields = student_fields(row)
        out[cache_key(row["input_sha256"], identity)] = teacher_row(fields)
    return out


def test_without_the_teacher_agreement_is_absent_not_zero() -> None:
    drawn, students = drawn_and_students()
    cache = full_cache(students)
    # The current week's texts were never labeled; the reference ones are in the cache.
    for k in drawn["telegram"]["current"]:
        del cache[cache_key(students[k]["input_sha256"], IDENTITIES["telegram"])]
    metrics, groups = evaluate(drawn, students, cache, CFG, "unreachable")

    current = groups["telegram"]["current"]
    assert current["status"] == "teacher_unreachable"
    assert current["teacher_answers"] == {"n": 0, "denominator": 4}
    assert "agreement" not in current
    assert not any(name.startswith("telegram.current.") for name in metrics)
    assert groups["telegram"]["reference"]["status"] == "ok"
    assert metrics["telegram.reference.role_accuracy"]["n"] == 4
    assert "current_minus_reference" not in groups["telegram"]
    assert groups["portals"]["current"]["status"] == "no_window"


def test_a_covered_sample_is_scored_in_both_windows() -> None:
    drawn, students = drawn_and_students()
    metrics, groups = evaluate(drawn, students, full_cache(students), CFG, "skipped")

    assert {groups["telegram"][w]["status"] for w in ("reference", "current")} == {"ok"}
    assert set(groups["telegram"]["current_minus_reference"]) <= set(REPORTED)
    assert metrics["telegram.current.macro_f1"]["value"] == 1.0
    for value in metrics.values():
        Metric.model_validate(value)


def test_no_server_means_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(client: httpx.Client, teacher: Any) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(drift_quality, "check_model", refuse)
    assert teacher_client("http://127.0.0.1:8096/v1") is None


def test_cache_keeps_answers_and_drops_failed_requests(tmp_path: Path) -> None:
    identity = IDENTITIES["portals"]
    fields = student_fields(student_row())
    assert fields is not None
    answered = cache_row(identity, "a", "habr:1", {"output": fields.model_dump_json(by_alias=True), "valid": True})
    failed = cache_row(identity, "b", "habr:2", {"error": "HTTP 500"})
    mismatch = cache_row(identity, "c", "habr:3", {"mismatch": True})
    path = tmp_path / "qwen.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (answered, failed, mismatch)), encoding="utf-8")

    cache = read_cache(path)
    assert set(cache) == {cache_key("a", identity), cache_key("c", identity)}
    # Another prompt version is another teacher: its answers are not reused.
    assert cache_key("a", IDENTITIES["telegram"]) not in cache


def test_pinned_teacher_matches_the_request_schema() -> None:
    check_identities()
