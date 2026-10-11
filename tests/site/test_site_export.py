import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from radar.config import ROOT
from radar.site import export
from radar.site.export import (
    Junior,
    Meta,
    Salary,
    SkillByRole,
    SkillTotal,
    WeeklyTelegram,
    WeeklyVacancies,
    WorkFormat,
    build,
    render,
    skills,
    weekly_telegram,
    write,
)

ROW_MODELS: dict[str, type[BaseModel]] = {
    "weekly_vacancies.json": WeeklyVacancies,
    "weekly_telegram.json": WeeklyTelegram,
    "skills.json": SkillTotal,
    "skills_by_role.json": SkillByRole,
    "work_format.json": WorkFormat,
    "junior.json": Junior,
    "salaries.json": Salary,
}


def weekly_row(week: str, source: str, bucket: str, **counts: int | None) -> dict[str, Any]:
    vacancy = source != "telegram"
    return {
        "week": week,
        "source": source,
        "role_bucket": bucket,
        "days_collected": 7 if vacancy else None,
        "complete_days": 7 if vacancy else None,
        "is_first_collected_week": 0,
        "vacancies_active": counts.get("active") if vacancy else None,
        "vacancies_new": counts.get("new") if vacancy else None,
        "vacancies_closed": counts.get("closed") if vacancy else None,
        "telegram_posts": None if vacancy else counts.get("posts"),
        "telegram_posts_before_dedup": None if vacancy else counts.get("before"),
        "telegram_channels": None if vacancy else counts.get("channels"),
    }


def skill_row(source: str, bucket: str, grade: str, skill: str, n: int) -> dict[str, Any]:
    return {
        "source": source,
        "role_bucket": bucket,
        "grade": grade,
        "skill": skill,
        "skill_category": "language",
        "vacancies_mentioning": n,
        "period_from": "2026-06-30",
        "period_to": "2026-10-10" if source != "telegram" else "2026-10-08",
    }


def junior_row(source: str, bucket: str, vacancies: int) -> dict[str, Any]:
    return {
        "source": source,
        "role_bucket": bucket,
        "vacancies": vacancies,
        "open_to_junior": 0,
        "graded_intern": 0,
        "graded_junior": 0,
        "no_experience_required": 0,
        "grade_known": 0,
        "experience_field_stated": None,
    }


@pytest.fixture
def marts() -> dict[str, list[dict[str, Any]]]:
    return {
        "mart_weekly_dynamics": [
            weekly_row("2026-07-06", "trudvsem_snapshot", "ml_ds", active=10, new=2, closed=1),
            weekly_row("2026-06-29", "trudvsem_snapshot", "backend", active=40, new=0, closed=3),
            weekly_row("2026-10-05", "habr", "ml_ds", active=5, new=5, closed=None),
            weekly_row("2024-01-01", "telegram", "ml_ds", posts=3, before=3, channels=1),
            weekly_row("2026-09-28", "telegram", "ml_ds", posts=7, before=9, channels=4),
            weekly_row("2026-10-05", "telegram", "multi_post", posts=2, before=2, channels=2),
        ],
        "mart_skills": [
            skill_row("trudvsem", "ml_ds", "junior", "Python", 4),
            skill_row("trudvsem", "ml_ds", "senior", "Python", 3),
            skill_row("trudvsem", "backend", "unknown", "Python", 2),
            skill_row("trudvsem", "backend", "unknown", "Go", 1),
            skill_row("telegram", "ml_ds", "unknown", "Python", 6),
        ],
        "mart_work_format": [
            {
                "source": "habr",
                "role_bucket": "ml_ds",
                "work_format": "remote",
                "vacancies": 3,
                "vacancies_in_bucket": 5,
            }
        ],
        "mart_junior": [
            junior_row("trudvsem", "ml_ds", 12),
            junior_row("trudvsem", "backend", 30),
            # a bucket where no skill was mentioned still counts in the source total
            junior_row("trudvsem", "design", 8),
            junior_row("telegram", "ml_ds", 9),
        ],
        "mart_salaries": [
            {
                "source": "habr",
                "role_bucket": "ml_ds",
                "grade": "senior",
                "region_group": "remote",
                "vacancies": 14,
                "with_salary_rub": 7,
                "with_salary_other_currency": 0,
                "with_from": 6,
                "with_to": 6,
                "salary_from_p25": 205000.0,
                "salary_from_median": 232500.0,
                "salary_from_p75": 263750.0,
                "salary_to_p25": None,
                "salary_to_median": None,
                "salary_to_p75": None,
            }
        ],
    }


def built(marts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return build(marts, datetime(2026, 10, 11, 9, 0, tzinfo=UTC), "abc1234")


def test_weekly_vacancies_leave_telegram_out_and_sort(marts: dict[str, list[dict[str, Any]]]) -> None:
    rows = built(marts)["weekly_vacancies.json"]
    assert [(r.source, r.week.isoformat()) for r in rows] == [
        ("habr", "2026-10-05"),
        ("trudvsem_snapshot", "2026-06-29"),
        ("trudvsem_snapshot", "2026-07-06"),
    ]
    assert rows[0].closed is None
    assert rows[1].first_collected_week is False


def test_telegram_keeps_only_the_latest_weeks(marts: dict[str, list[dict[str, Any]]]) -> None:
    rows = weekly_telegram(marts["mart_weekly_dynamics"], weeks=2)
    assert [(r.week.isoformat(), r.role_bucket) for r in rows] == [
        ("2026-09-28", "ml_ds"),
        ("2026-10-05", "multi_post"),
    ]
    assert len(built(marts)["weekly_telegram.json"]) == 2, "2024 is older than the default window"


def test_skills_add_up_grades_and_take_bucket_sizes_from_junior(marts: dict[str, list[dict[str, Any]]]) -> None:
    sizes = export.bucket_sizes(marts["mart_junior"])
    totals, by_role = skills(marts["mart_skills"], sizes, min_mentions=3)

    python_trudvsem = next(r for r in totals if r.source == "trudvsem" and r.skill == "Python")
    assert (python_trudvsem.vacancies_mentioning, python_trudvsem.vacancies_total) == (9, 12 + 30 + 8)
    assert any(r.skill == "Go" for r in totals), "per-source totals keep every skill"

    assert [(r.source, r.role_bucket, r.skill, r.vacancies_mentioning, r.vacancies_in_bucket) for r in by_role] == [
        ("telegram", "ml_ds", "Python", 6, 9),
        ("trudvsem", "ml_ds", "Python", 7, 12),
    ]


def test_more_mentions_than_vacancies_stops_the_export() -> None:
    with pytest.raises(ValueError, match="mentions of"):
        skills([skill_row("habr", "qa", "middle", "SQL", 5)], {("habr", "qa"): 4})


def test_meta_dates_and_counts(marts: dict[str, list[dict[str, Any]]]) -> None:
    docs = built(marts)
    meta: Meta = docs["meta.json"]
    assert meta.data_through == date(2026, 10, 10)
    assert meta.snapshot_first_week == date(2026, 6, 29)
    assert meta.snapshot_last_week == date(2026, 7, 6)
    assert meta.telegram_weeks_shown == 2
    assert meta.files == {name: len(rows) for name, rows in docs.items() if name != "meta.json"}
    assert set(meta.periods) == {"trudvsem", "telegram"}


def test_salary_amounts_come_out_as_integers(marts: dict[str, list[dict[str, Any]]]) -> None:
    (row,) = built(marts)["salaries.json"]
    assert row.salary_from_median == 232500
    assert json.loads(render([row]))[0]["salary_from_median"] == 232500


def test_a_field_the_contract_does_not_know_is_rejected() -> None:
    row = {"source": "habr", "role_bucket": "qa", "work_format": "remote", "vacancies": 1, "vacancies_in_bucket": 2}
    with pytest.raises(ValidationError):
        WorkFormat(**row, title="Senior QA")


def test_exported_fields_are_aggregates_only() -> None:
    """Pinned on purpose: a new field in the public data is a decision, not a side effect."""
    fields = {name: set(model.model_fields) for name, model in ROW_MODELS.items()}
    quantiles = {f"salary_{b}_{q}" for b in ("from", "to") for q in ("p25", "median", "p75")}
    assert fields == {
        "weekly_vacancies.json": {
            *("week", "source", "role_bucket", "days_collected", "complete_days", "first_collected_week"),
            *("active", "new", "closed"),
        },
        "weekly_telegram.json": {"week", "role_bucket", "posts", "posts_before_dedup", "channels"},
        "skills.json": {"source", "skill", "category", "vacancies_mentioning", "vacancies_total"},
        "skills_by_role.json": {
            *("source", "role_bucket", "skill", "category", "vacancies_mentioning", "vacancies_in_bucket"),
        },
        "work_format.json": {"source", "role_bucket", "work_format", "vacancies", "vacancies_in_bucket"},
        "junior.json": {
            *("source", "role_bucket", "vacancies", "open_to_junior", "graded_intern", "graded_junior"),
            *("no_experience_required", "grade_known", "experience_field_stated"),
        },
        "salaries.json": {
            *("source", "role_bucket", "grade", "region_group", "vacancies", "with_salary_rub"),
            *("with_salary_other_currency", "with_from", "with_to"),
            *quantiles,
        },
    }


def test_files_are_one_row_per_line_and_rewritten_whole(marts: dict[str, list[dict[str, Any]]], tmp_path: Path) -> None:
    docs = built(marts)
    sizes = write(docs, tmp_path)
    assert set(sizes) == set(docs)
    lines = (tmp_path / "weekly_vacancies.json").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[" and lines[-1] == "]" and len(lines) == 2 + len(docs["weekly_vacancies.json"])
    assert write(docs, tmp_path) == sizes
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["commit"] == "abc1234"


SITE_DATA = ROOT / "site" / "src" / "data"


@pytest.mark.skipif(not (SITE_DATA / "meta.json").exists(), reason="no export in site/src/data yet")
def test_committed_site_data_matches_the_contract() -> None:
    meta = Meta.model_validate_json((SITE_DATA / "meta.json").read_text(encoding="utf-8"))
    assert set(meta.files) == set(ROW_MODELS)
    for name, model in ROW_MODELS.items():
        rows = json.loads((SITE_DATA / name).read_text(encoding="utf-8"))
        assert len(rows) == meta.files[name], name
        for row in rows:
            model.model_validate(row)


def test_site_texts_have_no_long_dash() -> None:
    files = [ROOT / "site" / "observablehq.config.js", *(ROOT / "site" / "src").rglob("*.md")]
    files += (ROOT / "site" / "src" / "components").rglob("*.js")
    offenders = [p.name for p in files if "—" in p.read_text(encoding="utf-8")]
    assert offenders == []
