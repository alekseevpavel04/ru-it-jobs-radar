import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from radar.config import ROOT
from radar.models.forecast import SiteForecast
from radar.site import export
from radar.site.export import (
    HISTORY,
    MARKET_FACTS,
    SKILL_EFFECTS,
    SKILL_TRENDS,
    ExtractedSalary,
    Forecast,
    ForecastHistory,
    Junior,
    Lifetime,
    LifetimeObserved,
    Meta,
    Salary,
    SkillByRole,
    SkillEffects,
    SkillJunior,
    SkillTotal,
    SkillTrends,
    TelegramYears,
    VacancyHistory,
    WeeklyTelegram,
    WeeklyVacancies,
    WorkFormat,
    build,
    forecast_files,
    lifetime,
    median_ci_coverage,
    render,
    salary_min_n,
    skill_effects,
    skill_trends,
    skills,
    skills_junior,
    telegram_years,
    vacancy_history,
    weekly_telegram,
    write,
)

ROW_MODELS: dict[str, type[BaseModel]] = {
    "weekly_vacancies.json": WeeklyVacancies,
    "weekly_telegram.json": WeeklyTelegram,
    "skills.json": SkillTotal,
    "skills_by_role.json": SkillByRole,
    "skills_junior.json": SkillJunior,
    "work_format.json": WorkFormat,
    "junior.json": Junior,
    "salaries.json": Salary,
    "salaries_telegram.json": ExtractedSalary,
    "lifetime_observed.json": LifetimeObserved,
    "forecast_history.json": ForecastHistory,
}
DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    "meta.json": Meta,
    "lifetime.json": Lifetime,
    "forecast.json": Forecast,
    "telegram_years.json": TelegramYears,
    "skill_effects.json": SkillEffects,
    "vacancy_history.json": VacancyHistory,
    "skill_trends.json": SkillTrends,
}
DOCUMENTS = tuple(DOCUMENT_MODELS)


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
        "telegram_channels_week": None if vacancy else counts.get("channels_week", counts.get("channels")),
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
        "open_to_junior": 5,
        "graded_intern": 1,
        "graded_junior": 4,
        "no_experience_required": 0,
        "grade_known": 0,
        "experience_field_stated": None,
    }


def extracted_salary_row(source: str, level: str, role: str, grade: str) -> dict[str, Any]:
    return {
        "source": source,
        "level": level,
        "role": role,
        "grade": grade,
        "vacancies": 829,
        "with_salary": 170,
        "with_salary_rub_month": 159,
        "with_from": 146,
        "with_to": 115,
        "salary_from_p25": 60000.0,
        "salary_from_median": 80000.0,
        "salary_from_p75": 100000.0,
        "salary_to_p25": 90000.0,
        "salary_to_median": 120000.0,
        "salary_to_p75": 150000.0,
        "period_from": "2026-07-06",
        "period_to": "2026-10-13",
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
                "level": "cell",
                "role_group": "ml_ds_data",
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
        "mart_extracted_salaries": [
            extracted_salary_row("telegram", "grade", "all", "junior"),
            # portal rows of the extracted mart stay out: the site shows the portals' own salary fields
            extracted_salary_row("habr", "grade", "all", "junior"),
        ],
        "mart_lifetime": [
            {
                "source": "trudvsem_snapshot",
                "level": "source",
                "role_group": "all",
                "vacancies": 120,
                "still_open": 60,
                "closed": 60,
                "closed_in_gap": 0,
                "censored": 60,
                "left_censored": 50,
                "closed_by_repost": 5,
                "closed_in_quantiles": 30,
                "duration_days_p25": 11.0,
                "duration_days_median": 28.0,
                "duration_days_p75": 31.0,
                "duration_days_p90": 45.8,
                "period_from": "2026-06-30",
                "period_to": "2026-10-10",
                "window_days": 103,
            }
        ],
    }


def median_metric(
    value: float | str, low: float | None, high: float | None, subjects: int, events: int
) -> dict[str, Any]:
    return {
        "value": value,
        "n": None,
        "denominator": None,
        "low": low,
        "high": high,
        "subjects": subjects,
        "events": events,
    }


SURVIVAL: dict[str, Any] = {
    "source_script": "src/radar/models/survival.py",
    "command": "python -m radar.models.survival",
    "commit": "a002729",
    "created_at": "2026-10-11T10:48:46+00:00",
    "definitions": {"event": "no open sighting on 7 consecutive collected snapshot days; closed on the first of them"},
    "window": {"first_dt": "2026-06-30", "last_dt": "2026-09-10"},
    "limits_ru": ["Закрытие это исчезновение из выгрузки, а не найм."],
    "metrics": {
        "median_days_overall": {
            **median_metric(32.0, 32.0, 33.0, 9560, 4606),
            "at_risk": {"0": 1475, "7": 3009, "30": 2302},
            "survival": {"7": 0.9159, "30": 0.6948},
        },
        "median_days_incident_from_first_sighting": median_metric(32.0, 31.0, 36.0, 4279, 1657),
        "median_days_grace_14": median_metric(41.0, 36.0, 44.0, 9560, 3822),
        "median_days_role_ml_ds_data": median_metric(23.0, 19.0, 27.0, 199, 125),
        "median_days_incident_experience_6+": median_metric(58.0, 32.0, None, 155, 46),
        "median_days_salary_not_given": {
            "value": "not estimable",
            "n": None,
            "denominator": None,
            "reason": "a risk set empties",
            "subjects": 229,
            "events": 156,
        },
        "hr_role=ml_ds_data": {"value": 1.99, "n": None, "denominator": None, "low": 1.4, "high": 2.8},
    },
}


FORECAST_RESULT: dict[str, Any] = {
    "source_script": "src/radar/models/forecast.py",
    "commit": "6d319c5",
    "created_at": "2026-10-13T20:52:29+00:00",
    "limits_ru": ["Навыки извлечены моделью из текста."],
}


def site_forecast(n_weeks: int = 6) -> SiteForecast:
    weeks = [date(2026, 8, 31) + i * (date(2026, 9, 7) - date(2026, 8, 31)) for i in range(n_weeks)]
    return SiteForecast(
        method="ets",
        weeks=weeks,
        skills=["Python", "SQL"],
        mentions=np.array([[10.0 + t for t in range(n_weeks)], [5.0] * n_weeks]),
        it_units=np.array([100.0] * n_weeks),
        forecast_weeks=[weeks[-1] + (weeks[1] - weeks[0]) * h for h in (1, 2)],
        point=np.array([[0.151234, 0.151234], [0.05, 0.05]]),
        low=np.array([[0.12, 0.11], [0.0, 0.0]]),
        high=np.array([[0.18, 0.19], [0.09, 0.1]]),
        origins=3,
        interval_quantile=0.8,
        interval_covered=7,
        interval_cells=10,
    )


PROVENANCE = {"source_script": "src/x.py", "commit": "1234567", "created_at": "2026-10-14T12:32:24+00:00"}


def share_metric(n: int, denominator: int) -> dict[str, Any]:
    return {"value": round(n / denominator, 4), "n": n, "denominator": denominator}


MARKET_RESULT: dict[str, Any] = {
    **PROVENANCE,
    "ml_roles": ["data_scientist", "ml_engineer", "mlops"],
    "panel_years": [2025, 2026],
    "salary_years": [2025, 2026],
    "metrics": {
        "telegram_recent_salary.ml.junior.stated": share_metric(52, 223),
        "telegram_recent_salary.ml.junior.from_median": {"value": 100000, "rows": 51},
        "telegram_recent_salary.ml.intern.stated": share_metric(8, 221),
        "telegram_panel.channels": share_metric(39, 58),
        "telegram_year.last_week": {"value": "2026-10-12"},
        **{
            f"{prefix}_year.{year}.{name}": metric
            for prefix in ("telegram", "telegram_panel")
            for year in (2025, 2026)
            for name, metric in (
                ("it_posts", {"value": 18606}),
                ("ml_posts", share_metric(1433, 18606)),
                ("junior_posts", share_metric(3215, 18606)),
                ("junior_of_graded", share_metric(3215, 12539)),
                ("ml_junior_of_graded", share_metric(208, 997)),
            )
        },
        "telegram_ml_skill.2026.RAG": share_metric(206, 1433),
        "telegram_panel_ml_skill.2026.RAG": share_metric(150, 1000),
        "telegram_work_format.2026.junior.remote": share_metric(1249, 2301),
        "telegram_ml_work_format.2026.other.remote": share_metric(439, 694),
        "ml_ds.habr.vacancies": {"value": 54},
    },
}


def effect(skill: str, rows: int, naive: float, pct: float, p_holm: float) -> dict[str, Any]:
    return {
        "value": 0.0,
        "skill": skill,
        "rows_with_skill": rows,
        "naive_pct": naive,
        "pct": pct,
        "pct_ci_low": pct - 10,
        "pct_ci_high": pct + 10,
        "p_holm": p_holm,
        "se": 0.07,
    }


EFFECTS_RESULT: dict[str, Any] = {
    **PROVENANCE,
    "scenario": {"cf_y": 0.03, "cf_d": 0.03, "level": 0.95},
    "limits_ru": ["Оценка описывает вакансии с опубликованной зарплатой."],
    "metrics": {
        "data_rows": {"value": 7429},
        "treatments": {"value": 2},
        "estimated": share_metric(2, 2),
        "significant_holm": share_metric(1, 2),
        "overlap.Kubernetes": share_metric(372, 7429),
        "effect.Kubernetes": effect("Kubernetes", 162, 106.1, -5.4, 1.0),
        "effect.Microservices": effect("Microservices", 127, 135.3, 24.2, 0.02),
    },
}

HISTORY_RESULT: dict[str, Any] = {
    **PROVENANCE,
    "metrics": {
        "vacancies_tracked_trudvsem_snapshot": {"value": 25992},
        "closed_trudvsem_snapshot": share_metric(15173, 25992),
        "reopened_trudvsem_snapshot": share_metric(2256, 25992),
        "reposts_trudvsem_snapshot": share_metric(2820, 15173),
        "closed_habr": share_metric(45, 1180),
    },
}

TRENDS_RESULT: dict[str, Any] = {
    **PROVENANCE,
    "recent_years": [2025, 2026],
    "base_year": 2022,
    "min_mentions": 20,
    "min_year_posts": 300,
    "metrics": {
        "posts.ml_all.recent": {"value": 3313},
        "posts.ml_all.2025": {"value": 1880},
        "share.ml_all.CI/CD.recent": {**share_metric(305, 3313), "own": True},
        "share.ml_all.CI/CD.2025": share_metric(170, 1880),
        "posts.ml_junior.recent": {"value": 444},
        "share.ml_junior.A/B-тесты.recent": share_metric(27, 444),
    },
}

RESULTS = {
    MARKET_FACTS: MARKET_RESULT,
    SKILL_EFFECTS: EFFECTS_RESULT,
    HISTORY: HISTORY_RESULT,
    SKILL_TRENDS: TRENDS_RESULT,
}


def built(marts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    doc, history = forecast_files(FORECAST_RESULT, site_forecast(), "results/models/forecast.json", weeks=4)
    return build(
        marts,
        SURVIVAL,
        datetime(2026, 10, 11, 9, 0, tzinfo=UTC),
        "abc1234",
        forecast_doc=doc,
        forecast_history=history,
        results=RESULTS,
    )


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
    documents = DOCUMENTS
    assert meta.files == {name: len(rows) for name, rows in docs.items() if name not in documents}
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
        "weekly_telegram.json": {"week", "role_bucket", "posts", "posts_before_dedup", "channels", "channels_week"},
        "skills.json": {"source", "skill", "category", "vacancies_mentioning", "vacancies_total"},
        "skills_by_role.json": {
            *("source", "role_bucket", "skill", "category", "vacancies_mentioning", "vacancies_in_bucket"),
        },
        "skills_junior.json": {"source", "skill", "category", "vacancies_mentioning", "vacancies_graded_junior"},
        "work_format.json": {"source", "role_bucket", "work_format", "vacancies", "vacancies_in_bucket"},
        "junior.json": {
            *("source", "role_bucket", "vacancies", "open_to_junior", "graded_intern", "graded_junior"),
            *("no_experience_required", "grade_known", "experience_field_stated"),
        },
        "salaries.json": {
            *("source", "level", "role_group", "role_bucket", "grade", "region_group", "vacancies"),
            *("with_salary_rub", "with_salary_other_currency", "with_from", "with_to"),
            *quantiles,
        },
        "salaries_telegram.json": {
            *("level", "role", "grade", "vacancies", "with_salary", "with_salary_rub_month", "with_from", "with_to"),
            *quantiles,
            *("period_from", "period_to"),
        },
        "lifetime_observed.json": {
            *("source", "level", "role_group", "vacancies", "still_open", "closed", "closed_in_gap", "censored"),
            *("left_censored", "closed_by_repost", "closed_in_quantiles", "period_from", "period_to", "window_days"),
            *(f"duration_days_{q}" for q in ("p25", "median", "p75", "p90")),
        },
        "forecast_history.json": {"week", "skill", "posts_mentioning", "posts_it"},
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
    for name, document in DOCUMENT_MODELS.items():
        document.model_validate_json((SITE_DATA / name).read_text(encoding="utf-8"))
    for name, model in ROW_MODELS.items():
        rows = json.loads((SITE_DATA / name).read_text(encoding="utf-8"))
        assert len(rows) == meta.files[name], name
        for row in rows:
            model.model_validate(row)


def test_site_texts_have_no_long_dash() -> None:
    files = [ROOT / "site" / "observablehq.config.js", *(ROOT / "site" / "src").rglob("*.md")]
    files += (ROOT / "site" / "src" / "components").rglob("*.js")
    dashes = {"—", "–"}
    offenders = [p.name for p in files if dashes & set(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_junior_skills_count_intern_and_junior_grades_over_their_own_base(
    marts: dict[str, list[dict[str, Any]]],
) -> None:
    rows = skills_junior(marts["mart_skills"], marts["mart_junior"])
    assert [(r.source, r.skill, r.vacancies_mentioning, r.vacancies_graded_junior) for r in rows] == [
        ("trudvsem", "Python", 4, 15),
    ]
    with pytest.raises(ValueError, match="junior-graded"):
        skills_junior(
            marts["mart_skills"], [{**r, "graded_intern": 0, "graded_junior": 1} for r in marts["mart_junior"]]
        )


def test_salary_cells_below_nine_have_no_median_interval_inside_their_range() -> None:
    assert salary_min_n() == 9
    assert median_ci_coverage(8) < 0.95 <= median_ci_coverage(9)
    assert round(median_ci_coverage(9), 4) == 0.9609


def test_lifetime_takes_medians_with_counts_and_skips_other_metrics() -> None:
    doc = lifetime(SURVIVAL, "results/models/survival.json")
    found = {(m.group, m.level, m.cohort): m for m in doc.medians}
    assert set(found) == {
        ("overall", "all", "prevalent"),
        ("overall", "all", "incident"),
        ("role", "ml_ds_data", "prevalent"),
        ("experience", "6+", "incident"),
        ("salary", "not_given", "prevalent"),
    }
    assert (found["overall", "all", "prevalent"].median_days, found["overall", "all", "prevalent"].events) == (
        32.0,
        4606,
    )
    assert found["experience", "6+", "incident"].high is None
    not_estimable = found["salary", "not_given", "prevalent"]
    assert (not_estimable.median_days, not_estimable.subjects) == (None, 229)
    assert doc.grace_days == 7
    assert [(p.day, p.at_risk) for p in doc.overall_curve] == [(7, 3009), (30, 2302)]
    assert doc.source_file == "results/models/survival.json"


def test_lifetime_needs_the_closure_rule() -> None:
    broken = {**SURVIVAL, "definitions": {"event": "closed when gone"}}
    with pytest.raises(ValueError, match="grace"):
        lifetime(broken, "results/models/survival.json")


def test_salary_rows_keep_their_level(marts: dict[str, list[dict[str, Any]]]) -> None:
    group = {**marts["mart_salaries"][0], "level": "group", "role_bucket": "all", "grade": "all", "region_group": "all"}
    rows = built({**marts, "mart_salaries": [*marts["mart_salaries"], group]})["salaries.json"]
    assert [(r.level, r.role_group, r.role_bucket) for r in rows] == [
        ("cell", "ml_ds_data", "ml_ds"),
        ("group", "ml_ds_data", "all"),
    ]
    with pytest.raises(ValidationError):
        Salary(**{**group, "level": "everything"})


def test_observed_lifetime_passes_through(marts: dict[str, list[dict[str, Any]]]) -> None:
    (row,) = built(marts)["lifetime_observed.json"]
    assert (row.closed_in_quantiles, row.duration_days_median, row.window_days) == (30, 28.0, 103)


def test_forecast_keeps_the_last_weeks_as_counts_with_their_base() -> None:
    doc, history = forecast_files(FORECAST_RESULT, site_forecast(6), "results/models/forecast.json", weeks=4)
    assert [r.week for r in history if r.skill == "Python"] == site_forecast(6).weeks[2:]
    assert [(r.posts_mentioning, r.posts_it) for r in history if r.skill == "Python"][-1] == (15, 100)
    assert doc.history_first_week == date(2026, 9, 14)
    assert doc.series_first_week == date(2026, 8, 31) and doc.last_week == date(2026, 10, 5) and doc.weeks == 6
    assert (doc.commit, doc.method, doc.origins, doc.horizon) == ("6d319c5", "ets", 3, 2)
    assert (doc.interval_check_covered, doc.interval_check_cells) == (7, 10)


def test_forecast_points_are_rounded_shares_per_skill_and_horizon() -> None:
    doc, _ = forecast_files(FORECAST_RESULT, site_forecast(), "results/models/forecast.json")
    assert len(doc.points) == 2 * 2
    first = doc.points[0]
    assert (first.skill, first.horizon, first.week) == ("Python", 1, date(2026, 10, 12))
    assert (first.forecast, first.low, first.high) == (0.1512, 0.12, 0.18)
    assert all(p.low <= p.forecast <= p.high for p in doc.points)


def test_forecast_with_more_mentions_than_posts_stops_the_export() -> None:
    broken = site_forecast()
    broken.mentions[0, -1] = 101.0
    with pytest.raises(ValueError, match="mentions of"):
        forecast_files(FORECAST_RESULT, broken, "results/models/forecast.json")


def test_lifetime_keeps_the_curve_of_each_estimable_group() -> None:
    curve = {"at_risk": {"0": 54, "7": 111}, "survival": {"7": 0.7765}}
    doc = {
        **SURVIVAL,
        "metrics": {
            **SURVIVAL["metrics"],
            "median_days_role_ml_ds_data": {**median_metric(26.0, 20.0, 29.0, 211, 152), **curve},
            "median_days_incident_role_ml_ds_data": {**median_metric(30.0, 21.0, 31.0, 139, 89), **curve},
        },
    }
    (only,) = lifetime(doc, "results/models/survival.json").group_curves
    assert (only.group, only.level) == ("role", "ml_ds_data"), "incident cohorts and groups without a curve stay out"
    assert [(p.day, p.survival, p.at_risk) for p in only.points] == [(7, 0.7765, 111)]


def test_telegram_years_keep_counts_with_their_base() -> None:
    doc = telegram_years(MARKET_RESULT, "results/market/facts.json")
    assert (doc.panel_channels, doc.channels, doc.last_week) == (39, 58, date(2026, 10, 12))
    row = next(r for r in doc.years if r.scope == "all" and r.year == 2026)
    assert (row.it_posts, row.ml_posts, row.junior_posts, row.graded_posts) == (18606, 1433, 3215, 12539)
    assert (row.ml_junior_posts, row.ml_graded_posts) == (208, 997)
    assert [(r.scope, r.skill, r.posts_mentioning, r.ml_posts) for r in doc.skills] == [
        ("all", "RAG", 206, 1433),
        ("panel", "RAG", 150, 1000),
    ]
    assert [(r.scope, r.group, r.work_format, r.posts, r.posts_with_format) for r in doc.work_format] == [
        ("it", "junior", "remote", 1249, 2301),
        ("ml", "other", "remote", 439, 694),
    ]
    assert doc.source_file == "results/market/facts.json" and doc.commit == "1234567"


def test_skill_effects_copy_the_estimates_and_check_their_count() -> None:
    doc = skill_effects(EFFECTS_RESULT, "results/models/skill_effects.json")
    assert [(e.skill, e.naive_pct, e.pct, e.p_holm) for e in doc.effects] == [
        ("Kubernetes", 106.1, -5.4, 1.0),
        ("Microservices", 135.3, 24.2, 0.02),
    ]
    assert (doc.rows, doc.estimated, doc.significant_holm, doc.ci_level) == (7429, 2, 1, 0.95)
    broken = {**EFFECTS_RESULT, "metrics": {**EFFECTS_RESULT["metrics"], "estimated": share_metric(3, 3)}}
    with pytest.raises(ValueError, match="estimated"):
        skill_effects(broken, "results/models/skill_effects.json")


def test_vacancy_history_takes_the_snapshot_counts() -> None:
    doc = vacancy_history(HISTORY_RESULT, "results/dbt/history.json")
    assert (doc.vacancies_tracked, doc.closed, doc.reopened, doc.reposted) == (25992, 15173, 2256, 2820)


def test_telegram_salaries_leave_the_portal_rows_of_the_extracted_mart_out(
    marts: dict[str, list[dict[str, Any]]],
) -> None:
    (row,) = built(marts)["salaries_telegram.json"]
    assert (row.level, row.grade, row.with_from, row.salary_from_median) == ("grade", "junior", 146, 80000)


def test_skill_trends_keep_the_pooled_cut_apart_from_the_years() -> None:
    doc = skill_trends(TRENDS_RESULT, "results/market/skill_trends.json")
    assert [(g.group, g.year, g.posts) for g in doc.groups] == [
        ("ml_all", None, 3313),
        ("ml_all", 2025, 1880),
        ("ml_junior", None, 444),
    ]
    cicd = [s for s in doc.shares if s.skill == "CI/CD"]
    assert [(s.year, s.posts_mentioning, s.posts) for s in cicd] == [(None, 305, 3313), (2025, 170, 1880)]
    assert any(s.skill == "A/B-тесты" and s.group == "ml_junior" for s in doc.shares)
    assert cicd[0].own is True and cicd[1].own is None and doc.base_year == 2022


def test_salary_grades_keep_medians_only_where_the_result_has_them() -> None:
    doc = telegram_years(MARKET_RESULT, "results/market/facts.json")
    cells = {(g.scope, g.grade): g for g in doc.salary_grades}
    assert cells["ml", "junior"].from_median == 100000 and cells["ml", "junior"].from_rows == 51
    assert cells["ml", "junior"].to_median is None and cells["ml", "junior"].stated == 52
    assert cells["ml", "intern"].from_median is None and cells["ml", "intern"].posts == 221
    assert [g.grade for g in doc.salary_grades] == ["intern", "junior"]
