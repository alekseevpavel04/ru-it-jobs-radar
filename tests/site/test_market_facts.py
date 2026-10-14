import pytest

from radar.results import Metric
from radar.site.market_facts import ml_ds_buckets, ml_ds_skills, panel_channels, salaries, telegram_years, work_formats

JUNIOR = [
    {
        "source": "telegram",
        "vacancies": 10,
        "open_to_junior": 3,
        "graded_intern": 1,
        "graded_junior": 2,
        "grade_known": 8,
    }
]
CELLS = [
    {"source": "telegram", "grade": "intern", "vacancies": 1},
    {"source": "telegram", "grade": "junior", "vacancies": 2},
    {"source": "telegram", "grade": "senior", "vacancies": 4},
    {"source": "telegram", "grade": "lead", "vacancies": 1},
    {"source": "telegram", "grade": "unknown", "vacancies": 2},
]


def test_buckets_carry_denominators() -> None:
    m = ml_ds_buckets(JUNIOR, CELLS)
    assert m["ml_ds.telegram.open_to_junior"] == {"value": 0.3, "n": 3, "denominator": 10}
    assert m["ml_ds.telegram.graded_junior"]["n"] == 3
    assert m["ml_ds.telegram.graded_senior"] == {"value": 0.625, "n": 5, "denominator": 8}
    for v in m.values():
        Metric.model_validate(v)


def test_buckets_refuse_incomplete_grade_cells() -> None:
    with pytest.raises(ValueError, match="add up"):
        ml_ds_buckets(JUNIOR, CELLS[:-1])


def test_junior_skills_include_the_top_of_the_bucket() -> None:
    rows = [
        {"source": "telegram", "grade": "senior", "skill": "Docker", "vacancies_mentioning": 5},
        {"source": "telegram", "grade": "junior", "skill": "Docker", "vacancies_mentioning": 1},
        {"source": "telegram", "grade": "junior", "skill": "Python", "vacancies_mentioning": 3},
    ]
    m = ml_ds_skills(rows, JUNIOR)
    assert m["ml_ds_skill.telegram.Docker"] == {"value": 0.6, "n": 6, "denominator": 10}
    assert m["ml_ds_junior_skill.telegram.Python"] == {"value": 1.0, "n": 3, "denominator": 3}
    assert m["ml_ds_junior_skill.telegram.Docker"]["n"] == 1
    assert m["ml_ds_skill.telegram.Python"]["n"] == 3


def test_skill_mentions_cannot_exceed_the_base() -> None:
    rows = [{"source": "telegram", "grade": "junior", "skill": "SQL", "vacancies_mentioning": 4}]
    with pytest.raises(ValueError, match="mentions"):
        ml_ds_skills(rows, JUNIOR)


def test_salaries_keep_grade_rows_and_ml_junior_cells() -> None:
    rows = [
        {
            "source": "telegram",
            "level": "grade",
            "role": "all",
            "grade": "junior",
            "vacancies": 20,
            "with_salary_rub_month": 5,
            "with_from": 4,
            "with_to": 0,
            "salary_from_median": 80000.0,
            "salary_to_median": None,
        },
        {
            "source": "telegram",
            "level": "cell",
            "role": "ml_engineer",
            "grade": "junior",
            "vacancies": 6,
            "with_salary_rub_month": 1,
            "with_from": 1,
            "with_to": 1,
            "salary_from_median": None,
            "salary_to_median": None,
        },
    ]
    m = salaries(rows)
    assert m["salary.telegram.junior.stated"] == {"value": 0.25, "n": 5, "denominator": 20}
    assert m["salary.telegram.junior.from_median"] == {"value": 80000, "rows": 4}
    assert "salary.telegram.junior.to_median" not in m
    assert m["salary_ml_cell.telegram.ml_engineer.junior.stated"]["n"] == 1


def channel_year(year: int, channel: str, it: int, ml: int = 0, junior: int = 0) -> dict[str, object]:
    counts = {"it_posts": it, "ml_posts": ml, "junior_posts": junior, "graded_posts": it // 2}
    return {"year": year, "channel": channel, **counts}


def test_panel_needs_every_year() -> None:
    rows = [channel_year(2022, "a", 1), channel_year(2023, "a", 1), channel_year(2023, "b", 5)]
    assert panel_channels(rows, (2022, 2023)) == {"a"}


def test_telegram_years_split_all_and_panel() -> None:
    years = [
        channel_year(2022, "a", 10, ml=1, junior=2),
        channel_year(2023, "a", 10, ml=2, junior=1),
        channel_year(2023, "b", 10, ml=8, junior=0),
    ]
    skills = [
        {"year": 2023, "channel": "a", "skill": "RAG", "posts": 1},
        {"year": 2023, "channel": "b", "skill": "RAG", "posts": 4},
    ]
    m = telegram_years(years, skills, years=(2022, 2023), panel_years=(2022, 2023))
    assert m["telegram_panel.channels"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert m["telegram_year.2023.ml_posts"] == {"value": 0.5, "n": 10, "denominator": 20}
    assert m["telegram_panel_year.2023.ml_posts"] == {"value": 0.2, "n": 2, "denominator": 10}
    assert m["telegram_year.2022.junior_of_graded"] == {"value": 0.4, "n": 2, "denominator": 5}
    assert m["telegram_ml_skill.2023.RAG"] == {"value": 0.5, "n": 5, "denominator": 10}
    assert m["telegram_panel_ml_skill.2023.RAG"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert m["telegram_ml_skill.2022.RAG"]["n"] == 0


def test_work_formats_split_juniors_from_the_rest() -> None:
    rows = [
        {"year": 2026, "junior": 1, "work_format": "remote", "posts": 54},
        {"year": 2026, "junior": 1, "work_format": "office", "posts": 46},
        {"year": 2026, "junior": 0, "work_format": "remote", "posts": 7},
        {"year": 2026, "junior": 0, "work_format": "hybrid", "posts": 3},
    ]
    out = work_formats(rows)
    assert out["telegram_work_format.2026.junior.remote"] == {"value": 0.54, "n": 54, "denominator": 100}
    assert out["telegram_work_format.2026.other.remote"]["n"] == 7
    assert out["telegram_work_format.2026.other.remote"]["denominator"] == 10
