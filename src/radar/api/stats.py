"""GET /stats: aggregates of marts.* read as the read-only ClickHouse account, never a row of a vacancy.

Filters arrive validated against the marts' accepted values and reach ClickHouse as query parameters,
so no request text is spliced into SQL. Every count comes with its denominator.
"""

from collections.abc import Callable
from typing import Any, Literal

from radar.agent.clickhouse import Result

Query = Callable[[str, dict[str, str]], Result]

WeeklySource = Literal["trudvsem_snapshot", "trudvsem", "habr", "telegram"]
SalarySource = Literal["trudvsem", "habr"]
SkillSource = Literal["trudvsem", "habr", "telegram"]
SalaryLevel = Literal["group_grade", "group", "grade", "source"]
RoleBucket = Literal[
    "ml_ds",
    "data_engineering",
    "data_analytics",
    "developer_1c",
    "business_system_analysis",
    "security",
    "devops",
    "qa",
    "mobile",
    "fullstack",
    "backend",
    "frontend",
    "embedded",
    "sysadmin_support",
    "design",
    "management",
    "software_engineer",
    "other_it",
]

# Rows of one week and source add up across role buckets; the day counts are the same on each of them.
# A count the source does not have (closures of the API, vacancies of Telegram) is NULL on every row
# and stays NULL in the sum.
WEEKLY_SQL = """
SELECT toString(week) AS week,
       any(days_collected) AS days_collected,
       any(complete_days) AS complete_days,
       toBool(any(is_first_collected_week)) AS is_first_collected_week,
       sumOrNull(vacancies_active) AS vacancies_active,
       sumOrNull(vacancies_new) AS vacancies_new,
       sumOrNull(vacancies_closed) AS vacancies_closed,
       sumOrNull(telegram_posts) AS telegram_posts
FROM marts.mart_weekly_dynamics
WHERE source = {source:String} AND ({bucket:String} = '' OR role_bucket = {bucket:String})
GROUP BY week
ORDER BY week DESC
LIMIT {weeks:UInt16}
"""

SALARIES_SQL = """
SELECT role_group, grade, vacancies, with_salary_rub, with_salary_other_currency,
       with_from, salary_from_p25, salary_from_median, salary_from_p75,
       with_to, salary_to_p25, salary_to_median, salary_to_p75,
       toString(period_from) AS period_from, toString(period_to) AS period_to
FROM marts.mart_salaries
WHERE source = {source:String} AND level = {level:String}
ORDER BY role_group, grade
"""

SKILLS_SQL = """
SELECT skill, any(skill_category) AS category, sum(vacancies_mentioning) AS vacancies_mentioning,
       toString(min(period_from)) AS period_from, toString(max(period_to)) AS period_to
FROM marts.mart_skills
WHERE source = {source:String} AND ({bucket:String} = '' OR role_bucket = {bucket:String})
GROUP BY skill
ORDER BY vacancies_mentioning DESC, skill
LIMIT {limit:UInt16}
"""

# mart_skills has a cell's size only on rows of a mentioned skill; mart_junior counts the same units
# for every bucket (radar.site.export.bucket_sizes), so it gives the denominator.
SKILLS_BASE_SQL = """
SELECT sum(vacancies) AS vacancies
FROM marts.mart_junior
WHERE source = {source:String} AND ({bucket:String} = '' OR role_bucket = {bucket:String})
"""


def records(result: Result) -> list[dict[str, Any]]:
    return [dict(zip(result.columns, row, strict=True)) for row in result.rows]


def weekly(query: Query, source: str, bucket: str | None, weeks: int) -> dict[str, Any]:
    rows = records(query(WEEKLY_SQL, {"source": source, "bucket": bucket or "", "weeks": str(weeks)}))
    return {
        "mart": "marts.mart_weekly_dynamics",
        "source": source,
        "role_bucket": bucket,
        "weeks": sorted(rows, key=lambda r: r["week"]),
    }


def salaries(query: Query, source: str, level: str) -> dict[str, Any]:
    rows = records(query(SALARIES_SQL, {"source": source, "level": level}))
    period_from = min((r.pop("period_from") for r in rows), default=None)
    period_to = max((r.pop("period_to") for r in rows), default=None)
    return {
        "mart": "marts.mart_salaries",
        "source": source,
        "level": level,
        "unit": "RUB per month as stated; quantiles are null below 5 stated bounds",
        "period_from": period_from,
        "period_to": period_to,
        "rows": rows,
    }


def skills(query: Query, source: str, bucket: str | None, limit: int) -> dict[str, Any]:
    params = {"source": source, "bucket": bucket or ""}
    rows = records(query(SKILLS_SQL, {**params, "limit": str(limit)}))
    base = records(query(SKILLS_BASE_SQL, params))[0]["vacancies"]
    period_from = min((r.pop("period_from") for r in rows), default=None)
    period_to = max((r.pop("period_to") for r in rows), default=None)
    return {
        "mart": "marts.mart_skills",
        "source": source,
        "role_bucket": bucket,
        "vacancies": base,
        "period_from": period_from,
        "period_to": period_to,
        "skills": rows,
    }
