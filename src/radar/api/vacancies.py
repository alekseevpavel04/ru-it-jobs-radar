"""Vacancy search over marts.mart_open_vacancies for the MCP tool find_vacancies.

Filters reach ClickHouse as query parameters of one fixed query, read as the read-only account; the
row count is bounded twice, by the LIMIT here and by the account's max_result_rows. Next to the rows
goes the number of open vacancies the filters match, so a short list is not mistaken for the market.
"""

from typing import Any

from radar.agent.clickhouse import QueryError
from radar.api.stats import Query, records

MART = "marts.mart_open_vacancies"
MAX_LIMIT = 50
UNKNOWN_TABLE = 60  # ClickHouse error code

_WHERE = """
WHERE ({source:String} = '' OR source = {source:String})
  AND ({bucket:String} = '' OR role_bucket = {bucket:String})
  AND ({grade:String} = '' OR grade = {grade:String})
  AND ({region:String} = '' OR region_group = {region:String})
  AND ({work_format:String} = '' OR work_format = {work_format:String})
  AND ({skill:String} = '' OR has(skills, {skill:String}))
  AND ({min_salary:UInt32} = 0 OR coalesce(salary_to, salary_from) >= {min_salary:UInt32})
"""

SEARCH_SQL = f"""
SELECT vacancy_entity_id, source, title, employer_name, role_bucket, grade, region_group, work_format,
       salary_from, salary_to, url, toString(first_seen) AS first_seen, toString(last_seen) AS last_seen
FROM {MART}
{_WHERE}
ORDER BY last_seen DESC, first_seen DESC, vacancy_entity_id
LIMIT {{limit:UInt8}}
"""

COUNT_SQL = f"""
SELECT count() AS matched
FROM {MART}
{_WHERE}
"""


class MartMissing(RuntimeError):
    pass


def find(
    query: Query,
    *,
    source: str | None = None,
    role_bucket: str | None = None,
    grade: str | None = None,
    region_group: str | None = None,
    work_format: str | None = None,
    skill: str | None = None,
    min_salary: int | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Open vacancies matching every given filter, the most recently seen first.

    `min_salary` keeps vacancies whose stated upper bound (or the lower one, when only it is stated)
    reaches the amount; a vacancy without a salary never passes it."""
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be within 1..{MAX_LIMIT}")
    filters = {
        "source": source,
        "role_bucket": role_bucket,
        "grade": grade,
        "region_group": region_group,
        "work_format": work_format,
        "skill": skill,
        "min_salary": min_salary,
    }
    params = {
        "source": source or "",
        "bucket": role_bucket or "",
        "grade": grade or "",
        "region": region_group or "",
        "work_format": work_format or "",
        "skill": skill or "",
        "min_salary": str(min_salary or 0),
    }
    try:
        matched = records(query(COUNT_SQL, params))[0]["matched"]
        items = records(query(SEARCH_SQL, {**params, "limit": str(limit)}))
    except QueryError as e:
        if e.code == UNKNOWN_TABLE:
            raise MartMissing(f"{MART} is not built in this warehouse yet") from e
        raise
    return {
        "mart": MART,
        "filters": {k: v for k, v in filters.items() if v is not None},
        "matched": matched,
        "items": items,
    }
