"""Vacancy search over marts.mart_open_vacancies for the MCP tool find_vacancies, and the rows of
given vacancies for /match.

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

_COLUMNS = """vacancy_entity_id, source, title, employer_name, role_bucket, grade, region_group, work_format,
       skills, salary_from, salary_to, url, toString(first_seen) AS first_seen, toString(last_seen) AS last_seen"""

SEARCH_SQL = f"""
SELECT {_COLUMNS}
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

BY_ID_SQL = f"""
SELECT {_COLUMNS}
FROM {MART}
WHERE has({{ids:Array(String)}}, vacancy_entity_id)
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
    matched = _read(query, COUNT_SQL, params)[0]["matched"]
    items = _read(query, SEARCH_SQL, {**params, "limit": str(limit)})
    return {
        "mart": MART,
        "filters": {k: v for k, v in filters.items() if v is not None},
        "matched": matched,
        "items": items,
    }


def by_id(query: Query, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Rows of the given vacancies that are open now, by vacancy_entity_id; a closed one is absent."""
    if not ids:
        return {}
    return {r["vacancy_entity_id"]: r for r in _read(query, BY_ID_SQL, {"ids": _array(ids)})}


def _read(query: Query, sql: str, params: dict[str, str]) -> list[dict[str, Any]]:
    try:
        return records(query(sql, params))
    except QueryError as e:
        if e.code == UNKNOWN_TABLE:
            raise MartMissing(f"{MART} is not built in this warehouse yet") from e
        raise


def _array(values: list[str]) -> str:
    """An Array(String) query parameter in ClickHouse's quoted text format."""
    quoted = (v.replace("\\", "\\\\").replace("'", "\\'") for v in values)
    return "[" + ",".join(f"'{v}'" for v in quoted) + "]"
