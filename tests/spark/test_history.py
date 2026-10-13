"""The Spark mirror of the dbt history (jobs/spark) on small hand-made cases.

Needs pyspark and Java (the WSL spark venv, docs/methods/spark.md); skipped elsewhere.
"""

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyspark")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jobs" / "spark"))

import history
import rules
from pyspark.sql import SparkSession

RULES = {"complete_day_ratio": 0.9, "absent_days_to_close": 7, "max_step_days": 2}
D0 = date(2026, 7, 1)


def day(i: int) -> date:
    return D0 + timedelta(days=i - 1)


@pytest.fixture(scope="module")
def spark() -> Any:
    session = (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def obs_row(
    source_id: str, i: int, content_hash: str = "h1", closed: bool = False, salary: float | None = None
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "dt": day(i),
        "content_hash": content_hash,
        "is_closed_state": closed,
        "title": "Python-разработчик",
        "employer_name": "ООО Ромашка",
        "salary_from": salary,
        "salary_to": None,
        "salary_currency_code": "RUB" if salary else None,
        "schedule": "Полный день",
        "employment": None,
        "remote_work": None,
        "status": None,
        "skills": [],
        "text_hash": 1,
    }


def scenario(spark: Any) -> Any:
    rows = [obs_row(f"filler-{k}", i) for k in range(20) for i in range(1, 15)]
    # a: a salary change on day 4, gone after day 5 for good
    rows += [obs_row("a", i) for i in range(1, 4)] + [obs_row("a", i, "h2", salary=100000.0) for i in (4, 5)]
    # b: missing on days 3-4, back within the grace window
    rows += [obs_row("b", i) for i in (1, 2, *range(5, 15))]
    # c: deleted on day 2, open again on day 10 after the closure was confirmed
    rows += [obs_row("c", 1), obs_row("c", 2, closed=True), *(obs_row("c", i) for i in range(10, 15))]
    schema = (
        "source_id string, dt date, content_hash string, is_closed_state boolean, title string, "
        "employer_name string, salary_from double, salary_to double, salary_currency_code string, "
        "schedule string, employment string, remote_work boolean, status string, skills array<string>, "
        "text_hash long"
    )
    return spark.createDataFrame([tuple(r.values()) for r in rows], schema)


def collect(df: Any, key: str) -> dict[Any, dict[str, Any]]:
    return {tuple(r[k] for k in key.split(",")): r.asDict() for r in df.collect()}


def test_scd2_versions_closures_and_reopenings(spark: Any) -> None:
    obs = scenario(spark)
    cal = history.calendar(obs, RULES)
    versions = collect(history.scd2(obs, cal, RULES).where("source_id NOT LIKE 'filler%'"), "source_id,version_no")

    assert set(versions) == {("a", 1), ("a", 2), ("b", 1), ("c", 1), ("c", 2)}
    a1, a2 = versions[("a", 1)], versions[("a", 2)]
    assert (a1["change_reason"], a1["valid_to"], a1["ended_by"], a1["changed_fields"]) == ("new", day(4), "changed", [])
    assert (a2["change_reason"], a2["valid_to"], a2["ended_by"]) == ("changed", day(6), "closed")
    assert a2["changed_fields"] == ["salary_from", "salary_currency_code"]
    b = versions[("b", 1)]
    assert (b["is_current"], b["days_seen"], b["last_seen"]) == (True, 12, day(14))
    c1, c2 = versions[("c", 1)], versions[("c", 2)]
    assert (c1["valid_to"], c1["ended_by"], c2["change_reason"], c2["valid_from"]) == (
        day(2),
        "closed",
        "reopened",
        day(10),
    )


def test_lifetime_censoring_and_durations(spark: Any) -> None:
    obs = scenario(spark)
    cal = history.calendar(obs, RULES)
    life = collect(history.lifetime(history.scd2(obs, cal, RULES), cal), "source_id")
    assert (life[("a",)]["closed_at"], life[("a",)]["duration_days"], life[("a",)]["is_censored"]) == (day(6), 5, False)
    assert (life[("a",)]["n_salary_changes"], life[("a",)]["is_left_censored"]) == (1, True)
    assert (life[("b",)]["is_censored"], life[("b",)]["duration_days"]) == (True, 14)
    assert (life[("c",)]["n_reopened"], life[("c",)]["is_closed"]) == (1, False)


def test_observations_keep_the_latest_row_of_a_day_and_parse_the_payload(spark: Any) -> None:
    def payload(**fields: Any) -> str:
        base = {"codeProfessionalSphere": "Transport", "deleted": "false", "salaryMin": "0", "salaryMax": "90000"}
        return json.dumps(base | fields, ensure_ascii=False)

    t1, t2 = datetime(2026, 7, 1, 10, tzinfo=UTC), datetime(2026, 7, 1, 12, tzinfo=UTC)
    raw = spark.createDataFrame(
        [
            # x: IT by sphere in the earlier row, not IT in the latest: not an IT vacancy that day
            ("x", day(1), t1, "h1", "Водитель", None, payload(codeProfessionalSphere="InformationTechnology"), t1),
            ("x", day(1), t2, "h2", "Водитель", None, payload(), t1),
            # y: IT by title, deleted, skills unsorted
            (
                "y",
                day(1),
                t1,
                "h3",
                "Ведущий DevOps",
                "  Текст\n\nвакансии ",
                payload(deleted="true", skills=["b", "a"]),
                t1,
            ),
        ],
        "source_id string, dt date, fetched_at timestamp, content_hash string, title string, text string, "
        "payload string, published_at timestamp",
    )
    out = collect(history.observations(raw, rules.it_title_pattern(rules.macros()), []), "source_id")
    assert set(out) == {("y",)}
    y = out[("y",)]
    assert (y["is_closed_state"], y["skills"], y["salary_from"], y["salary_to"]) == (True, ["a", "b"], None, 90000.0)
    assert (y["salary_currency_code"], y["title_norm"], y["text_hash"] is not None) == ("RUB", "ведущий devops", True)
