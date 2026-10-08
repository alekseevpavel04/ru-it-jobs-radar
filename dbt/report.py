"""results/dbt/build.json: what the last `dbt build` did and what staging holds; results/dbt/history.json:
what the vacancy history (models/history) found and the memory its statements took in that build.

Reads dbt/target/build_results.json (kept by scripts/dbt.sh after each build) and, when present,
dbt/target/sources.json of the last freshness check; counts rows in ClickHouse. Run it through the
wrapper right after a build:

    bash scripts/dbt.sh build && bash scripts/dbt.sh source freshness && bash scripts/dbt.sh report
"""

import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from radar.config import ROOT, load_dotenv
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse

TARGET = ROOT / "dbt" / "target"
STAGING = ("stg_trudvsem_api", "stg_trudvsem_snapshot", "stg_habr", "stg_vacancies")


def share(n: int, denominator: int) -> dict[str, Any]:
    return {"value": round(n / denominator, 4) if denominator else 0, "n": n, "denominator": denominator}


def rows(ch: ClickHouse, sql: str) -> list[list[str]]:
    return [line.split("\t") for line in ch.query(sql + " FORMAT TSV").splitlines() if line]


def build_metrics(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kinds: dict[str, Counter[str]] = {"model": Counter(), "seed": Counter(), "test": Counter()}
    for result in run["results"]:
        kind = result["unique_id"].split(".", 1)[0]
        if kind in kinds:
            kinds[kind][result["status"]] += 1
    models, seeds, tests = kinds["model"], kinds["seed"], kinds["test"]
    return {
        "models_ok": share(models["success"], sum(models.values())),
        "seeds_ok": share(seeds["success"], sum(seeds.values())),
        "tests_passed": share(tests["pass"], sum(tests.values())),
        "tests_warned": {"value": tests["warn"], "denominator": sum(tests.values()), "n": tests["warn"]},
        "tests_failed": {
            "value": tests["fail"] + tests["error"],
            "n": tests["fail"] + tests["error"],
            "denominator": sum(tests.values()),
        },
        "build_seconds": {"value": round(run["elapsed_time"], 1)},
    }


def freshness_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for result in doc["results"]:
        name = result["unique_id"].rsplit(".", 1)[-1]
        out[f"freshness_{name}"] = {
            "value": result["status"],
            "hours_since_last_fetch": round(result.get("max_loaded_at_time_ago_in_s", 0) / 3600, 1),
            "checked_at": doc["metadata"]["generated_at"],
        }
    return out


def staging_metrics(ch: ClickHouse) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model in STAGING:
        for source, n in rows(ch, f"SELECT source, count() FROM staging.{model} GROUP BY source ORDER BY source"):
            out[f"rows_{model}_{source}"] = {"value": int(n)}
    for dt, it, total in rows(
        ch,
        "SELECT dt, countIf(is_it_candidate), count() FROM staging.stg_trudvsem_snapshot GROUP BY dt ORDER BY dt",
    ):
        out[f"it_candidates_trudvsem_snapshot_{dt}"] = share(int(it), int(total))
    for dt, it, total in rows(
        ch, "SELECT dt, countIf(is_it_candidate), count() FROM staging.stg_trudvsem_api GROUP BY dt ORDER BY dt"
    ):
        out[f"it_candidates_trudvsem_api_{dt}"] = share(int(it), int(total))
    for dt, future, total in rows(
        ch,
        "SELECT dt, countIf(has_future_published_at), countIf(has_card) FROM staging.stg_habr GROUP BY dt ORDER BY dt",
    ):
        out[f"habr_future_date_posted_{dt}"] = share(int(future), int(total))
    return out


def history_metrics(ch: ClickHouse) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source, days, complete, first, last, span in rows(
        ch,
        "SELECT source, count(), countIf(is_complete), min(dt), max(dt), dateDiff('day', min(dt), max(dt)) + 1 "
        "FROM staging.hist_calendar GROUP BY source ORDER BY source",
    ):
        out[f"collected_days_{source}"] = {"value": int(days), "first_dt": first, "last_dt": last}
        out[f"complete_days_{source}"] = share(int(complete), int(days))
        out[f"calendar_span_days_{source}"] = {"value": int(span)}

    for source, tracked, closed, reopened, salary, left in rows(
        ch,
        "SELECT source, count(), countIf(is_closed), countIf(n_reopened > 0), countIf(n_salary_changes > 0), "
        "countIf(is_left_censored) FROM staging.hist_vacancy_lifetime GROUP BY source ORDER BY source",
    ):
        out[f"vacancies_tracked_{source}"] = {"value": int(tracked)}
        out[f"closed_{source}"] = share(int(closed), int(tracked))
        out[f"reopened_{source}"] = share(int(reopened), int(tracked))
        out[f"salary_changed_{source}"] = share(int(salary), int(tracked))
        out[f"left_censored_{source}"] = share(int(left), int(tracked))

    changed: dict[str, int] = {}
    for source, versions, n_changed in rows(
        ch,
        "SELECT source, count(), countIf(change_reason = 'changed') FROM staging.hist_vacancy_scd2 "
        "GROUP BY source ORDER BY source",
    ):
        changed[source] = int(n_changed)
        out[f"versions_{source}"] = {"value": int(versions)}
        out[f"versions_changed_{source}"] = share(int(n_changed), int(versions))
    for source, field, n in rows(
        ch,
        "SELECT source, arrayJoin(changed_fields) AS field, count() FROM staging.hist_vacancy_scd2 "
        "WHERE change_reason = 'changed' GROUP BY source, field ORDER BY source, field",
    ):
        out[f"changed_field_{source}_{field}"] = share(int(n), changed[source])

    # Lifetimes of closed snapshot vacancies: all of them, and those first seen after the first
    # snapshot day, whose first sighting is close to their real start.
    for subset, condition in (("all", "1"), ("not_left_censored", "NOT is_left_censored")):
        for median, n in rows(
            ch,
            "SELECT quantileExact(0.5)(duration_days), count() FROM staging.hist_vacancy_lifetime "
            f"WHERE source = 'trudvsem_snapshot' AND is_closed AND {condition}",
        ):
            out[f"median_lifetime_days_closed_snapshot_{subset}"] = {"value": int(median), "vacancies": int(n)}

    # A snapshot vacancy closes on the first complete snapshot without it, yet some come back: how
    # many closures were followed by a reopening, and after how many days.
    for source, closures, reopened_after, le1, le3, le7 in rows(
        ch,
        "SELECT source, countIf(ended_by = 'closed'), countIf(next_reason = 'reopened'), "
        "countIf(next_reason = 'reopened' AND gap <= 1), countIf(next_reason = 'reopened' AND gap <= 3), "
        "countIf(next_reason = 'reopened' AND gap <= 7) FROM (SELECT source, ended_by, "
        "leadInFrame(change_reason) OVER w AS next_reason, "
        "dateDiff('day', valid_to, leadInFrame(valid_from) OVER w) AS gap FROM staging.hist_vacancy_scd2 "
        "WINDOW w AS (PARTITION BY source, source_id ORDER BY version_no "
        "ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING)) GROUP BY source ORDER BY source",
    ):
        out[f"closures_{source}"] = {"value": int(closures)}
        out[f"closures_reopened_{source}"] = share(int(reopened_after), int(closures))
        for limit, n in ((1, le1), (3, le3), (7, le7)):
            out[f"reopened_within_{limit}_days_{source}"] = share(int(n), int(reopened_after))

    # The history tracks a snapshot vacancy on the days it passes the IT filter; one that drops out
    # of the filter while still listed reads as absent on those days.
    for flips, ever_it in rows(
        ch,
        "SELECT countIf(n_it < n), count() FROM (SELECT source_id, countIf(is_it_candidate) AS n_it, count() AS n "
        "FROM staging.stg_trudvsem_snapshot WHERE source_id IN "
        "(SELECT source_id FROM staging.stg_trudvsem_snapshot WHERE is_it_candidate) GROUP BY source_id)",
    ):
        out["it_filter_left_while_listed_trudvsem_snapshot"] = share(int(flips), int(ever_it))

    closed_by_source = {
        source: int(count)
        for source, count in rows(
            ch, "SELECT source, countIf(is_closed) FROM staging.hist_vacancy_lifetime GROUP BY source"
        )
    }
    for source, reposts, *within in rows(
        ch,
        "SELECT source, countIf(is_repost), countIf(gap_days <= 0), countIf(gap_days <= 3), "
        "countIf(gap_days <= 7), countIf(gap_days <= 14), countIf(gap_days <= 30) "
        "FROM staging.hist_reposts GROUP BY source ORDER BY source",
    ):
        out[f"reposts_{source}"] = share(int(reposts), closed_by_source.get(source, 0))
        for gap, pairs in zip((0, 3, 7, 14, 30), within, strict=True):
            out[f"repost_candidates_{source}_gap_le_{gap}"] = share(int(pairs), closed_by_source.get(source, 0))
    return out


def history_memory(ch: ClickHouse, run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Peak memory of the statements that wrote each history table during the last build."""
    # A minute of margin on both ends: the server's clock (WSL) and the client's drift apart.
    generated = datetime.fromisoformat(run["metadata"]["generated_at"].replace("Z", "+00:00"))
    started = generated - timedelta(seconds=run["elapsed_time"] + 60)
    finished = generated + timedelta(seconds=60)
    ch.query("SYSTEM FLUSH LOGS")
    out: dict[str, dict[str, Any]] = {}
    # dbt opens every statement with a JSON comment that names its node.
    for model, peak, statements in rows(
        ch,
        'SELECT extract(query, \'"node_id": "model\\\\.radar\\\\.([a-z0-9_]+)"\') AS model, '
        "max(memory_usage), count() FROM system.query_log "
        "WHERE type = 'QueryFinish' AND query_kind = 'Insert' "
        f"AND event_time BETWEEN toDateTime('{started:%Y-%m-%d %H:%M:%S}', 'UTC') "
        f"AND toDateTime('{finished:%Y-%m-%d %H:%M:%S}', 'UTC') "
        "AND startsWith(model, 'hist_') GROUP BY model ORDER BY model",
    ):
        out[f"peak_memory_mib_{model}"] = {"value": round(int(peak) / 2**20, 1), "statements": int(statements)}
    return out


def main() -> int:
    build_path = TARGET / "build_results.json"
    if not build_path.exists():
        print("no dbt/target/build_results.json; run `bash scripts/dbt.sh build` first", file=sys.stderr)
        return 1
    run = json.loads(build_path.read_text(encoding="utf-8"))
    load_dotenv()
    metrics = build_metrics(run) | freshness_metrics(TARGET / "sources.json")
    with ClickHouse() as ch:
        metrics |= staging_metrics(ch)
        history = history_metrics(ch) | history_memory(ch, run)
    history_path = write_result(
        "dbt/history.json",
        history,
        source_script="dbt/report.py",
        dbt_version=run["metadata"]["dbt_version"],
        build_finished_at=run["metadata"]["generated_at"],
        notes={
            "rules": "closure, grace and repost window: dbt/macros/history.sql",
            "closed": "vacancies with a closure observed, of vacancies tracked; trudvsem (API) never closes by absence",
            "median_lifetime_days_closed_snapshot": "closed_at - first_seen of closed snapshot vacancies",
            "reposts": "pairs with is_repost (gap within the window), of closed vacancies of the source",
            "repost_candidates_gap_le": "pairs of hist_reposts with gap_days at most N, of closed vacancies",
            "peak_memory_mib": "max memory_usage of one INSERT into the table during the build (system.query_log)",
        },
    )
    print(history_path)
    path = write_result(
        "dbt/build.json",
        metrics,
        source_script="dbt/report.py",
        dbt_version=run["metadata"]["dbt_version"],
        build_finished_at=run["metadata"]["generated_at"],
        notes={
            "it_candidates": "wide IT filter is_it_candidate, not measured against labels yet",
            "habr_future_date_posted": "cards whose datePosted is later than the fetch, of cards read that day",
        },
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
