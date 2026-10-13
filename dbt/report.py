"""results/dbt/build.json: what the last `dbt build` did and what staging holds; results/dbt/history.json:
what the vacancy history (models/history) found; results/dbt/marts.json: rows of the marts, what they
stand on and leave out, and the headline numbers README and the site cite, each with its denominator.
history.json and marts.json also carry the memory (and marts.json the time) of their models in that build;
results/dbt/extracted_marts.json: rows and coverage of the marts on extracted fields, when they are built.

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

import yaml

from radar.config import ROOT, load_dotenv
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse

TARGET = ROOT / "dbt" / "target"
STAGING = ("stg_trudvsem_api", "stg_trudvsem_snapshot", "stg_habr", "stg_superjob", "stg_vacancies")
MARTS = (
    "mart_weekly_dynamics",
    "mart_salaries",
    "mart_skills",
    "mart_work_format",
    "mart_employers",
    "mart_junior",
    "mart_skills_weekly",
    "mart_lifetime",
)
EXTRACTED_MARTS = (
    "mart_extracted_skills_weekly",
    "mart_extracted_fields_weekly",
    "mart_extracted_salaries",
    "mart_extraction_coverage",
)
# Models and seeds of the marts, for their build time and memory.
MART_MODELS = (
    "stg_telegram",
    "int_vacancy_",
    "int_extracted_",
    "mart_",
    "skill_dictionary",
    "telegram_channels",
    "role_groups",
)
IT_DATA = ("ml_ds", "data_engineering", "data_analytics")


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
    for dt, it, total in rows(
        ch, "SELECT dt, countIf(is_it_candidate), count() FROM staging.stg_superjob GROUP BY dt ORDER BY dt"
    ):
        out[f"it_candidates_superjob_{dt}"] = share(int(it), int(total))
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


def peak_memory(ch: ClickHouse, run: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Peak memory of the statements that wrote the tables of the given models during the last build."""
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
        "AND model != '' GROUP BY model ORDER BY model",
    ):
        if model.startswith(prefixes):
            out[f"peak_memory_mib_{model}"] = {"value": round(int(peak) / 2**20, 1), "statements": int(statements)}
    return out


def model_seconds(run: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    total = 0.0
    for result in run["results"]:
        kind, _, name = result["unique_id"].split(".", 2)
        if kind in ("model", "seed") and name.startswith(prefixes):
            out[f"build_seconds_{name}"] = {"value": round(result["execution_time"], 2)}
            total += result["execution_time"]
    out["build_seconds_total"] = {"value": round(total, 1)}
    return out


def it_data_buckets() -> str:
    return ", ".join(f"'{b}'" for b in IT_DATA)


def marts_metrics(ch: ClickHouse) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mart in MARTS:
        out[f"rows_{mart}"] = {"value": ch.scalar(f"SELECT count() FROM marts.{mart}")}

    # What the marts stand on: IT records per source, vacancies (entities) per family, and what is
    # left out of every mart.
    for source, records, with_entity in rows(
        ch,
        "SELECT source, count(), countIf(has_dedup_entity) FROM staging.int_vacancy_records "
        "GROUP BY source ORDER BY source",
    ):
        out[f"it_records_{source}"] = {"value": int(records)}
        out[f"records_with_dedup_entity_{source}"] = share(int(with_entity), int(records))
    for family, entities, not_it, both, sole in rows(
        ch,
        "SELECT family, count(), countIf(role_bucket = 'not_it'), "
        "countIf(has(sources, 'trudvsem') AND has(sources, 'trudvsem_snapshot')), "
        "countIf(is_sole_proprietor AND role_bucket != 'not_it') "
        "FROM staging.int_vacancy_entities GROUP BY family ORDER BY family",
    ):
        it = int(entities) - int(not_it)
        out[f"vacancies_{family}"] = {"value": int(entities)}
        out[f"not_it_titles_left_out_{family}"] = share(int(not_it), int(entities))
        out[f"sole_proprietor_left_out_of_employers_{family}"] = share(int(sole), it)
        if family == "trudvsem":
            out["vacancies_in_api_and_snapshot_counted_once"] = share(int(both), int(entities))
    # SuperJob flags the vacancies it passed on to trudvsem; how many of them the dedup links to a
    # trudvsem record is a recall check of the cross-source link (the copy may also be outside the
    # trudvsem IT slice or not collected yet, so this is a lower bound).
    for linked, flagged in rows(
        ch,
        "SELECT countIf(has(e.sources, 'trudvsem') OR has(e.sources, 'trudvsem_snapshot')), count() "
        "FROM (SELECT DISTINCT source_id FROM staging.stg_superjob "
        "WHERE is_it_candidate AND uploaded_to_trudvsem) AS s "
        "INNER JOIN (SELECT source_id, vacancy_entity_id FROM staging.vacancy_entities WHERE source = 'superjob') AS v "
        "ON s.source_id = v.source_id "
        "INNER JOIN (SELECT vacancy_entity_id, groupUniqArray(source) AS sources FROM staging.vacancy_entities "
        "GROUP BY vacancy_entity_id) AS e ON v.vacancy_entity_id = e.vacancy_entity_id",
    ):
        out["superjob_uploaded_to_trudvsem_linked"] = share(int(linked), int(flagged))
    for posts, not_it, copies, network_keys, week_keys in rows(
        ch,
        "SELECT count(), countIf(role_bucket = 'not_it'), countIf(is_week_duplicate), "
        "uniqExact((published_week, network, text_hash)), uniqExact((published_week, text_hash)) "
        "FROM staging.stg_telegram",
    ):
        out["telegram_vacancy_posts"] = {"value": int(posts)}
        out["telegram_not_it_posts_left_out"] = share(int(not_it), int(posts))
        out["telegram_week_copies_removed"] = share(int(copies), int(posts))
        # Dedup only inside a publisher network would keep the copies that cross networks.
        out["telegram_posts_after_network_week_dedup"] = share(int(network_keys), int(posts))
        out["telegram_posts_after_week_dedup"] = share(int(week_keys), int(posts))

    # Weekly headlines: shares of the IT vacancies active in the source's latest collected week.
    for source in ("trudvsem_snapshot", "trudvsem", "habr", "superjob"):
        for week, days, data, ml, total in rows(
            ch,
            # vacancies_active is Nullable: a sum over no matching rows is NULL, not 0.
            f"SELECT week, any(days_collected), "
            f"ifNull(sumIf(vacancies_active, role_bucket IN ({it_data_buckets()})), 0), "
            "ifNull(sumIf(vacancies_active, role_bucket = 'ml_ds'), 0), ifNull(sum(vacancies_active), 0) "
            f"FROM marts.mart_weekly_dynamics WHERE source = '{source}' "
            f"AND week = (SELECT max(week) FROM marts.mart_weekly_dynamics WHERE source = '{source}') GROUP BY week",
        ):
            extra = {"week": week, "days_collected": int(days)}
            out[f"ml_ds_data_share_latest_week_{source}"] = share(int(data), int(total)) | extra
            out[f"ml_ds_share_latest_week_{source}"] = share(int(ml), int(total)) | extra
    for week, posts, before in rows(
        ch,
        "SELECT week, sum(telegram_posts), sum(telegram_posts_before_dedup) FROM marts.mart_weekly_dynamics "
        "WHERE source = 'telegram' AND week = (SELECT max(week) FROM marts.mart_weekly_dynamics "
        "WHERE source = 'telegram' AND week < toMonday(today())) GROUP BY week",
    ):
        out["telegram_posts_latest_full_week"] = share(int(posts), int(before)) | {"week": week}

    for source, salary, total, other in rows(
        ch,
        "SELECT source, sum(with_salary_rub), sum(vacancies), sum(with_salary_other_currency) "
        "FROM marts.mart_salaries WHERE level = 'source' GROUP BY source ORDER BY source",
    ):
        out[f"salary_stated_rub_{source}"] = share(int(salary), int(total))
        out[f"salary_other_currency_{source}"] = share(int(other), int(total))
    # Amounts are kept as stated; this many lower bounds look like hourly rates or typos.
    for source, low, stated in rows(
        ch,
        "SELECT family, countIf(salary_from < 10000), countIf(salary_from IS NOT NULL) "
        "FROM staging.int_vacancy_entities WHERE role_bucket != 'not_it' AND salary_currency_code = 'RUB' "
        "GROUP BY family ORDER BY family",
    ):
        out[f"salary_from_below_10000_rub_{source}"] = share(int(low), int(stated))

    for source, bucket, junior, total in rows(
        ch,
        f"SELECT source, role_bucket, open_to_junior, vacancies FROM marts.mart_junior "
        f"WHERE role_bucket IN ({it_data_buckets()}) "
        "UNION ALL SELECT source, 'all_it', sum(open_to_junior), sum(vacancies) FROM marts.mart_junior "
        "WHERE role_bucket != 'multi_post' GROUP BY source ORDER BY 1, 2",
    ):
        out[f"open_to_junior_{bucket}_{source}"] = share(int(junior), int(total))

    for source, remote, hybrid, unknown, total in rows(
        ch,
        "SELECT source, sumIf(vacancies, work_format = 'remote'), sumIf(vacancies, work_format = 'hybrid'), "
        "sumIf(vacancies, work_format = 'unknown'), sum(vacancies) FROM marts.mart_work_format "
        "WHERE role_bucket != 'multi_post' GROUP BY source ORDER BY source",
    ):
        out[f"remote_{source}"] = share(int(remote), int(total))
        out[f"hybrid_{source}"] = share(int(hybrid), int(total))
        out[f"work_format_unknown_{source}"] = share(int(unknown), int(total))

    for source, python, sql, total in rows(
        ch,
        "SELECT s.source, s.python, s.sql_, f.total FROM (SELECT source, "
        "sumIf(vacancies_mentioning, skill = 'Python') AS python, sumIf(vacancies_mentioning, skill = 'SQL') AS sql_ "
        "FROM marts.mart_skills WHERE role_bucket = 'ml_ds' GROUP BY source) AS s "
        "INNER JOIN (SELECT source, sum(vacancies) AS total FROM marts.mart_work_format "
        "WHERE role_bucket = 'ml_ds' GROUP BY source) AS f ON s.source = f.source ORDER BY s.source",
    ):
        out[f"ml_ds_mentioning_python_{source}"] = share(int(python), int(total))
        out[f"ml_ds_mentioning_sql_{source}"] = share(int(sql), int(total))

    for source, employers, with_lifetime in rows(
        ch,
        "SELECT source, count(), countIf(lifetime_median_days IS NOT NULL) FROM marts.mart_employers "
        "GROUP BY source ORDER BY source",
    ):
        out[f"employers_with_lifetime_{source}"] = share(int(with_lifetime), int(employers))
    return out


def extraction_model(run: dict[str, Any]) -> str:
    """var extraction_model of the build: a --vars override, else dbt_project.yml."""
    override = (run.get("args") or {}).get("vars") or {}
    if isinstance(override, dict) and "extraction_model" in override:
        return str(override["extraction_model"])
    project = yaml.safe_load((ROOT / "dbt" / "dbt_project.yml").read_text(encoding="utf-8"))
    return str(project["vars"]["extraction_model"])


def extracted_metrics(ch: ClickHouse) -> dict[str, dict[str, Any]]:
    """The marts on extracted fields (models/extracted): rows, and per source how many of the units they
    count have fields, valid fields and IT roles, and how many skill spellings the dictionary v1 maps."""
    out: dict[str, dict[str, Any]] = {}
    for mart in EXTRACTED_MARTS:
        out[f"rows_{mart}"] = {"value": ch.scalar(f"SELECT count() FROM marts.{mart}")}
    for source, units, with_fields, valid, it, linking_habr, written, in_dictionary in rows(
        ch,
        "SELECT source, units, units_with_fields, units_valid, units_it, units_linking_habr, skills_written, "
        "skills_in_dictionary FROM marts.mart_extraction_coverage WHERE level = 'source' ORDER BY source",
    ):
        out[f"coverage_{source}"] = share(int(with_fields), int(units))
        out[f"valid_{source}"] = share(int(valid), int(with_fields))
        out[f"it_{source}"] = share(int(it), int(valid))
        out[f"skills_in_dictionary_{source}"] = share(int(in_dictionary), int(written))
        if source == "telegram":
            out["telegram_linking_habr"] = share(int(linking_habr), int(units))
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
        history = history_metrics(ch) | peak_memory(ch, run, ("hist_",))
        marts = marts_metrics(ch) | model_seconds(run, MART_MODELS) | peak_memory(ch, run, MART_MODELS)
        extracted = (
            extracted_metrics(ch)
            if ch.scalar(
                "SELECT count() FROM system.tables WHERE database = 'marts' AND name = 'mart_extraction_coverage'"
            )
            else None
        )
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
    marts_path = write_result(
        "dbt/marts.json",
        marts,
        source_script="dbt/report.py",
        dbt_version=run["metadata"]["dbt_version"],
        build_finished_at=run["metadata"]["generated_at"],
        notes={
            "rules": "role bucket, grade, work format, region group, skill dictionary: dbt/macros/marts.sql",
            "vacancies": "dedup entities within a source family (trudvsem = API and snapshots), latest record",
            "not_it_titles_left_out": "vacancies whose title names no IT role, left out of every mart",
            "ml_ds_data_share_latest_week": "ml_ds, data_engineering and data_analytics of the IT vacancies "
            "active in the source's latest collected week (mart_weekly_dynamics)",
            "salary_stated_rub": "vacancies stating a RUB amount, of the IT vacancies of the collection period",
            "open_to_junior": "graded intern or junior, or no experience required, of the IT vacancies of the bucket",
            "telegram": "vacancy posts; copies of one text within a week count once",
            "build_seconds": "execution time of the model or seed in the build (run_results.json)",
            "peak_memory_mib": "max memory_usage of one INSERT into the table during the build (system.query_log)",
        },
    )
    print(marts_path)
    if extracted is not None:
        extracted_path = write_result(
            "dbt/extracted_marts.json",
            extracted,
            source_script="dbt/report.py",
            dbt_version=run["metadata"]["dbt_version"],
            build_finished_at=run["metadata"]["generated_at"],
            extraction_model=extraction_model(run),
            notes={
                "units": "portal vacancies (dedup entities; all_portals counts a vacancy on several portals once) "
                "and Telegram single-vacancy posts after the weekly cross-post dedup, each once over the period "
                "(mart_extraction_coverage level source)",
                "coverage": "units with extracted fields, of units; Telegram digests and link stubs are never "
                "extracted, so its coverage stays below 1",
                "valid": "units whose answer passed the schema, of units with fields",
                "it": "valid units with a role other than non_it, of valid units",
                "skills_in_dictionary": "extracted skill spellings whose lookup key is in the dictionary v1, "
                "of all spellings written for the units",
                "telegram_linking_habr": "Telegram units that link a Habr Career vacancy, of Telegram units",
            },
        )
        print(extracted_path)
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
