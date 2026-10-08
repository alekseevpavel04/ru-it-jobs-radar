"""Write the outcome of Airflow DAG runs to results/ from the Airflow REST API.

    python airflow/run_report.py radar_collect:manual__2026-10-08 radar_load:collect__2026-10-08 \
        radar_dbt:load__2026-10-08 --out airflow/first_run.json

Per run: state, logical date, wall time (run start to end, failed tries and retry waits included)
and the span of the task tries that count; per task: state, tries and seconds of the last try, and
the JSON summary a load task left as its XCom (rows in Parquet and in ClickHouse). Runs on the host
against the API server of the orchestration profile, account admin (AIRFLOW_ADMIN_PASSWORD).
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any

import httpx

from radar.collect import setup_logging
from radar.config import require
from radar.results import write_result


def seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 1)


def xcom_summary(client: httpx.Client, dag_id: str, run_id: str, task_id: str) -> dict[str, Any] | None:
    response = client.get(f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/xcomEntries/return_value")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    value = response.json()["value"]
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_record(client: httpx.Client, dag_id: str, run_id: str) -> dict[str, Any]:
    run = client.get(f"/api/v2/dags/{dag_id}/dagRuns/{run_id}").raise_for_status().json()
    instances = client.get(f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances").raise_for_status().json()
    tasks = []
    for ti in sorted(instances["task_instances"], key=lambda t: t["start_date"] or ""):
        task: dict[str, Any] = {
            "task_id": ti["task_id"],
            "state": ti["state"],
            "try_number": ti["try_number"],
            "seconds": seconds(ti["start_date"], ti["end_date"]),
        }
        if ti["task_id"].startswith("load_") and ti["state"] == "success":
            task["summary"] = xcom_summary(client, dag_id, run_id, ti["task_id"])
        tasks.append(task)
    starts = [ti["start_date"] for ti in instances["task_instances"] if ti["start_date"]]
    ends = [ti["end_date"] for ti in instances["task_instances"] if ti["end_date"]]
    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "logical_date": run["logical_date"],
        "state": run["state"],
        "start_date": run["start_date"],
        "end_date": run["end_date"],
        "seconds": seconds(run["start_date"], run["end_date"]),
        # First start to last end over the tries that count: excludes failed tries and retry waits.
        "seconds_tasks_span": seconds(min(starts, default=None), max(ends, default=None)),
        "tasks": tasks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("runs", nargs="+", help="dag_id:run_id")
    parser.add_argument("--out", required=True, help="path under results/, e.g. airflow/first_run.json")
    parser.add_argument("--url", default="http://localhost:8090", help="Airflow API server")
    args = parser.parse_args(argv)
    setup_logging()

    token = httpx.post(
        f"{args.url}/auth/token",
        json={"username": "admin", "password": require("AIRFLOW_ADMIN_PASSWORD")},
        timeout=30,
    ).raise_for_status()
    headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
    with httpx.Client(base_url=args.url, headers=headers, timeout=30) as client:
        runs = [run_record(client, *target.split(":", 1)) for target in args.runs]

    succeeded = sum(r["state"] == "success" for r in runs)
    loads = [t["summary"] for r in runs for t in r["tasks"] if t.get("summary")]
    metrics: dict[str, dict[str, Any]] = {
        "runs_success": {"value": succeeded, "n": succeeded, "denominator": len(runs)},
        "seconds_wall": {"value": round(sum(r["seconds"] or 0 for r in runs), 1)},
        "seconds_tasks_span": {"value": round(sum(r["seconds_tasks_span"] or 0 for r in runs), 1)},
        "rows_replayed": {"value": sum(s["rows_parquet"] for s in loads if s.get("replayed"))},
    }
    path = write_result(
        args.out,
        metrics,
        source_script="airflow/run_report.py",
        method=(
            "states and start/end times of the DAG runs and task instances from the Airflow REST API; seconds are "
            "end minus start as Airflow recorded them, for a task its last try; seconds_tasks_span runs from the "
            "first task start to the last task end; load summaries are the XCom each load task printed"
        ),
        runs=runs,
    )
    print(f"{path}: {succeeded} of {len(runs)} runs succeeded")
    return 0 if succeeded == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
