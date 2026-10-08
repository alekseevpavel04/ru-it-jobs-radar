"""Freshness SLA of the raw layer: is the latest complete partition of each source recent enough.

Airflow 3 dropped SLA misses. Deadline alerts replace them, but they measure how late a DAG run
finishes, not whether data arrived, and their callbacks need a triggerer this stack does not run.
So an hourly task checks the data itself, apart from the pipeline, so that a stuck pipeline still
raises the alert: per source the hours since its latest complete partition was due locally
(airflow/include/raw_layer.py; thresholds of the dbt source freshness). Every check sets the gauge
`radar.raw.lag_hours` per source; a source past a threshold writes a `raw freshness breach` log
line with its fields and increments the counter `radar.sla.breach` (tags source, level). The task
itself stays green: the alert lives in the log and in the metrics (statsd-exporter, the monitoring reads
it from there).
"""

from datetime import UTC, datetime

from airflow.sdk import DAG, task
from airflow.timetables.trigger import CronTriggerTimetable

with DAG(
    dag_id="radar_sla",
    description="Hourly freshness check of the raw layer per source; breaches to the log and to metrics",
    doc_md=__doc__,
    schedule=CronTriggerTimetable("15 * * * *", timezone="UTC"),
    start_date=datetime(2026, 10, 8, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 0},
    tags=["radar", "monitoring"],
) as dag:

    @task(task_id="raw_freshness")
    def raw_freshness() -> list[dict[str, object]]:
        import os
        from pathlib import Path

        import structlog
        from airflow.sdk.observability import stats
        from raw_layer import freshness

        log = structlog.get_logger("radar.sla")
        rows = freshness(Path(os.environ["RADAR_DATA_DIR"]) / "raw", datetime.now(UTC))
        for row in rows:
            if row["lag_hours"] is not None:
                stats.gauge("radar.raw.lag_hours", row["lag_hours"], tags={"source": row["source"]})
            if row["status"] == "pass":
                log.info("raw freshness ok", **row)
                continue
            stats.incr("radar.sla.breach", tags={"source": row["source"], "level": row["status"]})
            if row["status"] == "error":
                log.error("raw freshness breach", **row)
            else:
                log.warning("raw freshness breach", **row)
        return rows

    raw_freshness()
