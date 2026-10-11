"""Daily entry point: pull the day the collection server gathered, then hand the date to radar_load.

The collectors run on the server by cron (docs/vps.md, DECISIONS 11); this machine is on only
sometimes, so here "collect" means pulling the server's complete partitions with
scripts/sync_from_vps.sh. The logical date is the partition date: the run for day D starts at
00:00 UTC of D+1 (03:00 Moscow), after the server's 21:30 Moscow run of D. Missed days are caught
up in order when the scheduler comes back.

It is also the DAG to backfill raw-layer dates that are already on disk (trudvsem snapshots,
Telegram history), one run per date:

    airflow backfill create --dag-id radar_collect --from-date 2026-07-22 --to-date 2026-07-28 --max-active-runs 1

A date without any complete partition is skipped once nothing more can arrive for it; radar_load
skips each source ClickHouse already holds, and dbt runs once after the last loaded date. One date
by hand is a backfill with the same --from-date and --to-date: the DAG takes no manual runs.
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG, task
from airflow.timetables.interval import CronDataIntervalTimetable

# Sources the server collects every day; the local backfills (trudvsem snapshots before
# 2026-10-09, Telegram) do not go through it.
SERVER_SOURCES = ("trudvsem", "habr", "trudvsem_snapshot")
# Topics, as in radar_load: a date goes on when any of them is complete.
RAW_SOURCES = ("trudvsem", "trudvsem_snapshot", "habr", "telegram", "superjob")
SYNC = f"bash /opt/radar/airflow/scripts/sync_from_vps.sh {' '.join(SERVER_SOURCES)}"
# The server's run of a day may still be going on when the run starts; it ends about 22:30 Moscow,
# so six hours past 03:00 Moscow is late enough to call the day missing.
WAIT_FOR_SERVER = timedelta(hours=6)
POKE_INTERVAL = timedelta(minutes=30)

with DAG(
    dag_id="radar_collect",
    description="Pull the server's raw partitions of the day, then trigger radar_load for it",
    doc_md=__doc__,
    schedule=CronDataIntervalTimetable("0 0 * * *", timezone="UTC"),
    start_date=datetime(2026, 10, 8, tzinfo=UTC),
    catchup=True,
    max_active_runs=1,
    # A manual run with logical date D gets the interval that ends at D, while the scheduled run of
    # D starts there. Once the schedule reaches D, Airflow 3.3 finds the manual run, derives the
    # next date from its interval, gets D again and creates no scheduled run any more: the run of
    # 2026-10-08 held the schedule until a backfill passed it on 2026-10-11.
    allowed_run_types=["scheduled", "backfill"],
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 2, "retry_delay": timedelta(minutes=15)},
    tags=["radar", "daily"],
) as dag:
    sync = BashOperator(
        task_id="sync_from_server",
        doc_md="scripts/sync_from_vps.sh through airflow/scripts/sync_from_vps.sh (ssh setup). Only adds files.",
        bash_command=SYNC,
        execution_timeout=timedelta(hours=1),
    )

    # Runs after a failed sync as well: a date already on disk (a backfill, or a day an earlier
    # sync brought) is loaded even when the server is unreachable; the failed sync stays red.
    @task.sensor(
        task_id="check_partitions",
        mode="reschedule",
        poke_interval=POKE_INTERVAL,
        timeout=WAIT_FOR_SERVER + timedelta(hours=1),
        retries=0,
        trigger_rule="all_done",
    )
    def check_partitions(*, ds: str, data_interval_end: datetime) -> bool:
        """True once any source has a complete partition (`_SUCCESS`) for the date; while the server's
        run may still be going on, syncs again and waits; after that skips the date."""
        import logging
        import os
        import subprocess
        from datetime import date
        from pathlib import Path

        from airflow.sdk.exceptions import AirflowSkipException
        from raw_layer import complete_sources

        log = logging.getLogger("radar.airflow.collect")
        raw = Path(os.environ["RADAR_DATA_DIR"]) / "raw"
        day = date.fromisoformat(ds)
        complete = complete_sources(raw, RAW_SOURCES, day)
        if not complete and datetime.now(UTC) < data_interval_end + WAIT_FOR_SERVER:
            log.info("dt=%s: no complete partition yet, syncing again", ds)
            subprocess.run(SYNC.split(), check=True)
            complete = complete_sources(raw, RAW_SOURCES, day)
        if complete:
            log.info("dt=%s: complete partitions of %s", ds, ", ".join(complete))
            return True
        if datetime.now(UTC) < data_interval_end + WAIT_FOR_SERVER:
            return False
        raise AirflowSkipException(f"dt={ds}: no complete partition of {', '.join(RAW_SOURCES)}, nothing to load")

    trigger_load = TriggerDagRunOperator(
        task_id="trigger_load",
        trigger_dag_id="radar_load",
        trigger_run_id="collect__{{ ds }}",
        logical_date="{{ ds }}",
        reset_dag_run=True,
    )

    sync >> check_partitions() >> trigger_load
