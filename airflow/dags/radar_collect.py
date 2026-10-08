"""Daily entry point: pull the day the collection server gathered, then hand the date to radar_load.

The collectors run on the server by cron (docs/vps.md, DECISIONS 11); this machine is on only
sometimes, so here "collect" means pulling the server's complete partitions with
scripts/sync_from_vps.sh. The logical date is the partition date: the run for day D starts at
00:00 UTC of D+1 (03:00 Moscow), after the server's 21:30 Moscow run of D. Missed days are caught
up in order when the scheduler comes back.
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable

# Sources the server collects every day; the local backfills (trudvsem snapshots before
# 2026-10-09, Telegram) do not go through it.
SERVER_SOURCES = ("trudvsem", "habr", "trudvsem_snapshot")
DAILY_SOURCES = ("trudvsem", "habr")

with DAG(
    dag_id="radar_collect",
    description="Pull the server's raw partitions of the day, then trigger radar_load for it",
    doc_md=__doc__,
    schedule=CronDataIntervalTimetable("0 0 * * *", timezone="UTC"),
    start_date=datetime(2026, 10, 8, tzinfo=UTC),
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 2, "retry_delay": timedelta(minutes=15)},
    tags=["radar", "daily"],
) as dag:
    sync = BashOperator(
        task_id="sync_from_server",
        doc_md="scripts/sync_from_vps.sh through airflow/scripts/sync_from_vps.sh (ssh setup). Only adds files.",
        bash_command=f"bash /opt/radar/airflow/scripts/sync_from_vps.sh {' '.join(SERVER_SOURCES)}",
        execution_timeout=timedelta(hours=1),
    )

    # The server's run of a day may still be going on: the check retries for six hours.
    check = BashOperator(
        task_id="check_partitions",
        doc_md="Fails unless at least one daily source has a complete partition (`_SUCCESS`) for the date.",
        bash_command="""
found=0
for source in {{ params.sources }}; do
  if [ -f "$RADAR_DATA_DIR/raw/$source/dt={{ ds }}/_SUCCESS" ]; then
    echo "$source: complete"
    found=1
  else
    echo "$source: no complete partition for {{ ds }}"
  fi
done
[ "$found" = 1 ]
""",
        params={"sources": " ".join(DAILY_SOURCES)},
        retries=12,
        retry_delay=timedelta(minutes=30),
    )

    trigger_load = TriggerDagRunOperator(
        task_id="trigger_load",
        trigger_dag_id="radar_load",
        trigger_run_id="collect__{{ ds }}",
        logical_date="{{ ds }}",
        reset_dag_run=True,
    )

    sync >> check >> trigger_load
