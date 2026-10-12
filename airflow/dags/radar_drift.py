"""Weekly drift and data quality of the incoming vacancies per source (radar.monitor.drift).

Runs on Monday once the daily chain has loaded Sunday: radar_collect starts at 00:00 UTC and waits
up to six hours for the server, so 10:00 UTC (13:00 Moscow) leaves time for the load and dbt. The
module finds the latest complete week of each source in the data itself, so a late run or a rerun
reports the same week. It writes results/monitor/drift.json and the Evidently HTML reports under
reports/evidently/<week>/, both bind-mounted from the repository (infra/compose). Drift does not fail
the task, it is in the result file; a failed query or an empty staging layer does.
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.trigger import CronTriggerTimetable

DRIFT = "/opt/radar/.venv/bin/python -m radar.monitor.drift"

with DAG(
    dag_id="radar_drift",
    description="Weekly Evidently drift and data quality report of the incoming vacancies",
    doc_md=__doc__,
    schedule=CronTriggerTimetable("0 10 * * 1", timezone="UTC"),
    start_date=datetime(2026, 10, 12, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 1, "retry_delay": timedelta(minutes=30)},
    tags=["radar", "monitoring"],
) as dag:
    BashOperator(task_id="drift_report", bash_command=DRIFT, execution_timeout=timedelta(minutes=30))
