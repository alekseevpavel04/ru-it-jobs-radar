"""Weekly drift and data quality of the incoming vacancies and of the extraction model.

Runs on Monday once the daily chain has loaded Sunday: radar_collect starts at 00:00 UTC and waits
up to six hours for the server, so 10:00 UTC (13:00 Moscow) leaves time for the load and dbt. The
module finds the latest complete week of each source in the data itself, so a late run or a rerun
reports the same week. It writes results/monitor/drift.json and the Evidently HTML reports under
reports/evidently/<week>/, both bind-mounted from the repository (infra/compose). Drift does not fail
the task, it is in the result file; a failed query or an empty staging layer does.

Then the extraction model on the same weeks: drift of the student's fields (radar.monitor.drift_output,
results/monitor/drift_output.json) and its agreement with the teacher on a fixed subsample
(radar.monitor.drift_quality, results/monitor/extraction_quality.json). The teacher runs on the Windows
host and is not reachable from here, so this task reads its answers from the cache the host run left
in reports/evidently/teacher/; a window the cache does not cover is reported without agreement. The
three reports do not depend on each other and only share the server's memory, so they run one at a
time, each whatever the state of the one before (trigger rule all_done).
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.trigger import CronTriggerTimetable

DRIFT = "/opt/radar/.venv/bin/python -m radar.monitor.drift"
OUTPUT_DRIFT = "/opt/radar/.venv/bin/python -m radar.monitor.drift_output"
# Agreement is scored with configs/eval.toml and the skills dictionary; without configs/ in the image
# the task is skipped (exit 99) instead of failing the run or writing a result it could not compute.
QUALITY = (
    "test -f /opt/radar/configs/eval.toml || { echo 'configs/ is not mounted: no agreement this week'; exit 99; }; "
    "/opt/radar/.venv/bin/python -m radar.monitor.drift_quality"
)

with DAG(
    dag_id="radar_drift",
    description="Weekly Evidently drift of the incoming vacancies and of the extraction model's output",
    doc_md=__doc__,
    schedule=CronTriggerTimetable("0 10 * * 1", timezone="UTC"),
    start_date=datetime(2026, 10, 12, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 1, "retry_delay": timedelta(minutes=30)},
    tags=["radar", "monitoring"],
) as dag:
    drift = BashOperator(task_id="drift_report", bash_command=DRIFT, execution_timeout=timedelta(minutes=30))
    output = BashOperator(
        task_id="output_drift",
        bash_command=OUTPUT_DRIFT,
        execution_timeout=timedelta(minutes=30),
        trigger_rule="all_done",
    )
    quality = BashOperator(
        task_id="extraction_quality",
        bash_command=QUALITY,
        execution_timeout=timedelta(minutes=30),
        trigger_rule="all_done",
    )
    drift >> output >> quality
