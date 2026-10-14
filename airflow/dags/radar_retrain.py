"""Weekly refit of the salary model when its inputs drift; a refit becomes the challenger in MLflow.

Runs on Monday at 12:00 UTC, two hours after radar_drift starts: its first task writes the summary
results/monitor/drift.json (bind-mounted from the repository) within half an hour, an hour with its
retry.

- decide: airflow/include/retrain_rule.py over the summary. The salary model trains on trudvsem (API
  and snapshots), Habr Career and SuperJob with the inputs role, grade, region group, work format,
  experience, employment type, source and skills. The rule refits when, for one of those sources
  with a drift report, a summary feature that measures one of those inputs (role_bucket, grade,
  region_group, work_format, experience_min_years, no_experience, n_skills) has drifted past the
  summary's threshold and past its own spread between the reference weeks. Otherwise the task
  logs why and the refit is skipped; a summary older than seven days is not acted on.
- refit_and_register: airflow/include/salary_retrain.py. The training run of radar.models.salary
  with its outputs in models/salary_retrain/salary-drift-<week>/, registered through
  radar.monitor.registry as a new version of radar-salary with the alias challenger. champion, the
  version the API serves from models/salary/, does not move; comparing the two and promoting is
  not this DAG's job. A week that already has its version is skipped.
"""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, task
from airflow.timetables.trigger import CronTriggerTimetable

DRIFT_SUMMARY = "/opt/radar/results/monitor/drift.json"
RETRAIN = "/opt/radar/.venv/bin/python /opt/radar/airflow/include/salary_retrain.py"
# salary_retrain.py exits with it when the week already has its version.
SKIP_EXIT_CODE = 99

with DAG(
    dag_id="radar_retrain",
    description="Salary model refit on input drift of the weekly summary, registered as challenger",
    doc_md=__doc__,
    schedule=CronTriggerTimetable("0 12 * * 1", timezone="UTC"),
    start_date=datetime(2026, 10, 12, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 0},
    tags=["radar", "ml"],
) as dag:

    @task.short_circuit(task_id="decide")
    def decide(*, ti) -> bool:
        import json
        import logging
        from pathlib import Path

        from retrain_rule import decide

        summary = json.loads(Path(DRIFT_SUMMARY).read_text(encoding="utf-8"))
        decision = decide(summary, datetime.now(UTC))
        logging.getLogger("radar.airflow.retrain").info(
            "%s: %s", "refit" if decision.retrain else "no refit", decision.reason
        )
        ti.xcom_push(key="decision", value=decision.as_dict() | {"summary_created_at": summary["created_at"]})
        return decision.retrain

    refit = BashOperator(
        task_id="refit_and_register",
        bash_command=RETRAIN,
        env={"RETRAIN_DECISION": "{{ ti.xcom_pull(task_ids='decide', key='decision') | tojson }}"},
        append_env=True,
        skip_on_exit_code=SKIP_EXIT_CODE,
        execution_timeout=timedelta(hours=2),
    )

    decide() >> refit
