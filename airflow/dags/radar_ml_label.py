"""Fields for the vacancy versions that have none yet, by the fine-tuned student on the CPU.

Scheduled by the asset marts, which the build of radar_dbt emits once the day's raw layer is staged.

extract_pending runs radar.extract.corpus_run, the runner of the whole-corpus extraction,
against the compose service `student`: the Q4_K_M GGUF in llama-server on two CPUs, rendered and
checked as in the corpus run (airflow/include/student_cpu.py). Pending is what the runner enumerates (the
latest staged version of every IT candidate of the portals, the latest version of every Telegram
post the corpus rules keep) minus the versions extract.vacancy_fields holds for this model
(qwen3_1p7b_r16_clean-q4_k_m) or for the corpus run's (qwen3_1p7b_r16_clean-awq). Rows go to
extract.vacancy_fields and data/corpus_run/<model_id>/ as in the corpus run, one line per version.

The cap. On the CPU a text takes about 20 s, and one collection day brings hundreds of new versions
(681 on 2026-10-13), so a run stops sending requests after params.max_records versions or
params.time_budget_minutes, whichever comes first. Requests in flight finish; what is left waits for
the next run, and the runner logs how much (also in the run's entry of meta.json). params.workers
requests go at a time: the server has two slots and /extract of the API asks the same server, so
the default of one leaves the API a slot.

summarize logs the run's entry of meta.json and lets rebuild_extracted_marts go on only when rows
were inserted: dbt build of stg_vacancy_fields and everything built on it (int_vacancy_fields,
int_extracted_fields, the extracted marts and their tests). The marts read the fields of the model
the dbt var extraction_model names, the AWQ checkpoint of the corpus run; until they read this
model's rows as well, the rows land in extract.vacancy_fields and leave the marts as they were.

hand_over emits the asset marts.extracted, which schedules radar_publish; it runs when nothing was
pending too, and not after a failure.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset, Param, task

STUDENT_URL = "http://student:8080"
STUDENT_MODEL = "qwen3_1p7b_r16_clean-q4_k_m"
CORPUS_MODEL = "qwen3_1p7b_r16_clean-awq"
RUNNER = "/opt/radar/.venv/bin/python -m radar.extract.corpus_run"
DBT = "bash /opt/radar/scripts/dbt.sh"
EXTRACT = (
    f"{RUNNER} run --backend student_cpu:backend --backend-opt base_url={STUDENT_URL}"
    f" --skip-done-by {CORPUS_MODEL} --limit {{{{ params.max_records }}}}"
    " --time-budget {{ params.time_budget_minutes * 60 }} --workers {{ params.workers }}"
    " --insert-every 10 --log-every 10"
)
# Emitted by radar_dbt; same name and URI there.
MARTS = Asset(name="marts", uri="clickhouse://clickhouse:8123/marts")
# Schedules radar_publish; same name and URI there.
EXTRACTED = Asset(name="marts.extracted", uri="clickhouse://clickhouse:8123/marts/extracted")

with DAG(
    dag_id="radar_ml_label",
    description="Student on the CPU over versions without fields, capped per run; extracted marts rebuilt",
    doc_md=__doc__,
    schedule=[MARTS],
    max_active_runs=1,
    is_paused_upon_creation=False,
    params={
        "max_records": Param(150, type="integer", minimum=1, description="versions per run at most"),
        "time_budget_minutes": Param(
            60, type="integer", minimum=1, maximum=240, description="no new request after this long"
        ),
        "workers": Param(1, type="integer", minimum=1, maximum=2, description="requests in flight (server slots)"),
    },
    default_args={"owner": "radar", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["radar", "daily", "ml"],
) as dag:
    extract = BashOperator(
        task_id="extract_pending",
        bash_command=EXTRACT,
        # the budget at most, plus enumeration and the last requests in flight
        execution_timeout=timedelta(hours=4, minutes=30),
    )

    @task.short_circuit(task_id="summarize", ignore_downstream_trigger_rules=False)
    def summarize(*, dag_run) -> bool:
        """The run's entry in meta.json; rows inserted means the extracted marts need a rebuild."""
        import json
        import logging
        import os
        from pathlib import Path

        log = logging.getLogger("radar.airflow.ml_label")
        meta = Path(os.environ["RADAR_DATA_DIR"]) / "corpus_run" / STUDENT_MODEL / "meta.json"
        run = json.loads(meta.read_text(encoding="utf-8"))["runs"][-1]
        if datetime.fromisoformat(run["started_at"]) < dag_run.start_date.replace(microsecond=0):
            raise RuntimeError(f"the last run in {meta} started at {run['started_at']}, before this DAG run")
        left = run["versions_found"] - run["rows"]
        log.info(
            "run %s: %d pending, %d rows written (%d requests, %d valid JSON, %d reused, %d failed), %d inserted, "
            "%d left for the next run%s",
            run["run_id"],
            run["versions_found"],
            run["rows"],
            run["requests"],
            run["valid_requests"],
            run["reused"],
            run["errors"],
            run["inserted"],
            left,
            " (time budget spent)" if run["stopped_by_time_budget"] else "",
        )
        return bool(run["inserted"])

    rebuild = BashOperator(
        task_id="rebuild_extracted_marts",
        bash_command=f"{DBT} build --select stg_vacancy_fields+",
        execution_timeout=timedelta(hours=1),
    )

    @task(task_id="hand_over", outlets=[EXTRACTED], trigger_rule="none_failed", retries=0)
    def hand_over() -> None:
        """Marks the extracted marts current for radar_publish."""

    extract >> summarize() >> rebuild >> hand_over()
