"""Replay the raw partitions of one date into Redpanda and wait until ClickHouse has them.

Triggered by radar_collect for its date, or by hand with a logical date (the partition date).
One task per source runs airflow/include/radar_load.py in the project's environment: a source
without a complete partition for the date is skipped, one ClickHouse already holds in full is
counted and left alone (param `force` replays it anyway), the rest is replayed with
radar.stream.replay and checked row for row. Then radar_dbt is triggered for the same date.
"""

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG, Param

# Topics, one per source.
SOURCES = ("trudvsem", "trudvsem_snapshot", "habr", "telegram", "superjob")
LOAD = "/opt/radar/.venv/bin/python /opt/radar/airflow/include/radar_load.py"
# radar_load.py exits with it when the partition is absent or incomplete.
SKIP_EXIT_CODE = 99

with DAG(
    dag_id="radar_load",
    description="Raw partitions of the date: Parquet to Redpanda to ClickHouse, counts checked",
    doc_md=__doc__,
    schedule=None,
    max_active_runs=1,
    # A snapshot day is about 550 thousand records; two replays at a time fit the worker's memory.
    max_active_tasks=2,
    is_paused_upon_creation=False,
    params={"force": Param(False, type="boolean", description="replay even if ClickHouse holds the partition")},
    default_args={"owner": "radar", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["radar", "daily"],
) as dag:
    loads = [
        BashOperator(
            task_id=f"load_{source}",
            bash_command=f"{LOAD} --source {source} --dt {{{{ ds }}}}{{% if params.force %}} --force{{% endif %}}",
            skip_on_exit_code=SKIP_EXIT_CODE,
            execution_timeout=timedelta(hours=1),
        )
        for source in SOURCES
    ]

    trigger_dbt = TriggerDagRunOperator(
        task_id="trigger_dbt",
        trigger_dag_id="radar_dbt",
        trigger_run_id="load__{{ ds }}",
        logical_date="{{ ds }}",
        reset_dag_run=True,
        trigger_rule="none_failed",
    )

    loads >> trigger_dbt
