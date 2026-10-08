"""Replay the raw partitions of one date into Redpanda and wait until ClickHouse has them.

Triggered by radar_collect for its date (daily, catch-up or backfill), or by hand with a logical
date (the partition date). One task per source runs airflow/include/radar_load.py in the project's
environment: a source without a complete partition for the date, or one ClickHouse already holds
in full, is skipped (param `force` replays it anyway); the rest is replayed with
radar.stream.replay and checked row for row.

The last task hands the replayed partitions over to dbt by emitting the asset raw.vacancies, which
schedules radar_dbt. While more radar_load or radar_collect runs are queued (a catch-up or a
backfill), it only remembers them (airflow/include/dbt_handover.py), so a batch of dates ends in
one dbt build rather than one per date. Nothing replayed, nothing emitted.
"""

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset, Param, task

# Topics, one per source.
SOURCES = ("trudvsem", "trudvsem_snapshot", "habr", "telegram", "superjob")
LOAD = "/opt/radar/.venv/bin/python /opt/radar/airflow/include/radar_load.py"
# radar_load.py exits with it when there is nothing to load.
SKIP_EXIT_CODE = 99
# Same name and URI in radar_dbt, which it schedules.
RAW_VACANCIES = Asset(name="raw.vacancies", uri="clickhouse://clickhouse:8123/raw/vacancies")

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

    # all_done, so that what the successful loads replayed is remembered even when another load
    # failed; the task then fails as well, which keeps the run failed (it is the only leaf).
    @task(task_id="hand_over_to_dbt", outlets=[RAW_VACANCIES], trigger_rule="all_done", retries=0)
    def hand_over_to_dbt(*, ti, run_id: str, outlet_events) -> None:
        import json
        import logging

        from airflow.sdk import Variable
        from airflow.sdk.exceptions import AirflowFailException, AirflowSkipException
        from dbt_handover import PENDING_VARIABLE, hand_over

        log = logging.getLogger("radar.airflow.load")
        load_ids = [t.task_id for t in loads]
        states = ti.get_task_states(dag_id="radar_load", task_ids=load_ids, run_ids=[run_id])[run_id]
        replayed = []
        for task_id in load_ids:
            if states.get(task_id) == "success":
                summary = json.loads(ti.xcom_pull(task_ids=task_id))
                replayed.append(f"{summary['source']}/{summary['dt']}")
        failed = sorted(t for t in load_ids if states.get(t) not in ("success", "skipped"))
        # Queued only: a running radar_collect may sit in its sensor for hours and then skip its
        # date, which would leave the pending partitions waiting for the next day.
        waiting = ti.get_dr_count(dag_id="radar_load", states=["queued"]) + ti.get_dr_count(
            dag_id="radar_collect", states=["queued"]
        )

        pending = Variable.get(PENDING_VARIABLE, default=[], deserialize_json=True)
        now, later = hand_over(pending, replayed, waiting if not failed else 1)
        Variable.set(PENDING_VARIABLE, later, serialize_json=True)
        log.info("replayed %s; queued runs %d; handing over %s; pending %s", replayed, waiting, now, later)
        if failed:
            raise AirflowFailException(f"load tasks did not succeed: {', '.join(failed)}")
        if not now:
            raise AirflowSkipException("nothing to hand over now" if not later else "a later run hands over")
        outlet_events[RAW_VACANCIES].extra = {"partitions": now}

    loads >> hand_over_to_dbt()
