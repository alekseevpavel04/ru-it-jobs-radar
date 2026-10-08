"""dbt build of staging, history and marts on ClickHouse, then the freshness check of the raw sources.

Scheduled by the asset raw.vacancies, which radar_load emits after the last date of a batch (one
daily date, or every date of a catch-up or backfill). Events that arrive while a build runs wait
and start one more run after it (max_active_runs=1); the scheduler folds every queued event into
that one run. The staging models rebuild only the raw days that got rows since the previous build
(dbt/macros/incremental_by_day.sql), history and marts are rebuilt once per run. A source past its
error threshold (dbt/models/staging/_sources.yml) fails the freshness task.
"""

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset

DBT = "bash /opt/radar/scripts/dbt.sh"
# Emitted by radar_load; same name and URI there.
RAW_VACANCIES = Asset(name="raw.vacancies", uri="clickhouse://clickhouse:8123/raw/vacancies")

with DAG(
    dag_id="radar_dbt",
    description="dbt build and source freshness on the loaded raw layer",
    doc_md=__doc__,
    schedule=[RAW_VACANCIES],
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["radar", "daily"],
) as dag:
    build = BashOperator(task_id="build", bash_command=f"{DBT} build", execution_timeout=timedelta(hours=1))
    freshness = BashOperator(task_id="source_freshness", bash_command=f"{DBT} source freshness")

    build >> freshness
