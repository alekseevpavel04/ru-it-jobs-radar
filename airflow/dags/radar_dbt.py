"""dbt build of staging, history and marts on ClickHouse, then the freshness check of the raw sources.

Triggered by radar_load for its date. The models read everything raw.vacancies holds, so a run for
any date brings the marts up to the data loaded so far; incremental models add only new days.
A source past its error threshold (dbt/models/staging/_sources.yml) fails the freshness task.
"""

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

DBT = "bash /opt/radar/scripts/dbt.sh"

with DAG(
    dag_id="radar_dbt",
    description="dbt build and source freshness on the loaded raw layer",
    doc_md=__doc__,
    schedule=None,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["radar", "daily"],
) as dag:
    build = BashOperator(task_id="build", bash_command=f"{DBT} build", execution_timeout=timedelta(hours=1))
    freshness = BashOperator(task_id="source_freshness", bash_command=f"{DBT} source freshness")

    build >> freshness
