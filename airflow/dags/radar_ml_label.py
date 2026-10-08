"""Skeleton, filled in later: extraction of new vacancies by the chosen model.

Planned: take the vacancies of the date that have no extracted fields yet, run the best extractor
from the final test, load the JSON into ClickHouse and rebuild the marts on extracted fields.
Paused until then.
"""

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG

with DAG(
    dag_id="radar_ml_label",
    description="Skeleton: LLM extraction of the date's new vacancies",
    doc_md=__doc__,
    schedule=None,
    is_paused_upon_creation=True,
    default_args={"owner": "radar"},
    tags=["radar", "skeleton"],
) as dag:
    EmptyOperator(
        task_id="extract_new_vacancies",
        doc_md="Placeholder: extractor over vacancies without fields, results into ClickHouse.",
    )
