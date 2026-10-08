"""Skeleton, filled in later (9.2): drift report and retraining of the registered models.

Planned: an Evidently report on incoming vacancies and extraction quality against the teacher on a
subsample, then retraining of the salary model when the drift rule fires, tracked in MLflow.
Paused until then.
"""

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG

with DAG(
    dag_id="radar_retrain",
    description="Skeleton: drift report and retraining (9.2)",
    doc_md=__doc__,
    schedule=None,
    is_paused_upon_creation=True,
    default_args={"owner": "radar"},
    tags=["radar", "skeleton"],
) as dag:
    EmptyOperator(
        task_id="drift_and_retrain",
        doc_md="Placeholder: Evidently drift report, retraining through MLflow when it fires.",
    )
