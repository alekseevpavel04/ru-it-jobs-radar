"""Skeleton, filled in later: retraining of the registered models.

Planned: retraining of the salary model when the drift rule fires, tracked in MLflow. The drift
report on incoming vacancies runs weekly in radar_drift; extraction quality against the
teacher on a subsample joins it after the corpus run. Paused until then.
"""

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG

with DAG(
    dag_id="radar_retrain",
    description="Skeleton: retraining when drift fires",
    doc_md=__doc__,
    schedule=None,
    is_paused_upon_creation=True,
    default_args={"owner": "radar"},
    tags=["radar", "skeleton"],
) as dag:
    EmptyOperator(
        task_id="retrain",
        doc_md="Placeholder: retraining through MLflow when the drift of radar_drift fires.",
    )
