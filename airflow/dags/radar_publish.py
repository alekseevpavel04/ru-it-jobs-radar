"""Skeleton, filled in later (10.3): aggregates for the public site.

Planned: export the marts into site/data and rebuild the static site; site/data holds aggregates
only, never vacancy texts. Paused until then.
"""

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG

with DAG(
    dag_id="radar_publish",
    description="Skeleton: mart aggregates for the site (10.3)",
    doc_md=__doc__,
    schedule=None,
    is_paused_upon_creation=True,
    default_args={"owner": "radar"},
    tags=["radar", "skeleton"],
) as dag:
    EmptyOperator(
        task_id="export_site_aggregates",
        doc_md="Placeholder: marts to site/data as aggregates only.",
    )
