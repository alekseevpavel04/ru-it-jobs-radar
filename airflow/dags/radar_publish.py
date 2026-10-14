"""Export of the marts to the site's data, checked and left in the working tree for review.

Scheduled by the asset marts.extracted, which radar_ml_label emits after the day's dbt build and
extraction, or run by hand. The Pages workflow cannot reach ClickHouse, so the site is built from
the export committed to the repository (.github/workflows/site.yml). This DAG makes that export and
stops: it does not commit or push, publishing stays a human step, and report_changes says what to
look at in the diff.

- keep_copy: site/src/data and results/site/export.json copied aside, in the worker's /tmp per run.
- export: python -m radar.site.export, aggregates of marts.* with their denominators, rewritten whole.
- check_aggregates: the per-file checks of src/radar/site/scan_dist.py, which the Pages workflow runs
  on the built site (declared files and row fields only, no per-vacancy keys, links or contacts).
- check_pii: scripts/scan_pii.py on the exported files (the detectors of the raw-layer cleaner).
- restore_on_failure: after a failed export or check the copy goes back, so the working tree never
  holds an export that failed; the run stays failed.
- report_changes: files changed, added and removed, row counts that moved, the last data day.

The worker has no git checkout, so the export records its commit as `unknown` in meta.json and in
results/site/export.json.
"""

from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Asset, task

ROOT = "/opt/radar"
PYTHON = f"{ROOT}/.venv/bin/python"
COPIES = "/tmp/radar_publish"
# Emitted by radar_ml_label; same name and URI there.
EXTRACTED = Asset(name="marts.extracted", uri="clickhouse://clickhouse:8123/marts/extracted")

with DAG(
    dag_id="radar_publish",
    description="Marts to site/src/data as aggregates, checked like the site build; no commit, no push",
    doc_md=__doc__,
    schedule=[EXTRACTED],
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"owner": "radar", "retries": 0},
    tags=["radar", "daily", "site"],
) as dag:

    @task(task_id="keep_copy")
    def keep_copy(*, run_id: str) -> dict[str, object]:
        from pathlib import Path

        from site_export import keep_copy, state

        keep_copy(Path(ROOT), Path(COPIES) / run_id.replace(":", "_"))
        return state(Path(ROOT))

    export = BashOperator(
        task_id="export",
        bash_command=f"cd {ROOT} && {PYTHON} -m radar.site.export",
        execution_timeout=timedelta(minutes=30),
    )

    @task(task_id="check_aggregates")
    def check_aggregates() -> None:
        import logging
        from pathlib import Path

        from airflow.sdk.exceptions import AirflowFailException
        from site_export import load_scan_dist, scan

        problems = scan(Path(ROOT), load_scan_dist(Path(ROOT) / "src" / "radar" / "site" / "scan_dist.py"))
        for problem in problems:
            logging.getLogger("radar.airflow.publish").error("%s", problem)
        if problems:
            raise AirflowFailException(f"{len(problems)} problems in the export, see the log")

    check_pii = BashOperator(
        task_id="check_pii",
        bash_command=f"cd {ROOT} && {PYTHON} scripts/scan_pii.py site/src/data",
        execution_timeout=timedelta(minutes=10),
    )

    @task(task_id="restore_on_failure", trigger_rule="one_failed")
    def restore_on_failure(*, run_id: str) -> None:
        import logging
        import shutil
        from pathlib import Path

        from site_export import restore

        copy = Path(COPIES) / run_id.replace(":", "_")
        restored = restore(Path(ROOT), copy)
        shutil.rmtree(copy)
        logging.getLogger("radar.airflow.publish").warning("export rejected, restored %s", ", ".join(restored))

    @task(task_id="report_changes")
    def report_changes(before: dict[str, object], *, run_id: str) -> dict[str, object]:
        import logging
        import shutil
        from pathlib import Path

        from site_export import changes, state

        diff = changes(before, state(Path(ROOT)))
        log = logging.getLogger("radar.airflow.publish")
        log.info("data through %s -> %s", *diff["data_through"])
        for name in ("changed", "added", "removed"):
            log.info("%s: %s", name, ", ".join(diff[name]) or "-")
        for name, (old, new) in diff["rows"].items():
            log.info("rows of %s: %s -> %s", name, old, new)
        log.info("left in the working tree for review; nothing committed or pushed")
        shutil.rmtree(Path(COPIES) / run_id.replace(":", "_"))
        return diff

    before = keep_copy()
    aggregates, pii = check_aggregates(), check_pii
    before >> export >> aggregates >> pii
    [export, aggregates, pii] >> restore_on_failure()
    pii >> report_changes(before)
