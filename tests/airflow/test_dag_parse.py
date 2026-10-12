"""DAG parse inside the running Airflow worker: the files as Airflow 3 imports them, with its operators.

Needs the orchestration profile up (scripts/up.sh core orchestration), so it is marked slow and
stays out of scripts/check.sh: `pytest -m slow tests/airflow`. Both commands parse the mounted
airflow/dags locally in the container rather than read what the DAG processor stored.
"""

import json
import os
import subprocess
from typing import Any

import pytest

pytestmark = pytest.mark.slow

CONTAINER = "radar-airflow-scheduler-1"
EXPECTED = {
    "radar_collect",
    "radar_load",
    "radar_dbt",
    "radar_sla",
    "radar_drift",
    "radar_ml_label",
    "radar_retrain",
    "radar_publish",
}


def airflow(*args: str) -> str:
    cmd = ["docker", "exec", CONTAINER, "airflow", *args]
    if os.name == "nt":
        cmd = ["wsl.exe", "-d", os.environ.get("RADAR_WSL_DISTRO", "radar"), "--exec", *cmd]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def last_json(output: str) -> Any:
    """The CLI prints log lines before its JSON."""
    lines = [line for line in output.splitlines() if line.startswith(("[", "{"))]
    return json.loads(lines[-1]) if lines else []


def test_no_import_errors() -> None:
    out = airflow("dags", "list-import-errors", "--local", "--output", "json")
    assert last_json(out) == [], out[-2000:]


def test_every_dag_parses_with_its_tasks() -> None:
    report = last_json(airflow("dags", "report", "--output", "json"))
    dags = {dag_id: int(row["task_num"]) for row in report for dag_id in row["dags"]}
    assert set(dags) == EXPECTED
    assert dags["radar_load"] == 6  # one task per raw topic plus the hand-over to radar_dbt
    assert all(n >= 1 for n in dags.values())
