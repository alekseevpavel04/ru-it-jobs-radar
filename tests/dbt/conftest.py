"""The dbt manifest, parsed once per session without a database.

`dbt parse` renders every model, macro and yml with placeholder credentials; no ClickHouse is
contacted.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("dbt")
    env = os.environ | {
        "CLICKHOUSE_URL": "http://localhost:1",
        "CLICKHOUSE_USER": "parse_only",
        "CLICKHOUSE_PASSWORD": "parse_only",
        "DBT_PROJECT_DIR": str(ROOT / "dbt"),
        "DBT_PROFILES_DIR": str(ROOT / "dbt"),
        "PYTHONIOENCODING": "utf-8",
    }
    args = ["parse", "--no-partial-parse", "--target-path", str(out / "target"), "--log-path", str(out / "logs")]
    run = subprocess.run(
        [sys.executable, "-P", "-c", "from dbt.cli.main import cli; cli()", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    doc: dict[str, Any] = json.loads((out / "target" / "manifest.json").read_text(encoding="utf-8"))
    return doc
