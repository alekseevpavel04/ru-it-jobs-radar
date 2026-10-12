"""The models over extract.vacancy_fields stay off until var extraction_model names the
extractor the marts read, and parse into staging views with their grain tested once it does.

`dbt parse` with placeholder credentials, as in conftest.py; no ClickHouse is contacted.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXTRACT = {"stg_vacancy_fields", "int_vacancy_fields"}


def _extract_models(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        n["name"]: n for n in manifest["nodes"].values() if n["resource_type"] == "model" and n["fqn"][1] == "extract"
    }


@pytest.fixture(scope="module")
def manifest_on(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("dbt_extract")
    env = os.environ | {
        "CLICKHOUSE_URL": "http://localhost:1",
        "CLICKHOUSE_USER": "parse_only",
        "CLICKHOUSE_PASSWORD": "parse_only",
        "DBT_PROJECT_DIR": str(ROOT / "dbt"),
        "DBT_PROFILES_DIR": str(ROOT / "dbt"),
        "PYTHONIOENCODING": "utf-8",
    }
    args = [
        "parse",
        "--no-partial-parse",
        "--vars",
        "{extraction_model: student-q4}",
        "--target-path",
        str(out / "target"),
        "--log-path",
        str(out / "logs"),
    ]
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


def test_extract_models_are_off_by_default(manifest: dict[str, Any]) -> None:
    assert not _extract_models(manifest)
    disabled = {n[0]["name"] for n in manifest["disabled"].values() if n[0]["resource_type"] == "model"}
    assert disabled >= EXTRACT


def test_extract_models_are_staging_views_with_their_grain(manifest_on: dict[str, Any]) -> None:
    models = _extract_models(manifest_on)
    assert set(models) == EXTRACT
    for name, model in models.items():
        assert model["schema"] == "staging", name
        assert model["config"]["materialized"] == "view", name
        assert model["description"].strip() and model["columns"], name
    grain = {
        (n["attached_node"].rsplit(".", 1)[-1], tuple(n["test_metadata"]["kwargs"]["columns"]))
        for n in manifest_on["nodes"].values()
        if n["resource_type"] == "test" and n.get("test_metadata", {}).get("name") == "unique_combination"
    }
    assert ("stg_vacancy_fields", ("source", "source_id", "content_hash", "model_id")) in grain
    assert ("int_vacancy_fields", ("source", "source_id")) in grain
    sources = {s["name"] for s in manifest_on["sources"].values() if s["source_name"] == "extract"}
    assert sources == {"vacancy_fields"}
