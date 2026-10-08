"""The dbt project parses without a database, and its documentation and grain tests stay in place.

`dbt parse` renders every model, macro and yml with placeholder credentials; no ClickHouse is
contacted. Building on real data is `bash scripts/dbt.sh build` (needs the stack).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
STAGING = {"stg_trudvsem_api", "stg_trudvsem_snapshot", "stg_habr"}
HISTORY = {
    "hist_snapshot_days",
    "hist_observations",
    "hist_calendar",
    "hist_vacancy_scd2",
    "hist_reposts",
    "hist_vacancy_lifetime",
}


@pytest.fixture(scope="module")
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


def _models(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["name"]: n for n in manifest["nodes"].values() if n["resource_type"] == "model"}


def _folder(manifest: dict[str, Any], folder: str) -> dict[str, dict[str, Any]]:
    return {name: m for name, m in _models(manifest).items() if m["fqn"][1] == folder}


def test_staging_models_exist_and_land_in_staging(manifest: dict[str, Any]) -> None:
    models = _folder(manifest, "staging")
    assert set(models) == STAGING | {"stg_vacancies"}
    assert {m["schema"] for m in models.values()} == {"staging"}
    for name in STAGING:
        assert models[name]["config"]["materialized"] == "incremental_by_day"


def test_every_model_and_column_is_described(manifest: dict[str, Any]) -> None:
    for name, model in _models(manifest).items():
        assert model["description"].strip(), name
        assert model["columns"], name
        for column, spec in model["columns"].items():
            assert spec["description"].strip(), f"{name}.{column}"


def test_grain_is_tested_on_each_source_model(manifest: dict[str, Any]) -> None:
    grain = {
        n["attached_node"].rsplit(".", 1)[-1]
        for n in manifest["nodes"].values()
        if n["resource_type"] == "test"
        and n.get("test_metadata", {}).get("name") == "unique_combination"
        and n["test_metadata"]["kwargs"]["columns"] == ["source_id", "dt"]
    }
    assert grain & set(_folder(manifest, "staging")) == STAGING


def test_union_row_count_test_exists(manifest: dict[str, Any]) -> None:
    tests = [n for n in manifest["nodes"].values() if n["resource_type"] == "test"]
    singular = {n["name"] for n in tests if "test_metadata" not in n}
    assert "assert_stg_vacancies_rows_match_sources" in singular


def test_every_source_has_its_own_freshness(manifest: dict[str, Any]) -> None:
    sources = {s["name"]: s for s in manifest["sources"].values()}
    assert set(sources) == {"trudvsem", "trudvsem_snapshot", "habr"}
    for name, spec in sources.items():
        freshness = spec["freshness"]
        assert freshness["warn_after"]["count"] and freshness["error_after"]["count"], name
        assert freshness["filter"] == f"source = '{name}'"
        assert spec["loaded_at_field"] == "fetched_at"


def test_history_models_and_their_tests(manifest: dict[str, Any]) -> None:
    models = _folder(manifest, "history")
    assert set(models) == HISTORY
    assert {m["schema"] for m in models.values()} == {"staging"}
    assert models["hist_snapshot_days"]["config"]["materialized"] == "incremental_by_day"
    unit = {n["model"] for n in manifest["unit_tests"].values()}
    assert {"hist_calendar", "hist_vacancy_scd2", "hist_reposts", "hist_vacancy_lifetime"} <= unit
    tests = [n for n in manifest["nodes"].values() if n["resource_type"] == "test"]
    singular = {n["name"] for n in tests if "test_metadata" not in n}
    assert {"assert_hist_scd2_intervals", "assert_hist_one_current_per_open_vacancy"} <= singular
    grain = {
        tuple(n["test_metadata"]["kwargs"]["columns"])
        for n in tests
        if n.get("test_metadata", {}).get("name") == "unique_combination"
        and n["attached_node"].endswith(".hist_vacancy_scd2")
    }
    assert ("source", "source_id", "valid_from") in grain
