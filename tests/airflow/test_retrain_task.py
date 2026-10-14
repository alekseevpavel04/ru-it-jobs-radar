"""radar_retrain's task script: outputs kept out of the published paths, challenger set, champion kept."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from radar.models import salary

INCLUDE = Path(__file__).resolve().parents[2] / "airflow" / "include"
spec = importlib.util.spec_from_file_location("airflow_include_salary_retrain", INCLUDE / "salary_retrain.py")
assert spec is not None and spec.loader is not None
retrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain)

DECISION = {"retrain": True, "reason": "input drift", "week": "2026-10-12", "triggers": []}


class FakeClient:
    def __init__(self, keys: list[str], aliases: dict[str, str]) -> None:
        self.keys, self.aliases = keys, aliases

    def search_model_versions(self, filter_string: str) -> list[Any]:
        return [SimpleNamespace(tags={retrain.KEY_TAG: k}) for k in self.keys]

    def get_registered_model(self, name: str) -> Any:
        return SimpleNamespace(aliases=dict(self.aliases))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version


class FakeRegistrar:
    moves_champion = False

    def __init__(self, client: FakeClient, registry: object) -> None:
        self.client = client

    def register(self, entry: Any) -> Any:
        assert entry.model == "radar-salary" and entry.alias is None
        assert [a.dest for a in entry.artifacts] == ["."]
        if self.moves_champion:
            self.client.aliases["champion"] = "9"
        return SimpleNamespace(version=8)


@pytest.fixture
def registrar(monkeypatch: pytest.MonkeyPatch) -> type[FakeRegistrar]:
    monkeypatch.setattr(retrain, "Registrar", FakeRegistrar)
    monkeypatch.setattr(retrain, "load_registry", lambda: None)
    return FakeRegistrar


def test_new_version_becomes_challenger_and_champion_stays(registrar: type[FakeRegistrar], tmp_path: Path) -> None:
    client = FakeClient([], {"champion": "3"})
    assert retrain.register(client, "salary-drift-2026-10-12", tmp_path, DECISION) == "8"
    assert client.aliases == {"champion": "3", "challenger": "8"}


def test_a_moved_champion_fails_the_task(
    registrar: type[FakeRegistrar], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registrar, "moves_champion", True)
    with pytest.raises(RuntimeError, match="champion moved"):
        retrain.register(FakeClient([], {"champion": "3"}), "k", tmp_path, DECISION)


def test_a_week_with_its_version_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRAIN_DECISION", json.dumps(DECISION))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(retrain, "setup_logging", lambda: None)
    monkeypatch.setattr(retrain, "MlflowClient", lambda tracking_uri: FakeClient(["salary-drift-2026-10-12"], {}))
    monkeypatch.setattr(retrain, "refit", lambda out, cfg: pytest.fail("refit ran for a registered week"))
    assert retrain.main() == retrain.SKIP_EXIT_CODE


def test_refit_writes_under_its_own_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frame = pd.DataFrame({"entity_id": ["e1"], "split": ["train"], "salary_mid": [100000.0]})
    seen: dict[str, Path] = {}

    def run(frame: pd.DataFrame, counts: dict[str, int], cfg: object, out: Path) -> dict[str, Any]:
        seen["out"], seen["figure"] = out, salary.SHAP_FIGURE
        return {"metrics": {"dev_gbm_cqr_mae_rub": {"value": 31000.0}}, "model_version": "v"}

    monkeypatch.setattr(retrain, "load_clickhouse", lambda model: (None, None, None))
    monkeypatch.setattr(retrain, "load_corpus_groups", lambda path: {})
    monkeypatch.setattr(retrain, "build_frame", lambda *a: (frame, {"rows": 1}))
    monkeypatch.setattr(salary, "run", run)
    monkeypatch.setattr(salary, "SHAP_FIGURE", salary.SHAP_FIGURE)
    cfg = SimpleNamespace(extraction_model="m", shares=None, min_monthly=10000)
    path = retrain.refit(tmp_path / "salary-drift-2026-10-12", cfg)
    assert seen == {"out": tmp_path / "salary-drift-2026-10-12", "figure": seen["out"] / "salary_shap.png"}
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["metrics"]["dev_gbm_cqr_mae_rub"]["value"] == 31000.0 and doc["model_version"] == "v"
    assert (path.parent / "frame.parquet").exists() and path.parent == tmp_path / "salary-drift-2026-10-12"
