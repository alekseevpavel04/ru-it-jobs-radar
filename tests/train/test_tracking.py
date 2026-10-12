import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from radar.train.tracking import LocalTracker, MLflowTracker, numeric, open_tracker


def test_local_tracker_keeps_params_metrics_and_status(tmp_path: Path) -> None:
    t = LocalTracker(tmp_path / "run")
    t.log_params({"lr": 2e-4})
    t.log_params({"r": 16})
    t.log_metrics({"loss": 1.5, "epoch": 0.1, "note": "x", "nan": float("nan")}, step=10)
    t.set_tags({"backend": "unsloth"})
    t.end("finished")
    assert json.loads((tmp_path / "run" / "params.json").read_text(encoding="utf-8")) == {"lr": 2e-4, "r": 16}
    lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"step": 10, "loss": 1.5, "epoch": 0.1}
    assert json.loads((tmp_path / "run" / "tags.json").read_text(encoding="utf-8")) == {
        "backend": "unsloth",
        "status": "finished",
    }


def test_numeric_drops_bools_strings_and_non_finite() -> None:
    assert numeric({"a": 1, "b": True, "c": "s", "d": float("inf"), "e": 0.5}) == {"a": 1.0, "e": 0.5}


def test_without_a_uri_only_the_local_copy(tmp_path: Path) -> None:
    assert isinstance(open_tracker(tmp_path, "x", None, resume=False), LocalTracker)


def test_a_uri_without_mlflow_installed_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(SystemExit, match="mlflow is not installed"):
        open_tracker(tmp_path, "x", "http://localhost:5000", resume=False)


class FakeMlflow:
    """The calls MLflowTracker makes, with the server's refusal to change a logged param."""

    def __init__(self, params: dict[str, str]) -> None:
        self.params = dict(params)
        self.tags: dict[str, str] = {}

    def set_tracking_uri(self, uri: str) -> None: ...

    def set_experiment(self, name: str) -> None: ...

    def start_run(self, run_id: str | None = None, run_name: str | None = None) -> Any:
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id or "new"), data=SimpleNamespace(params=self.params))

    def log_params(self, params: dict[str, str]) -> None:
        for k, v in params.items():
            if k in self.params and self.params[k] != v:
                raise ValueError(f"changing param {k}")
        self.params |= params

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags |= tags


def test_a_resumed_run_keeps_logged_params_and_tags_the_changed_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMlflow({"commit": "aaa", "lr": "0.0002"})
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    (tmp_path / "mlflow_run_id.txt").write_text("run1\n", encoding="utf-8")
    tracker = MLflowTracker("http://localhost:5000", tmp_path, "x", resume=True, experiment="e")
    tracker.log_params({"commit": "bbb", "lr": 0.0002, "mode": "resume"})
    assert fake.params == {"commit": "aaa", "lr": "0.0002", "mode": "resume"}
    assert fake.tags == {"resume.commit": "bbb"}
