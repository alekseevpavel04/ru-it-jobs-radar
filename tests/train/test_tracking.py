import json
import sys
from pathlib import Path

import pytest

from radar.train.tracking import LocalTracker, numeric, open_tracker


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
