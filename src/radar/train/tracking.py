"""Run tracking: MLflow (compose profile ml, MLFLOW_TRACKING_URI) and a local copy next to the adapter.

The local copy (metrics.jsonl, params.json, tags.json in the run directory) is written in every case,
so a run is readable without the server; MLflow is where runs are compared. A resumed run continues
the same MLflow run, whose id is kept in the run directory.
"""

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("radar.train.tracking")

EXPERIMENT = "radar-sft"
RUN_ID_FILE = "mlflow_run_id.txt"


class Tracker(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...

    def set_tags(self, tags: Mapping[str, str]) -> None: ...

    def end(self, status: str) -> None: ...


def numeric(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in values.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    }


class LocalTracker:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _merge(self, name: str, values: Mapping[str, Any]) -> None:
        path = self.dir / name
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        path.write_text(json.dumps(current | dict(values), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._merge("params.json", params)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **numeric(metrics)}) + "\n")

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._merge("tags.json", tags)

    def end(self, status: str) -> None:
        self._merge("tags.json", {"status": status})


class MLflowTracker:
    def __init__(self, uri: str, run_dir: Path, run_name: str, resume: bool, experiment: str) -> None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as e:
            raise SystemExit(
                "MLFLOW_TRACKING_URI is set but mlflow is not installed in this venv "
                "(install mlflow-skinny into ~/venvs/train, or pass --no-mlflow)"
            ) from e
        self.mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        id_file = run_dir / RUN_ID_FILE
        run_id = id_file.read_text(encoding="utf-8").strip() if resume and id_file.exists() else None
        run = mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name)
        id_file.write_text(run.info.run_id + "\n", encoding="utf-8")
        log.info("MLflow run %s at %s", run.info.run_id, uri)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.mlflow.log_params({k: str(v)[:6000] for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.mlflow.log_metrics(numeric(metrics), step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.mlflow.set_tags(dict(tags))

    def end(self, status: str) -> None:
        self.mlflow.set_tag("status", status)
        self.mlflow.end_run(status="FINISHED" if status == "finished" else "FAILED")


class Both:
    def __init__(self, *trackers: Tracker) -> None:
        self.trackers = trackers

    def log_params(self, params: Mapping[str, Any]) -> None:
        for t in self.trackers:
            t.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for t in self.trackers:
            t.log_metrics(metrics, step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        for t in self.trackers:
            t.set_tags(tags)

    def end(self, status: str) -> None:
        for t in self.trackers:
            t.end(status)


def open_tracker(run_dir: Path, run_name: str, uri: str | None, resume: bool, experiment: str = EXPERIMENT) -> Tracker:
    local = LocalTracker(run_dir)
    if uri is None:
        return local
    return Both(local, MLflowTracker(uri, run_dir, run_name, resume, experiment))
