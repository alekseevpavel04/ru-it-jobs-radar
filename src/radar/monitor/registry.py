"""MLflow model registry: the extractor and the salary model, with aliases.

What is registered is declared in infra/compose/mlflow/registry.toml. Every entry becomes one run of
the `radar-registry` experiment, which holds the model files under `model/`, the result files its
metrics come from under `results/` and provenance files under `provenance/`, and one version of a
registered model whose source is that run's `model/`. Aliases rather than stages, which MLflow
deprecated in 2.9: `champion` is the version in service, `challenger` a candidate measured against it.

Re-running is the workflow. An entry already registered is found by its key and left alone without
reading its local files: the registry keeps its own copy, so a training run may overwrite
models/salary/ with the next version. A new entry is fingerprinted (sha256 of its model files), and
files identical to a registered version stop the run, so one set of weights never gets two keys.
Aliases are synced to the file.

    python -m radar.monitor.registry           register new entries, sync aliases
    python -m radar.monitor.registry --list    models, versions, aliases and headline metrics
"""

import argparse
import logging
import sys
import time
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Run, RunStatus
from mlflow.entities.model_registry import ModelVersion

from radar.collect import setup_logging
from radar.config import ROOT, require
from radar.monitor.fingerprint import Artifact, fingerprint
from radar.results import ResultFile

log = logging.getLogger("radar.monitor.registry")

REGISTRY_FILE = ROOT / "infra" / "compose" / "mlflow" / "registry.toml"
EXPERIMENT = "radar-registry"
ALIASES = ("champion", "challenger")
MODEL_DIR = "model"
KEY_TAG = "radar.key"
MODEL_TAG = "radar.model"
FINGERPRINT_TAG = "radar.fingerprint"
TRAIN_RUN_TAG = "radar.train_run_id"


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResultRef:
    path: Path
    prefix: str = ""
    include: tuple[str, ...] = ()


@dataclass(frozen=True)
class Entry:
    model: str
    key: str
    alias: str | None
    description: str
    artifacts: tuple[Artifact, ...]
    results: tuple[ResultRef, ...]
    provenance: tuple[Path, ...] = ()
    train_run_id_file: Path | None = None


@dataclass(frozen=True)
class ModelSpec:
    description: str
    headline: tuple[str, ...] = ()


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelSpec]
    entries: tuple[Entry, ...]

    def desired_aliases(self, model: str) -> dict[str, str]:
        """alias -> key of the entry it should point at."""
        return {e.alias: e.key for e in self.entries if e.model == model and e.alias}


@dataclass
class Outcome:
    registered: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    aliases_set: list[str] = field(default_factory=list)
    aliases_removed: list[str] = field(default_factory=list)


def load_registry(path: Path = REGISTRY_FILE, root: Path = ROOT) -> Registry:
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    models = {
        name: ModelSpec(description=spec["description"], headline=tuple(spec.get("headline", [])))
        for name, spec in doc.get("models", {}).items()
    }
    entries = tuple(
        Entry(
            model=raw["model"],
            key=raw["key"],
            alias=raw.get("alias"),
            description=raw.get("description", ""),
            artifacts=tuple(Artifact(root / a["path"], a["dest"]) for a in raw["artifacts"]),
            results=tuple(
                ResultRef(root / r["path"], r.get("prefix", ""), tuple(r.get("include", []))) for r in raw["results"]
            ),
            provenance=tuple(root / p for p in raw.get("provenance", [])),
            train_run_id_file=root / raw["train_run_id_file"] if "train_run_id_file" in raw else None,
        )
        for raw in doc.get("version", [])
    )
    registry = Registry(models, entries)
    validate(registry)
    return registry


def validate(registry: Registry) -> None:
    keys: set[tuple[str, str]] = set()
    aliases: set[tuple[str, str]] = set()
    for e in registry.entries:
        where = f"{e.model}/{e.key}"
        if e.model not in registry.models:
            raise RegistryError(f"{where}: model {e.model} has no [models.{e.model}] section")
        if (e.model, e.key) in keys:
            raise RegistryError(f"{where}: the key is declared twice")
        keys.add((e.model, e.key))
        if e.alias is not None:
            if e.alias not in ALIASES:
                raise RegistryError(f"{where}: alias {e.alias!r} is not one of {', '.join(ALIASES)}")
            if (e.model, e.alias) in aliases:
                raise RegistryError(f"{where}: alias {e.alias} is given to two versions")
            aliases.add((e.model, e.alias))
        if not e.artifacts:
            raise RegistryError(f"{where}: no model files")


def require_files(entry: Entry) -> None:
    paths = [*(a.path for a in entry.artifacts), *(r.path for r in entry.results), *entry.provenance]
    if entry.train_run_id_file is not None:
        paths.append(entry.train_run_id_file)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise RegistryError(f"{entry.model}/{entry.key}: missing {', '.join(str(p) for p in missing)}")


def metrics_of(ref: ResultRef) -> dict[str, float]:
    """Numeric metrics of a result file, with their bootstrap bounds as `<name>_ci_low`/`_ci_high`."""
    doc = ResultFile.model_validate_json(ref.path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for name, metric in doc.metrics.items():
        if ref.include and not name.startswith(ref.include):
            continue
        if isinstance(metric.value, str):
            continue
        out[ref.prefix + name] = float(metric.value)
        for bound in ("ci_low", "ci_high"):
            value = (metric.model_extra or {}).get(bound)
            if isinstance(value, int | float):
                out[f"{ref.prefix}{name}_{bound}"] = float(value)
    return out


def entry_metrics(entry: Entry) -> dict[str, float]:
    merged: dict[str, float] = {}
    for ref in entry.results:
        for name, value in metrics_of(ref).items():
            if name in merged and merged[name] != value:
                raise RegistryError(f"{entry.key}: metric {name} differs between result files")
            merged[name] = value
    return merged


def _batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Registrar:
    def __init__(self, client: MlflowClient, registry: Registry) -> None:
        self.client = client
        self.registry = registry
        self.outcome = Outcome()

    def experiment_id(self) -> str:
        experiment = self.client.get_experiment_by_name(EXPERIMENT)
        if experiment is not None:
            return str(experiment.experiment_id)
        return self.client.create_experiment(EXPERIMENT)

    def find_run(self, experiment_id: str, entry: Entry) -> Run | None:
        runs = self.client.search_runs(
            [experiment_id],
            filter_string=f"tags.`{MODEL_TAG}` = '{entry.model}' and tags.`{KEY_TAG}` = '{entry.key}'",
        )
        finished = [r for r in runs if r.info.status == RunStatus.to_string(RunStatus.FINISHED)]
        for stale in (r for r in runs if r.info.status == RunStatus.to_string(RunStatus.RUNNING)):
            # An upload that died half way: the run is closed as failed and the entry uploaded again.
            log.warning("%s: run %s never finished, marking it failed", entry.key, stale.info.run_id)
            self.client.set_terminated(stale.info.run_id, RunStatus.to_string(RunStatus.FAILED))
        if len(finished) > 1:
            raise RegistryError(f"{entry.model}/{entry.key}: {len(finished)} finished runs carry this key")
        return finished[0] if finished else None

    def upload(self, experiment_id: str, entry: Entry, digest: str) -> Run:
        tags = {MODEL_TAG: entry.model, KEY_TAG: entry.key, FINGERPRINT_TAG: digest}
        if entry.train_run_id_file is not None:
            tags[TRAIN_RUN_TAG] = entry.train_run_id_file.read_text(encoding="utf-8").strip()
        if entry.description:
            tags["mlflow.note.content"] = entry.description
        run = self.client.create_run(experiment_id, run_name=entry.key, tags=tags)
        run_id = run.info.run_id
        stamp = int(time.time() * 1000)
        metrics = [Metric(k, v, stamp, 0) for k, v in sorted(entry_metrics(entry).items())]
        for batch in _batches(metrics, 1000):
            self.client.log_batch(run_id, metrics=list(batch))
        for a in entry.artifacts:
            dest = MODEL_DIR if a.dest in ("", ".") else f"{MODEL_DIR}/{a.dest.strip('/')}"
            if a.path.is_dir():
                self.client.log_artifacts(run_id, str(a.path), dest)
            else:
                self.client.log_artifact(run_id, str(a.path), dest)
            log.info("%s: uploaded %s", entry.key, a.path)
        for ref in entry.results:
            self.client.log_artifact(run_id, str(ref.path), "results")
        for path in entry.provenance:
            self.client.log_artifact(run_id, str(path), "provenance")
        self.client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        return self.client.get_run(run_id)

    def ensure_model(self, name: str) -> None:
        spec = self.registry.models[name]
        found = self.client.search_registered_models(filter_string=f"name = '{name}'")
        if not found:
            self.client.create_registered_model(name, description=spec.description)
        elif found[0].description != spec.description:
            self.client.update_registered_model(name, description=spec.description)

    def ensure_version(self, entry: Entry, run: Run) -> ModelVersion:
        versions = self.client.search_model_versions(f"name = '{entry.model}' and run_id = '{run.info.run_id}'")
        if versions:
            return versions[0]
        version = self.client.create_model_version(
            entry.model,
            source=f"{run.info.artifact_uri}/{MODEL_DIR}",
            run_id=run.info.run_id,
            tags={KEY_TAG: entry.key, FINGERPRINT_TAG: run.data.tags[FINGERPRINT_TAG]},
            description=entry.description or None,
        )
        # The client waits for the server to finish the version (up to 300 s) before returning.
        if version.status != "READY":
            raise RegistryError(f"{entry.model} version {version.version}: {version.status}")
        return version

    def registered_as(self, experiment_id: str, model: str, digest: str) -> str | None:
        """Key of a finished run of `model` that already holds exactly these files."""
        runs = self.client.search_runs(
            [experiment_id],
            filter_string=f"tags.`{MODEL_TAG}` = '{model}' and tags.`{FINGERPRINT_TAG}` = '{digest}'",
        )
        finished = [r for r in runs if r.info.status == RunStatus.to_string(RunStatus.FINISHED)]
        return finished[0].data.tags.get(KEY_TAG) if finished else None

    def register(self, entry: Entry) -> ModelVersion:
        experiment_id = self.experiment_id()
        run = self.find_run(experiment_id, entry)
        if run is None:
            require_files(entry)
            digest = fingerprint(entry.artifacts)
            twin = self.registered_as(experiment_id, entry.model, digest)
            if twin is not None:
                raise RegistryError(f"{entry.model}/{entry.key}: these model files are already registered as {twin}")
            log.info("%s: registering %s", entry.model, entry.key)
            run = self.upload(experiment_id, entry, digest)
            self.outcome.registered.append(f"{entry.model}/{entry.key}")
        else:
            self.outcome.unchanged.append(f"{entry.model}/{entry.key}")
        self.ensure_model(entry.model)
        return self.ensure_version(entry, run)

    def sync_aliases(self, model: str, version_of: dict[str, str]) -> None:
        desired = {alias: version_of[key] for alias, key in self.registry.desired_aliases(model).items()}
        current = dict(self.client.get_registered_model(model).aliases)
        for alias, version in sorted(desired.items()):
            if current.get(alias) != version:
                self.client.set_registered_model_alias(model, alias, version)
                self.outcome.aliases_set.append(f"{model}@{alias} -> v{version}")
        for alias in sorted(set(current) - set(desired)):
            self.client.delete_registered_model_alias(model, alias)
            self.outcome.aliases_removed.append(f"{model}@{alias}")

    def apply(self) -> Outcome:
        version_of: dict[str, dict[str, str]] = {}
        for entry in self.registry.entries:
            version = self.register(entry)
            version_of.setdefault(entry.model, {})[entry.key] = str(version.version)
        for model, versions in version_of.items():
            self.sync_aliases(model, versions)
        return self.outcome


def listing(client: MlflowClient, registry: Registry) -> list[str]:
    lines = []
    for name, spec in registry.models.items():
        found = client.search_registered_models(filter_string=f"name = '{name}'")
        if not found:
            lines.append(f"{name}: not registered")
            continue
        aliases_of: dict[str, list[str]] = {}
        for alias, version in found[0].aliases.items():
            aliases_of.setdefault(str(version), []).append(alias)
        lines.append(name)
        for v in sorted(client.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version)):
            aliases = ",".join(sorted(aliases_of.get(str(v.version), []))) or "-"
            lines.append(f"  v{v.version}  @{aliases}  key={v.tags.get(KEY_TAG)}  run={v.run_id}  {v.status}")
            metrics = client.get_run(v.run_id).data.metrics if v.run_id else {}
            for metric in spec.headline:
                if metric in metrics:
                    lines.append(f"      {metric} = {metrics[metric]:.4f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    # The MLflow client prints run links with an emoji, which a cp1251 console cannot encode.
    setup_logging()
    ap = argparse.ArgumentParser(prog="python -m radar.monitor.registry", description=__doc__.splitlines()[0])
    ap.add_argument("--registry", type=Path, default=REGISTRY_FILE)
    ap.add_argument("--list", action="store_true", help="print the registry instead of changing it")
    args = ap.parse_args(argv)
    client = MlflowClient(tracking_uri=require("MLFLOW_TRACKING_URI"))
    registry = load_registry(args.registry)
    if not args.list:
        outcome = Registrar(client, registry).apply()
        log.info(
            "registered %s; unchanged %s; aliases set %s; removed %s",
            outcome.registered or "-",
            outcome.unchanged or "-",
            outcome.aliases_set or "-",
            outcome.aliases_removed or "-",
        )
    print("\n".join(listing(client, registry)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
