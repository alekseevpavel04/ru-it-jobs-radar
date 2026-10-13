import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from radar.monitor.fingerprint import Artifact, artifact_files, fingerprint
from radar.monitor.registry import (
    FINGERPRINT_TAG,
    KEY_TAG,
    MODEL_TAG,
    REGISTRY_FILE,
    Registrar,
    RegistryError,
    ResultRef,
    entry_metrics,
    load_registry,
    metrics_of,
)


def result_file(path: Path, metrics: dict[str, Any]) -> Path:
    doc = {
        "source_script": "s.py",
        "command": "python -m x",
        "commit": "abc",
        "created_at": "2026-10-13",
        "metrics": metrics,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "models" / "ext" / "adapter").mkdir(parents=True)
    (tmp_path / "models" / "ext" / "adapter" / "weights.bin").write_bytes(b"w1")
    (tmp_path / "models" / "ext" / "adapter" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "ext" / "q4.gguf").write_bytes(b"gguf")
    (tmp_path / "models" / "salary.pkl").write_bytes(b"pkl")
    (tmp_path / "results").mkdir()
    result_file(
        tmp_path / "results" / "dev.json",
        {
            "macro_f1": {"value": 0.9, "ci_low": 0.88, "ci_high": 0.92},
            "valid_json": {"value": 1.0, "n": 165, "denominator": 165},
            "note": {"value": "text"},
        },
    )
    result_file(tmp_path / "results" / "salary.json", {"dev_mae": {"value": 29000.0}, "data_rows": {"value": 9594}})
    (tmp_path / "train_run.txt").write_text("abc123\n", encoding="utf-8")
    return tmp_path


TOML = """
[models.radar-extractor]
description = "extractor"
headline = ["dev/macro_f1"]

[models.radar-salary]
description = "salary"

[[version]]
model = "radar-extractor"
key = "ext-v1"
alias = "{ext_alias}"
train_run_id_file = "train_run.txt"
artifacts = [{{ path = "models/ext/adapter", dest = "adapter" }}, {{ path = "models/ext/q4.gguf", dest = "gguf" }}]
results = [{{ path = "results/dev.json", prefix = "dev/" }}]

[[version]]
model = "radar-salary"
key = "salary-v0"
alias = "champion"
artifacts = [{{ path = "models/salary.pkl", dest = "." }}]
results = [{{ path = "results/salary.json", include = ["dev_"] }}]
"""


def write_toml(repo: Path, text: str = TOML, **fmt: str) -> Path:
    path = repo / "registry.toml"
    path.write_text(text.format(**({"ext_alias": "champion"} | fmt)), encoding="utf-8")
    return path


def test_the_repo_registry_file_is_valid() -> None:
    registry = load_registry(REGISTRY_FILE)
    assert {e.model for e in registry.entries} == {"radar-extractor", "radar-salary"}
    for model in registry.models:
        assert registry.desired_aliases(model).get("champion"), model


def test_alias_outside_champion_and_challenger_is_refused(repo: Path) -> None:
    with pytest.raises(RegistryError, match="not one of"):
        load_registry(write_toml(repo, ext_alias="production"), root=repo)


def test_one_alias_on_two_versions_is_refused(repo: Path) -> None:
    second = TOML.replace('key = "salary-v0"', 'key = "salary-v1"').split("[[version]]")[2]
    text = TOML + "\n[[version]]" + second
    with pytest.raises(RegistryError, match="given to two versions"):
        load_registry(write_toml(repo, text), root=repo)


def test_metrics_carry_prefix_bounds_and_skip_text(repo: Path) -> None:
    got = metrics_of(ResultRef(repo / "results" / "dev.json", prefix="dev/"))
    assert got == {
        "dev/macro_f1": 0.9,
        "dev/macro_f1_ci_low": 0.88,
        "dev/macro_f1_ci_high": 0.92,
        "dev/valid_json": 1.0,
    }
    assert metrics_of(ResultRef(repo / "results" / "salary.json", include=("dev_",))) == {"dev_mae": 29000.0}


def test_conflicting_metric_across_result_files_is_an_error(repo: Path) -> None:
    result_file(repo / "results" / "other.json", {"macro_f1": {"value": 0.5}})
    entry = load_registry(write_toml(repo), root=repo).entries[0]
    clash = type(entry)(
        **{**entry.__dict__, "results": (*entry.results, ResultRef(repo / "results" / "other.json", "dev/"))}
    )
    with pytest.raises(RegistryError, match="differs"):
        entry_metrics(clash)


def test_fingerprint_follows_contents_and_layout(repo: Path) -> None:
    artifacts = [
        Artifact(repo / "models" / "ext" / "adapter", "adapter"),
        Artifact(repo / "models" / "ext" / "q4.gguf", "gguf"),
    ]
    names = [name for name, _ in artifact_files(artifacts)]
    assert names == ["adapter/config.json", "adapter/weights.bin", "gguf/q4.gguf"]
    before = fingerprint(artifacts)
    assert fingerprint(artifacts) == before
    (repo / "models" / "ext" / "adapter" / "weights.bin").write_bytes(b"w2")
    changed = fingerprint(artifacts)
    assert changed != before
    assert fingerprint([Artifact(artifacts[0].path, "lora"), artifacts[1]]) != changed


class FakeClient:
    """The slice of MlflowClient the registrar uses, in memory."""

    def __init__(self) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, SimpleNamespace] = {}
        self.uploads: list[tuple[str, str, str]] = []
        self.models: dict[str, SimpleNamespace] = {}
        self.versions: dict[str, list[SimpleNamespace]] = {}

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        return SimpleNamespace(experiment_id=self.experiments[name]) if name in self.experiments else None

    def create_experiment(self, name: str) -> str:
        self.experiments[name] = str(len(self.experiments) + 1)
        return self.experiments[name]

    def search_runs(self, experiment_ids: list[str], filter_string: str) -> list[SimpleNamespace]:
        wanted = dict(re.findall(r"tags\.`([^`]+)` = '([^']*)'", filter_string))
        return [r for r in self.runs.values() if all(r.data.tags.get(k) == v for k, v in wanted.items())]

    def create_run(self, experiment_id: str, run_name: str, tags: dict[str, str]) -> SimpleNamespace:
        run_id = f"run{len(self.runs) + 1}"
        info = SimpleNamespace(
            run_id=run_id, status="RUNNING", artifact_uri=f"mlflow-artifacts:/{experiment_id}/{run_id}"
        )
        self.runs[run_id] = SimpleNamespace(info=info, data=SimpleNamespace(tags=dict(tags), metrics={}))
        return self.runs[run_id]

    def log_batch(self, run_id: str, metrics: list[Any]) -> None:
        self.runs[run_id].data.metrics.update({m.key: m.value for m in metrics})

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str) -> None:
        self.uploads.append((run_id, local_dir, artifact_path))

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.uploads.append((run_id, local_path, artifact_path))

    def set_terminated(self, run_id: str, status: str) -> None:
        self.runs[run_id].info.status = status

    def get_run(self, run_id: str) -> SimpleNamespace:
        return self.runs[run_id]

    def search_registered_models(self, filter_string: str) -> list[SimpleNamespace]:
        name = re.search(r"name = '([^']+)'", filter_string)
        return [self.models[name.group(1)]] if name and name.group(1) in self.models else []

    def create_registered_model(self, name: str, description: str) -> None:
        self.models[name] = SimpleNamespace(name=name, description=description, aliases={})
        self.versions[name] = []

    def update_registered_model(self, name: str, description: str) -> None:
        self.models[name].description = description

    def get_registered_model(self, name: str) -> SimpleNamespace:
        return self.models[name]

    def search_model_versions(self, filter_string: str) -> list[SimpleNamespace]:
        name = re.search(r"name = '([^']+)'", filter_string)
        run = re.search(r"run_id = '([^']+)'", filter_string)
        found = self.versions.get(name.group(1), []) if name else []
        return [v for v in found if run is None or v.run_id == run.group(1)]

    def create_model_version(
        self, name: str, source: str, run_id: str, tags: dict[str, str], description: str | None
    ) -> SimpleNamespace:
        version = SimpleNamespace(
            version=str(len(self.versions[name]) + 1), source=source, run_id=run_id, tags=tags, status="READY"
        )
        self.versions[name].append(version)
        return version

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.models[name].aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        del self.models[name].aliases[alias]


def test_first_apply_registers_and_second_changes_nothing(repo: Path) -> None:
    client = FakeClient()
    registry = load_registry(write_toml(repo), root=repo)
    outcome = Registrar(client, registry).apply()
    assert outcome.registered == ["radar-extractor/ext-v1", "radar-salary/salary-v0"]
    assert sorted(outcome.aliases_set) == ["radar-extractor@champion -> v1", "radar-salary@champion -> v1"]

    ext_run = client.versions["radar-extractor"][0]
    run = client.runs[ext_run.run_id]
    assert run.info.status == "FINISHED"
    assert run.data.tags[KEY_TAG] == "ext-v1" and run.data.tags["radar.train_run_id"] == "abc123"
    assert run.data.metrics["dev/macro_f1"] == 0.9
    assert ext_run.source == f"{run.info.artifact_uri}/model"
    assert {dest for rid, _, dest in client.uploads if rid == run.info.run_id} == {
        "model/adapter",
        "model/gguf",
        "results",
    }

    uploads = len(client.uploads)
    again = Registrar(client, registry).apply()
    assert again.registered == [] and again.aliases_set == [] and again.aliases_removed == []
    assert len(client.uploads) == uploads


def test_registered_entry_is_not_reread_and_aliases_follow_the_file(repo: Path) -> None:
    client = FakeClient()
    Registrar(client, load_registry(write_toml(repo), root=repo)).apply()
    # The next training overwrites the files of a registered entry: nothing to re-check, nothing fails.
    (repo / "models" / "ext" / "q4.gguf").write_bytes(b"next weights")
    text = TOML.replace('alias = "{ext_alias}"\n', "")
    outcome = Registrar(client, load_registry(write_toml(repo, text), root=repo)).apply()
    assert outcome.unchanged == ["radar-extractor/ext-v1", "radar-salary/salary-v0"]
    assert outcome.aliases_removed == ["radar-extractor@champion"]
    assert client.models["radar-extractor"].aliases == {}


def test_same_files_under_a_new_key_are_refused(repo: Path) -> None:
    client = FakeClient()
    Registrar(client, load_registry(write_toml(repo), root=repo)).apply()
    text = TOML.replace('key = "ext-v1"', 'key = "ext-v2"')
    with pytest.raises(RegistryError, match="already registered as ext-v1"):
        Registrar(client, load_registry(write_toml(repo, text), root=repo)).apply()


def test_half_uploaded_run_is_failed_and_uploaded_again(repo: Path) -> None:
    client = FakeClient()
    client.create_experiment("radar-registry")
    stale = client.create_run("1", "ext-v1", {MODEL_TAG: "radar-extractor", KEY_TAG: "ext-v1", FINGERPRINT_TAG: "x"})
    outcome = Registrar(client, load_registry(write_toml(repo), root=repo)).apply()
    assert stale.info.status == "FAILED"
    assert "radar-extractor/ext-v1" in outcome.registered
    assert client.versions["radar-extractor"][0].run_id != stale.info.run_id


def test_missing_model_file_stops_a_new_entry(repo: Path) -> None:
    (repo / "models" / "salary.pkl").unlink()
    with pytest.raises(RegistryError, match="missing"):
        Registrar(FakeClient(), load_registry(write_toml(repo), root=repo)).apply()
