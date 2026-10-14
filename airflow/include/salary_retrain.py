"""Refit of the salary model for radar_retrain, registered in MLflow with the alias challenger.

    RETRAIN_DECISION='<json of retrain_rule.Decision>' /opt/radar/.venv/bin/python \
        /opt/radar/airflow/include/salary_retrain.py

Runs in the project's environment. The decision names the drift week, and the version key is
salary-drift-<week>: a rerun for a week that already has its version registers nothing and exits
with SKIP_EXIT_CODE (the task shows as skipped).

The refit is `python -m radar.models.salary` with every output moved to models/salary_retrain/<key>/:
the training frame, both pickles, the result file and the SHAP figure. models/salary/ (what the API
serves), data/salary/frame.parquet (the frame the held-out test was scored on) and results/models/
are not written. The frame is a fresh query, so its split holds today's data: the challenger's dev
metrics are on its own dev rows, not on the champion's, and comparing the two is a step after this.

Registration takes the path every registered version takes, radar.monitor.registry.Registrar: one
run of the radar-registry experiment with model/, results/ and provenance/, a fingerprint, and a
stop when the files equal a registered version. Then the alias challenger moves to the new version.
champion is read before and after and must not have moved.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from radar.collect import setup_logging
from radar.config import ROOT, data_dir, require
from radar.models import salary
from radar.models.salary_data import build_frame, load_clickhouse, load_corpus_groups
from radar.monitor.fingerprint import Artifact
from radar.monitor.registry import KEY_TAG, Entry, Registrar, ResultRef, load_registry
from radar.results import Metric, ResultFile, git_commit, invocation

log = logging.getLogger("radar.airflow.retrain")

SKIP_EXIT_CODE = 99
MODEL = "radar-salary"
OUT_ROOT = ROOT / "models" / "salary_retrain"
RESULT = "salary.json"


def version_key(week: str) -> str:
    return f"salary-drift-{week}"


def registered_keys(client: MlflowClient) -> set[str]:
    return {v.tags.get(KEY_TAG, "") for v in client.search_model_versions(f"name = '{MODEL}'")}


def refit(out: Path, cfg: salary.Config) -> Path:
    """The training run of radar.models.salary with its outputs in `out`; returns the result file."""
    entities, records, student = load_clickhouse(cfg.extraction_model)
    corpus = load_corpus_groups(data_dir() / "corpus" / "split.parquet")
    frame, counts = build_frame(entities, records, student, corpus, cfg.shares, cfg.min_monthly)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "frame.parquet", index=False)
    (out / "frame.counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    log.info("frame: %s", counts)

    # run() draws the SHAP figure at this module path, which by default is the published one in results/.
    salary.SHAP_FIGURE = out / "salary_shap.png"
    doc = salary.run(frame, counts, cfg, out)
    metrics = doc.pop("metrics")
    result = ResultFile(
        source_script="airflow/include/salary_retrain.py",
        command=invocation(),
        commit=git_commit(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        metrics={k: Metric.model_validate(v) for k, v in metrics.items()},
        definitions=salary.DEFINITIONS,
        **doc,
    )
    path = out / RESULT
    path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def register(client: MlflowClient, key: str, out: Path, decision: dict[str, Any]) -> str:
    entry = Entry(
        model=MODEL,
        key=key,
        alias=None,
        description=f"Refit by radar_retrain on the drift of week {decision['week']}: {decision['reason']}",
        artifacts=(Artifact(out / salary.ARTIFACT, "."),),
        results=(ResultRef(out / RESULT, include=("dev_",)),),
        provenance=(salary.CONFIG, out / "decision.json"),
    )
    champion = client.get_registered_model(MODEL).aliases.get("champion")
    version = str(Registrar(client, load_registry()).register(entry).version)
    client.set_registered_model_alias(MODEL, "challenger", version)
    after = client.get_registered_model(MODEL).aliases.get("champion")
    if after != champion:
        raise RuntimeError(f"{MODEL}@champion moved from v{champion} to v{after} during the refit")
    log.info("%s v%s registered as %s, alias challenger; champion stays v%s", MODEL, version, key, champion)
    return version


def main() -> int:
    setup_logging()
    decision = json.loads(require("RETRAIN_DECISION"))
    if not decision["retrain"]:
        raise SystemExit(f"the decision says no refit: {decision['reason']}")
    key = version_key(decision["week"])
    client = MlflowClient(tracking_uri=require("MLFLOW_TRACKING_URI"))
    if key in registered_keys(client):
        log.info("%s already has a version %s; nothing to refit", MODEL, key)
        return SKIP_EXIT_CODE

    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("refitting %s for week %s into %s: %s", MODEL, decision["week"], out, decision["reason"])
    refit(out, salary.load_config())
    register(client, key, out, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
