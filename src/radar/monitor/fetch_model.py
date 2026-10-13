"""Download the files of the model version an alias points at, from MLflow over REST.

The Helm chart runs this as the init container of the API, so the salary model a pod serves is the
registry's `champion`: promoting a version is moving the alias and restarting the deployment. The
files are checked against the version's `radar.fingerprint` tag (radar.monitor.registry) before they
replace what the target directory held. httpx only: the API image has no mlflow client.

    python -m radar.monitor.fetch_model --model radar-salary --alias champion --out /models/salary
"""

import argparse
import logging
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from radar.collect import setup_logging
from radar.config import require
from radar.monitor.fingerprint import directory_fingerprint

log = logging.getLogger("radar.monitor.fetch_model")

FINGERPRINT_TAG = "radar.fingerprint"
SCHEME = "mlflow-artifacts:/"


class FetchError(RuntimeError):
    pass


def version_of(client: httpx.Client, model: str, alias: str) -> dict[str, Any]:
    response = client.get("/api/2.0/mlflow/registered-models/alias", params={"name": model, "alias": alias})
    if response.status_code in (400, 404):
        # An unknown alias comes back as 400 INVALID_PARAMETER_VALUE, an unknown model as 404.
        raise FetchError(f"{model}@{alias}: {response.json().get('message', response.text)}")
    response.raise_for_status()
    version: dict[str, Any] = response.json()["model_version"]
    return version


def files_under(client: httpx.Client, path: str) -> Iterator[str]:
    """Artifact paths of every file below `path` (the proxied artifact store, not a local one)."""
    response = client.get("/api/2.0/mlflow-artifacts/artifacts", params={"path": path})
    response.raise_for_status()
    for entry in response.json().get("files", []):
        child = f"{path}/{entry['path']}"
        if entry.get("is_dir"):
            yield from files_under(client, child)
        else:
            yield child


def fetch(client: httpx.Client, model: str, alias: str, out: Path) -> str:
    version = version_of(client, model, alias)
    source = str(version["source"])
    if not source.startswith(SCHEME):
        raise FetchError(f"{model}@{alias}: source {source} is not in the server's artifact store")
    root = source.removeprefix(SCHEME).strip("/")
    tags = {t["key"]: t["value"] for t in version.get("tags", [])}
    expected = tags.get(FINGERPRINT_TAG)
    if expected is None:
        raise FetchError(f"{model} version {version['version']} has no {FINGERPRINT_TAG} tag")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}-") as tmp:
        staging = Path(tmp) / "model"
        for path in files_under(client, root):
            target = staging / path.removeprefix(root).strip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            with client.stream("GET", f"/api/2.0/mlflow-artifacts/artifacts/{path}") as response:
                response.raise_for_status()
                with target.open("wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
        got = directory_fingerprint(staging)
        if got != expected:
            raise FetchError(f"{model} version {version['version']}: fingerprint {got} instead of {expected}")
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
    log.info("%s@%s: version %s into %s", model, alias, version["version"], out)
    return str(version["version"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radar.monitor.fetch_model", description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", type=Path, required=True, help="directory that receives the version's model files")
    args = ap.parse_args(argv)
    setup_logging()
    with httpx.Client(base_url=require("MLFLOW_TRACKING_URI"), timeout=60.0) as client:
        fetch(client, args.model, args.alias, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
