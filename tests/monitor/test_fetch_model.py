import hashlib
from pathlib import Path

import httpx
import pytest

from radar.monitor.fetch_model import FetchError, fetch
from radar.monitor.fingerprint import Artifact, fingerprint

ROOT = "4/run1/artifacts/model"
FILES = {"salary.pkl": b"model bytes", "meta/config.json": b"{}"}


def registered_fingerprint(tmp_path: Path) -> str:
    """The digest the registrar computes from the local files at registration."""
    local = tmp_path / "local"
    for name, data in FILES.items():
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(data)
    return fingerprint([Artifact(local / "salary.pkl", "."), Artifact(local / "meta", "meta")])


def server(tags: dict[str, str], files: dict[str, bytes] = FILES) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/2.0/mlflow/registered-models/alias":
            if request.url.params["alias"] != "champion":
                return httpx.Response(400, json={"message": "Registered model alias challenger not found."})
            version = {
                "version": "3",
                "source": f"mlflow-artifacts:/{ROOT}",
                "tags": [{"key": k, "value": v} for k, v in tags.items()],
            }
            return httpx.Response(200, json={"model_version": version})
        if path == "/api/2.0/mlflow-artifacts/artifacts":
            prefix = request.url.params["path"].removeprefix(ROOT).strip("/")
            children: dict[str, bool] = {}
            for name in files:
                if prefix and not name.startswith(prefix + "/"):
                    continue
                head = name.removeprefix(prefix).strip("/").split("/")[0]
                children[head] = children.get(head, False) or "/" in name.removeprefix(prefix).strip("/")
            return httpx.Response(200, json={"files": [{"path": k, "is_dir": d} for k, d in children.items()]})
        name = path.removeprefix(f"/api/2.0/mlflow-artifacts/artifacts/{ROOT}/")
        return httpx.Response(200, content=files[name])

    return httpx.MockTransport(handler)


def test_downloads_the_alias_and_checks_the_fingerprint(tmp_path: Path) -> None:
    tags = {"radar.fingerprint": registered_fingerprint(tmp_path)}
    out = tmp_path / "models" / "salary"
    out.mkdir(parents=True)
    (out / "stale.pkl").write_bytes(b"old")
    with httpx.Client(base_url="http://mlflow", transport=server(tags)) as client:
        assert fetch(client, "radar-salary", "champion", out) == "3"
    assert {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == set(FILES)
    assert (out / "salary.pkl").read_bytes() == b"model bytes"


def test_tampered_download_leaves_the_old_files(tmp_path: Path) -> None:
    tags = {"radar.fingerprint": registered_fingerprint(tmp_path)}
    out = tmp_path / "models" / "salary"
    out.mkdir(parents=True)
    (out / "salary.pkl").write_bytes(b"old")
    transport = server(tags, {**FILES, "salary.pkl": b"other bytes"})
    with (
        httpx.Client(base_url="http://mlflow", transport=transport) as client,
        pytest.raises(FetchError, match="fingerprint"),
    ):
        fetch(client, "radar-salary", "champion", out)
    assert (out / "salary.pkl").read_bytes() == b"old"


def test_missing_alias_names_the_model(tmp_path: Path) -> None:
    with (
        httpx.Client(base_url="http://mlflow", transport=server({})) as client,
        pytest.raises(FetchError, match="challenger"),
    ):
        fetch(client, "radar-salary", "challenger", tmp_path / "out")


def test_fingerprint_does_not_depend_on_listing_order(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    assert fingerprint([Artifact(a, "x"), Artifact(b, "y")]) == fingerprint([Artifact(b, "y"), Artifact(a, "x")])
    # A single file: sha256 of its one line, which is what the first registrations were tagged with.
    line = f"x/a.bin\t{hashlib.sha256(b'a').hexdigest()}\n"
    assert fingerprint([Artifact(a, "x")]) == hashlib.sha256(line.encode()).hexdigest()
