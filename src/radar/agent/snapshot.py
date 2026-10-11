"""The frozen copy of the marts the control set is scored on.

`export` writes every mart to `data/agent/snapshot/` (Parquet and its CREATE TABLE) with a manifest of
row counts and sha256; `eval/agent/snapshot.json` pins that manifest in git. `load` restores the files
into the `agent_snapshot` database, which the read-only account may read, after checking the hashes
against the pin. The answers of the control set are computed there and the agent's SQL is scored
there, so the numbers stay fixed while the live marts grow.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radar.agent.sqlcheck import DATABASE, TABLES
from radar.config import ROOT, data_dir
from radar.stream.clickhouse import ClickHouse, ClickHouseError

SNAPSHOT_DATABASE = "agent_snapshot"
PIN = ROOT / "eval" / "agent" / "snapshot.json"


def snapshot_dir() -> Path:
    return data_dir() / "agent" / "snapshot"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _post(ch: ClickHouse, query: str, body: bytes | None = None) -> bytes:
    if body is None:
        response = ch.client.post("/", content=query.encode("utf-8"))
    else:
        response = ch.client.post("/", params={"query": query}, content=body)
    if response.status_code != 200:
        raise ClickHouseError(response.text.strip()[:2000])
    return response.content


def export(ch: ClickHouse, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, Any] = {}
    for name in sorted(TABLES):
        ddl = _post(ch, f"SHOW CREATE TABLE {DATABASE}.{name} FORMAT TSVRaw").decode("utf-8").strip()
        (out / f"{name}.sql").write_text(ddl + "\n", encoding="utf-8")
        parquet = out / f"{name}.parquet"
        parquet.write_bytes(_post(ch, f"SELECT * FROM {DATABASE}.{name} ORDER BY ALL FORMAT Parquet"))
        rows = int(_post(ch, f"SELECT count() FROM {DATABASE}.{name}").decode().strip())
        tables[name] = {"rows": rows, "sha256": _sha256(parquet), "ddl_sha256": _sha256(out / f"{name}.sql")}
    manifest = {"exported_at": datetime.now(UTC).isoformat(timespec="seconds"), "tables": tables}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def pinned() -> dict[str, Any]:
    pin: dict[str, Any] = json.loads(PIN.read_text(encoding="utf-8"))
    return pin


def pin_id(pin: dict[str, Any]) -> str:
    """One hash for the whole snapshot: the control set records it next to every answer."""
    canonical = json.dumps(pin["tables"], sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def verify_files(src: Path, pin: dict[str, Any]) -> None:
    for name, expected in pin["tables"].items():
        for suffix, key in ((".parquet", "sha256"), (".sql", "ddl_sha256")):
            path = src / f"{name}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; the snapshot is in data/agent/snapshot")
            if _sha256(path) != expected[key]:
                raise ValueError(f"{path} differs from the pin in {PIN.relative_to(ROOT).as_posix()}")


def load(ch: ClickHouse, src: Path, pin: dict[str, Any]) -> dict[str, int]:
    verify_files(src, pin)
    _post(ch, f"CREATE DATABASE IF NOT EXISTS {SNAPSHOT_DATABASE}")
    loaded = {}
    for name in sorted(pin["tables"]):
        ddl = (src / f"{name}.sql").read_text(encoding="utf-8")
        prefix = f"CREATE TABLE {DATABASE}.{name}"
        if not ddl.startswith(prefix):
            raise ValueError(f"{name}.sql does not start with {prefix}")
        _post(ch, f"DROP TABLE IF EXISTS {SNAPSHOT_DATABASE}.{name}")
        _post(ch, f"CREATE TABLE {SNAPSHOT_DATABASE}.{name}" + ddl[len(prefix) :])
        _post(ch, f"INSERT INTO {SNAPSHOT_DATABASE}.{name} FORMAT Parquet", (src / f"{name}.parquet").read_bytes())
        rows = int(_post(ch, f"SELECT count() FROM {SNAPSHOT_DATABASE}.{name}").decode().strip())
        if rows != pin["tables"][name]["rows"]:
            raise ValueError(f"{SNAPSHOT_DATABASE}.{name} has {rows} rows, the pin {pin['tables'][name]['rows']}")
        loaded[name] = rows
    return loaded


def is_loaded(ch: ClickHouse, pin: dict[str, Any]) -> bool:
    counts = {}
    for name in pin["tables"]:
        exists = _post(ch, f"EXISTS TABLE {SNAPSHOT_DATABASE}.{name}").decode().strip() == "1"
        if not exists:
            return False
        counts[name] = int(_post(ch, f"SELECT count() FROM {SNAPSHOT_DATABASE}.{name}").decode().strip())
    return all(counts[n] == t["rows"] for n, t in pin["tables"].items())
