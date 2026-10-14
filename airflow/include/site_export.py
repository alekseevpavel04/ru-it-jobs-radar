"""What radar_publish keeps around the site export: a copy to fall back on, the check, the changes.

Standard library only, like raw_layer: the DAG's tasks import it under Airflow's interpreter.

The export (python -m radar.site.export) rewrites site/src/data whole and results/site/export.json.
Before it runs, both are copied aside; when a check of the new files fails, the copy goes back, so
the working tree never holds an export that failed its check. The check is the one the Pages
workflow runs on the built site (src/radar/site/scan_dist.py), applied file by file to the export:
only the files and row fields the export declares, no key naming per-vacancy content, no long
strings, no link to a single vacancy, no e-mail or phone.
"""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

DATA = Path("site") / "src" / "data"
RESULT = Path("results") / "site" / "export.json"
META = "meta.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(root: Path) -> dict[str, Any]:
    """sha256 of every data file, rows per file and the last data day as meta.json declares them."""
    data = root / DATA
    files = {p.name: _sha256(p) for p in sorted(data.glob("*")) if p.is_file()}
    meta = json.loads((data / META).read_text(encoding="utf-8")) if (data / META).exists() else {}
    return {"files": files, "rows": meta.get("files", {}), "data_through": meta.get("data_through")}


def keep_copy(root: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(root / DATA, dest / "data")
    if (root / RESULT).exists():
        shutil.copy2(root / RESULT, dest / RESULT.name)


def restore(root: Path, dest: Path) -> list[str]:
    """Put the copy back: the data directory as it was (files the export added go), and its result file."""
    if not (dest / "data").is_dir():
        raise FileNotFoundError(f"no copy in {dest}: the data directory is left as it is")
    data = root / DATA
    for path in data.glob("*"):
        if path.is_file():
            path.unlink()
    restored = []
    for path in sorted((dest / "data").glob("*")):
        shutil.copy2(path, data / path.name)
        restored.append(path.name)
    if (dest / RESULT.name).exists():
        shutil.copy2(dest / RESULT.name, root / RESULT)
    return restored


def changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old, new = before["files"], after["files"]
    return {
        "changed": sorted(n for n in new if n in old and new[n] != old[n]),
        "unchanged": sorted(n for n in new if old.get(n) == new[n]),
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "rows": {
            n: [before["rows"].get(n), after["rows"].get(n)]
            for n in sorted(set(before["rows"]) | set(after["rows"]))
            if before["rows"].get(n) != after["rows"].get(n)
        },
        "data_through": [before["data_through"], after["data_through"]],
    }


def load_scan_dist(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("scan_dist", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan(root: Path, scan_dist: ModuleType) -> list[str]:
    """Problems scan_dist finds in the exported files, as it reports them for a built site."""
    files = sorted(p for p in (root / DATA).glob("*") if p.is_file())
    if not files:
        return [f"{DATA.as_posix()}: no data files"]
    problems: list[str] = []
    for path in files:
        if path.suffix != ".json":
            problems.append(f"{path.name}: data must be JSON written by the export")
            continue
        problems += scan_dist.scan_data_file(path, root / DATA.parent)
        problems += scan_dist.scan_text(path, root / DATA.parent)
    return problems
