"""Result files: the only place a published number may come from.

A result file is JSON with provenance (`source_script`, `command`, `commit`, `created_at`) and a
`metrics` mapping. Every metric has `value`; a share also carries `n` (numerator) and
`denominator`, so "37 of 174" can always be printed next to "21%".
"""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from radar.config import RESULTS, ROOT


class Metric(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: float | int | str
    n: int | None = None
    denominator: int | None = None

    @model_validator(mode="after")
    def _share_has_both_parts(self) -> "Metric":
        if (self.n is None) != (self.denominator is None):
            raise ValueError("n and denominator go together")
        return self


class ResultFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_script: str
    command: str
    commit: str
    created_at: str
    metrics: dict[str, Metric]


# What can change a computed number: code, configs, SQL, labeled inputs under version control and the
# lock file. A modified or untracked file there marks the commit "-dirty". Other paths are left out:
# results/ itself changes while a session writes one result after another, and logs, docs or the
# built site do not take part in any computation.
CODE_PATHS = (
    "src",
    "scripts",
    "configs",
    "dbt",
    "airflow",
    "jobs",
    "training",
    "infra",
    "deploy",
    "eval",
    "data",
    "pyproject.toml",
    "uv.lock",
)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=ROOT, capture_output=True, text=True
    ).stdout
    return out.stdout.strip() + ("-dirty" if dirty.strip() else "")


def invocation() -> str:
    """The command line as typed: `python -m radar.eval ...` rather than `__main__.py ...`."""
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec:
        head = f"python -m {spec.name.removesuffix('.__main__')}"
    else:
        script = Path(sys.argv[0]).resolve()
        head = f"python {script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else script.name}"
    return " ".join([head, *sys.argv[1:]])


def write_result(relpath: str, metrics: dict[str, dict[str, Any]], source_script: str, **extra: Any) -> Path:
    doc = ResultFile(
        source_script=source_script,
        command=invocation(),
        commit=git_commit(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        metrics={k: Metric.model_validate(v) for k, v in metrics.items()},
        **extra,
    )
    path = RESULTS / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def lookup(doc: Any, dotted: str) -> Any:
    node = doc
    for part in dotted.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(dotted)
    return node
