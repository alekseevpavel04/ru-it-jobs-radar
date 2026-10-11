"""Schema context for SQL generation: the mart descriptions of `_marts.yml` with column types.

The descriptions are the ones dbt documents and tests, so the agent reads the same definitions of
"vacancy", "active" and "open" as the site. Types come from the server (DESCRIBE as the read-only
account) and are passed in, so tests can build the context without ClickHouse.
"""

from pathlib import Path
from typing import Any

import yaml

from radar.agent.sqlcheck import DATABASE, TABLES
from radar.config import ROOT

MARTS_YML = ROOT / "dbt" / "models" / "marts" / "_marts.yml"


def _accepted_values(column: dict[str, Any]) -> list[str]:
    for test in column.get("data_tests", []):
        if isinstance(test, dict) and "accepted_values" in test:
            return [str(v) for v in test["accepted_values"]["arguments"]["values"]]
    return []


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def schema_context(types: dict[str, list[tuple[str, str]]], path: Path = MARTS_YML) -> str:
    models = {m["name"]: m for m in yaml.safe_load(path.read_text(encoding="utf-8"))["models"]}
    missing = TABLES - models.keys()
    if missing:
        raise ValueError(f"{path} does not describe {sorted(missing)}")
    blocks = []
    for name in sorted(TABLES):
        model = models[name]
        documented = {c["name"]: c for c in model.get("columns", [])}
        lines = [f"TABLE {DATABASE}.{name}: {_clean(model['description'])}"]
        for column, ch_type in types[name]:
            doc = documented.get(column, {})
            line = f"  - {column} {ch_type}"
            if doc.get("description"):
                line += f": {_clean(doc['description'])}"
            values = _accepted_values(doc)
            if values:
                line += f" Values: {', '.join(values)}."
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
