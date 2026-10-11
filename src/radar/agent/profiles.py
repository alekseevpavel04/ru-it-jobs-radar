"""What the agent sees: the live marts, or the frozen snapshot the control set is scored on.

The two drift apart: the snapshot keeps the marts and the `_marts.yml` of the day it was exported
(eval/agent/snapshot_marts.yml), the live marts gain columns and tables. A profile ties a database to
its table list, its schema description and the SQL prompt written for that schema, so a run never
describes one schema and queries the other.
"""

from dataclasses import dataclass
from pathlib import Path

from radar.agent import prompts
from radar.agent.schema import MARTS_YML
from radar.agent.snapshot import SNAPSHOT_DATABASE, SNAPSHOT_YML, pinned
from radar.agent.sqlcheck import DATABASE, TABLES


@dataclass(frozen=True)
class Profile:
    name: str
    database: str
    tables: frozenset[str]
    yml: Path
    sql_system: str


def live() -> Profile:
    return Profile("live", DATABASE, TABLES, MARTS_YML, prompts.SQL_SYSTEM_LIVE)


def frozen() -> Profile:
    tables = frozenset(pinned()["tables"])
    if not tables <= TABLES:
        raise ValueError(f"the snapshot holds tables outside the whitelist: {sorted(tables - TABLES)}")
    return Profile("snapshot", SNAPSHOT_DATABASE, tables, SNAPSHOT_YML, prompts.SQL_SYSTEM_SNAPSHOT)
