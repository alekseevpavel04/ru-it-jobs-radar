"""The read-only ClickHouse account of the agent, applied from `infra/clickhouse/agent/readonly_user.sql`.

`python scripts/apply_agent_user.py` runs it with the full-rights account; it is idempotent, so it is
safe after every `up`. The name and the password are spliced into the SQL text, so their alphabet is
checked instead of escaped (the same rule as infra/clickhouse/init.sh and .env.example).
"""

import re
from pathlib import Path

from radar.config import ROOT, ConfigError, require
from radar.stream.clickhouse import ClickHouse, statements

USER_SQL = ROOT / "infra" / "clickhouse" / "agent" / "readonly_user.sql"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PASSWORD = re.compile(r"[A-Za-z0-9_.~+/=-]+")


def render(ro_user: str, ro_password: str, admin_user: str, path: Path = USER_SQL) -> list[str]:
    if not _IDENTIFIER.fullmatch(ro_user):
        raise ConfigError("CLICKHOUSE_RO_USER must be a plain identifier (letters, digits, underscore)")
    if not _PASSWORD.fullmatch(ro_password):
        raise ConfigError("CLICKHOUSE_RO_PASSWORD may contain only letters, digits and _.~+/=-")
    if ro_user == admin_user:
        raise ConfigError("CLICKHOUSE_RO_USER must differ from CLICKHOUSE_USER")
    sql = path.read_text(encoding="utf-8").replace("{ro_user}", ro_user).replace("{ro_password}", ro_password)
    return statements(sql)


def apply() -> str:
    ro_user = require("CLICKHOUSE_RO_USER")
    rendered = render(ro_user, require("CLICKHOUSE_RO_PASSWORD"), require("CLICKHOUSE_USER"))
    with ClickHouse(timeout_s=60) as ch:
        for statement in rendered:
            ch.query(statement)
    return ro_user
