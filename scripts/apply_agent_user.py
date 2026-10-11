"""Create or update the read-only ClickHouse account of the analyst agent (idempotent).

Usage: .venv/Scripts/python.exe scripts/apply_agent_user.py
Needs CLICKHOUSE_URL, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_RO_USER, CLICKHOUSE_RO_PASSWORD.
"""

import sys

from radar.agent.access import USER_SQL, apply
from radar.config import ROOT, load_dotenv


def main() -> int:
    load_dotenv()
    user = apply()
    print(f"applied {USER_SQL.relative_to(ROOT).as_posix()}: {user} reads marts and agent_snapshot only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
