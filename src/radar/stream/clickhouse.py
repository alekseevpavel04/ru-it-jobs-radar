"""ClickHouse over HTTP for the stream: the DDL of `infra/clickhouse/*.sql` and counts of what landed.

`python -m radar.stream.clickhouse` applies the DDL to a running server. init.sh does the same on
`up`; this path is for a server whose volume predates a new file. Every statement is idempotent.
"""

import logging
import re
import sys
from datetime import date
from pathlib import Path
from types import TracebackType

import httpx

from radar.collect import setup_logging
from radar.config import ROOT, require

log = logging.getLogger("radar.stream.clickhouse")

DDL_DIR = ROOT / "infra" / "clickhouse"
TABLE = "raw.vacancies"
_STATEMENT_END = re.compile(r";[ \t]*(?:\n|$)")


class ClickHouseError(RuntimeError):
    pass


def statements(sql: str) -> list[str]:
    """Statements of a DDL file: whole-line `--` comments dropped, a `;` at a line end closes one."""
    body = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [s.strip() for s in _STATEMENT_END.split(body) if s.strip()]


class ClickHouse:
    def __init__(self, timeout_s: float = 300.0) -> None:
        self.client = httpx.Client(
            base_url=require("CLICKHOUSE_URL"),
            headers={
                "X-ClickHouse-User": require("CLICKHOUSE_USER"),
                "X-ClickHouse-Key": require("CLICKHOUSE_PASSWORD"),
            },
            timeout=timeout_s,
        )

    def query(self, sql: str, **params: str) -> str:
        """Run one statement; `{name:Type}` placeholders take `params` (server-side binding)."""
        response = self.client.post(
            "/", content=sql.encode("utf-8"), params={f"param_{k}": v for k, v in params.items()}
        )
        if response.status_code != 200:
            raise ClickHouseError(response.text.strip()[:2000])
        return response.text

    def scalar(self, sql: str, **params: str) -> int:
        return int(self.query(sql, **params).strip())

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ClickHouse":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def apply_schema(ch: ClickHouse, ddl_dir: Path = DDL_DIR) -> list[str]:
    applied = []
    for path in sorted(ddl_dir.glob("*.sql")):
        for statement in statements(path.read_text(encoding="utf-8")):
            ch.query(statement)
        applied.append(path.name)
    return applied


def rows_final(ch: ClickHouse, source: str, dt: date) -> int:
    """Rows of a source and day after deduplication by key, whatever the state of background merges."""
    return ch.scalar(
        f"SELECT count() FROM {TABLE} FINAL WHERE source = {{source:String}} AND dt = {{dt:Date}}",
        source=source,
        dt=dt.isoformat(),
    )


def rows_stored(ch: ClickHouse, source: str, dt: date) -> int:
    """Rows as stored right now, replayed copies included until a merge folds them."""
    return ch.scalar(
        f"SELECT count() FROM {TABLE} WHERE source = {{source:String}} AND dt = {{dt:Date}}",
        source=source,
        dt=dt.isoformat(),
    )


def main() -> int:
    setup_logging()
    with ClickHouse() as ch:
        for name in apply_schema(ch):
            log.info("applied %s", name)
        log.info("tables in raw: %s", " ".join(ch.query("SHOW TABLES FROM raw").split()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
