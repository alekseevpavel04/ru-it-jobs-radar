"""Runs checked SELECTs as the read-only account and returns typed rows.

The account is the second line of defence: whatever the code check lets through, the server refuses
writes, DDL, setting changes, other databases and queries over the profile limits
(infra/clickhouse/agent/readonly_user.sql).
"""

import re
from dataclasses import dataclass
from typing import Any

import httpx

from radar.config import require

_CODE = re.compile(r"Code: (\d+)\.")
_WIDE_INT = re.compile(r"^(Nullable\()?(U?Int(64|128|256))\)?$")
_DECIMAL = re.compile(r"^(Nullable\()?(Decimal|Float)")
# ClickHouse error codes: the server refused the account, or a profile limit stopped the query.
READONLY = 164
ACCESS_DENIED = 497
TIMEOUT_EXCEEDED = 159
TOO_MANY_ROWS_OR_BYTES = 396
SERVER_REFUSALS = {READONLY, ACCESS_DENIED}


class QueryError(RuntimeError):
    def __init__(self, message: str, code: int | None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Result:
    columns: list[str]
    types: list[str]
    rows: list[list[Any]]


def _convert(value: Any, ch_type: str) -> Any:
    # JSON output quotes 64-bit integers and decimals as strings and prints a whole float without a
    # point; the account cannot change those settings (readonly = 1), so the column types from the
    # response metadata decide what the value is.
    if value is None or isinstance(value, bool):
        return value
    if _WIDE_INT.match(ch_type):
        return int(value)
    if _DECIMAL.match(ch_type):
        return float(value)
    return value


def parse_json_compact(payload: dict[str, Any]) -> Result:
    names = [m["name"] for m in payload["meta"]]
    types = [m["type"] for m in payload["meta"]]
    rows = [[_convert(v, t) for v, t in zip(row, types, strict=True)] for row in payload["data"]]
    return Result(names, types, rows)


class ReadOnlyClickHouse:
    def __init__(self, timeout_s: float = 30.0) -> None:
        self.client = httpx.Client(
            base_url=require("CLICKHOUSE_URL"),
            headers={
                "X-ClickHouse-User": require("CLICKHOUSE_RO_USER"),
                "X-ClickHouse-Key": require("CLICKHOUSE_RO_PASSWORD"),
            },
            timeout=timeout_s,
        )

    def raw(self, sql: str, params: dict[str, str] | None = None) -> httpx.Response:
        """One statement as is: the live tests send hostile SQL here to see the server refuse it."""
        return self.client.post("/", content=sql.encode("utf-8"), params=params)

    def run(self, sql: str, params: dict[str, str] | None = None) -> Result:
        """`params` fill `{name:Type}` placeholders as ClickHouse query parameters (`param_<name>`)."""
        query = {f"param_{k}": v for k, v in (params or {}).items()}
        response = self.raw(sql + "\nFORMAT JSONCompact", query or None)
        if response.status_code != 200:
            text = response.text.strip()
            match = _CODE.search(text)
            raise QueryError(text[:1500], int(match.group(1)) if match else None)
        return parse_json_compact(response.json())

    def describe(self, table: str) -> list[tuple[str, str]]:
        result = self.run(f"DESCRIBE TABLE {table}")
        return [(str(row[0]), str(row[1])) for row in result.rows]

    def close(self) -> None:
        self.client.close()
