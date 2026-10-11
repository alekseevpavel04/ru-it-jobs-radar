import re

import pytest

from radar.agent.__main__ import main
from radar.agent.access import render
from radar.agent.clickhouse import parse_json_compact
from radar.agent.schema import schema_context
from radar.agent.sqlcheck import TABLES
from radar.config import ConfigError


def test_rendered_account_is_readonly_with_limits_and_select_on_two_databases() -> None:
    sql = "\n".join(render("agent_ro", "s3cret-pass", "radar"))
    assert re.search(r"readonly = 1\b", sql)
    for limit in ("max_execution_time", "max_result_rows", "max_rows_to_read", "max_memory_usage"):
        assert limit in sql
    grants = re.findall(r"^GRANT .+$", sql, flags=re.MULTILINE)
    assert grants == ["GRANT SELECT ON marts.* TO agent_ro", "GRANT SELECT ON agent_snapshot.* TO agent_ro"]
    assert "REVOKE ALL ON *.* FROM agent_ro" in sql
    assert "{ro_" not in sql


@pytest.mark.parametrize(
    ("user", "password", "admin"),
    [
        ("agent; DROP USER radar", "pw", "radar"),
        ("agent", "pw' SETTINGS readonly = 0 --", "radar"),
        ("radar", "pw", "radar"),
    ],
)
def test_names_that_would_change_the_sql_are_refused(user: str, password: str, admin: str) -> None:
    with pytest.raises(ConfigError):
        render(user, password, admin)


def test_json_compact_turns_quoted_wide_ints_and_whole_floats_into_numbers() -> None:
    payload = {
        "meta": [
            {"name": "n", "type": "UInt64"},
            {"name": "m", "type": "Nullable(Float64)"},
            {"name": "s", "type": "LowCardinality(String)"},
            {"name": "d", "type": "Date"},
            {"name": "k", "type": "UInt8"},
        ],
        "data": [["871", 232500, "habr", "2026-10-10", 1], ["0", None, "trudvsem", "2026-10-10", 0]],
    }
    result = parse_json_compact(payload)
    assert result.rows[0] == [871, 232500.0, "habr", "2026-10-10", 1]
    assert isinstance(result.rows[0][1], float)
    assert result.rows[1][1] is None


def test_schema_context_has_every_mart_column_types_and_values() -> None:
    types = {name: [("source", "LowCardinality(String)"), ("vacancies", "UInt64")] for name in TABLES}
    text = schema_context(types)
    for name in TABLES:
        assert f"TABLE marts.{name}:" in text
    assert "Values: trudvsem, habr, telegram." in text
    assert "- vacancies UInt64:" in text


def test_test_part_is_refused_without_final_test(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--split", "test"]) == 2
    assert "scored once" in capsys.readouterr().err
    assert main(["eval", "--split", "dev", "--final-test"]) == 2
