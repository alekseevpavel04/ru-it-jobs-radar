"""The second line of defence on a running server: the read-only account itself refuses.

These send hostile SQL straight to ClickHouse as CLICKHOUSE_RO_USER, bypassing the code check, so a
pass means the server, not the agent, stopped it. Needs the core stack, the account
(scripts/apply_agent_user.py) and the loaded snapshot (python -m radar.agent snapshot load):
`pytest -m slow tests/agent`.
"""

import re
from collections.abc import Iterator

import pytest

from radar.agent.clickhouse import (
    ACCESS_DENIED,
    READONLY,
    TIMEOUT_EXCEEDED,
    TOO_MANY_ROWS_OR_BYTES,
    QueryError,
    ReadOnlyClickHouse,
)
from radar.agent.compare import results_match
from radar.agent.control_set import load
from radar.agent.evaluate import system_prompt
from radar.agent.profiles import frozen, live
from radar.agent.snapshot import SNAPSHOT_DATABASE
from radar.agent.sqlcheck import check
from radar.config import load_dotenv
from radar.stream.clickhouse import ClickHouse

pytestmark = pytest.mark.slow
DATABASE_ACCESS_DENIED = 291  # file() outside user_files is refused before the grant check
REFUSED = {READONLY, ACCESS_DENIED, DATABASE_ACCESS_DENIED}
_CODE = re.compile(r"Code: (\d+)\.")


@pytest.fixture(scope="module")
def ro() -> Iterator[ReadOnlyClickHouse]:
    load_dotenv()
    client = ReadOnlyClickHouse(timeout_s=60)
    yield client
    client.close()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO marts.mart_junior (source) VALUES ('x')",
        "INSERT INTO marts.mart_junior SELECT * FROM marts.mart_junior",
        "ALTER TABLE marts.mart_skills DELETE WHERE 1",
        "ALTER TABLE marts.mart_junior ADD COLUMN x UInt8",
        "DROP TABLE marts.mart_salaries",
        "DROP DATABASE marts",
        "TRUNCATE TABLE marts.mart_junior",
        "RENAME TABLE marts.mart_junior TO marts.mart_junior_old",
        "CREATE TABLE marts.tmp (x UInt8) ENGINE = Memory",
        "CREATE TEMPORARY TABLE tmp (x UInt8)",
        "OPTIMIZE TABLE marts.mart_junior FINAL",
        "SET readonly = 0",
        "SELECT * FROM marts.mart_junior SETTINGS readonly = 0",
        "SELECT * FROM marts.mart_junior SETTINGS max_result_rows = 0",
        "SELECT count() FROM raw.vacancies",
        "SELECT count() FROM staging.int_vacancy_entities",
        "SELECT name FROM system.users",
        "SELECT count() FROM system.query_log",
        "SELECT * FROM url('http://127.0.0.1:8123/?query=SELECT+1', CSV, 'a String')",
        "SELECT * FROM file('/etc/passwd', 'LineAsString')",
        "SELECT * FROM file('a.csv', 'LineAsString')",
        "SELECT * FROM remote('127.0.0.1', system.users)",
    ],
)
def test_server_refuses_writes_settings_and_other_databases(ro: ReadOnlyClickHouse, sql: str) -> None:
    response = ro.raw(sql)
    code = _CODE.search(response.text)
    assert response.status_code != 200 and code and int(code.group(1)) in REFUSED, response.text[:300]


def test_url_parameter_cannot_lift_readonly(ro: ReadOnlyClickHouse) -> None:
    response = ro.raw("SELECT 1", {"readonly": "0"})
    assert response.status_code != 200 and f"Code: {READONLY}." in response.text


def test_marts_and_snapshot_are_readable(ro: ReadOnlyClickHouse) -> None:
    assert ro.run("SELECT count() FROM marts.mart_junior").rows[0][0] > 0
    assert ro.run(f"SELECT count() FROM {SNAPSHOT_DATABASE}.mart_junior").rows[0][0] > 0


def test_heavy_query_is_stopped_by_the_time_limit(ro: ReadOnlyClickHouse) -> None:
    sql = "SELECT count() FROM marts.mart_skills AS a, marts.mart_skills AS b, marts.mart_skills AS c"
    with pytest.raises(QueryError) as e:
        ro.run(check(sql).sql)
    assert e.value.code == TIMEOUT_EXCEEDED


def test_large_result_is_stopped_by_the_row_limit(ro: ReadOnlyClickHouse) -> None:
    with pytest.raises(QueryError) as e:
        ro.run("SELECT * FROM marts.mart_skills AS a, marts.mart_skills AS b")
    assert e.value.code == TOO_MANY_ROWS_OR_BYTES


def test_writes_left_nothing_behind() -> None:
    with ClickHouse(timeout_s=60) as admin:
        assert admin.query("EXISTS TABLE marts.tmp").strip() == "0"
        assert admin.query("EXISTS TABLE marts.mart_junior_old").strip() == "0"
        assert int(admin.query("SELECT countIf(source = 'x') FROM marts.mart_junior").strip()) == 0


def test_reference_answers_recompute_on_the_snapshot(ro: ReadOnlyClickHouse) -> None:
    for item in load():
        result = ro.run(check(item.sql, database=SNAPSHOT_DATABASE).sql)
        assert results_match(item.answer.rows, result.rows) and len(result.rows) == len(item.answer.rows), item.id


def test_both_profiles_describe_tables_that_exist(ro: ReadOnlyClickHouse) -> None:
    for profile in (live(), frozen()):
        prompt = system_prompt(ro, profile)
        for name in profile.tables:
            assert f"TABLE marts.{name}:" in prompt, (profile.name, name)
        assert ro.run(f"SELECT count() FROM {profile.database}.mart_salaries").rows[0][0] > 0


@pytest.mark.parametrize("database", ["marts", SNAPSHOT_DATABASE])
def test_aggregate_only_sources_never_name_an_employer(ro: ReadOnlyClickHouse, database: str) -> None:
    sql = (
        "SELECT source, countIf(employer_name IS NOT NULL), countIf(employer_key IS NOT NULL), count() "
        "FROM marts.mart_employers GROUP BY source"
    )
    rows = {r[0]: r[1:] for r in ro.run(check(sql, database=database).sql).rows}
    assert rows["trudvsem"][0] == rows["trudvsem"][2] > 0
    for source, (names, keys, n) in rows.items():
        if source != "trudvsem":
            assert names == keys == 0 and n > 0, source
