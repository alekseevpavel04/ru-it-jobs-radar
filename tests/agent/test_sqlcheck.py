import pytest

from radar.agent.sqlcheck import MAX_ROWS, Blocked, Unparsable, check

ATTACKS = [
    "INSERT INTO marts.mart_junior (source) VALUES ('x')",
    "DROP TABLE marts.mart_salaries",
    "DROP DATABASE marts",
    "ALTER TABLE marts.mart_skills DELETE WHERE 1",
    "ALTER TABLE marts.mart_junior RENAME COLUMN source TO s",
    "RENAME TABLE marts.mart_junior TO marts.mart_junior_old",
    "TRUNCATE TABLE marts.mart_junior",
    "CREATE TABLE marts.tmp (x UInt8) ENGINE = Memory",
    "CREATE TABLE marts.tmp ENGINE = Memory AS SELECT * FROM marts.mart_junior",
    "SET readonly = 0",
    "SYSTEM DROP DNS CACHE",
    "GRANT ALL ON *.* TO soliver",
    "KILL QUERY WHERE 1",
    "OPTIMIZE TABLE marts.mart_junior FINAL",
    "SELECT 1 FROM marts.mart_junior; DROP TABLE marts.mart_junior",
    "SELECT * FROM raw.vacancies",
    "SELECT * FROM staging.int_vacancy_entities",
    "SELECT name FROM system.users",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM marts.mart_junior AS j JOIN raw.vacancies AS v ON j.source = v.source",
    "SELECT * FROM marts.mart_junior WHERE source IN (SELECT source FROM raw.vacancies)",
    "WITH x AS (SELECT * FROM system.query_log) SELECT * FROM x",
    "SELECT * FROM numbers(1000000000000)",
    "SELECT * FROM url('http://example.com/a.csv', CSV, 'a String')",
    "SELECT * FROM file('/etc/passwd', 'LineAsString')",
    "SELECT * FROM remote('127.0.0.1', system.users)",
    "SELECT * FROM marts.mart_junior SETTINGS readonly = 0",
    "SELECT * FROM marts.mart_junior SETTINGS max_result_rows = 0",
    "SELECT * FROM marts.mart_junior FORMAT TSV",
    "SELECT * FROM marts.mart_junior INTO OUTFILE '/tmp/x'",
    "SELECT sleep(3) FROM marts.mart_junior",
    "SELECT sleepEachRow(1) FROM marts.mart_junior",
    "SELECT dictGet('d', 'x', 1) FROM marts.mart_junior",
    "SELECT * FROM mart_unknown",
    "SELECT * FROM marts.vacancies",
    "SELECT 1",
    "EXPLAIN SELECT * FROM marts.mart_junior",
    "SHOW TABLES FROM raw",
    "DESCRIBE TABLE raw.vacancies",
    "",
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_attack_is_not_let_through(sql: str) -> None:
    with pytest.raises((Blocked, Unparsable)):
        check(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE marts.mart_salaries",
        "INSERT INTO marts.mart_junior (source) VALUES ('x')",
        "SELECT name FROM system.users",
        "SELECT * FROM numbers(10)",
        "SELECT * FROM marts.mart_junior SETTINGS readonly = 0",
    ],
)
def test_unsafe_query_is_blocked_not_repairable(sql: str) -> None:
    with pytest.raises(Blocked):
        check(sql)


def test_select_over_marts_passes_and_gets_a_limit() -> None:
    checked = check("SELECT source, sum(vacancies) FROM marts.mart_junior GROUP BY source")
    assert checked.tables == ("mart_junior",)
    assert checked.sql.endswith(f"LIMIT {MAX_ROWS}")


def test_unqualified_mart_is_qualified() -> None:
    assert "FROM marts.mart_junior" in check("SELECT * FROM mart_junior").sql


def test_small_limit_is_kept_large_one_is_clamped() -> None:
    assert check("SELECT * FROM marts.mart_junior LIMIT 5").sql.endswith("LIMIT 5")
    assert check("SELECT * FROM marts.mart_junior LIMIT 100000").sql.endswith(f"LIMIT {MAX_ROWS}")


def test_limit_by_and_union_are_bounded_from_outside() -> None:
    by = check("SELECT source, skill FROM marts.mart_skills ORDER BY source LIMIT 3 BY source")
    assert "LIMIT 3 BY source" in by.sql and by.sql.endswith(f"LIMIT {MAX_ROWS}")
    union = check("SELECT source FROM marts.mart_junior UNION ALL SELECT source FROM marts.mart_salaries")
    assert union.sql.startswith("SELECT * FROM (") and union.sql.endswith(f"LIMIT {MAX_ROWS}")
    assert union.tables == ("mart_junior", "mart_salaries")


def test_cte_and_subqueries_over_marts_pass() -> None:
    sql = """
    WITH t AS (SELECT role_bucket, vacancies FROM marts.mart_junior WHERE source = 'habr')
    SELECT role_bucket FROM t WHERE vacancies > (SELECT avg(vacancies) FROM marts.mart_junior)
    """
    assert check(sql).tables == ("mart_junior",)


def test_snapshot_database_replaces_marts_in_the_rendered_sql() -> None:
    checked = check(
        "SELECT count() FROM marts.mart_skills AS s JOIN mart_junior USING (source)", database="agent_snapshot"
    )
    assert "agent_snapshot.mart_skills" in checked.sql and "agent_snapshot.mart_junior" in checked.sql
    assert "marts." not in checked.sql


def test_trailing_semicolon_and_fence_free_text() -> None:
    assert check("SELECT count() FROM marts.mart_junior;").tables == ("mart_junior",)


def test_garbage_is_unparsable() -> None:
    with pytest.raises(Unparsable):
        check("SELECT FROM WHERE (")
