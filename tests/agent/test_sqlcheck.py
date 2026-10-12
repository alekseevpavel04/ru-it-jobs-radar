import pytest
import sqlglot
from sqlglot import exp

from radar.agent.sqlcheck import EMPLOYERS, HIDDEN_COLUMNS, MAX_ROWS, NAMED_SOURCES, Blocked, Unparsable, check

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
    "SELECT 1 FROM marts.mart_junior WHERE source IN raw.vacancies",
    "SELECT 1 FROM marts.mart_junior WHERE employer_name IN marts.mart_employers",
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


def _employer_reads(sql: str) -> list[exp.Select]:
    tree = sqlglot.parse_one(sql, dialect="clickhouse")
    return [t.parent_select for t in tree.find_all(exp.Table) if t.name == EMPLOYERS and t.db]


@pytest.mark.parametrize(
    ("sql", "n_reads"),
    [
        ("SELECT employer_name, employer_key FROM marts.mart_employers WHERE source = 'habr'", 1),
        ("SELECT e.employer_name FROM marts.mart_employers AS e FINAL JOIN marts.mart_junior AS j USING (source)", 1),
        ("SELECT marts.mart_employers.employer_name FROM mart_employers", 1),
        ("WITH mart_employers AS (SELECT * FROM marts.mart_employers) SELECT employer_key FROM mart_employers", 1),
        ("SELECT employer_name FROM marts.mart_employers UNION ALL SELECT employer_name FROM mart_employers", 2),
        ("SELECT count() FROM marts.mart_junior WHERE source IN (SELECT source FROM marts.mart_employers)", 1),
    ],
)
def test_every_read_of_employers_goes_through_the_mask(sql: str, n_reads: int) -> None:
    rendered = check(sql, database="agent_snapshot").sql
    reads = _employer_reads(rendered)
    assert len(reads) == n_reads
    for select in reads:
        star = select.expressions[0]
        assert isinstance(star, exp.Star)
        masked = {alias.alias: alias.this for alias in star.args["replace"]}
        assert set(masked) == set(HIDDEN_COLUMNS)
        for column, expr in masked.items():
            keep = [str(v.this) for v in expr.find(exp.In).expressions]
            assert keep == list(NAMED_SOURCES) and expr.find(exp.Null) is not None, column
        assert select.args["from_"].this.db == "agent_snapshot"
    assert "FINAL" not in rendered and "marts." not in rendered


def test_mask_keeps_the_alias_and_other_tables() -> None:
    rendered = check("SELECT e.employer_rank, j.vacancies FROM marts.mart_employers e, marts.mart_junior j").sql
    assert ") AS e" in rendered and "marts.mart_junior AS j" in rendered
    assert NAMED_SOURCES == ("trudvsem",)
