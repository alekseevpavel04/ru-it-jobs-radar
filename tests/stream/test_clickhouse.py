from radar.stream.clickhouse import DDL_DIR, statements


def test_statements_split_on_line_end_semicolons_and_drop_comments() -> None:
    sql = "-- header; not a statement\nCREATE TABLE a (x UInt8)\nENGINE = Memory;\n\n  -- note\nSELECT 'a;b';\n"
    assert statements(sql) == ["CREATE TABLE a (x UInt8)\nENGINE = Memory", "SELECT 'a;b'"]


def test_every_ddl_statement_is_idempotent() -> None:
    files = sorted(DDL_DIR.glob("*.sql"))
    assert files
    for path in files:
        found = statements(path.read_text(encoding="utf-8"))
        assert found, path.name
        for statement in found:
            assert statement.startswith("CREATE ") and " IF NOT EXISTS " in statement, statement[:80]
