"""The code-side check of generated SQL: one SELECT over the marts whitelist, with a row limit.

What reaches ClickHouse is the SQL rendered back from the checked syntax tree, never the model's text,
so the parser that checks and the text that runs cannot disagree. The read-only account
(infra/clickhouse/agent/readonly_user.sql) enforces the same rules on the server; this check rejects
early, with a reason the agent can report.
"""

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

DATABASE = "marts"
TABLES = frozenset(
    {
        "mart_weekly_dynamics",
        "mart_salaries",
        "mart_skills",
        "mart_work_format",
        "mart_employers",
        "mart_junior",
    }
)
MAX_ROWS = 1000
# Scalar functions that wait, read files or URLs, or reach dictionaries and Join tables outside marts.
FORBIDDEN_FUNCTIONS = frozenset(
    {"sleep", "sleepeachrow", "file", "url", "s3", "remote", "remotesecure", "input", "joinget", "joingetornull"}
)
FORBIDDEN_PREFIXES = ("dictget", "dicthas", "dictisin")

logging.getLogger("sqlglot").setLevel(logging.ERROR)


class Blocked(ValueError):
    """The query is not allowed: not repaired, the agent refuses."""


class Unparsable(ValueError):
    """The text is not a single parsable query: the agent may try once more."""


@dataclass(frozen=True)
class CheckedQuery:
    sql: str
    tables: tuple[str, ...]


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower()


def _check_tables(tree: exp.Query) -> list[exp.Table]:
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    tables = []
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise Blocked(f"table functions are not allowed: {table.sql(dialect='clickhouse')[:80]}")
        if table.catalog:
            raise Blocked(f"only {DATABASE}.* tables may be read: {table.sql(dialect='clickhouse')}")
        if not table.db and table.name in ctes:
            continue
        if table.db and table.db != DATABASE:
            raise Blocked(f"only {DATABASE}.* tables may be read, not {table.db}.{table.name}")
        if table.name not in TABLES:
            raise Blocked(f"unknown table {table.name}; allowed: {', '.join(sorted(TABLES))}")
        tables.append(table)
    if not tables:
        raise Blocked("the query reads no mart table")
    return tables


def _check_clauses(tree: exp.Query) -> None:
    for select in tree.find_all(exp.Select, exp.SetOperation):
        for clause in ("settings", "format", "into"):
            if select.args.get(clause):
                raise Blocked(f"{clause.upper()} clause is not allowed")
    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name in FORBIDDEN_FUNCTIONS or name.startswith(FORBIDDEN_PREFIXES):
            raise Blocked(f"function {name} is not allowed")


def _plain_limit(node: exp.Query) -> int | None:
    limit = node.args.get("limit")
    if not isinstance(limit, exp.Limit) or limit.expressions:
        return None
    value = limit.expression
    if isinstance(value, exp.Literal) and not value.is_string:
        return int(value.this)
    return None


def _bound_rows(tree: exp.Query, max_rows: int) -> exp.Query:
    limit = tree.args.get("limit")
    current = _plain_limit(tree)
    if current is not None and current <= max_rows:
        return tree
    if isinstance(tree, exp.Select) and (limit is None or current is not None):
        return tree.limit(max_rows, copy=False)
    # LIMIT BY, a computed LIMIT or a set operation: bound the whole query from outside.
    return exp.select("*").from_(tree.subquery("bounded")).limit(max_rows)


def check(sql: str, database: str = DATABASE, max_rows: int = MAX_ROWS) -> CheckedQuery:
    """Check `sql` against the whitelist; render it to run against `database` (the frozen snapshot in eval)."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise Unparsable("empty query")
    try:
        trees = [t for t in sqlglot.parse(text, dialect="clickhouse") if t is not None]
    except (ParseError, TokenError) as e:
        raise Unparsable(str(e).splitlines()[0][:300]) from e
    if len(trees) != 1:
        raise Blocked("exactly one statement is allowed")
    tree = trees[0]
    if not isinstance(tree, exp.Query):
        raise Blocked(f"only SELECT is allowed, got {tree.key.upper()}")
    _check_clauses(tree)
    tables = _check_tables(tree)
    names = tuple(sorted({t.name for t in tables}))
    for table in tables:
        table.set("db", exp.to_identifier(database))
    bounded = _bound_rows(tree, max_rows)
    return CheckedQuery(bounded.sql(dialect="clickhouse"), names)
