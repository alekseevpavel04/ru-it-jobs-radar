"""Rules of the Spark mirror of the dbt history (jobs/spark) come from dbt itself."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jobs" / "spark"))

import rules  # noqa: E402

COMPILED_STG = ROOT / "dbt" / "target" / "compiled" / "radar" / "models" / "staging" / "stg_trudvsem_snapshot.sql"


def test_title_pattern_has_single_backslashes_as_clickhouse_receives_it() -> None:
    pattern = rules.it_title_pattern(rules.macros())
    assert r"\bsre\b" in pattern and r"c\+\+" in pattern and r"\.net" in pattern
    assert "\\\\" not in pattern
    assert re.search(pattern, "senior python-разработчик") and re.search(pattern, "ведущий ит-специалист")
    assert not re.search(pattern, "водитель погрузчика")


@pytest.mark.skipif(not COMPILED_STG.exists(), reason="dbt compile has not been run")
def test_title_pattern_equals_the_compiled_sql_literal() -> None:
    sql = COMPILED_STG.read_text(encoding="utf-8")
    literal = re.search(r"match\(lowerUTF8\(coalesce\(title, ''\)\), '(.*?)'\)", sql)
    assert literal is not None
    assert literal.group(1).replace("\\\\", "\\") == rules.it_title_pattern(rules.macros())


def test_rules_and_tracked_fields_follow_dbt() -> None:
    text = rules.macros()
    assert rules.history_rules(text) == {"complete_day_ratio": 0.9, "absent_days_to_close": 7, "max_step_days": 2}
    assert rules.tracked_fields_in_dbt(text) == rules.TRACKED_FIELDS
    assert "Программист" in rules.it_typical_positions()
