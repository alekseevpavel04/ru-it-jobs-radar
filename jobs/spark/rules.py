"""The rules of the dbt history that are data, read from dbt itself so Spark cannot drift from them.

No Spark here: these are checked by the project's tests on any machine.
"""

import ast
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DBT = ROOT / "dbt"
SOURCE = "trudvsem_snapshot"

# Macro history_tracked_fields: fields whose change is named in changed_fields of the SCD2 rows.
TRACKED_FIELDS = (
    "title",
    "employer_name",
    "salary_from",
    "salary_to",
    "salary_currency_code",
    "schedule",
    "employment",
    "remote_work",
    "status",
    "skills",
    "text_hash",
)
SALARY_FIELDS = ("salary_from", "salary_to", "salary_currency_code")


def macros(folder: Path = DBT / "macros") -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.sql")))


def _macro_body(name: str, text: str) -> str:
    match = re.search(r"{%\s*macro " + name + r"\(.*?\)\s*-?%}(.*?){%-?\s*endmacro", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"macro {name} not found in the dbt macros")
    return match.group(1)


def it_title_pattern(text: str) -> str:
    """The regex of the IT title filter as ClickHouse receives it (macro it_title_pattern).

    The parts are Jinja string literals that become SQL string literals, so a regex backslash is
    written four times in the macro: Python literal rules leave two, the SQL literal leaves one.
    """
    body = _macro_body("it_title_pattern", text)
    listed = re.search(r"set parts = (\[.*?\])\s*-?%}", body, flags=re.DOTALL)
    if listed is None:
        raise ValueError("it_title_pattern has no parts list")
    parts: list[str] = ast.literal_eval(listed.group(1))
    return "|".join(p.replace("\\\\", "\\") for p in parts)


def _int_macro(name: str, text: str) -> int:
    body = _macro_body(name, text)
    value = re.search(r"var\('\w+', (\d+)\)", body) or re.fullmatch(r"\s*(\d+)\s*", body)
    if value is None:
        raise ValueError(f"macro {name} is not a number or a var with a default")
    return int(value.group(1))


def history_rules(text: str) -> dict[str, float]:
    """Numbers of the history rules for the snapshot source (dbt/macros/history.sql)."""
    ratio = re.fullmatch(r"\s*([0-9.]+)\s*", _macro_body("history_complete_day_ratio", text))
    absent = re.search(r"= '" + SOURCE + r"', {{ history_grace_days\(\) }}", text)
    if ratio is None or absent is None:
        raise ValueError("history macros changed shape: complete-day ratio or snapshot grace days not found")
    return {
        "complete_day_ratio": float(ratio.group(1)),
        "absent_days_to_close": _int_macro("history_grace_days", text),
        "max_step_days": _int_macro("history_max_step_days", text),
    }


def it_typical_positions(path: Path = DBT / "seeds" / "it_typical_positions.csv") -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row["typical_position"] for row in csv.DictReader(f)]


def tracked_fields_in_dbt(text: str) -> tuple[str, ...]:
    body = _macro_body("history_tracked_fields", text)
    listed = re.search(r"return\((\[.*?\])\)", body, flags=re.DOTALL)
    if listed is None:
        raise ValueError("history_tracked_fields has no list")
    return tuple(ast.literal_eval(listed.group(1)))
