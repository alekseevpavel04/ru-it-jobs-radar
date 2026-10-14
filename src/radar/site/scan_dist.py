"""Check a built site (site/dist) before it is published: aggregates only (DECISIONS 2, 3).

    python src/radar/site/scan_dist.py site/dist

Standard library only, so the Pages workflow runs it with the runner's python3 and no project
install. Fails when:

- a data file is not one the export writes, or a row of it carries a field outside the export's
  list (tests/site keeps ALLOWED equal to the row models of radar.site.export), or a nested value;
- any JSON anywhere in the output has a key that names per-vacancy content (title, text, url, ...);
- a row string is longer than a label, which is how a vacancy text would show;
- a page or a data file holds a link to a single vacancy or post, an e-mail or a phone number.

Library code (_npm, _observablehq) is not scanned for links and contacts: d3 and Plot carry
example URLs and digit runs that are not data.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

ROW_FILES: dict[str, frozenset[str]] = {
    "weekly_vacancies": frozenset(
        {
            "week",
            "source",
            "role_bucket",
            "days_collected",
            "complete_days",
            "first_collected_week",
            "active",
            "new",
            "closed",
        }
    ),
    "weekly_telegram": frozenset({"week", "role_bucket", "posts", "posts_before_dedup", "channels", "channels_week"}),
    "skills": frozenset({"source", "skill", "category", "vacancies_mentioning", "vacancies_total"}),
    "skills_by_role": frozenset(
        {"source", "role_bucket", "skill", "category", "vacancies_mentioning", "vacancies_in_bucket"}
    ),
    "skills_junior": frozenset({"source", "skill", "category", "vacancies_mentioning", "vacancies_graded_junior"}),
    "work_format": frozenset({"source", "role_bucket", "work_format", "vacancies", "vacancies_in_bucket"}),
    "junior": frozenset(
        {
            "source",
            "role_bucket",
            "vacancies",
            "open_to_junior",
            "graded_intern",
            "graded_junior",
            "no_experience_required",
            "grade_known",
            "experience_field_stated",
        }
    ),
    "salaries": frozenset(
        {
            "source",
            "level",
            "role_group",
            "role_bucket",
            "grade",
            "region_group",
            "vacancies",
            "with_salary_rub",
            "with_salary_other_currency",
            "with_from",
            "with_to",
        }
        | {f"salary_{b}_{q}" for b in ("from", "to") for q in ("p25", "median", "p75")}
    ),
}
ROW_FILES["lifetime_observed"] = frozenset(
    {
        "source",
        "level",
        "role_group",
        "vacancies",
        "still_open",
        "closed",
        "closed_in_gap",
        "censored",
        "left_censored",
        "closed_by_repost",
        "closed_in_quantiles",
        "period_from",
        "period_to",
        "window_days",
    }
    | {f"duration_days_{q}" for q in ("p25", "median", "p75", "p90")}
)
ROW_FILES["forecast_history"] = frozenset({"week", "skill", "posts_mentioning", "posts_it"})
DOCUMENT_FILES = frozenset({"meta", "lifetime", "forecast"})
ALLOWED = {**ROW_FILES, **{name: frozenset() for name in DOCUMENT_FILES}}

FORBIDDEN_KEYS = frozenset(
    {
        "title",
        "text",
        "description",
        "body",
        "url",
        "link",
        "source_id",
        "vacancy_id",
        "entity_id",
        "content_hash",
        "payload",
        "employer",
        "employer_name",
        "employer_key",
        "inn",
        "company",
        "contact",
        "contacts",
        "phone",
        "email",
        "channel",
        "author",
        "published_at",
        "fetched_at",
        "modified_at",
    }
)
# Skill and category names are the longest strings of a row (e.g. "Information security standards").
MAX_ROW_STRING = 60

VACANCY_LINK = re.compile(
    r"trudvsem\.ru/vacanc|career\.habr\.com/vacancies/\d|superjob\.ru/vakansii/|t\.me/(?:s/)?[A-Za-z0-9_]{4,}/\d+"
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\d.])(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?![\d.])")
LIBRARY_DIRS = ("_npm", "_observablehq", "_node")
TEXT_SUFFIXES = {".html", ".json", ".js", ".csv", ".md", ".txt"}

# The hash Framework appends to a file name: skills_by_role.0123abcd.json -> skills_by_role.
_HASHED = re.compile(r"^(?P<stem>.+?)(?:\.[0-9a-f]{8})?$")


def data_name(path: Path) -> str:
    match = _HASHED.match(path.stem)
    return match["stem"] if match else path.stem


def forbidden_keys(node: Any, where: str) -> list[str]:
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.lower() in FORBIDDEN_KEYS:
                found.append(f"{where}: forbidden key {key!r}")
            found += forbidden_keys(value, f"{where}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += forbidden_keys(value, f"{where}[{i}]")
    return found


def check_rows(name: str, rows: Any, where: str) -> list[str]:
    if not isinstance(rows, list):
        return [f"{where}: a row file must hold a list"]
    allowed = ROW_FILES[name]
    problems = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"{where}[{i}]: a row must be an object")
            continue
        extra = set(row) - allowed
        if extra:
            problems.append(f"{where}[{i}]: fields outside the export: {sorted(extra)}")
        for key, value in row.items():
            if isinstance(value, dict | list):
                problems.append(f"{where}[{i}].{key}: nested value in an aggregate row")
            elif isinstance(value, str) and len(value) > MAX_ROW_STRING:
                problems.append(f"{where}[{i}].{key}: string of {len(value)} characters")
    return problems


def scan_data_file(path: Path, root: Path) -> list[str]:
    where = path.relative_to(root).as_posix()
    name = data_name(path)
    if name not in ALLOWED:
        return [f"{where}: not a file the export writes"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    problems = forbidden_keys(doc, where)
    if name in ROW_FILES:
        problems += check_rows(name, doc, where)
    return problems


def scan_text(path: Path, root: Path) -> list[str]:
    where = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    problems = []
    for label, pattern in (("vacancy link", VACANCY_LINK), ("e-mail", EMAIL), ("phone", PHONE)):
        for match in pattern.finditer(text):
            problems.append(f"{where}: {label} {match.group(0)!r}")
    return problems


def scan(root: Path) -> list[str]:
    if not (root / "index.html").exists():
        return [f"{root}: no index.html, not a built site"]
    problems: list[str] = []
    data_files = sorted((root / "_file" / "data").glob("*"))
    if not data_files:
        problems.append(f"{root}: no data files under _file/data")
    for path in data_files:
        if path.suffix != ".json":
            problems.append(f"{path.relative_to(root).as_posix()}: data must be JSON written by the export")
        else:
            problems += scan_data_file(path, root)
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and path.suffix in TEXT_SUFFIXES and rel.parts[0] not in LIBRARY_DIRS:
            problems += scan_text(path, root)
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: scan_dist.py <built site dir>", file=sys.stderr)
        return 2
    root = Path(argv[1])
    problems = scan(root)
    for problem in problems:
        print(problem)
    files = sum(1 for p in root.rglob("*") if p.is_file())
    print(f"scanned {files} files in {root}: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
