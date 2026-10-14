import json
from pathlib import Path
from typing import Any

import pytest

from radar.config import ROOT
from radar.site import export
from radar.site.scan_dist import ALLOWED, ROW_FILES, main, scan

ROW_MODELS = {
    "weekly_vacancies": export.WeeklyVacancies,
    "weekly_telegram": export.WeeklyTelegram,
    "skills": export.SkillTotal,
    "skills_by_role": export.SkillByRole,
    "skills_junior": export.SkillJunior,
    "work_format": export.WorkFormat,
    "junior": export.Junior,
    "salaries": export.Salary,
    "lifetime_observed": export.LifetimeObserved,
    "forecast_history": export.ForecastHistory,
}


def test_scanner_allows_exactly_the_fields_the_export_writes() -> None:
    assert {name: set(fields) for name, fields in ROW_FILES.items()} == {
        name: set(model.model_fields) for name, model in ROW_MODELS.items()
    }
    assert set(ALLOWED) == set(ROW_FILES) | {"meta", "lifetime", "forecast"}


def site(tmp_path: Path, files: dict[str, Any], pages: dict[str, str] | None = None) -> Path:
    root = tmp_path / "dist"
    data = root / "_file" / "data"
    data.mkdir(parents=True)
    (root / "index.html").write_text("<p>Срез рынка</p>", encoding="utf-8")
    for name, doc in files.items():
        (data / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    for name, text in (pages or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


SKILL_ROW = {"source": "habr", "skill": "SQL", "category": "language", "vacancies_mentioning": 3, "vacancies_total": 9}
META = {"exported_at": "2026-10-11T10:57:35Z", "periods": {"telegram": {"period_from": "2026-07-01"}}}


def test_a_clean_build_passes(tmp_path: Path) -> None:
    root = site(
        tmp_path,
        {"skills.0123abcd.json": [SKILL_ROW], "meta.89abcdef.json": META},
        {"_npm/d3/x.js": "mailto:someone@example.com +7 495 123-45-67", "skills.html": "<a href='./methodology'>"},
    )
    assert scan(root) == []


@pytest.mark.parametrize(
    ("files", "pages", "expected"),
    [
        ({"skills.0123abcd.json": [{**SKILL_ROW, "title": "Senior SQL"}]}, {}, "forbidden key 'title'"),
        ({"skills.0123abcd.json": [{**SKILL_ROW, "employer_rank": 1}]}, {}, "fields outside the export"),
        ({"skills.0123abcd.json": [{**SKILL_ROW, "skill": "x" * 400}]}, {}, "string of 400 characters"),
        ({"skills.0123abcd.json": [{**SKILL_ROW, "category": ["a"]}]}, {}, "nested value"),
        ({"meta.89abcdef.json": {**META, "sample": [{"text": "Ищем аналитика"}]}}, {}, "forbidden key 'text'"),
        ({"posts.0123abcd.json": []}, {}, "not a file the export writes"),
        ({"skills.0123abcd.json": [SKILL_ROW]}, {"skills.html": "https://t.me/s/datajobs/1234"}, "vacancy link"),
        (
            {"skills.0123abcd.json": [SKILL_ROW]},
            {"skills.html": "https://career.habr.com/vacancies/1000150001"},
            "link",
        ),
        ({"skills.0123abcd.json": [SKILL_ROW]}, {"_import/x.js": "hr@company.ru"}, "e-mail"),
        ({"skills.0123abcd.json": [SKILL_ROW]}, {"junior.html": "звоните +7 (495) 123-45-67"}, "phone"),
    ],
)
def test_what_must_not_be_published_fails_the_scan(
    tmp_path: Path, files: dict[str, Any], pages: dict[str, str], expected: str
) -> None:
    problems = scan(site(tmp_path, files, pages))
    assert any(expected in p for p in problems), problems


def test_not_a_built_site(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scan_dist.py", str(tmp_path)]) == 1
    assert "no index.html" in capsys.readouterr().out


DIST = ROOT / "site" / "dist"


@pytest.mark.skipif(not (DIST / "index.html").exists(), reason="site is not built (npm run build in site/)")
def test_the_local_build_passes() -> None:
    assert scan(DIST) == []
