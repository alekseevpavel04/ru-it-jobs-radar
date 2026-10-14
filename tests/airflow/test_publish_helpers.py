"""The keep-check-restore helpers of radar_publish around the site export."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


site_export = load(ROOT / "airflow" / "include" / "site_export.py", "airflow_include_site_export")
scan_dist = site_export.load_scan_dist(ROOT / "src" / "radar" / "site" / "scan_dist.py")

ROW = {
    "week": "2026-10-05",
    "source": "habr",
    "role_bucket": "ml_ds",
    "days_collected": 7,
    "complete_days": 7,
    "first_collected_week": False,
    "active": 40,
    "new": 9,
    "closed": 3,
}


def export(root: Path, rows: list[dict[str, object]], data_through: str) -> None:
    data = root / site_export.DATA
    data.mkdir(parents=True, exist_ok=True)
    (data / "weekly_vacancies.json").write_text(json.dumps(rows), encoding="utf-8")
    meta = {"data_through": data_through, "files": {"weekly_vacancies.json": len(rows)}}
    (data / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / site_export.RESULT).parent.mkdir(parents=True, exist_ok=True)
    (root / site_export.RESULT).write_text(json.dumps({"rows": len(rows)}), encoding="utf-8")


def test_changes_name_files_rows_and_the_data_day(tmp_path: Path) -> None:
    export(tmp_path, [ROW], "2026-10-11")
    before = site_export.state(tmp_path)
    export(tmp_path, [ROW, ROW | {"source": "trudvsem"}], "2026-10-13")
    (tmp_path / site_export.DATA / "skills.json").write_text("[]", encoding="utf-8")
    diff = site_export.changes(before, site_export.state(tmp_path))
    assert diff["changed"] == ["meta.json", "weekly_vacancies.json"]
    assert diff["added"] == ["skills.json"] and diff["removed"] == [] and diff["unchanged"] == []
    assert diff["rows"] == {"weekly_vacancies.json": [1, 2]}
    assert diff["data_through"] == ["2026-10-11", "2026-10-13"]


def test_a_rejected_export_is_rolled_back_to_the_copy(tmp_path: Path) -> None:
    export(tmp_path, [ROW], "2026-10-11")
    before = site_export.state(tmp_path)
    site_export.keep_copy(tmp_path, tmp_path / "copy")
    export(tmp_path, [ROW | {"title": "Senior ML engineer at a named company"}], "2026-10-13")
    (tmp_path / site_export.DATA / "extra.json").write_text("{}", encoding="utf-8")

    problems = site_export.scan(tmp_path, scan_dist)
    assert any("fields outside the export: ['title']" in p for p in problems)
    assert any("extra.json: not a file the export writes" in p for p in problems)

    assert site_export.restore(tmp_path, tmp_path / "copy") == ["meta.json", "weekly_vacancies.json"]
    assert site_export.state(tmp_path) == before
    assert json.loads((tmp_path / site_export.RESULT).read_text(encoding="utf-8")) == {"rows": 1}


def test_a_clean_export_passes_the_build_check(tmp_path: Path) -> None:
    export(tmp_path, [ROW], "2026-10-11")
    assert site_export.scan(tmp_path, scan_dist) == []


def test_contacts_in_an_export_are_caught(tmp_path: Path) -> None:
    export(tmp_path, [ROW | {"role_bucket": "write to hr@example.com"}], "2026-10-11")
    assert any("e-mail" in p for p in site_export.scan(tmp_path, scan_dist))


def test_restore_without_a_copy_leaves_the_data_alone(tmp_path: Path) -> None:
    export(tmp_path, [ROW], "2026-10-11")
    with pytest.raises(FileNotFoundError):
        site_export.restore(tmp_path, tmp_path / "missing")
    assert (tmp_path / site_export.DATA / "weekly_vacancies.json").exists()
