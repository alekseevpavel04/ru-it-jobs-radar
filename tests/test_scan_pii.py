"""scripts/scan_pii.py on synthetic values: what it reports, what it leaves to the code, and the allowlist."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scanner() -> Any:
    spec = importlib.util.spec_from_file_location("scan_pii", ROOT / "scripts" / "scan_pii.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_check_digits(first10: str) -> str:
    d = [int(c) for c in first10]
    n11 = sum(w * x for w, x in zip((7, 2, 4, 10, 3, 5, 9, 4, 6, 8), d, strict=True)) % 11 % 10
    d.append(n11)
    n12 = sum(w * x for w, x in zip((3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8), d, strict=True)) % 11 % 10
    return first10 + str(n11) + str(n12)


def _kinds(scanner: Any, text: str, code: bool = False) -> list[str]:
    return [kind for _, kind, _, _ in scanner.scan_string(text, scanner.keep_handles(), code=code)]


def test_unlabelled_inn_needs_a_valid_checksum(scanner: Any) -> None:
    inn = _with_check_digits("7700000000")
    assert scanner.is_person_inn(inn)
    wrong = inn[:-1] + str((int(inn[-1]) + 1) % 10)
    assert not scanner.is_person_inn(wrong)
    assert _kinds(scanner, f"номер {inn} в тексте") == ["inn_unlabelled"]
    assert _kinds(scanner, f"номер {wrong} в тексте") == []
    assert _kinds(scanner, f"hash a{inn}b") == []


def test_contacts_names_and_account_roots_are_found(scanner: Any) -> None:
    text = "Пишите: hr@example.com, +7 900 000-00-00, @ivan_petrov_hr\nИванов Иван Иванович"
    assert set(_kinds(scanner, text)) == {"email", "phone", "nick", "person_name"}
    assert _kinds(scanner, "https://github.com/someone_private") == ["account_link"]
    # a repository, the author's own account and service pages are not people
    assert _kinds(scanner, "https://github.com/someone_private/project") == []
    assert _kinds(scanner, "https://huggingface.co/alekseevpavel04") == []
    assert _kinds(scanner, "https://huggingface.co/datasets") == []


def test_code_decorators_at_rules_fractions_and_preview_links_are_not_findings(scanner: Any) -> None:
    code = "@dataclass\nclass A:\n    pass\n@media (max-width: 1px) {}\nx = '@ivan_petrov_hr'\n"
    assert _kinds(scanner, code, code=True) == ["nick"]
    assert _kinds(scanner, "@dataclass", code=False) == ["nick"]
    assert _kinds(scanner, "share 0.3333333333 of rows") == []
    assert _kinds(scanner, "read through https://t.me/s and t.me/robots.txt") == []
    assert _kinds(scanner, "https://t.me/ivan_petrov_hr") == ["profile_link"]
    # channels of the config are not people
    assert _kinds(scanner, "@datasciencejobs") == []


def test_json_values_are_read_decoded_and_parquet_by_cell(scanner: Any, tmp_path: Path) -> None:
    doc = tmp_path / "a.json"
    doc.write_text(json.dumps({"contact": "Иванов Иван Иванович"}), encoding="utf-8")  # escaped by json.dumps
    hits = list(scanner.scan_file(doc, "a.json", scanner.keep_handles()))
    assert [(h.where, h.kind) for h in hits] == [("$.contact", "person_name")]

    table = tmp_path / "t.parquet"
    pq.write_table(pa.table({"id": ["1"], "text": ["звоните +7 900 000-00-00"]}), table)
    hits = list(scanner.scan_file(table, "t.parquet", scanner.keep_handles()))
    assert [(h.where, h.kind) for h in hits] == [("row 0 text", "phone")]
    assert all(len(h.fingerprint) == 12 for h in hits)


def test_allowlist_matches_path_place_kind_and_fingerprint(scanner: Any, tmp_path: Path) -> None:
    hit = scanner.Hit("data/x.parquet", "row 3 vacancy_id", "phone", "", "abc123abc123", "9000000000")
    allow = tmp_path / "allow.tsv"
    allow.write_text(
        "# comment\ndata/*.parquet::row * vacancy_id\tphone\t*\tportal id\nother/*\t*\tabc123abc123\tsomething else\n",
        encoding="utf-8",
    )
    rules = scanner.load_allowlist(allow)
    assert rules[0].covers(hit) and not rules[1].covers(hit)
    assert not rules[0].covers(scanner.Hit("data/x.parquet", "row 3 text", "phone", "", "abc123abc123", "9"))
    allow.write_text("data/*\tphone\t*\t \n", encoding="utf-8")
    with pytest.raises(SystemExit):
        scanner.load_allowlist(allow)


def test_main_exits_one_on_open_findings_and_never_prints_values(
    scanner: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "doc.md").write_text("Контакт: hr@example.com\n", encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    empty = tmp_path / "allow.tsv"
    empty.write_text("", encoding="utf-8")
    report = tmp_path / "report.json"
    assert scanner.main([str(tmp_path), "--allowlist", str(empty), "--report", str(report)]) == 1
    out = capsys.readouterr().out
    assert "hr@example.com" not in out and "email" in out and "pic.png" in out
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["open_by_kind"] == {"email": 1} and "value" not in saved["open"][0]

    fp = scanner.fingerprint("hr@example.com")
    empty.write_text(f"*doc.md\temail\t{fp}\tsynthetic\n", encoding="utf-8")
    assert scanner.main([str(tmp_path), "--allowlist", str(empty)]) == 0
