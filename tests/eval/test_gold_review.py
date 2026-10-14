"""radar.eval.gold_review: an exported file parses back unchanged, an edit and a read mark are counted."""

import json
from pathlib import Path

import pytest

from radar.eval import gold_review

FIELDS = {
    "role": "qa",
    "grade": "middle",
    "skills": ["Android"],
    "salary": {"from": None, "to": 350000, "currency": "RUB", "gross_net": None, "period": "month"},
    "work_format": "remote",
    "cities": [],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    gold, out = tmp_path / "gold", tmp_path / "review"
    gold.mkdir()
    rows = [{"id": f"telegram:c/{i}", "source": "telegram", "fields": FIELDS} for i in (1, 2)]
    (gold / "tg_test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    texts = [{"id": r["id"], "source": "telegram", "input": "Manual QA (middle), до 350 000 руб."} for r in rows]
    (gold / "tg_test_inputs.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts), encoding="utf-8")
    return gold, out


def test_untouched_file_is_unread_and_unchanged(dirs: tuple[Path, Path]) -> None:
    gold, out = dirs
    assert gold_review.export(gold, out, "tg_test") is not None
    assert gold_review.export(gold, out, "tg_test") is None
    summary = gold_review.compare(gold, out, "tg_test")
    assert summary is not None
    assert summary["records_read"]["n"] == 0 and summary["records_changed"]["n"] == 0


def test_edit_and_read_mark_are_counted(dirs: tuple[Path, Path]) -> None:
    gold, out = dirs
    gold_review.export(gold, out, "tg_test")
    path = out / "tg_test.md"
    text = path.read_text(encoding="utf-8").replace('"grade": "middle"', '"grade": "senior"', 1)
    text = text.replace(gold_review.UNREAD, gold_review.READ)
    path.write_text(text, encoding="utf-8")
    summary = gold_review.compare(gold, out, "tg_test")
    assert summary is not None
    assert summary["records_read"]["n"] == 2
    assert summary["records_changed"]["n"] == 1 and summary["field_changed.grade"]["n"] == 1
    corrected = [json.loads(x) for x in (out / "tg_test.corrected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert corrected[0]["fields"]["grade"] == "senior" and corrected[1]["fields"]["grade"] == "middle"


def test_broken_block_names_the_record(dirs: tuple[Path, Path]) -> None:
    gold, out = dirs
    gold_review.export(gold, out, "tg_test")
    path = out / "tg_test.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace('"grade": "middle"', '"grade": "boss"', 1), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="record 1"):
        gold_review.compare(gold, out, "tg_test")
