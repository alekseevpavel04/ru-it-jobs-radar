import json
from pathlib import Path
from typing import Any

import pytest

from radar import config, results
from radar.extract import teacher
from radar.extract.baseline import extract
from radar.extract.teacher import sha256_text

FIELDS = {
    "role": "backend",
    "grade": None,
    "skills": ["Python", "PostgreSQL"],
    "salary": None,
    "work_format": "remote",
    "cities": ["Москва"],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}


def text(i: int) -> str:
    return f"Должность: Python-разработчик {i}\nАдрес: г Москва\n\nУдалённая работа, Python и PostgreSQL."


def row(i: int, output: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": f"t:{i}",
        "output": None if output is None else json.dumps(output, ensure_ascii=False),
        "valid": output is not None,
        "input_sha256": sha256_text(text(i)),
        "error": None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_teacher(data: Path, name: str, split: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(data / "teacher" / name / f"{split}.jsonl", rows)
    meta = {"model": name, "prompt_sha256": "p", "schema_sha256": "s", "runs": []}
    (data / "teacher" / name / f"{split}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    monkeypatch.setattr(teacher, "load_dotenv", lambda: None)
    monkeypatch.setenv("RADAR_DATA_DIR", str(root))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    return root


def share(m: dict[str, Any]) -> tuple[int, int]:
    return m["n"], m["denominator"]


def test_agreement_on_train_pool_and_dev_with_gold_reference(data: Path, tmp_path: Path) -> None:
    write_teacher(data, "qwen", "train_pool", [row(i, FIELDS) for i in range(3)])
    write_teacher(data, "yandex", "train_pool", [row(0, FIELDS), row(1, {**FIELDS, "role": "qa"}), row(2, None)])
    write_teacher(data, "qwen", "dev", [row(i, FIELDS) for i in range(2)])
    swapped = {**FIELDS, "skills": ["postgresql", "python"]}
    write_teacher(data, "yandex", "dev", [row(0, swapped), row(1, {**FIELDS, "role": "qa"})])
    write_jsonl(data / "gold" / "dev.jsonl", [{"id": f"t:{i}", "fields": FIELDS} for i in range(2)])
    write_jsonl(
        data / "gold" / "dev_inputs.jsonl", [{"id": f"t:{i}", "source": "habr", "input": text(i)} for i in range(2)]
    )

    assert teacher.main(["--agreement"]) == 0

    doc = json.loads((tmp_path / "results" / "teacher" / "agreement.json").read_text(encoding="utf-8"))
    m = doc["metrics"]
    assert m["train_pool_records_labeled_by_both"]["value"] == 3
    assert share(m["train_pool_qwen_valid"]) == (3, 3)
    assert share(m["train_pool_yandex_valid"]) == (2, 3)
    assert share(m["train_pool_role_exact"]) == (1, 2)
    assert share(m["train_pool_skills_exact"]) == (2, 2)
    assert share(m["train_pool_all_fields_equal"]) == (1, 2)
    assert m["train_pool_skills_f1_dictionary"]["value"] == 1.0

    assert share(m["dev_skills_exact"]) == (2, 2)
    assert share(m["dev_gold_role_agreed_correct"]) == (1, 1)
    assert share(m["dev_gold_role_disagreed_qwen_correct"]) == (1, 1)
    assert share(m["dev_gold_role_disagreed_yandex_correct"]) == (0, 1)
    rules_right = int(extract(text(1)).role == "backend")
    assert share(m["dev_gold_role_disagreed_rules_correct"]) == (rules_right, 1)
    assert doc["teachers"]["qwen/dev"]["model"] == "qwen"


def test_agreement_needs_both_teachers(data: Path) -> None:
    write_teacher(data, "qwen", "train_pool", [row(0, FIELDS)])
    with pytest.raises(SystemExit, match="label train_pool with yandex"):
        teacher.main(["--agreement"])
