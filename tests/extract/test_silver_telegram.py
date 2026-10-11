import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar import config, results
from radar.extract import silver
from radar.extract.baseline import extract
from radar.extract.schema import dump
from radar.extract.teacher import sha256_text

FIELDS = {
    "role": "data_scientist",
    "grade": "senior",
    "skills": ["Python", "PyTorch"],
    "salary": {"from": 3000, "to": 6000, "currency": "USD", "gross_net": None, "period": None},
    "work_format": "hybrid",
    "cities": ["Санкт-Петербург"],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}
CONFIG = """
[candidates]
trudvsem_habr = ["rules"]
telegram = ["rules", "qwen_all"]
[chosen]
trudvsem_habr = "rules"
telegram = "qwen_all"
[rules_sample.trudvsem_habr]
limit = 10
[compositions.rules]
[compositions.qwen_all]
role = "qwen"
grade = "qwen"
skills = "qwen"
salary = "qwen"
work_format = "qwen"
cities = "qwen"
experience_years = "qwen"
employment_type = "qwen"
lang = "qwen"
"""


def post(i: int) -> str:
    return f"#ml #spb\nSenior Data Scientist {i}\nВилка: 3000 $ - 6000 $\nPython, PyTorch"


def s1_text(i: int) -> str:
    return f"Должность: Python-разработчик {i}\nАдрес: г Москва\n\nPython и PostgreSQL."


def row(rid: str, text: str, output: dict[str, Any] | None, exclude: str | None = None) -> dict[str, Any]:
    return {
        "id": rid,
        "output": None if output is None else json.dumps(output, ensure_ascii=False),
        "valid": output is not None,
        "input_sha256": sha256_text(text),
        "error": None,
        "exclude": exclude,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("RADAR_DATA_DIR", str(root))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    return root


def write_s1(data: Path) -> None:
    corpus = data / "corpus"
    corpus.mkdir(parents=True)
    records = [{"id": f"t:{i}", "source": "trudvsem", "role_proxy": "dev", "input": s1_text(i)} for i in range(2)]
    pq.write_table(pa.Table.from_pylist(records), corpus / "corpus.parquet")
    split = [{"split": "train_pool", "member_ids": ["t:0", "t:1"]}]
    pq.write_table(pa.Table.from_pylist(split), corpus / "split.parquet")
    for name in ("qwen", "yandex"):
        rows = [{k: v for k, v in row(f"t:{i}", s1_text(i), FIELDS).items() if k != "exclude"} for i in range(2)]
        write_jsonl(data / "teacher" / name / "train_pool.jsonl", rows)


def write_telegram(data: Path) -> None:
    root = data / "corpus" / "telegram"
    root.mkdir(parents=True)
    records = [
        {
            "id": f"telegram:c/{i}",
            "group_id": "g0" if i < 2 else f"g{i}",
            "network": "geekjob",
            "role_proxy": "ml_ds_data",
            "published_at": f"2026-10-0{i + 1}T10:00:00+00:00",
            "input": post(i),
        }
        for i in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(records), root / "records.parquet")
    split = [
        {"split": "train_pool", "member_ids": ["telegram:c/0", "telegram:c/1"]},
        {"split": "train_pool", "member_ids": ["telegram:c/2"]},
        {"split": "train_pool", "member_ids": ["telegram:c/3"]},
        {"split": "dev", "member_ids": ["telegram:c/4"]},
    ]
    pq.write_table(pa.Table.from_pylist(split), root / "split.parquet")
    # c/0 was labeled while it was its group's latest post; c/1 arrived later and was not labeled
    qwen = [
        row("telegram:c/0", post(0), FIELDS),
        row("telegram:c/2", post(2), FIELDS, exclude="multi_vacancy"),
        row("telegram:c/3", post(3), FIELDS),
        row("telegram:c/4", post(4), FIELDS),
    ]
    yandex = [
        row("telegram:c/0", post(0), {**FIELDS, "role": "ml_engineer"}),
        row("telegram:c/2", post(2), FIELDS),
        row("telegram:c/3", post(3), FIELDS, exclude="not_vacancy"),
        row("telegram:c/4", post(4), FIELDS),
    ]
    write_jsonl(data / "teacher" / "qwen" / "tg_train_pool.jsonl", qwen)
    write_jsonl(data / "teacher" / "yandex" / "tg_train_pool.jsonl", yandex)


def write_meta(data: Path) -> None:
    for name in ("qwen", "yandex"):
        for split in ("train_pool", "tg_train_pool"):
            meta = {"model": name, "prompt_sha256": "p", "schema_sha256": split, "runs": []}
            (data / "teacher" / name / f"{split}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_build_composes_each_family_on_its_own_and_drops_excluded_posts(data: Path, tmp_path: Path) -> None:
    write_s1(data)
    write_telegram(data)
    write_meta(data)
    cfg = tmp_path / "s.toml"
    cfg.write_text(CONFIG, encoding="utf-8")

    assert silver.main(["--config", str(cfg), "build"]) == 0

    rows = {
        r["id"]: r for r in map(json.loads, (data / "silver" / "train.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert sorted(rows) == ["t:0", "t:1", "telegram:c/0"]
    assert json.loads(rows["telegram:c/0"]["output"]) == FIELDS
    assert json.loads(rows["t:0"]["output"]) == dump(extract(s1_text(0)))

    doc = json.loads((tmp_path / "results" / "silver" / "build.json").read_text(encoding="utf-8"))
    m = doc["metrics"]
    assert m["records_written"]["value"] == 3
    assert (m["telegram.records_written"]["n"], m["telegram.records_written"]["denominator"]) == (1, 4)
    assert m["telegram.dropped_excluded"]["n"] == 2
    assert m["telegram.dropped_not_in_train_pool"]["n"] == 1
    assert (m["telegram.excluded_by_qwen"]["n"], m["telegram.excluded_by_yandex"]["n"]) == (1, 1)
    assert m["trudvsem_habr.records_written"]["n"] == 2
    assert "trudvsem_habr.dropped_excluded" not in m
    assert doc["compositions"] == {"trudvsem_habr": "rules", "telegram": "qwen_all"}
    assert doc["teachers"]["telegram"]["qwen"]["schema_sha256"] == "tg_train_pool"
    assert "trudvsem_habr" not in doc["teachers"]
    assert doc["families_not_built"] == []


def test_dev_on_tg_dev_writes_predictions_and_waits_for_gold(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_jsonl(data / "gold" / "tg_dev_inputs.jsonl", [{"id": "telegram:c/4", "source": "telegram", "input": post(4)}])
    for name in ("qwen", "yandex"):
        write_jsonl(data / "teacher" / name / "tg_dev.jsonl", [row("telegram:c/4", post(4), FIELDS)])
    cfg = tmp_path / "s.toml"
    cfg.write_text(CONFIG, encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(silver, "evaluate", lambda argv: calls.append(argv) or 0)

    assert silver.main(["--config", str(cfg), "dev", "--family", "telegram"]) == 0
    assert calls == []
    for name in ("rules", "qwen_all"):
        assert (data / "predictions" / "tg_dev" / f"silver_{name}.jsonl").exists()

    write_jsonl(data / "gold" / "tg_dev.jsonl", [{"id": "telegram:c/4", "source": "telegram", "fields": FIELDS}])
    assert silver.main(["--config", str(cfg), "dev", "--family", "telegram", "--composition", "qwen_all"]) == 0
    assert calls == [["--system", "silver_qwen_all", "--split", "tg_dev", "--compare", "regex"]]
