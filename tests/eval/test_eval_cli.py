import json
from pathlib import Path
from typing import Any

import pytest

from radar import config, results
from radar.eval import cli


def fields(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "role": None,
        "grade": None,
        "skills": [],
        "salary": None,
        "work_format": None,
        "cities": [],
        "experience_years": None,
        "employment_type": None,
        "lang": "ru",
    }
    return base | kw


GOLD = [
    {"id": "a", "source": "trudvsem_snapshot", "fields": fields(role="qa", skills=["Python", "SQL"])},
    {"id": "b", "source": "trudvsem_snapshot", "fields": fields(role="backend", skills=["Java"])},
    {"id": "c", "source": "habr", "fields": fields(role="devops"), "note": "annotator note is ignored"},
    {"id": "d", "source": "habr", "fields": fields(role="data_analyst", lang="en")},
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def answer(**kw: Any) -> str:
    return json.dumps(fields(**kw), ensure_ascii=False)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    write_jsonl(tmp_path / "data" / "gold" / "dev.jsonl", GOLD)
    write_jsonl(tmp_path / "data" / "gold" / "test.jsonl", GOLD)
    preds = tmp_path / "data" / "predictions"
    write_jsonl(
        preds / "dev" / "good.jsonl",
        [
            {"id": "a", "output": answer(role="qa", skills=["python", "SQL"])},
            {"id": "b", "output": answer(role="backend", skills=["Java", "Spring"])},
            {"id": "c", "output": '{"role": "devops"'},  # broken JSON
            {"id": "zzz", "output": answer()},  # not in gold
        ],  # "d" is missing
    )
    write_jsonl(preds / "dev" / "weak.jsonl", [{"id": g["id"], "output": answer()} for g in GOLD])
    write_jsonl(preds / "test" / "good.jsonl", [{"id": g["id"], "output": answer()} for g in GOLD])
    return tmp_path


@pytest.mark.parametrize("split", ["test", "tg_test"])
def test_test_split_refused_without_final_test(repo: Path, capsys: pytest.CaptureFixture[str], split: str) -> None:
    assert cli.main(["--system", "good", "--split", split]) == 2
    assert "refused" in capsys.readouterr().err
    assert not (repo / "results").exists()


@pytest.mark.parametrize("split", ["dev", "tg_dev"])
def test_final_test_flag_needs_test_split(repo: Path, split: str) -> None:
    assert cli.main(["--system", "good", "--split", split, "--final-test"]) == 2
    assert not (repo / "results").exists()


def test_bad_system_name(repo: Path) -> None:
    assert cli.main(["--system", "../good", "--split", "dev"]) == 2


def test_dev_report(repo: Path) -> None:
    assert cli.main(["--system", "good", "--split", "dev"]) == 0
    doc = json.loads((repo / "results" / "extract" / "dev_good.json").read_text(encoding="utf-8"))
    results.ResultFile.model_validate(doc)
    m = doc["metrics"]
    # valid answers for a and b only: c is broken, d is missing
    assert (m["valid_json"]["n"], m["valid_json"]["denominator"]) == (2, 4)
    assert m["role_accuracy"]["value"] == 0.5
    # skills: tp 3 (python, SQL, Java), predicted 4, gold 3 -> F1 6/7
    assert m["skills_f1"]["value"] == pytest.approx(6 / 7, abs=1e-6)
    for entry in m.values():
        assert entry["ci_low"] <= entry["value"] <= entry["ci_high"]
    assert doc["n_examples"] == 4
    assert doc["n_missing_predictions"] == 1
    assert doc["n_extra_predictions_ignored"] == 1
    assert doc["bootstrap"]["n_resamples"] >= 1000
    assert doc["gold_sha256"] and doc["predictions_sha256"]
    assert set(doc["by_source"]) == {"habr", "trudvsem_snapshot"}
    assert doc["by_source"]["trudvsem_snapshot"]["role_accuracy"]["value"] == 1.0
    assert doc["by_source"]["habr"]["role_accuracy"]["value"] == 0.0
    assert "salary_currency_accuracy" in doc["undefined_metrics"]


def test_compare_writes_paired_difference(repo: Path) -> None:
    assert cli.main(["--system", "good", "--split", "dev", "--compare", "weak"]) == 0
    doc = json.loads((repo / "results" / "extract" / "dev_good_vs_weak.json").read_text(encoding="utf-8"))
    acc = doc["metrics"]["role_accuracy"]
    assert (acc["a"], acc["b"]) == (0.5, 0.0)
    assert acc["value"] == 0.5
    assert doc["n_missing_predictions"] == {"good": 1, "weak": 0}


def test_config_demands_enough_resamples(tmp_path: Path) -> None:
    text = cli.CONFIG.read_text(encoding="utf-8").replace("n_resamples = 2000", "n_resamples = 200")
    (tmp_path / "eval.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="1000"):
        cli.load_config(tmp_path / "eval.toml")


def test_gold_must_match_schema(repo: Path) -> None:
    bad = [{"id": "a", "fields": fields(role="astronaut")}]
    write_jsonl(repo / "data" / "gold" / "dev.jsonl", bad)
    with pytest.raises(ValueError, match=r"dev\.jsonl:1"):
        cli.main(["--system", "good", "--split", "dev"])
