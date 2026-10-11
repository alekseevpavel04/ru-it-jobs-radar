import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar import config, results
from radar.extract import silver
from radar.extract.baseline import extract
from radar.extract.schema import VacancyFields, dump
from radar.extract.silver import Composition, compose, load_config
from radar.extract.teacher import sha256_text

FIELDS = {
    "role": "backend",
    "grade": None,
    "skills": ["Python", "PostgreSQL"],
    "salary": {"from": 100000, "to": None, "currency": "RUB", "gross_net": None, "period": None},
    "work_format": "remote",
    "cities": ["Москва"],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}
RULES = VacancyFields.model_validate({**FIELDS, "role": "software_engineer", "work_format": None})


def answer(**changes: Any) -> VacancyFields:
    return VacancyFields.model_validate({**FIELDS, **changes})


def composition(**sources: str) -> Composition:
    return Composition("t", {f: sources.get(f, "rules") for f in VacancyFields.model_fields})


def config_text(compositions: str, chosen: str = "a", candidates: str = '["a"]') -> str:
    return f'[candidates]\ntrudvsem_habr = {candidates}\n[chosen]\ntrudvsem_habr = "{chosen}"\n{compositions}'


def write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- config -------------------------------------------------------------------------------------


def test_unlisted_fields_come_from_rules(tmp_path: Path) -> None:
    text = config_text('[compositions.a]\nrole = "qwen"\n[compositions.b]\n', candidates='["a", "b"]')
    cfg = load_config(write_config(tmp_path / "s.toml", text))
    assert cfg.chosen == {"trudvsem_habr": "a"}
    assert cfg.candidates == {"trudvsem_habr": ["a", "b"]}
    assert cfg.compositions["a"].sources["role"] == "qwen"
    assert cfg.compositions["a"].sources["skills"] == "rules"
    assert set(cfg.compositions["b"].sources.values()) == {"rules"}


def test_a_family_may_have_no_choice_yet(tmp_path: Path) -> None:
    text = config_text('[compositions.a]\nrole = "qwen"\n[compositions.b]\n').replace(
        "[chosen]", 'telegram = ["a", "b"]\n[chosen]'
    )
    cfg = load_config(write_config(tmp_path / "s.toml", text))
    assert cfg.candidates["telegram"] == ["a", "b"]
    assert "telegram" not in cfg.chosen


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (config_text('[compositions.a]\nsalary_text = "qwen"\n'), "not schema fields"),
        (config_text('[compositions.a]\nrole = "gpt"\n'), "a source is one of"),
        (config_text("[compositions.a]\n", chosen="b"), "not one of candidates"),
        (config_text("[compositions.a]\n", candidates='["a", "c"]'), "not in"),
        (config_text("[compositions.a]\n", candidates='["a", "a", "a", "a", "a"]'), "at most 4"),
        (config_text("[compositions.A]\n", chosen="A", candidates='["A"]'), "lowercase"),
        ('[chosen]\nhh = "a"\n[compositions.a]\n', "not a source family"),
    ],
)
def test_bad_config_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(write_config(tmp_path / "s.toml", text))


def test_repository_config_loads() -> None:
    cfg = load_config()
    assert cfg.chosen["trudvsem_habr"] == "rules"
    assert set(cfg.candidates) == {"trudvsem_habr", "telegram"}
    assert all(len(names) <= 4 for names in cfg.candidates.values())


# --- composition --------------------------------------------------------------------------------


def test_rules_only_is_the_rules_answer() -> None:
    composed = compose(composition(), RULES, {"qwen": answer(), "yandex": None}, fallback=False)
    assert composed.output == RULES


def test_a_teacher_field_comes_from_that_teacher_only() -> None:
    answers = {"qwen": answer(role="data_engineer"), "yandex": answer(role="qa")}
    composed = compose(composition(role="yandex"), RULES, answers, fallback=False)
    assert composed.output is not None
    assert composed.output.role == "qa"
    assert composed.output.work_format is None


def test_agreement_ignores_list_order_and_case_and_takes_the_first_teachers_value() -> None:
    answers = {"qwen": answer(), "yandex": answer(skills=["postgresql", "python"])}
    composed = compose(composition(skills="agree", role="agree"), RULES, answers, fallback=False)
    assert composed.output is not None
    assert composed.output.skills == ["Python", "PostgreSQL"]
    assert composed.output.role == "backend"
    assert composed.agreed == {"skills", "role"}


def test_disagreement_or_a_missing_teacher_falls_back_to_rules_under_agree() -> None:
    sources = composition(role="agree", work_format="agree")
    disagree = compose(sources, RULES, {"qwen": answer(), "yandex": answer(role="qa")}, fallback=False)
    assert disagree.output is not None
    assert (disagree.output.role, disagree.output.work_format) == ("software_engineer", "remote")
    assert disagree.agreed == {"work_format"}
    missing = compose(sources, RULES, {"qwen": answer(), "yandex": None}, fallback=False)
    assert missing.output is not None
    assert (missing.output.role, missing.output.work_format, missing.agreed) == ("software_engineer", None, set())


def test_missing_named_teacher_drops_the_record_or_falls_back() -> None:
    sources = composition(role="qwen", work_format="yandex")
    answers = {"qwen": None, "yandex": answer()}
    dropped = compose(sources, RULES, answers, fallback=False)
    assert (dropped.output, dropped.missing) == (None, "qwen")
    kept = compose(sources, RULES, answers, fallback=True)
    assert kept.output is not None
    assert (kept.output.role, kept.output.work_format) == ("software_engineer", "remote")
    assert kept.fallback == {"role"}


# --- runs ---------------------------------------------------------------------------------------


def text(i: int) -> str:
    return f"Должность: Python-разработчик {i}\nАдрес: г Москва\n\nТребуется опыт работы с Python и PostgreSQL."


def teacher_row(rid: str, input_text: str, output: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    return {
        "id": rid,
        "output": None if output is None else json.dumps(output, ensure_ascii=False),
        "valid": output is not None,
        "input_sha256": sha256_text(input_text),
        "error": error,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_meta(data: Path, split: str) -> None:
    for name in ("qwen", "yandex"):
        meta = {"model": name, "prompt_sha256": "p", "schema_sha256": "s", "runs": []}
        (data / "teacher" / name / f"{split}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("RADAR_DATA_DIR", str(root))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    return root


def test_build_drops_records_by_reason_and_writes_counts(data: Path, tmp_path: Path) -> None:
    corpus = data / "corpus"
    corpus.mkdir(parents=True)
    records = [{"id": f"t:{i}", "source": "trudvsem", "role_proxy": "dev", "input": text(i)} for i in range(5)]
    pq.write_table(pa.Table.from_pylist(records), corpus / "corpus.parquet")
    split = [
        {"split": "train_pool", "member_ids": ["t:0", "t:1", "t:2", "t:3"]},
        {"split": "dev", "member_ids": ["t:4"]},
    ]
    pq.write_table(pa.Table.from_pylist(split), corpus / "split.parquet")
    qwen = [teacher_row(f"t:{i}", text(i), FIELDS) for i in range(5)]
    qwen[1]["input_sha256"] = sha256_text("an older input")
    yandex = [teacher_row(f"t:{i}", text(i), {**FIELDS, "role": "qa"}) for i in range(5)]
    yandex[2] = teacher_row("t:2", text(2), None)
    write_jsonl(data / "teacher" / "qwen" / "train_pool.jsonl", qwen)
    write_jsonl(data / "teacher" / "yandex" / "train_pool.jsonl", yandex)
    write_meta(data, "train_pool")
    cfg = write_config(
        tmp_path / "s.toml", config_text('[compositions.y]\nrole = "yandex"\nskills = "agree"\n', "y", '["y"]')
    )

    assert silver.main(["--config", str(cfg), "build"]) == 0

    rows = [json.loads(x) for x in (data / "silver" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in rows) == ["t:0", "t:3"]
    label = json.loads(rows[0]["output"])
    assert label["role"] == "qa"
    assert label["skills"] == FIELDS["skills"]
    assert label["cities"] == extract(rows[0]["input"]).cities
    doc = json.loads((tmp_path / "results" / "silver" / "build.json").read_text(encoding="utf-8"))
    m = {k.removeprefix("trudvsem_habr."): v for k, v in doc["metrics"].items()}
    assert m["records_labeled_by_both"]["value"] == 5
    assert (m["records_written"]["n"], m["records_written"]["denominator"]) == (2, 5)
    assert m["dropped_not_in_train_pool"]["n"] == 1
    assert m["dropped_input_changed"]["n"] == 1
    assert m["dropped_yandex_invalid"]["n"] == 1
    assert m["dropped_qwen_invalid"]["n"] == 0
    assert "dropped_excluded" not in m
    assert (m["agreed_skills"]["n"], m["agreed_skills"]["denominator"]) == (2, 2)
    assert doc["sources"]["trudvsem_habr"]["role"] == "yandex"
    assert doc["strata"] == {"trudvsem_habr": {"trudvsem|dev": 2}}
    assert doc["families_not_built"] == ["telegram"]


def test_dev_scores_a_composition_against_regex_with_rules_for_a_failed_request(data: Path, tmp_path: Path) -> None:
    gold_dir = data / "gold"
    inputs = [{"id": f"t:{i}", "source": "trudvsem", "input": text(i)} for i in range(3)]
    write_jsonl(gold_dir / "dev_inputs.jsonl", inputs)
    gold = [{"id": f"t:{i}", "source": "trudvsem", "fields": FIELDS} for i in range(3)]
    write_jsonl(gold_dir / "dev.jsonl", gold)
    qwen = [teacher_row(f"t:{i}", text(i), FIELDS) for i in range(3)]
    qwen[2] = teacher_row("t:2", text(2), None, error="HTTP 500")
    write_jsonl(data / "teacher" / "qwen" / "dev.jsonl", qwen)
    write_jsonl(data / "teacher" / "yandex" / "dev.jsonl", [teacher_row(f"t:{i}", text(i), FIELDS) for i in range(3)])
    regex = [{"id": f"t:{i}", "output": json.dumps(dump(extract(text(i))), ensure_ascii=False)} for i in range(3)]
    write_jsonl(data / "predictions" / "dev" / "regex.jsonl", regex)
    cfg = write_config(tmp_path / "s.toml", config_text('[compositions.q]\nrole = "qwen"\n', "q", '["q"]'))

    assert silver.main(["--config", str(cfg), "dev"]) == 0

    preds = [
        json.loads(x)
        for x in (data / "predictions" / "dev" / "silver_q.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [p["fallback"] for p in preds] == [[], [], ["role"]]
    assert json.loads(preds[0]["output"])["role"] == "backend"
    assert json.loads(preds[2]["output"])["role"] == extract(text(2)).role
    vs = json.loads((tmp_path / "results" / "extract" / "dev_silver_q_vs_regex.json").read_text(encoding="utf-8"))
    assert vs["compare"] == "regex"


def test_dev_refuses_a_teacher_answer_for_another_input(data: Path, tmp_path: Path) -> None:
    write_jsonl(data / "gold" / "dev_inputs.jsonl", [{"id": "t:0", "source": "trudvsem", "input": text(0)}])
    write_jsonl(data / "teacher" / "qwen" / "dev.jsonl", [teacher_row("t:0", "something else", FIELDS)])
    write_jsonl(data / "teacher" / "yandex" / "dev.jsonl", [teacher_row("t:0", text(0), FIELDS)])
    cfg = write_config(tmp_path / "s.toml", config_text('[compositions.q]\nrole = "qwen"\n', "q", '["q"]'))
    with pytest.raises(SystemExit, match="another input"):
        silver.main(["--config", str(cfg), "dev"])


def test_a_rules_only_family_is_a_sample_of_the_current_pool_without_teachers(data: Path, tmp_path: Path) -> None:
    corpus = data / "corpus"
    corpus.mkdir(parents=True)
    proxies = ["ml_ds_data", "dev", "dev", "dev", "qa", "dev"]
    records = [
        {"id": f"t:{i}", "source": "trudvsem", "role_proxy": proxy, "input": text(i)} for i, proxy in enumerate(proxies)
    ]
    pq.write_table(pa.Table.from_pylist(records), corpus / "corpus.parquet")
    split = [
        {"split": "train_pool", "member_ids": [f"t:{i}" for i in range(5)]},
        {"split": "dev", "member_ids": ["t:5"]},
    ]
    pq.write_table(pa.Table.from_pylist(split), corpus / "split.parquet")
    text_ = config_text("[compositions.r]\n", "r", '["r"]')
    cfg = write_config(
        tmp_path / "s.toml", text_ + '[rules_sample.trudvsem_habr]\nlimit = 3\nkeep_all = ["ml_ds_data"]\n'
    )

    assert silver.main(["--config", str(cfg), "build"]) == 0

    rows = [json.loads(x) for x in (data / "silver" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert "t:0" in {r["id"] for r in rows}
    assert "t:5" not in {r["id"] for r in rows}
    for r in rows:
        assert json.loads(r["output"]) == dump(extract(r["input"]))
    doc = json.loads((tmp_path / "results" / "silver" / "build.json").read_text(encoding="utf-8"))
    m = doc["metrics"]
    assert (m["trudvsem_habr.records_written"]["n"], m["trudvsem_habr.records_written"]["denominator"]) == (3, 5)
    assert m["trudvsem_habr.kept_all_ml_ds_data"]["n"] == 1
    assert doc["rules_sample"] == {"trudvsem_habr": {"limit": 3, "keep_all": ["ml_ds_data"]}}
    assert doc["teachers"] == {}


def test_a_chosen_rules_only_composition_needs_a_rules_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rules_sample"):
        load_config(write_config(tmp_path / "s.toml", config_text("[compositions.a]\n")))
