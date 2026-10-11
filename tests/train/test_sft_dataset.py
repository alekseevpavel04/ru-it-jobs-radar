import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar import config, results
from radar.train import dataset
from radar.train.dataset import (
    Built,
    DatasetConfig,
    Origin,
    answer_text,
    build,
    cap_input,
    in_val,
    load_config,
    origins,
    quantiles,
    report,
)

FIELDS = {
    "role": "data_engineer",
    "grade": None,
    "skills": ["Python", "Airflow"],
    "salary": {"from": 200000, "to": None, "currency": "RUB", "gross_net": None, "period": None},
    "work_format": "remote",
    "cities": ["Москва"],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}
OUTPUT = json.dumps(FIELDS, ensure_ascii=False)


class WordTokens:
    """One token per whitespace-separated word; the template adds 5 tokens around the turns."""

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    def lengths(self, system: str, user: str, answer: str) -> tuple[int, int]:
        return len(system.split()) + len(user.split()) + 5, len(answer.split()) + 1


def cfg(**changes: Any) -> DatasetConfig:
    base = {
        "tokenizer": "t",
        "prompt": Path("p.md"),
        "max_input_tokens": 50,
        "max_seq_len": 100,
        "val_share": 0.2,
        "seed": 7,
    }
    return DatasetConfig(**(base | changes))


def origin(rid: str, group: str | None = None) -> Origin:
    return Origin("trudvsem_habr", "habr", f"trudvsem_habr:{group or rid}", "dev")


# --- text ---------------------------------------------------------------------------------------


def test_cap_keeps_a_short_input_whole() -> None:
    text = "один два три"
    assert cap_input(text, WordTokens().spans(text), 3) == (text, False)


def test_cap_cuts_at_the_end_of_the_last_kept_token() -> None:
    text = "один  два три четыре"
    assert cap_input(text, WordTokens().spans(text), 2) == ("один  два", True)


def test_answer_is_compact_json_in_schema_key_order() -> None:
    shuffled = json.dumps(dict(reversed(list(FIELDS.items()))), ensure_ascii=False)
    answer = answer_text(shuffled)
    assert answer.startswith('{"role":"data_engineer","grade":null,"skills":["Python","Airflow"]')
    assert ", " not in answer and ": " not in answer


def test_a_label_outside_the_schema_is_refused() -> None:
    with pytest.raises(ValueError):
        answer_text(json.dumps(FIELDS | {"role": "astronaut"}))


# --- split and build ----------------------------------------------------------------------------


def test_val_membership_is_a_seeded_function_of_the_group() -> None:
    groups = [f"g{i}" for i in range(2000)]
    first = [in_val(g, 0.1, 7) for g in groups]
    assert first == [in_val(g, 0.1, 7) for g in groups]
    assert first != [in_val(g, 0.1, 8) for g in groups]
    assert 150 < sum(first) < 250


def test_build_keeps_groups_whole_and_counts_what_it_drops() -> None:
    rows = [{"id": f"r{i}", "input": "слово " * 10, "output": OUTPUT} for i in range(60)]
    rows.append({"id": "long", "input": "слово " * 200, "output": OUTPUT})
    rows.append({"id": "moved", "input": "слово", "output": OUTPUT})
    where = {r["id"]: origin(r["id"], group=f"g{i % 20}") for i, r in enumerate(rows) if r["id"] != "moved"}
    built = build(rows, where, cfg(), "система", WordTokens())

    assert built.outside == ["moved"]
    assert built.too_long == []  # cut to 50 input tokens, the sequence fits
    long = next(e for e in built.train + built.val if e["id"] == "long")
    assert long["truncated"] and long["input_tokens"] == 200
    assert len(long["messages"][1]["content"].split()) == 50
    assert {e["group"] for e in built.train}.isdisjoint({e["group"] for e in built.val})
    assert len(built.train) + len(built.val) == 61
    ex = built.train[0]
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["messages"][2]["content"] == answer_text(OUTPUT)
    assert ex["role"] == "data_engineer"


def test_an_example_over_max_seq_len_is_dropped_not_cut() -> None:
    rows = [{"id": "a", "input": "слово " * 40, "output": OUTPUT}]
    built = build(rows, {"a": origin("a")}, cfg(max_seq_len=30), "система", WordTokens())
    assert built.too_long == ["a"] and not built.train and not built.val


def test_report_publishes_denominators_and_refuses_shared_groups() -> None:
    def ex(i: str, group: str, truncated: bool = False) -> dict[str, Any]:
        return {
            "id": i,
            "family": "trudvsem_habr",
            "source": "habr",
            "group": group,
            "role_proxy": "dev",
            "role": "backend",
            "input_tokens": 10,
            "truncated": truncated,
            "prompt_tokens": 20,
            "completion_tokens": 5,
        }

    metrics, extra = report(Built([ex("a", "g1", True), ex("b", "g2")], [ex("c", "g3")], ["d"], ["e"]), 5)
    assert metrics["train_examples"] == {"value": 0.4, "n": 2, "denominator": 5}
    assert metrics["truncated_inputs"] == {"value": 0.3333, "n": 1, "denominator": 3}
    assert metrics["dropped_not_in_train_pool"]["n"] == 1
    assert metrics["train_sequence_tokens"]["value"] == 50
    assert extra["by_source"] == {"train": {"habr": 2}, "val": {"habr": 1}}
    assert extra["by_role_proxy"]["train"] == {"trudvsem_habr|dev": 2}
    with pytest.raises(AssertionError):
        report(Built([ex("a", "g1")], [ex("b", "g1")], [], []), 2)


def test_quantiles_are_nearest_rank() -> None:
    assert quantiles(list(range(1, 101))) == {"p50": 50, "p90": 90, "p99": 99, "max": 100}
    assert quantiles([7]) == {"p50": 7, "p90": 7, "p99": 7, "max": 7}


# --- origins from the corpora ---------------------------------------------------------------------


def write_corpora(root: Path) -> None:
    tg = root / "telegram"
    tg.mkdir(parents=True)
    split = {
        "split": ["train_pool", "dev", "train_pool"],
        "member_ids": [["habr:1", "trudvsem:old", "trudvsem:new"], ["habr:2"], ["habr:2b"]],
    }
    pq.write_table(pa.table(split), root / "split.parquet")
    members = {
        "id": ["habr:1", "trudvsem:old", "trudvsem:new", "habr:2", "habr:2b"],
        "source": ["habr", "trudvsem", "trudvsem", "habr", "habr"],
        "group_id": ["g1", "g1", "g1", "g2", "g3"],
    }
    pq.write_table(pa.table(members), root / "members.parquet")
    corpus = {"id": ["habr:1", "trudvsem:new", "habr:2"], "role_proxy": ["dev", "ml_ds_data", "dev"]}
    pq.write_table(pa.table(corpus), root / "corpus.parquet")
    tg_split = {"split": ["train_pool", "test"], "member_ids": [["telegram:a/1", "telegram:b/2"], ["telegram:c/3"]]}
    pq.write_table(pa.table(tg_split), tg / "split.parquet")
    records = {
        "id": ["telegram:a/1", "telegram:b/2", "telegram:c/3"],
        "group_id": ["t1", "t1", "t2"],
        "role_proxy": ["ml_ds_data", "ml_ds_data", "dev"],
    }
    pq.write_table(pa.table(records), tg / "records.parquet")


def test_origins_take_train_pool_members_only(tmp_path: Path) -> None:
    write_corpora(tmp_path)
    wanted = ["habr:1", "trudvsem:old", "habr:2", "telegram:a/1", "telegram:c/3"]
    got = origins(tmp_path, wanted)
    assert set(got) == {"habr:1", "trudvsem:old", "telegram:a/1"}
    assert got["habr:1"] == Origin("trudvsem_habr", "habr", "trudvsem_habr:g1", "dev")
    assert got["trudvsem:old"].role_proxy == "not_in_corpus"
    assert got["telegram:a/1"] == Origin("telegram", "telegram", "telegram:t1", "ml_ds_data")


# --- config and CLI -----------------------------------------------------------------------------


def test_project_config_loads_and_its_prompt_is_short() -> None:
    c = load_config()
    assert c.max_input_tokens < c.max_seq_len
    prompt = dataset.system_prompt(c.prompt)
    assert 0 < len(prompt) < 2000
    for value in ("ml_engineer", "non_it", "labor_contract", "hybrid", "KZT"):
        assert value in prompt


@pytest.mark.parametrize("line", ["val_share = 0.7", "max_input_tokens = 4096"])
def test_config_rejects_bad_values(tmp_path: Path, line: str) -> None:
    text = dataset.CONFIG.read_text(encoding="utf-8")
    key = line.split(" = ")[0]
    text = re.sub(rf"^{key} = .*$", line, text, flags=re.M)
    (tmp_path / "d.toml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(tmp_path / "d.toml")


def test_main_writes_examples_meta_and_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    write_corpora(data / "corpus")
    silver = data / "silver" / "train.jsonl"
    silver.parent.mkdir(parents=True)
    rows = [
        {"id": "habr:1", "input": "Должность: Data Engineer\n\nAirflow", "output": OUTPUT},
        {"id": "telegram:a/1", "input": "#вакансия Data Engineer, удалённо", "output": OUTPUT},
        {"id": "habr:2", "input": "dev record", "output": OUTPUT},
    ]
    silver.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setenv("RADAR_DATA_DIR", str(data))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(dataset, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(dataset, "HFTokens", lambda name: WordTokens())

    assert dataset.main([]) == 0
    train = dataset.read_jsonl(data / "sft" / "train.jsonl")
    val = dataset.read_jsonl(data / "sft" / "val.jsonl")
    assert sorted(e["id"] for e in train + val) == ["habr:1", "telegram:a/1"]
    meta = json.loads((data / "sft" / "meta.json").read_text(encoding="utf-8"))
    assert meta["system_prompt"] == dataset.system_prompt(dataset.load_config().prompt)
    assert meta["max_input_tokens"] == dataset.load_config().max_input_tokens
    doc = json.loads((tmp_path / "results" / "sft" / "dataset.json").read_text(encoding="utf-8"))
    assert doc["metrics"]["dropped_not_in_train_pool"] == {"value": 0.3333, "n": 1, "denominator": 3}
    assert doc["dropped_not_in_train_pool_ids"] == ["habr:2"]
    assert set(doc["tokens"]) == {"trudvsem_habr", "telegram"}


def test_main_refuses_a_silver_file_its_build_does_not_describe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    (data / "silver").mkdir(parents=True)
    (data / "silver" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "results" / "silver").mkdir(parents=True)
    (tmp_path / "results" / "silver" / "build.json").write_text('{"output_sha256": "other"}', encoding="utf-8")
    monkeypatch.setenv("RADAR_DATA_DIR", str(data))
    monkeypatch.setattr(dataset, "RESULTS", tmp_path / "results")
    with pytest.raises(SystemExit, match="silver build again"):
        dataset.main([])
