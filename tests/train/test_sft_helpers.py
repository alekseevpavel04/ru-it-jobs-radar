from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from radar.config import ROOT
from radar.train.config import load_run
from radar.train.predict import batches, dataset_meta
from radar.train.sft import check_dataset, forecast, has_checkpoint, probe_rows, rows_for_trainer

RUN = load_run(ROOT / "configs" / "sft" / "qwen3_1p7b.toml")


class ChatTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def example(i: int, tokens: int = 100, system: str = "S") -> dict[str, Any]:
    return {
        "id": f"r{i}",
        "prompt_tokens": tokens,
        "completion_tokens": 10,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": '{"role":null}'},
        ],
    }


def test_rows_end_the_answer_with_the_end_of_turn_token() -> None:
    rows = rows_for_trainer([example(1)], ChatTokenizer())
    assert rows == [{"prompt": "<system>S<user>u1<assistant>", "completion": '{"role":null}<|im_end|>'}]


def test_dataset_must_match_the_run() -> None:
    meta = {"max_seq_len": RUN.train.max_seq_len, "system_prompt": "S"}
    check_dataset([example(1), example(2)], meta, RUN)
    with pytest.raises(SystemExit, match="max_seq_len"):
        check_dataset([example(1)], meta | {"max_seq_len": 1024}, RUN)
    with pytest.raises(SystemExit, match="system prompt"):
        check_dataset([example(1), example(2, system="other")], meta, RUN)


def test_probe_takes_the_longest_examples_first() -> None:
    examples = [example(i, tokens=100 + i) for i in range(20)]
    rows = probe_rows(examples, 10, batch=2)
    assert len(rows) == 10
    assert [r["id"] for r in rows[:4]] == ["r19", "r18", "r17", "r16"]
    assert len({r["id"] for r in rows}) == 10


def test_forecast_skips_the_first_step() -> None:
    watch = SimpleNamespace(seconds=[30.0, 2.0, 2.0], tokens=[1000, 1000, 1000])
    out = forecast(watch, train_tokens=3_600_000, epochs=2)
    assert out["probe_tokens_per_s"] == 500.0
    assert out["forecast_epoch_h"] == 2.0
    assert out["forecast_run_h"] == 4.0
    assert forecast(SimpleNamespace(seconds=[1.0], tokens=[0]), 10, 1) == {}


def test_checkpoints_are_found(tmp_path: Path) -> None:
    assert not has_checkpoint(tmp_path)
    (tmp_path / "checkpoint-100").mkdir()
    assert has_checkpoint(tmp_path)


def test_batches_cover_every_index_once() -> None:
    assert batches([4, 2, 0, 1, 3], 2) == [[4, 2], [0, 1], [3]]


def test_a_base_model_reads_the_dataset_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path))
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "meta.json").write_text('{"max_input_tokens": 7}', encoding="utf-8")
    assert dataset_meta(None) == {"max_input_tokens": 7}
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "dataset_meta.json").write_text('{"max_input_tokens": 9}', encoding="utf-8")
    assert dataset_meta(adapter) == {"max_input_tokens": 9}
