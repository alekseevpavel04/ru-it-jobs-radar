from pathlib import Path
from typing import Any

import pytest

from radar.config import ROOT
from radar.train.config import load_dpo
from radar.train.dpo import (
    check_pairs,
    forecast_from_probe,
    last_logged,
    loss_settings,
    probe_rows,
    rows_for_trainer,
)

CONFIG = ROOT / "configs" / "dpo" / "qwen3_1p7b_clean.toml"


class ChatTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def pair(i: int, user_len: int = 10) -> dict[str, Any]:
    return {"id": f"p{i}", "user": "u" * user_len, "chosen": '{"role":"qa"}', "rejected": '{"role":"backend"}'}


def test_config_loads_and_points_at_the_final_student() -> None:
    cfg = load_dpo(CONFIG)
    assert cfg.sft_adapter == ROOT / "models" / "sft" / "qwen3_1p7b_r16_clean" / "adapter"
    assert cfg.output_dir == ROOT / "models" / "dpo" / "qwen3_1p7b_r16_clean_dpo"
    assert cfg.pairs.corrupted() == {"work_format": 200, "role": 200, "drop_field": 50, "broken_json": 50}
    assert cfg.params()["train.beta"] == 0.1


def test_config_rejects_unknown_keys_and_bad_values(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("beta = 0.1", "beta = 0.1\nbetta = 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_dpo(bad)
    bad.write_text(text.replace("holdout_share = 0.1", "holdout_share = 0.7"), encoding="utf-8")
    with pytest.raises(ValueError, match="holdout_share"):
        load_dpo(bad)
    bad.write_text(text.replace("beta = 0.1", "beta = 0"), encoding="utf-8")
    with pytest.raises(ValueError, match="beta"):
        load_dpo(bad)


def test_rows_render_the_student_prompt_and_close_both_answers() -> None:
    rows = rows_for_trainer([pair(1, 3)], ChatTokenizer(), "S")
    assert rows == [
        {
            "prompt": "<system>S<user>uuu<assistant>",
            "chosen": '{"role":"qa"}<|im_end|>',
            "rejected": '{"role":"backend"}<|im_end|>',
        }
    ]


def test_loss_adds_the_nll_term_only_with_a_weight() -> None:
    assert loss_settings(1.0) == (["sigmoid", "sft"], [1.0, 1.0])
    assert loss_settings(0.0) == (["sigmoid"], None)


def test_probe_takes_the_longest_rows_first() -> None:
    rows = [{"prompt": "x" * (10 + i), "chosen": "a", "rejected": "b"} for i in range(20)]
    out = probe_rows(rows, 6, batch=2)
    assert [len(r["prompt"]) for r in out[:4]] == [29, 28, 27, 26]
    assert len(out) == 6


def test_forecast_from_one_pair_steps() -> None:
    out = forecast_from_probe([40.0, 2.0, 4.0, 3.0], batch=1, train_pairs=1200, epochs=2)
    assert out == {"probe_seconds_per_pair": 3.0, "forecast_run_h": 2.0}
    assert forecast_from_probe([40.0], batch=1, train_pairs=10, epochs=1) == {}


def test_last_logged_takes_the_last_value_of_each_key() -> None:
    history = [{"loss": 0.7}, {"eval_loss": 0.6, "eval_rewards/accuracies": 0.5}, {"loss": 0.4}]
    out = last_logged(history)
    assert out["train_loss_last"] == 0.4
    assert out["eval_reward_accuracy_last"] == 0.5
    assert out["eval_reward_margin_last"] is None


def test_pairs_must_match_the_adapter_prompt() -> None:
    check_pairs({"prompt_sha256": "a"}, {"prompt_sha256": "a"})
    with pytest.raises(SystemExit, match="prompt"):
        check_pairs({"prompt_sha256": "a"}, {"prompt_sha256": "b"})
