import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from radar.train import quantize


class Encoding(dict[str, Any]):
    @property
    def input_ids(self) -> list[int]:
        ids: list[int] = self["input_ids"]
        return ids


class CharTokenizer:
    """One token per character, the chat template as tags."""

    def __call__(self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False) -> Encoding:
        assert add_special_tokens is False
        return Encoding(input_ids=list(range(len(text))), offset_mapping=[(i, i + 1) for i in range(len(text))])

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def test_run_name_follows_the_run_directory() -> None:
    assert quantize.run_name(Path("models/sft/qwen3_1p7b_r16_clean/adapter")) == "qwen3_1p7b_r16_clean"
    assert quantize.run_name(Path("models/sft/qwen3_1p7b_r16/epoch-1")) == "qwen3_1p7b_r16_epoch-1"


def test_gguf_files_sit_under_the_model_name() -> None:
    path = quantize.gguf_path("m", "Q4_K_M")
    assert path.parent == quantize.GGUF / "m"
    assert path.name == "m-Q4_K_M.gguf"


def test_prompts_reuse_the_student_cap_and_template() -> None:
    meta = {"system_prompt": "S", "max_input_tokens": 4}
    rows = quantize.prompt_rows([{"id": "a", "input": "abcdef"}, {"id": "b", "input": "ab"}], CharTokenizer(), meta)
    assert rows[0] == {
        "id": "a",
        "system": "S",
        "user": "abcd",
        "prompt": "<system>S<user>abcd<assistant>",
        "prompt_tokens": len("<system>S<user>abcd<assistant>"),
        "truncated_input": True,
    }
    assert rows[1]["user"] == "ab"
    assert rows[1]["truncated_input"] is False


def test_manifest_entry_hashes_the_file(tmp_path: Path) -> None:
    f = tmp_path / "m-Q8_0.gguf"
    f.write_bytes(b"x" * 10)
    entry = quantize.manifest_entry(f, "Q8_0", None)
    assert entry["size_bytes"] == 10
    assert entry["type"] == "Q8_0"
    assert entry["sha256"] == hashlib.sha256(b"x" * 10).hexdigest()
    assert entry["from"] is None


def _result(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


def _jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(json.dumps({"id": i, "output": o}) + "\n" for i, o in rows), encoding="utf-8")


def _sidecar(system: str, workers: int = 4) -> dict[str, Any]:
    return {
        "system": system,
        "workers": workers,
        "records_per_s": 5.0,
        "latency_s": {"p50": 0.5, "p95": 0.9, "max": 1.0},
        "vram_peak_mib": 3000,
        "backend": "llama",
        "weights": {"file": f"models/gguf/m/{system}.gguf", "size_mib": 1.0},
        "decoding": {"json_schema_constrained": False},
        "server": {"total_slots": workers},
        "template_match": {"n": 3, "denominator": 3},
        "prompt_token_mismatch": 0,
    }


def test_summary_reads_scores_from_eval_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, data, merged, gguf = (tmp_path / d for d in ("results", "data", "merged", "gguf"))
    monkeypatch.setattr(quantize, "RESULTS", results)
    monkeypatch.setattr(quantize, "MERGED", merged)
    monkeypatch.setattr(quantize, "GGUF", gguf)
    monkeypatch.setenv("RADAR_DATA_DIR", str(data))
    f1 = {"value": 0.9, "ci_low": 0.88, "ci_high": 0.92}
    valid = {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0, "n": 3, "denominator": 3}
    for split in quantize.SPLITS:
        _result(results / "extract" / f"{split}_ref.json", {"macro_f1": f1, "valid_json": valid})
        _result(results / "extract" / f"{split}_q4.json", {"macro_f1": f1, "valid_json": valid})
        _result(
            results / "extract" / f"{split}_q4_vs_ref.json",
            {"macro_f1": {"value": -0.01, "ci_low": -0.02, "ci_high": 0.0, "a": 0.89, "b": 0.9}},
        )
        side = data / "predictions" / split / "q4.meta.json"
        side.parent.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps(_sidecar("q4")), encoding="utf-8")
        ref = {"records": 10, "wall_seconds": 4.0, "batch": 8, "backend": "unsloth", "adapter": "a"}
        (data / "predictions" / split / "ref.meta.json").write_text(json.dumps(ref), encoding="utf-8")
        _jsonl(data / "predictions" / split / "ref.jsonl", [("a", '{"x": 1}'), ("b", "{}")])
        _jsonl(data / "predictions" / split / "q4.jsonl", [("a", '{"x":1}'), ("b", '{"y": 2}')])
    (merged / "m").mkdir(parents=True)
    (merged / "m" / "model.safetensors").write_bytes(b"x" * 2**20)
    (gguf / "m").mkdir(parents=True)
    (gguf / "m" / "manifest.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    bench = tmp_path / "q4_w1.meta.json"
    bench.write_text(json.dumps(_sidecar("q4", workers=1)), encoding="utf-8")

    metrics, extra = quantize.summarize("m", "ref", ["q4"], [bench])

    assert metrics["dev/ref/macro_f1"] == f1
    assert metrics["tg_dev/q4/valid_json"]["n"] == 3
    assert metrics["dev/q4/macro_f1_minus_ref"] == {"value": -0.01, "ci_low": -0.02, "ci_high": 0.0}
    assert "dev/ref/macro_f1_minus_ref" not in metrics
    assert metrics["dev/q4/records_per_s"] == {"value": 5.0, "workers": 4}
    assert metrics["bench/q4/workers_1/latency_p95_s"] == {"value": 0.9}
    assert metrics["size_mib/merged_bf16"] == {"value": 1.0}
    assert metrics["size_mib/q4"] == {"value": 1.0}
    assert metrics["dev/q4/same_answer_as_ref"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert metrics["dev/ref/records_per_s"] == {"value": 2.5, "batch": 8, "backend": "unsloth"}
    assert extra["served"]["ref"]["adapter"] == "a"
    assert extra["served"]["q4"]["template_match"] == {"n": 3, "denominator": 3}


def test_quantize_needs_the_f16_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quantize, "GGUF", tmp_path)
    with pytest.raises(SystemExit, match="convert"):
        quantize.cmd_quantize("m", ["Q4_K_M"])


def test_awq_calibration_uses_training_rows_in_the_training_format() -> None:
    from radar.train.awq import calibration_texts

    class Tok(CharTokenizer):
        eos_token = "<eos>"

    rows = [{"messages": [{"content": "S"}, {"content": f"u{i}"}, {"content": f"a{i}"}]} for i in range(10)]
    texts = calibration_texts(rows, Tok(), 3, seed=1)
    assert len(texts) == 3
    assert texts == calibration_texts(rows, Tok(), 3, seed=1)
    i = texts[0].split("<user>u")[1][0]
    assert texts[0] == f"<system>S<user>u{i}<assistant>a{i}<eos>"
    assert len(calibration_texts(rows, Tok(), 50, seed=1)) == 10


def test_same_answers_never_match_broken_json() -> None:
    same = quantize.same_answers({"a": "{", "b": None}, {"a": "{", "b": None})
    assert same == {"value": 0.0, "n": 0, "denominator": 2}
