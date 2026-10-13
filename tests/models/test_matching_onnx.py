"""The ONNX query encoder of /match without the exported model: a word-level tokenizer and a scripted
session stand in for e5's, so pooling, masking, truncation and the refusals are checked offline."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from radar.models import matching_onnx
from radar.models.matching_onnx import OnnxEncoder, load_encoder, pool

VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "query:": 4, "python": 5, "sql": 6, "docker": 7}


def word_tokenizer(max_length: int) -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    tokenizer.enable_truncation(max_length)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    return tokenizer


class Session:
    """The hidden state of a token is its id in every dimension, so a mean over tokens is visible."""

    def __init__(self) -> None:
        self.feeds: list[dict[str, Any]] = []

    def run(self, outputs: list[str], feeds: dict[str, Any]) -> list[Any]:
        assert outputs == ["last_hidden_state"]
        self.feeds.append(feeds)
        ids = feeds["input_ids"].astype(np.float32)
        return [np.repeat(ids[..., None], 3, axis=2)]


def test_mean_pooling_skips_padding_and_normalizes() -> None:
    hidden = np.array([[[3.0, 4.0], [100.0, 100.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 1]], dtype=np.int64)
    out = pool(hidden, mask, "mean")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.6, 0.8], [2**-0.5, 2**-0.5]], rtol=1e-6)
    np.testing.assert_allclose(pool(hidden, mask, "cls"), [[0.6, 0.8], [1.0, 0.0]], rtol=1e-6)


def test_a_zero_vector_stays_finite() -> None:
    out = pool(np.zeros((1, 2, 3), dtype=np.float32), np.ones((1, 2), dtype=np.int64), "mean")
    assert np.isfinite(out).all()


def test_encoder_pads_a_batch_and_counts_truncated_texts() -> None:
    session = Session()
    encode = OnnxEncoder(word_tokenizer(max_length=4), session, "mean", max_length=4)
    vectors, truncated = encode(["query: python", "query: python sql docker"])
    feeds = session.feeds[0]
    assert feeds["input_ids"].tolist() == [[1, 4, 5, 2], [1, 4, 5, 2]]  # the second cut to 4 tokens
    assert feeds["attention_mask"].dtype == np.int64
    assert truncated == 2  # both reach max_length, as the torch encoder counts them
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)

    vectors, truncated = OnnxEncoder(word_tokenizer(8), Session(), "mean", 8)(["query: python", "query: sql"])
    assert truncated == 0
    np.testing.assert_allclose(vectors[0], np.full(3, 3**-0.5), rtol=1e-6)


def export_dir(tmp_path: Path, model: str, with_model: bool = True) -> Path:
    meta = {"model": model, "pad_id": 0, "pad_token": "<pad>"}
    (tmp_path / matching_onnx.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    word_tokenizer(16).save(str(tmp_path / matching_onnx.TOKENIZER_FILE))
    (tmp_path / matching_onnx.DATA_FILE).write_bytes(b"")
    if with_model:
        (tmp_path / matching_onnx.MODEL_FILE).write_bytes(b"")
    return tmp_path


def test_an_export_of_another_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BAAI/bge-m3"):
        load_encoder(export_dir(tmp_path, "BAAI/bge-m3"), "intfloat/multilingual-e5-small", 512)


def test_an_export_without_its_graph_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_encoder(
            export_dir(tmp_path, "intfloat/multilingual-e5-small", with_model=False),
            "intfloat/multilingual-e5-small",
            512,
        )


def test_the_loader_sets_max_length_and_padding_from_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnxruntime as ort  # type: ignore[import-untyped]

    made: list[str] = []

    def session(path: str, options: Any, providers: list[str]) -> Session:
        made.append(path)
        assert providers == ["CPUExecutionProvider"]
        return Session()

    monkeypatch.setattr(ort, "InferenceSession", session)
    encode = load_encoder(export_dir(tmp_path, "intfloat/multilingual-e5-small"), "intfloat/multilingual-e5-small", 3)
    assert made == [str(tmp_path / matching_onnx.MODEL_FILE)] and encode.pooling == "mean"
    _, truncated = encode(["query: python sql docker"])
    assert truncated == 1
