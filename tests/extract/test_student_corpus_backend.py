import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from radar.extract import corpus_run, student_server
from radar.extract.schema import json_schema, try_parse

ANSWER = (
    '{"role":"ml_engineer","grade":null,"skills":["Python"],"salary":null,"work_format":"remote",'
    '"cities":[],"experience_years":null,"employment_type":null,"lang":"ru"}'
)
META = {"system_prompt": "S", "max_input_tokens": 4, "prompt_sha256": "abc123"}


class Encoding(dict[str, Any]):
    @property
    def input_ids(self) -> list[int]:
        ids: list[int] = self["input_ids"]
        return ids


class CharTokenizer:
    """One token per character, the chat template as tags."""

    def __call__(self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False) -> Encoding:
        return Encoding(input_ids=list(range(len(text))), offset_mapping=[(i, i + 1) for i in range(len(text))])

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


def vllm_server(
    seen: list[dict[str, Any]], root: str = "/mnt/d/p/models/awq/m", extra_tokens: int = 0, status: int = 200
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m-awq", "root": root, "max_model_len": 4096}]})
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        seen.append(body)
        if status != 200:
            return httpx.Response(status, text="engine error")
        return httpx.Response(
            200,
            json={
                "choices": [{"text": ANSWER, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(body["prompt"]) + extra_tokens, "completion_tokens": 40},
            },
        )

    return httpx.MockTransport(handle)


@pytest.fixture
def merged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "dataset_meta.json").write_text(json.dumps(META), encoding="utf-8")
    monkeypatch.setattr(student_server, "MERGED", tmp_path)
    return tmp_path


def connect(transport: httpx.MockTransport, schema: bool = True) -> student_server.CorpusBackend:
    client = httpx.Client(base_url="http://test", transport=transport)
    return student_server.connect("m", CharTokenizer(), client, "vllm", "awq", schema, 512)


def test_inputs_are_rendered_like_the_scored_prompts_and_sent_with_the_schema(merged: Path) -> None:
    seen: list[dict[str, Any]] = []
    backend = connect(vllm_server(seen))
    assert backend.model_id == "m-awq"
    assert backend.prompt_sha256 == "abc123"
    answers = backend.extract(["abcdef", "ab"])
    assert [b["prompt"] for b in seen] == ["<system>S<user>abcd<assistant>", "<system>S<user>ab<assistant>"]
    assert seen[0]["structured_outputs"] == {"json": json_schema()}
    assert seen[0]["temperature"] == 0 and seen[0]["max_tokens"] == 512
    assert [a.input_truncated for a in answers] == [True, False]
    assert answers[0].finish_reason == "stop" and answers[0].error is None
    assert corpus_run.checked_output(answers[0]) is not None
    assert try_parse(answers[0].raw or "") is not None


def test_free_decoding_is_another_model_id(merged: Path) -> None:
    seen: list[dict[str, Any]] = []
    backend = connect(vllm_server(seen), schema=False)
    backend.extract(["ab"])
    assert backend.model_id == "m-awq-noschema"
    assert "structured_outputs" not in seen[0]


def test_other_served_weights_are_refused(merged: Path) -> None:
    with pytest.raises(SystemExit, match="serves"):
        connect(vllm_server([], root="/mnt/d/p/models/merged/m"))


def test_a_prompt_the_server_counts_differently_stops_the_run(merged: Path) -> None:
    backend = connect(vllm_server([], extra_tokens=1))
    with pytest.raises(RuntimeError, match="prompt tokens"):
        backend.extract(["ab"])


def test_a_failed_request_is_an_error_answer(merged: Path) -> None:
    answers = connect(vllm_server([], status=500)).extract(["abcdef"])
    assert answers[0].raw is None
    assert answers[0].error == "HTTP 500: engine error"
    assert answers[0].input_truncated is True


def test_served_weights_of_a_directory_and_a_gguf_file() -> None:
    assert student_server.served_weights("models/awq/m") == ("awq", "m")
    assert student_server.served_weights("D:\\p\\models\\gguf\\m\\m-Q4_K_M.gguf") == ("m", "m-Q4_K_M")
