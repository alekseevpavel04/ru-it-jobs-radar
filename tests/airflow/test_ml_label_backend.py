"""The CPU student's corpus backend (airflow/include/student_cpu.py) against a fake llama-server."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from radar.extract import student_server
from radar.extract.schema import json_schema

INCLUDE = Path(__file__).resolve().parents[2] / "airflow" / "include"
spec = importlib.util.spec_from_file_location("airflow_include_student_cpu", INCLUDE / "student_cpu.py")
assert spec is not None and spec.loader is not None
student_cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_cpu)

ANSWER = (
    '{"role":"ml_engineer","grade":null,"skills":["Python"],"salary":null,"work_format":"remote",'
    '"cities":[],"experience_years":null,"employment_type":null,"lang":"ru"}'
)
META = {"system_prompt": "S", "max_input_tokens": 3, "prompt_sha256": "abc123"}
WORDS = ["<|im_start|>", "<|im_end|>", "system", "user", "assistant", "S", "ищем", "ml", "инженера", "в", "команду"]


@pytest.fixture
def merged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(["[UNK]", *WORDS])}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    (tmp_path / "m").mkdir()
    tok.save(str(tmp_path / "m" / "tokenizer.json"))
    (tmp_path / "m" / "dataset_meta.json").write_text(json.dumps(META), encoding="utf-8")
    monkeypatch.setattr(student_cpu, "MERGED", tmp_path)
    return tmp_path


def template(messages: list[dict[str, str]]) -> str:
    turns = "".join(f"<|im_start|> {m['role']} {m['content']} <|im_end|> " for m in messages)
    return turns + "<|im_start|> assistant"


def server(seen: list[dict[str, Any]], model: str = "/weights/m-Q4_K_M.gguf", extra: int = 0) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"model_path": model, "total_slots": 2, "default_generation_settings": {"n_ctx": 4096}}
            )
        body = json.loads(request.content)
        if request.url.path == "/apply-template":
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"prompt": template(body["messages"])})
        assert request.url.path == "/completion"
        seen.append(body)
        prompt = body["prompt"]
        n = (len(prompt) if isinstance(prompt, list) else len(prompt.split())) + extra
        return httpx.Response(
            200, json={"content": ANSWER, "stop_type": "eos", "tokens_evaluated": n, "tokens_predicted": 40}
        )

    return httpx.MockTransport(handle)


def test_input_is_cut_rendered_by_the_server_and_sent_with_the_schema(merged: Path) -> None:
    seen: list[dict[str, Any]] = []
    backend = student_cpu.backend("http://student:8080", name="m", transport=server(seen))
    assert backend.model_id == "m-q4_k_m" and backend.prompt_sha256 == "abc123"
    long, short = backend.extract(["ищем ml инженера в команду", "ищем ml"])
    # the server gets the token ids of the rendered prompt, not its text to tokenize again
    rendered = template([{"role": "system", "content": "S"}, {"role": "user", "content": "ищем ml инженера"}])
    assert seen[0]["prompt"] == Tokenizer.from_file(str(merged / "m" / "tokenizer.json")).encode(rendered).ids
    assert seen[0]["json_schema"] == json_schema() and seen[0]["temperature"] == 0
    assert seen[0]["n_predict"] == student_server.MAX_NEW_TOKENS
    assert (long.input_truncated, short.input_truncated) == (True, False)
    assert long.raw == ANSWER and long.finish_reason == "stop"


def test_other_weights_are_refused(merged: Path) -> None:
    with pytest.raises(SystemExit, match="serves m-Q8_0"):
        student_cpu.backend("http://student:8080", name="m", transport=server([], model="/weights/m-Q8_0.gguf"))


def test_a_prompt_the_server_counts_differently_stops_the_run(merged: Path) -> None:
    backend = student_cpu.backend("http://student:8080", name="m", transport=server([], extra=1))
    with pytest.raises(RuntimeError, match="prompt tokens"):
        backend.extract(["ищем ml"])
