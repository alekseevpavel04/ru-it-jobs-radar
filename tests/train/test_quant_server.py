import json
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest

from radar.extract import student_server

ANSWER = (
    '{"role":"ml_engineer","grade":null,"skills":["Python"],"salary":null,"work_format":"remote",'
    '"cities":[],"experience_years":null,"employment_type":null,"lang":"ru"}'
)


def prompt_row(i: int) -> dict[str, Any]:
    prompt = f"<system>S<user>u{i}<assistant>"
    return {
        "id": f"r{i}",
        "system": "S",
        "user": f"u{i}",
        "prompt": prompt,
        "prompt_tokens": 7,
        "truncated_input": i == 0,
    }


def fake_server(seen: list[dict[str, Any]], template_ok: bool = True, stop_type: str = "eos") -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "model_path": "D:/nowhere/m-Q4_K_M.gguf",
                    "model_alias": "m-Q4_K_M",
                    "total_slots": 2,
                    "default_generation_settings": {"n_ctx": 4096},
                },
            )
        body = json.loads(request.content)
        seen.append({"path": request.url.path, **body})
        if request.url.path == "/apply-template":
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            system, user = (m["content"] for m in body["messages"])
            prompt = f"<system>{system}<user>{user}<assistant>"
            return httpx.Response(200, json={"prompt": prompt if template_ok else prompt + " "})
        assert request.url.path == "/completion"
        return httpx.Response(
            200,
            json={
                "content": " " + ANSWER + "\n",
                "stop_type": stop_type,
                "tokens_evaluated": 7,
                "tokens_predicted": 40,
            },
        )

    return httpx.MockTransport(handle)


def llama(seen: list[dict[str, Any]], schema: dict[str, Any] | None = None, **kw: Any) -> student_server.LlamaServer:
    client = httpx.Client(base_url="http://test", transport=fake_server(seen, **kw))
    return student_server.LlamaServer(client, 512, schema)


def test_completion_is_greedy_and_schema_is_optional() -> None:
    assert llama([]).body("p") == {"prompt": "p", "n_predict": 512, "temperature": 0, "cache_prompt": True}
    assert llama([], {"type": "object"}).body("p")["json_schema"] == {"type": "object"}


def test_predict_sends_the_rendered_prompt_and_checks_the_template() -> None:
    seen: list[dict[str, Any]] = []
    row = llama(seen).predict(prompt_row(0))
    assert seen[1]["prompt"] == "<system>S<user>u0<assistant>"
    assert "json_schema" not in seen[1]
    assert row["output"] == ANSWER
    assert row["valid"] is True
    assert row["finish"] == "stop"
    assert row["template_match"] is True
    assert row["truncated_input"] is True
    assert (row["prompt_tokens"], row["completion_tokens"]) == (7, 40)


def test_a_cut_answer_and_a_template_mismatch_are_recorded() -> None:
    row = llama([], template_ok=False, stop_type="limit").predict(prompt_row(1))
    assert row["finish"] == "length"
    assert row["template_match"] is False


def test_vllm_completion_with_a_text_prompt() -> None:
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": "m-awq", "root": "/mnt/d/x/m-awq", "max_model_len": 4096}]}
            )
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"text": ANSWER, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 512},
            },
        )

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
    backend = student_server.Vllm(client, 512, {"type": "object"})
    assert backend.info()["model_path"] == "/mnt/d/x/m-awq"
    row = backend.predict(prompt_row(2))
    assert seen[0] == {
        "model": "m-awq",
        "prompt": "<system>S<user>u2<assistant>",
        "max_tokens": 512,
        "temperature": 0,
        "structured_outputs": {"json": {"type": "object"}},
    }
    assert row["finish"] == "length"
    assert row["template_match"] is None


def test_wsl_paths_map_to_windows_drives() -> None:
    assert student_server.local_path("/mnt/d/VScode_projects/x") == Path("D:/VScode_projects/x")
    assert student_server.local_path("D:/x/m.gguf") == Path("D:/x/m.gguf")


def test_template_share_skips_unknown() -> None:
    assert student_server.share([True, False, None]) == {"n": 1, "denominator": 2}
    assert student_server.share([None]) is None


def test_quantile_is_nearest_rank() -> None:
    xs = [float(i) for i in range(1, 101)]
    assert student_server.quantile(xs, 0.5) == 50.0
    assert student_server.quantile(xs, 0.95) == 95.0
    assert student_server.quantile([3.0], 0.95) == 3.0


def test_peak_sampler_keeps_the_largest_value() -> None:
    values = iter([100, 300, 200] + [150] * 100)
    with student_server.PeakSampler(lambda: next(values), every=0.001) as s:
        pass
    assert s.peak == 300


def test_peak_sampler_without_a_card() -> None:
    with student_server.PeakSampler(lambda: None, every=0.001) as s:
        pass
    assert s.peak is None


def test_main_writes_predictions_and_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    merged = tmp_path / "merged"
    (merged / "m" / "prompts").mkdir(parents=True)
    student_server.write_jsonl(merged / "m" / "prompts" / "dev.jsonl", [prompt_row(i) for i in range(3)])
    (merged / "m" / "dataset_meta.json").write_text(
        json.dumps({"prompt_sha256": "abc", "max_input_tokens": 1536}), encoding="utf-8"
    )
    monkeypatch.setattr(student_server, "MERGED", merged)
    monkeypatch.setattr(student_server, "nvidia_smi_used", lambda: 1234)
    monkeypatch.setattr(student_server, "weights_info", lambda path: {"file": path})
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(httpx, "Client", partial(httpx.Client, transport=fake_server(seen)))

    assert student_server.main(["--name", "m", "--split", "dev", "--system", "q4", "--workers", "2", "--schema"]) == 0

    out = tmp_path / "data" / "predictions" / "dev" / "q4.jsonl"
    rows = student_server.read_jsonl(out)
    assert [r["id"] for r in rows] == ["r0", "r1", "r2"]
    side = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert side["valid_json"] == {"n": 3, "denominator": 3}
    assert side["template_match"] == {"n": 3, "denominator": 3}
    assert side["prompt_token_mismatch"] == 0
    assert side["truncated_inputs"] == 1
    assert side["decoding"]["json_schema_constrained"] is True
    assert side["vram_peak_mib"] == 1234
    assert all("json_schema" in s for s in seen if s["path"] == "/completion")


def test_main_refuses_more_workers_than_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    merged = tmp_path / "merged"
    (merged / "m" / "prompts").mkdir(parents=True)
    student_server.write_jsonl(merged / "m" / "prompts" / "dev.jsonl", [prompt_row(0)])
    (merged / "m" / "dataset_meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(student_server, "MERGED", merged)
    monkeypatch.setattr(httpx, "Client", partial(httpx.Client, transport=fake_server([])))
    with pytest.raises(SystemExit, match="slots"):
        student_server.main(["--name", "m", "--split", "dev", "--system", "q4", "--workers", "4"])
