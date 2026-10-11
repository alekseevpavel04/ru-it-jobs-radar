import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar import config, results
from radar.extract import teacher
from radar.extract.teacher import (
    TEACHERS,
    Item,
    ServerError,
    label_one,
    request_body,
    request_schema,
    reusable,
    sha256_text,
    stratified,
    validate,
)

ANSWER = {
    "role": "backend",
    "grade": None,
    "skills": ["Python", "PostgreSQL"],
    "salary": {"from": 100000, "to": None, "currency": "RUB", "gross_net": None, "period": None},
    "work_format": None,
    "cities": ["Москва"],
    "experience_years": None,
    "employment_type": None,
    "lang": "ru",
}
BODY = request_body(TEACHERS["qwen"], "rules", "Должность: Программист", request_schema("train_pool"))


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 4500, "completion_tokens": 120},
    }


def client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))


def item(rid: str = "trudvsem:1", stratum: str = "trudvsem|dev", text: str = "Должность: Программист") -> Item:
    return Item(rid, stratum, text)


# --- request and answer ------------------------------------------------------------------------


def test_object_without_bounds_becomes_null_and_is_marked() -> None:
    raw = json.dumps(
        {
            **ANSWER,
            "salary": {"from": None, "to": None, "currency": "RUB", "gross_net": None, "period": None},
            "experience_years": {"min": None, "max": None},
        }
    )
    output, repaired = validate(raw)
    assert repaired is True
    assert output is not None
    fields = json.loads(output)
    assert fields["salary"] is None
    assert fields["experience_years"] is None
    assert validate(json.dumps(ANSWER)) == (json.dumps(ANSWER, ensure_ascii=False), False)


def test_broken_json_is_invalid() -> None:
    assert validate('{"role": "backend"') == (None, False)
    assert validate(None) == (None, False)


def test_request_body_switches_qwen_thinking_off_only() -> None:
    qwen = request_body(TEACHERS["qwen"], "rules", "text", {"type": "object"})
    yandex = request_body(TEACHERS["yandex"], "rules", "text", {"type": "object"})
    assert qwen["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in yandex
    assert qwen["temperature"] == 0
    assert qwen["response_format"]["type"] == "json_schema"
    assert [m["role"] for m in qwen["messages"]] == ["system", "user"]


# --- one request --------------------------------------------------------------------------------


def test_valid_answer_is_normalized() -> None:
    raw = json.dumps(ANSWER, ensure_ascii=False, indent=2)
    row = label_one(client(lambda r: httpx.Response(200, json=completion(raw))), BODY, item())
    assert row["valid"] is True
    assert row["raw"] == raw
    assert json.loads(row["output"]) == ANSWER
    assert "\n" not in row["output"]
    assert (row["prompt_tokens"], row["completion_tokens"], row["error"]) == (4500, 120, None)
    assert row["input_sha256"] == sha256_text("Должность: Программист")


def test_schema_violation_is_kept_raw_and_invalid() -> None:
    bad = json.dumps({**ANSWER, "experience_years": {"min": 5, "max": 1}})
    row = label_one(client(lambda r: httpx.Response(200, json=completion(bad))), BODY, item())
    assert row["valid"] is False
    assert row["output"] is None
    assert row["raw"] == bad
    assert row["error"] is None


def test_cut_answer_is_invalid_even_if_it_parses() -> None:
    row = label_one(client(lambda r: httpx.Response(200, json=completion(json.dumps(ANSWER), "length"))), BODY, item())
    assert row["valid"] is False
    assert row["finish_reason"] == "length"


def test_server_error_is_retried() -> None:
    statuses = iter([500, 200])

    def handle(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, json=completion(json.dumps(ANSWER)) if status == 200 else {"error": "busy"})

    assert label_one(client(handle), BODY, item(), pause_s=0)["valid"] is True


def test_persistent_http_error_becomes_a_failed_row() -> None:
    row = label_one(
        client(lambda r: httpx.Response(500, json={"error": {"message": "Context size has been exceeded."}})),
        BODY,
        item(),
        pause_s=0,
    )
    assert row["valid"] is False
    assert "Context size" in row["error"]


def test_unreachable_server_stops_the_run() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ServerError, match="ConnectError"):
        label_one(client(handle), BODY, item(), pause_s=0)


# --- selection and resume -----------------------------------------------------------------------


def test_stratified_is_proportional_and_keeps_listed_proxies() -> None:
    items = [item(f"trudvsem:a{i}", "trudvsem|dev") for i in range(60)]
    items += [item(f"trudvsem:b{i}", "trudvsem|non_it_noise") for i in range(30)]
    items += [item(f"habr:c{i}", "habr|ml_ds_data") for i in range(5)]
    sample = stratified(items, 30)
    strata = [i.stratum for i in sample]
    assert len(sample) == 30
    assert strata.count("trudvsem|dev") == 19
    assert strata.count("trudvsem|non_it_noise") == 9
    assert sample[0].id == "trudvsem:a0"

    kept = stratified(items, 25, keep_all=["ml_ds_data"])
    assert [i.id for i in kept[:5]] == [f"habr:c{i}" for i in range(5)]
    assert len(kept) == 25


def test_resume_keeps_good_rows_and_relabels_errors_and_changed_inputs() -> None:
    a, b, c = item("x:a", text="one"), item("x:b", text="two"), item("x:c", text="three")
    rows = [
        {"id": "x:a", "input_sha256": sha256_text("one"), "error": None},
        {"id": "x:b", "input_sha256": sha256_text("old two"), "error": None},
        {"id": "x:c", "input_sha256": sha256_text("three"), "error": "HTTP 500"},
        {"id": "x:z", "input_sha256": "elsewhere", "error": None},
    ]
    kept, done = reusable(rows, [a, b, c])
    assert done == {"x:a"}
    assert [r["id"] for r in kept] == ["x:a", "x:z"]


# --- end to end ---------------------------------------------------------------------------------


def write_corpus(root: Path) -> None:
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    rows = [
        {"id": f"trudvsem:{i}", "source": "trudvsem", "role_proxy": "dev", "input": f"Должность: Программист {i}"}
        for i in range(6)
    ]
    pq.write_table(pa.Table.from_pylist(rows), corpus / "corpus.parquet")
    split = [
        {"split": "train_pool", "member_ids": ["trudvsem:0", "trudvsem:1", "trudvsem:2", "trudvsem:3"]},
        {"split": "train_pool", "member_ids": ["trudvsem:4"]},
        {"split": "dev", "member_ids": ["trudvsem:5"]},
        {"split": "withheld", "member_ids": ["trudvsem:4"]},
    ]
    pq.write_table(pa.Table.from_pylist(split), corpus / "split.parquet")


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"alias": "Qwen3-8B-Q4_K_M", "requests": 0, "fail": set()}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": state["alias"]}]})
        state["requests"] += 1
        text = json.loads(request.content)["messages"][1]["content"]
        if text in state["fail"]:
            return httpx.Response(400, json={"error": "too long"})
        return httpx.Response(200, json=completion(json.dumps(ANSWER, ensure_ascii=False)))

    real_client = httpx.Client

    def fake_client(**kwargs: Any) -> httpx.Client:
        return real_client(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(teacher.httpx, "Client", fake_client)
    monkeypatch.setattr(teacher, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    write_corpus(tmp_path / "data")
    return state


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_train_pool_run_skips_held_out_records_and_resumes(server: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "data" / "teacher" / "qwen" / "train_pool.jsonl"
    server["fail"] = {"Должность: Программист 1"}
    assert teacher.main(["--teacher", "qwen", "--split", "train_pool", "--workers", "2"]) == 0
    rows = lines(out)
    assert sorted(r["id"] for r in rows) == ["trudvsem:0", "trudvsem:1", "trudvsem:2", "trudvsem:3"]
    assert sum(r["valid"] for r in rows) == 3

    with pytest.raises(SystemExit, match="exists"):
        teacher.main(["--teacher", "qwen", "--split", "train_pool"])

    server["fail"] = set()
    server["requests"] = 0
    assert teacher.main(["--teacher", "qwen", "--split", "train_pool", "--resume"]) == 0
    assert server["requests"] == 1
    rows = lines(out)
    assert len(rows) == 4 and all(r["valid"] for r in rows)

    meta = json.loads((out.parent / "train_pool.meta.json").read_text(encoding="utf-8"))
    assert meta["prompt_sha256"] == sha256_text(teacher.PROMPT.read_text(encoding="utf-8"))
    assert [run["n_labeled"] for run in meta["runs"]] == [4, 1]


def test_resume_refuses_another_prompt(server: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert teacher.main(["--teacher", "qwen", "--split", "train_pool", "--limit", "2"]) == 0
    prompt = tmp_path / "prompt.md"
    prompt.write_text("другие правила", encoding="utf-8")
    monkeypatch.setattr(teacher, "PROMPT", prompt)
    with pytest.raises(SystemExit, match="prompt_sha256"):
        teacher.main(["--teacher", "qwen", "--split", "train_pool", "--resume"])


def test_wrong_model_on_the_server_is_refused(server: dict[str, Any]) -> None:
    server["alias"] = "something-else"
    with pytest.raises(ServerError, match="Qwen3-8B"):
        teacher.main(["--teacher", "qwen", "--split", "train_pool"])


def test_probe_report(server: dict[str, Any], tmp_path: Path) -> None:
    for name, alias in (("qwen", "Qwen3-8B-Q4_K_M"), ("yandex", "YandexGPT-5-Lite-8B-instruct-Q4_K_M")):
        server["alias"] = alias
        assert teacher.main(["--teacher", name, "--split", "train_pool", "--limit", "3"]) == 0
    assert teacher.main(["--probe-report", "3"]) == 0
    doc = json.loads((tmp_path / "results" / "teacher" / "probe.json").read_text(encoding="utf-8"))
    m = doc["metrics"]
    assert (m["qwen_valid_json"]["n"], m["qwen_valid_json"]["denominator"]) == (3, 3)
    assert (m["agreement_role_exact"]["n"], m["agreement_role_exact"]["denominator"]) == (3, 3)
    assert m["qwen_forecast_train_pool_hours"]["records"] == 4
    assert doc["probe"]["n"] == 3
