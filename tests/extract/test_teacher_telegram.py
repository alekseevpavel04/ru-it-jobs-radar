import json
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radar import config, results
from radar.extract import teacher
from radar.extract.schema import json_schema
from radar.extract.teacher import (
    SPLITS,
    TEACHERS,
    Item,
    label_one,
    request_body,
    request_schema,
    select,
    sha256_text,
    split_exclude,
    tg_train_pool_items,
)

ANSWER = {
    "role": "backend",
    "grade": "middle",
    "skills": ["Python", "FastAPI"],
    "salary": {"from": 250000, "to": 350000, "currency": None, "gross_net": "net", "period": None},
    "work_format": "remote",
    "cities": ["Казань"],
    "experience_years": {"min": 3, "max": None},
    "employment_type": None,
    "lang": "ru",
}
QWEN = "Qwen3-8B-Q4_K_M"
YANDEX = "YandexGPT-5-Lite-8B-instruct-Q4_K_M"


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 6900, "completion_tokens": 100},
    }


def ask(split: str, content: str) -> dict[str, Any]:
    body = request_body(TEACHERS["qwen"], "rules", "text", request_schema(split))
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=completion(content)))
    client = httpx.Client(base_url="http://llm.test/v1", transport=transport)
    return label_one(client, body, Item("telegram:c/1", "geekjob|dev", "text"))


# --- the exclude key ----------------------------------------------------------------------------


def test_s1_request_schema_is_the_extraction_schema() -> None:
    for split in ("train_pool", "dev"):
        assert json.dumps(request_schema(split), sort_keys=True) == json.dumps(json_schema(), sort_keys=True)


def test_telegram_request_schema_asks_exclude_last() -> None:
    schema = request_schema("tg_train_pool")
    assert list(schema["properties"])[-1] == "exclude"
    assert schema["required"][-1] == "exclude"
    assert schema["properties"]["exclude"]["anyOf"][0]["enum"] == ["multi_vacancy", "not_vacancy"]
    assert set(json_schema()["properties"]) == set(schema["properties"]) - {"exclude"}
    assert request_schema("tg_dev") == schema


def test_exclude_is_split_off_the_fields() -> None:
    fields, value = split_exclude(json.dumps({**ANSWER, "exclude": "multi_vacancy"}))
    assert value == "multi_vacancy"
    assert fields is not None and json.loads(fields) == ANSWER
    assert split_exclude(json.dumps({**ANSWER, "exclude": None}))[1] is None
    assert split_exclude(json.dumps({**ANSWER, "exclude": "digest"})) == (None, None)
    assert split_exclude(json.dumps(ANSWER)) == (None, None)
    assert split_exclude('{"role": ') == ('{"role": ', None)


def test_telegram_row_keeps_exclude_out_of_output() -> None:
    row = ask("tg_train_pool", json.dumps({**ANSWER, "exclude": "not_vacancy"}))
    assert row["valid"] is True
    assert row["exclude"] == "not_vacancy"
    assert json.loads(row["output"]) == ANSWER
    assert "exclude" in json.loads(row["raw"])

    single = ask("tg_dev", json.dumps({**ANSWER, "exclude": None}))
    assert (single["valid"], single["exclude"]) == (True, None)


def test_telegram_answer_without_a_valid_exclude_is_invalid() -> None:
    for content in (json.dumps(ANSWER), json.dumps({**ANSWER, "exclude": "digest"})):
        row = ask("tg_train_pool", content)
        assert (row["valid"], row["output"], row["exclude"]) == (False, None, None)


def test_s1_row_has_no_exclude_key() -> None:
    row = ask("train_pool", json.dumps(ANSWER))
    assert row["valid"] is True
    assert "exclude" not in row


# --- selection ----------------------------------------------------------------------------------


def write_telegram_corpus(data: Path) -> None:
    root = data / "corpus" / "telegram"
    root.mkdir(parents=True)

    def rec(rid: str, group: str, network: str, proxy: str, day: str) -> dict[str, Any]:
        return {
            "id": rid,
            "group_id": group,
            "network": network,
            "role_proxy": proxy,
            "published_at": f"2026-10-{day}T10:00:00+00:00",
            "input": f"Вакансия {rid}",
        }

    records = [
        rec("telegram:a/1", "g1", "geekjob", "dev", "01"),
        rec("telegram:b/1", "g1", "job_x", "dev", "03"),  # a later repost of the same vacancy
        rec("telegram:a/2", "g2", "geekjob", "ml_ds_data", "02"),
        rec("telegram:a/3", "g3", "geekjob", "qa_devops_sys", "02"),
        rec("telegram:a/4", "g4", "interns", "dev", "02"),
        rec("telegram:a/5", "g5", "geekjob", "dev", "02"),
        rec("telegram:a/6", "g6", "geekjob", "dev", "02"),
    ]
    pq.write_table(pa.Table.from_pylist(records), root / "records.parquet")
    split = [
        {"split": "train_pool", "member_ids": ["telegram:a/1", "telegram:b/1"]},
        {"split": "train_pool", "member_ids": ["telegram:a/2"]},
        {"split": "train_pool", "member_ids": ["telegram:a/3"]},
        {"split": "train_pool", "member_ids": ["telegram:a/4"]},
        {"split": "dev", "member_ids": ["telegram:a/5"]},
        {"split": "withheld", "member_ids": ["telegram:a/6"]},
    ]
    pq.write_table(pa.Table.from_pylist(split), root / "split.parquet")
    gold = data / "gold"
    gold.mkdir(parents=True)
    line = {"id": "telegram:a/5", "source": "telegram", "input": "Вакансия telegram:a/5"}
    (gold / "tg_dev_inputs.jsonl").write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")


def test_tg_train_pool_takes_the_latest_post_of_each_group(tmp_path: Path) -> None:
    write_telegram_corpus(tmp_path)
    items = tg_train_pool_items(tmp_path / "corpus" / "telegram", seed=1)
    assert sorted(i.id for i in items) == ["telegram:a/2", "telegram:a/3", "telegram:a/4", "telegram:b/1"]
    assert {i.id: i.stratum for i in items}["telegram:b/1"] == "job_x|dev"
    every = tg_train_pool_items(tmp_path / "corpus" / "telegram", seed=1, one_per_group=False)
    assert len(every) == 5


def test_tg_splits_and_no_test_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path))
    write_telegram_corpus(tmp_path)
    assert [i.id for i in select("tg_dev", None, [], seed=1)] == ["telegram:a/5"]
    kept = select("tg_train_pool", 2, ["ml_ds_data"], seed=1)
    assert kept[0].id == "telegram:a/2"
    assert len(kept) == 2
    assert not {"test", "tg_test"} & set(SPLITS)
    for split in ("test", "tg_test"):
        with pytest.raises(ValueError, match="not served"):
            select(split, None, [], seed=1)
        with pytest.raises(SystemExit):
            teacher.main(["--teacher", "qwen", "--split", split])


# --- end to end ---------------------------------------------------------------------------------


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"alias": QWEN, "exclude": {}}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": state["alias"]}]})
        body = json.loads(request.content)
        text = body["messages"][1]["content"]
        answer = dict(ANSWER)
        if "exclude" in body["response_format"]["json_schema"]["schema"]["properties"]:
            answer["exclude"] = state["exclude"].get(text)
        return httpx.Response(200, json=completion(json.dumps(answer, ensure_ascii=False)))

    real_client = httpx.Client

    def fake_client(**kwargs: Any) -> httpx.Client:
        return real_client(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(teacher.httpx, "Client", fake_client)
    monkeypatch.setattr(teacher, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    write_telegram_corpus(tmp_path / "data")
    return state


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_tg_train_pool_run_writes_exclude_and_its_own_schema(server: dict[str, Any], tmp_path: Path) -> None:
    server["exclude"] = {"Вакансия telegram:a/3": "multi_vacancy"}
    assert teacher.main(["--teacher", "qwen", "--split", "tg_train_pool", "--workers", "2"]) == 0
    out = tmp_path / "data" / "teacher" / "qwen"
    rows = {r["id"]: r for r in lines(out / "tg_train_pool.jsonl")}
    assert sorted(rows) == ["telegram:a/2", "telegram:a/3", "telegram:a/4", "telegram:b/1"]
    assert rows["telegram:a/3"]["exclude"] == "multi_vacancy"
    assert rows["telegram:a/2"]["exclude"] is None
    assert all(r["valid"] and "exclude" not in json.loads(r["output"]) for r in rows.values())
    assert rows["telegram:b/1"]["input_sha256"] == sha256_text("Вакансия telegram:b/1")

    meta = json.loads((out / "tg_train_pool.meta.json").read_text(encoding="utf-8"))
    assert meta["asks_exclude"] is True
    assert meta["schema_sha256"] == sha256_text(json.dumps(request_schema("tg_train_pool"), sort_keys=True))
    assert meta["schema_sha256"] != sha256_text(json.dumps(json_schema(), sort_keys=True))


def test_tg_probe_report(server: dict[str, Any], tmp_path: Path) -> None:
    server["exclude"] = {"Вакансия telegram:a/3": "not_vacancy"}
    for name, alias in (("qwen", QWEN), ("yandex", YANDEX)):
        server["alias"] = alias
        assert teacher.main(["--teacher", name, "--split", "tg_train_pool"]) == 0
    assert teacher.main(["--probe-report", "4", "--split", "tg_train_pool"]) == 0
    doc = json.loads((tmp_path / "results" / "teacher" / "probe_tg.json").read_text(encoding="utf-8"))
    m = doc["metrics"]
    assert (m["qwen_exclude_not_vacancy"]["n"], m["qwen_exclude_not_vacancy"]["denominator"]) == (1, 4)
    assert m["yandex_exclude_multi_vacancy"]["n"] == 0
    assert m["qwen_forecast_tg_train_pool_hours"]["records"] == 4
    assert "qwen_tg_train_pool_wall_s_per_record_w1" in m
    assert doc["probe"]["split"] == "tg_train_pool"
    assert not (tmp_path / "results" / "teacher" / "probe.json").exists()
