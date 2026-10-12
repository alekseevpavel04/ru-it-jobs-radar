import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import radar.results
from radar.eval import human_check as hc


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _fields(role: str = "backend") -> dict[str, Any]:
    return {
        "role": role,
        "grade": None,
        "skills": ["Python"],
        "salary": {"from": 100000, "to": None, "currency": "RUB", "gross_net": None, "period": None},
        "work_format": None,
        "cities": [],
        "experience_years": None,
        "employment_type": None,
        "lang": "ru",
    }


@pytest.fixture
def gold_dir(tmp_path: Path) -> Path:
    gold = tmp_path / "gold"
    plan = {"test": [("trudvsem", 60), ("trudvsem_snapshot", 70), ("habr", 20)], "tg_test": [("telegram", 40)]}
    for split, sources in plan.items():
        records, inputs = [], []
        for source, n in sources:
            for k in range(n):
                rid = f"{source}:{k}"
                records.append({"id": rid, "source": source, "fields": _fields()})
                inputs.append({"id": rid, "source": source, "input": f"Вакансия {rid}, нужен Python"})
        _write(gold / f"{split}.jsonl", records)
        _write(gold / f"{split}_inputs.jsonl", inputs)
    return gold


def test_allocation_on_the_real_gold_test_sizes_keeps_eight_habr_and_is_proportional_elsewhere() -> None:
    sizes = {"habr": 65, "telegram": 130, "trudvsem": 181, "trudvsem_snapshot": 194}
    assert hc.allocate(sizes, 50, 8) == {"habr": 8, "telegram": 11, "trudvsem": 15, "trudvsem_snapshot": 16}


def test_allocation_floor_takes_a_whole_small_stratum_and_never_more() -> None:
    out = hc.allocate({"a": 3, "b": 100, "c": 100}, 20, 8)
    assert out == {"a": 3, "b": 9, "c": 8}
    assert sum(out.values()) == 20
    with pytest.raises(ValueError):
        hc.allocate({"a": 3}, 4, 1)


def test_draw_is_fixed_by_the_seed_and_stratified(gold_dir: Path) -> None:
    first, second = hc.draw(gold_dir, 7), hc.draw(gold_dir, 7)
    assert first == second
    assert first["composition"] == {"habr": 8, "telegram": 10, "trudvsem": 15, "trudvsem_snapshot": 17}
    ids = [r["id"] for r in first["records"]]
    assert len(set(ids)) == 50
    assert first["ids_sha256"] == hc.ids_sha(ids)
    assert hc.draw(gold_dir, 8)["ids_sha256"] != first["ids_sha256"]
    assert {r["split"] for r in first["records"] if r["source"] == "telegram"} == {"tg_test"}


def test_saved_sample_is_reused_and_an_edited_one_is_refused(gold_dir: Path, tmp_path: Path) -> None:
    review = tmp_path / "review"
    with pytest.raises(FileNotFoundError):
        hc.load_sample(review, gold_dir, 7, create=False)
    sample = hc.load_sample(review, gold_dir, 7, create=True)
    assert hc.load_sample(review, gold_dir, 99, create=False) == sample
    edited = sample | {"records": sample["records"][1:]}
    (review / hc.SAMPLE_FILE).write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="edited"):
        hc.load_sample(review, gold_dir, 7, create=False)


def _answer(item: hc.Item, wrong: tuple[str, ...] = (), verdict: str | None = "agree", **extra: Any) -> hc.Answer:
    marks = {f: ("wrong" if f in wrong else "ok") for f in hc.FIELDS}
    return hc.Answer.model_validate(
        {"id": item.id, "fields_sha": item.fields_sha, "marks": marks, "verdict": verdict, **extra}
    )


def test_report_counts_complete_answers_and_lists_only_ids_and_fields(gold_dir: Path, tmp_path: Path) -> None:
    items = hc.load_items(hc.draw(gold_dir, 7), gold_dir)
    answers = {
        items[0].id: _answer(items[0]),
        items[1].id: _answer(
            items[1],
            wrong=("skills", "salary"),
            verdict="disagree",
            comments={"skills": "нет SQL в тексте"},
            cases={"skills": [48]},
        ),
        items[2].id: _answer(items[2], verdict=None),
        items[3].id: _answer(items[3]).model_copy(update={"fields_sha": "0" * 64}),
    }
    metrics, extra = hc.summarize(items, answers)
    assert metrics["records_checked"] == {"value": 0.04, "n": 2, "denominator": 50}
    assert metrics["records_all_fields_correct"] == {"value": 0.5, "n": 1, "denominator": 2}
    assert metrics["field_skills_correct"]["n"] == 1
    assert metrics["field_role_correct"]["n"] == 2
    assert metrics["verdict_disagree"]["n"] == 1
    assert extra["disagreements"] == [
        {"id": items[1].id, "source": items[1].source, "field": "skills", "guide_cases": [48]},
        {"id": items[1].id, "source": items[1].source, "field": "salary", "guide_cases": []},
    ]
    assert extra["stale_answers"] == [items[3].id]
    assert len(extra["unchecked_ids"]) == 48
    assert "нет SQL" not in json.dumps(extra, ensure_ascii=False)


def test_report_command_writes_counts_and_the_sample_identity(
    gold_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RADAR_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(radar.results, "RESULTS", tmp_path / "results")
    assert hc.main(["sample"]) == 0
    sample = json.loads((tmp_path / "review" / hc.SAMPLE_FILE).read_text(encoding="utf-8"))
    items = hc.load_items(sample, gold_dir)
    answers = tmp_path / "review" / hc.ANSWERS_FILE
    answers.write_text(
        _answer(items[0], comments={"role": "текст вакансии"}).model_dump_json() + "\n", encoding="utf-8"
    )

    assert hc.main(["report"]) == 0
    doc = json.loads((tmp_path / "results" / "gold" / "human_check.json").read_text(encoding="utf-8"))
    assert doc["metrics"]["records_all_fields_correct"] == {"value": 1.0, "n": 1, "denominator": 1}
    assert doc["sample"]["ids_sha256"] == sample["ids_sha256"]
    assert doc["sample"]["seed"] == hc.load_config().seed
    assert len(doc["sample"]["ids"]) == 50
    assert "текст вакансии" not in json.dumps(doc, ensure_ascii=False)


def test_report_refuses_when_nothing_is_checked(gold_dir: Path) -> None:
    items = hc.load_items(hc.draw(gold_dir, 7), gold_dir)
    with pytest.raises(ValueError, match="no record"):
        hc.summarize(items, {})


@pytest.fixture
def server(gold_dir: Path, tmp_path: Path) -> Iterator[hc.ReviewServer]:
    items = hc.load_items(hc.draw(gold_dir, 7), gold_dir)
    srv = hc.ReviewServer(0, items, tmp_path / "review" / hc.ANSWERS_FILE)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _request(
    srv: hc.ReviewServer, method: str, path: str, body: Any = None, host: str | None = None
) -> tuple[int, bytes]:
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}", "Content-Type": "application/json"}
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    resp = conn.getresponse()
    out = resp.status, resp.read()
    conn.close()
    return out


def test_server_serves_the_page_and_state_and_appends_answers(server: hc.ReviewServer) -> None:
    status, page = _request(server, "GET", "/")
    assert status == 200 and b"<title>" in page
    status, raw = _request(server, "GET", "/api/state")
    state = json.loads(raw)
    assert status == 200 and len(state["items"]) == 50 and state["answers"] == {}
    item = state["items"][0]

    body = {"id": item["id"], "fields_sha": item["fields_sha"], "marks": {"role": "ok"}}
    assert _request(server, "POST", "/api/answer", body)[0] == 200
    body["marks"] = {"role": "wrong"}
    assert _request(server, "POST", "/api/answer", body)[0] == 200

    lines = server.answers_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[-1])["answered_at"]
    answers = json.loads(_request(server, "GET", "/api/state")[1])["answers"]
    assert answers[item["id"]]["marks"] == {"role": "wrong"}


def test_server_refuses_unknown_records_fields_stale_labels_and_foreign_hosts(server: hc.ReviewServer) -> None:
    item = json.loads(_request(server, "GET", "/api/state")[1])["items"][0]
    ok = {"id": item["id"], "fields_sha": item["fields_sha"]}
    assert _request(server, "POST", "/api/answer", ok | {"id": "nope"})[0] == 400
    assert _request(server, "POST", "/api/answer", ok | {"marks": {"company": "ok"}})[0] == 400
    assert _request(server, "POST", "/api/answer", ok | {"marks": {"role": "maybe"}})[0] == 400
    assert _request(server, "POST", "/api/answer", ok | {"fields_sha": "0" * 64})[0] == 400
    assert _request(server, "GET", "/api/state", host="evil.example:80")[0] == 403
    assert not server.answers_path.exists()


def test_a_verdict_without_field_marks_counts_for_the_verdict_only() -> None:
    items = [hc.Item(id=f"r{i}", source="habr", split="test", input="t", fields={"role": "qa"}) for i in range(3)]
    sha = items[0].fields_sha
    answers = {
        "r0": hc.Answer(id="r0", fields_sha=sha, marks={f: "ok" for f in hc.FIELDS}, verdict="agree"),
        "r1": hc.Answer(id="r1", fields_sha=sha, verdict="agree"),
    }
    metrics, extra = hc.summarize(items, answers)
    assert (metrics["records_judged"]["n"], metrics["records_judged"]["denominator"]) == (2, 3)
    assert (metrics["records_checked"]["n"], metrics["records_checked"]["denominator"]) == (1, 3)
    assert (metrics["verdict_agree"]["n"], metrics["verdict_agree"]["denominator"]) == (2, 2)
    assert metrics["field_role_correct"]["denominator"] == 1
    assert extra["unchecked_ids"] == ["r2"]
