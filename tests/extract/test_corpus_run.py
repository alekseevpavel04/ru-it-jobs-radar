import argparse
import json
import re
import sys
import types
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, get_args

import pytest

from radar.extract import schema
from radar.extract.corpus_run import (
    DDL,
    SOURCES,
    TABLE,
    Answer,
    ClickHouseStore,
    Key,
    RegexBackend,
    Version,
    cmd_run,
    load_backend,
    open_meta,
    plan_metrics,
    sha256_text,
    table_row,
    versions,
)
from radar.extract.inputs import input_for_record

POST = (
    "Ищем Python-разработчика в команду платежей. Удалённо, полный день, зарплата от 250 000 руб. "
    "Нужны Python, PostgreSQL, Kafka и опыт от 3 лет. Пишите в личные сообщения."
)
DIGEST = "Вакансии недели для аналитиков: смотрите подборку ниже, там много интересного и полезного для всех."


class FakeStore:
    """ClickHouse in memory: staged keys, raw rows, Telegram posts and the extraction table."""

    def __init__(self) -> None:
        self.staged: dict[str, list[tuple[str, str, str]]] = {s: [] for s in SOURCES if s != "telegram"}
        self.raw: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.posts: list[dict[str, Any]] = []
        self.table: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.exists = False
        self.inserts = 0
        self.fail_inserts = 0

    def add(self, source: str, source_id: str, title: str, text: str, payload: Mapping[str, Any], dt: str) -> None:
        content_hash = sha256_text(f"{title}|{text}|{json.dumps(payload, sort_keys=True)}")[:16]
        self.staged[source].append((source_id, dt, content_hash))
        self.raw[(source, source_id, dt, content_hash)] = {
            "source_id": source_id,
            "dt": dt,
            "content_hash": content_hash,
            "title": title,
            "text": text,
            "payload": json.dumps(payload, ensure_ascii=False),
            "published_at": f"{dt} 10:00:00.000000",
        }

    def post(self, source_id: str, text: str, is_vacancy: bool = True, published: str = "2026-10-01") -> None:
        self.posts.append(
            {
                "source_id": source_id,
                "dt": "2026-10-08",
                "content_hash": sha256_text(text)[:16],
                "published_at": f"{published} 09:00:00.000000",
                "text": text,
                "payload": "{}",
                "is_vacancy": is_vacancy,
                "multi_vacancy": False,
            }
        )

    def latest_keys(self, source: str) -> list[tuple[str, str, str]]:
        return list(self.staged[source])

    def raw_rows(self, source: str, keys: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
        return [dict(self.raw[(source, *k)]) for k in keys if (source, *k) in self.raw]

    def telegram_posts(self, since: date | None, part: int, parts: int) -> list[dict[str, Any]]:
        if part:
            return []
        return [dict(p) for p in self.posts if since is None or p["published_at"][:10] >= since.isoformat()]

    def table_exists(self) -> bool:
        return self.exists

    def ensure_table(self) -> None:
        self.exists = True

    def done_keys(self, model_id: str) -> set[Key]:
        return {(s, i, h) for (s, i, h, m) in self.table if m == model_id}

    def prompt_shas(self, model_id: str) -> set[str | None]:
        return {r["prompt_sha256"] for (_, _, _, m), r in self.table.items() if m == model_id}

    def insert(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.fail_inserts:
            self.fail_inserts -= 1
            raise RuntimeError("ClickHouse is down")
        self.inserts += 1
        for r in rows:
            self.table[(r["source"], r["source_id"], r["content_hash"], r["model_id"])] = dict(r)

    def summary(self, model_id: str) -> list[dict[str, Any]]:
        return []


ANSWER = {
    "role": "backend",
    "grade": None,
    "skills": ["Python", "PostgreSQL"],
    "salary": {"from": 250000, "to": None, "currency": "RUB", "gross_net": None, "period": "month"},
    "work_format": "remote",
    "cities": [],
    "experience_years": {"min": 3, "max": None},
    "employment_type": None,
    "lang": "ru",
}


class FakeBackend:
    model_id = "fake-student"
    prompt_sha256: str | None = "p1"

    def __init__(self, fail: Sequence[str] = (), invalid: Sequence[str] = (), cut: Sequence[str] = ()) -> None:
        self.fail, self.invalid, self.cut = set(fail), set(invalid), set(cut)
        self.seen: list[str] = []

    def extract(self, inputs: Sequence[str]) -> list[Answer]:
        out = []
        for text in inputs:
            self.seen.append(text)
            if any(m in text for m in self.fail):
                out.append(Answer(None, error="HTTP 500: busy"))
            elif any(m in text for m in self.invalid):
                out.append(Answer('{"role": "astronaut"}', "stop"))
            elif any(m in text for m in self.cut):
                out.append(Answer(json.dumps(ANSWER), "length"))
            else:
                out.append(Answer(json.dumps(ANSWER), "stop", input_truncated=False, seconds=0.2))
        return out


def args(tmp_path: Path, **kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "sources": list(SOURCES),
        "telegram_since": None,
        "limit": None,
        "batch_size": 2,
        "workers": 2,
        "insert_every": 3,
        "dry_run": False,
        "state_dir": tmp_path / "state",
        "backend": "fake",
        "log_every": 1,
    }
    return argparse.Namespace(**(base | kw))


@pytest.fixture
def store() -> FakeStore:
    s = FakeStore()
    payload = {"salary_min": 100000, "salary_max": 150000, "region": {"name": "Город Москва"}}
    s.add("trudvsem", "a1", "Аналитик данных", "Обязанности: SQL, отчёты", payload, "2026-10-11")
    # the snapshot renders the same page: an identical input, one request for both
    s.add(
        "trudvsem_snapshot",
        "a1",
        "Аналитик данных",
        "Обязанности: SQL, отчёты",
        {"salaryMin": 100000, "salaryMax": 150000, "regionName": "Город Москва"},
        "2026-10-10",
    )
    s.add(
        "habr", "1000123", "Backend-разработчик", "Python и Go", {"salary": {"formatted": "от 300 000 ₽"}}, "2026-10-11"
    )
    s.add(
        "superjob", "55", "Тестировщик", "Ручное тестирование", {"payment_from": 90000, "currency": "rub"}, "2026-10-12"
    )
    s.post("devjobs/101", POST)
    s.post("devjobs/102", DIGEST)
    s.post("devjobs/103", "Короткий пост https://example.com/job")
    s.post("devjobs/104", POST.replace("Python", "Go"), is_vacancy=False)
    return s


def test_ddl_enumerations_match_the_schema() -> None:
    ddl = DDL.read_text(encoding="utf-8")
    columns = {
        "role": schema.Role,
        "grade": schema.Grade,
        "salary_currency": schema.Currency,
        "salary_gross_net": schema.GrossNet,
        "salary_period": schema.Period,
        "work_format": schema.WorkFormat,
        "employment_type": schema.EmploymentType,
        "lang": schema.Lang,
    }
    for column, literal in columns.items():
        found = re.search(rf"^\s+{column} Nullable\(Enum8\((.*?)\)\)", ddl, flags=re.MULTILINE | re.DOTALL)
        assert found, column
        assert re.findall(r"'(\w+)' = \d+", found.group(1)) == list(get_args(literal)), column


def test_ddl_columns_are_the_row_the_runner_writes() -> None:
    body = DDL.read_text(encoding="utf-8").split(f"CREATE TABLE IF NOT EXISTS {TABLE}", 1)[1]
    names = re.findall(r"^    (\w+) ", body, flags=re.MULTILINE)
    row = {
        "source": "habr",
        "source_id": "1",
        "content_hash": "h",
        "dt": "2026-10-11",
        "model_id": "m",
        "prompt_sha256": None,
        "schema_version": 1,
        "input_sha256": "s",
        "input_truncated": None,
        "output": json.dumps(ANSWER),
        "valid": True,
        "finish_reason": "stop",
        "reused": False,
        "run_id": "r",
        "created_at": "2026-10-12T10:00:00.123+00:00",
    }
    assert names == [*table_row(row), "inserted_at"]


def test_versions_follow_the_corpus_rules(store: FakeStore) -> None:
    drops: Counter[str] = Counter()
    found = list(versions(store, SOURCES, None, set(), drops))
    assert [v.id for v in found] == [
        "trudvsem:a1",
        "trudvsem_snapshot:a1",
        "habr:1000123",
        "superjob:55",
        "telegram:devjobs/101",
    ]
    assert drops == {"digest": 1, "link_stub": 1, "not_vacancy": 1}
    by_id = {v.id: v for v in found}
    assert by_id["trudvsem:a1"].input == by_id["trudvsem_snapshot:a1"].input
    assert by_id["superjob:55"].input.startswith("Должность: Тестировщик\nЗарплата: от 90 000 руб.")
    assert by_id["telegram:devjobs/101"].input == input_for_record(
        {"source": "telegram", "title": None, "text": POST, "payload": "{}"}
    )


def test_staged_version_without_raw_row_stops_the_enumeration(store: FakeStore) -> None:
    store.staged["habr"].append(("999", "2026-10-11", "nohash"))
    with pytest.raises(RuntimeError, match="no raw row"):
        list(versions(store, ["habr"], None, set(), Counter()))


def test_run_extracts_each_input_once_and_loads_typed_rows(store: FakeStore, tmp_path: Path) -> None:
    backend = FakeBackend()
    cmd_run(args(tmp_path), store, backend, seed=7)
    assert len(backend.seen) == 4  # five versions, the API and the snapshot share one input
    rows = {(s, i): r for (s, i, _, _), r in store.table.items()}
    assert len(rows) == 5
    assert sum(r["reused"] for r in rows.values()) == 1
    r = rows[("telegram", "devjobs/101")]
    assert r["valid"] is True and r["role"] == "backend" and r["salary_from"] == 250000 and r["salary_to"] is None
    assert r["salary_currency"] == "RUB" and r["experience_min"] == 3 and r["skills"] == ["Python", "PostgreSQL"]
    assert r["prompt_sha256"] == "p1" and r["model_id"] == "fake-student"
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}", r["created_at"])

    meta = json.loads((tmp_path / "state" / "meta.json").read_text(encoding="utf-8"))
    run = meta["runs"][-1]
    assert run["finished"] and run["requests"] == 4 and run["reused"] == 1 and run["inserted"] == 5

    again = FakeBackend()
    cmd_run(args(tmp_path), store, again, seed=7)
    assert again.seen == []


def test_new_version_is_extracted_and_old_answer_reused(store: FakeStore, tmp_path: Path) -> None:
    cmd_run(args(tmp_path), store, FakeBackend(), seed=7)
    # a repost of a known text under a new id: no request, the earlier answer is copied
    store.post("devjobs/205", POST)
    backend = FakeBackend()
    cmd_run(args(tmp_path), store, backend, seed=7)
    assert backend.seen == []
    assert store.table[("telegram", "devjobs/205", sha256_text(POST)[:16], "fake-student")]["reused"] is True


def test_failed_request_is_kept_out_of_the_table_and_retried(store: FakeStore, tmp_path: Path) -> None:
    cmd_run(args(tmp_path), store, FakeBackend(fail=["Тестировщик"]), seed=7)
    assert ("superjob", "55") not in {(s, i) for (s, i, _, _) in store.table}
    file_rows = (tmp_path / "state" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(json.loads(line)["error"] is not None for line in file_rows) == 1
    retry = FakeBackend()
    cmd_run(args(tmp_path), store, retry, seed=7)
    assert len(retry.seen) == 1 and "Тестировщик" in retry.seen[0]


def test_invalid_and_cut_answers_are_rows_with_empty_fields(store: FakeStore, tmp_path: Path) -> None:
    cmd_run(args(tmp_path), store, FakeBackend(invalid=["Тестировщик"], cut=["Backend"]), seed=7)
    rows = {(s, i): r for (s, i, _, _), r in store.table.items()}
    for key in (("superjob", "55"), ("habr", "1000123")):
        assert rows[key]["valid"] is False and rows[key]["role"] is None and rows[key]["skills"] == []
    assert rows[("habr", "1000123")]["finish_reason"] == "length"


def test_rows_the_table_missed_are_loaded_before_the_next_run(store: FakeStore, tmp_path: Path) -> None:
    store.fail_inserts = 1
    with pytest.raises(RuntimeError, match="down"):
        cmd_run(args(tmp_path, insert_every=100, workers=1), store, FakeBackend(), seed=7)
    assert store.table == {}
    meta = json.loads((tmp_path / "state" / "meta.json").read_text(encoding="utf-8"))
    run = meta["runs"][-1]
    assert run["finished"] is False and run["rows"] == 5 and run["inserted"] == 0
    backend = FakeBackend()
    cmd_run(args(tmp_path), store, backend, seed=7)
    assert backend.seen == [] and len(store.table) == 5


def test_limit_and_sources_select_a_probe(store: FakeStore, tmp_path: Path) -> None:
    backend = FakeBackend()
    cmd_run(args(tmp_path, limit=2, sources=["habr", "superjob", "telegram"]), store, backend, seed=7)
    assert len(backend.seen) == 2 and len(store.table) == 2


def test_dry_run_writes_the_file_only(store: FakeStore, tmp_path: Path) -> None:
    cmd_run(args(tmp_path, dry_run=True), store, RegexBackend(), seed=7)
    assert store.table == {} and not store.exists
    rows = (tmp_path / "state" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5 and all(json.loads(line)["valid"] for line in rows)


def test_telegram_since_filters_posts(store: FakeStore) -> None:
    store.post("devjobs/1", POST.replace("платежей", "поиска"), published="2019-05-01")
    found = list(versions(store, ["telegram"], date(2026, 1, 1), set(), Counter()))
    assert [v.source_id for v in found] == ["devjobs/101"]


def test_a_model_id_with_another_prompt_is_refused(store: FakeStore, tmp_path: Path) -> None:
    cmd_run(args(tmp_path), store, FakeBackend(), seed=7)
    changed = FakeBackend()
    changed.prompt_sha256 = "p2"
    with pytest.raises(SystemExit, match="prompt"):
        open_meta(tmp_path / "state" / "meta.json", changed, "fake")
    with pytest.raises(SystemExit, match="prompt"):
        cmd_run(args(tmp_path, state_dir=tmp_path / "other"), store, changed, seed=7)


def test_plan_counts_requests_and_forecasts(store: FakeStore) -> None:
    drops: Counter[str] = Counter()
    found = list(versions(store, SOURCES, None, set(), drops))
    done = {("habr", "1000123", found[2].content_hash)}
    m = plan_metrics(found, done, drops, {"hf": 1.0, "llama": 0.25})
    assert m["habr_done"]["value"] == 1 and m["habr_to_extract"]["value"] == 0
    assert m["total_to_extract"]["value"] == 4 and m["total_requests"]["value"] == 3
    assert m["requests_saved_by_identical_inputs"] == {"value": 0.25, "n": 1, "denominator": 4}
    assert m["forecast_hf_hours"]["value"] == round(3 / 3600, 1)
    assert m["forecast_llama_hours"]["assumed_s_per_request"] == 0.25
    assert m["telegram_to_extract_published_2026"]["value"] == 1
    assert m["telegram_dropped_digest"]["value"] == 1


def test_regex_backend_answers_validate() -> None:
    answers = RegexBackend().extract(["Должность: Data Scientist\n\nPython, SQL, удалённо"])
    assert answers[0].error is None and schema.try_parse(answers[0].raw) is not None


def test_backend_is_loaded_from_module_and_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fake_backends")

    def make(slots: str = "1") -> FakeBackend:
        backend = FakeBackend()
        backend.model_id = f"fake-{slots}"
        return backend

    module.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_backends", module)
    assert load_backend("fake_backends:make", {"slots": "4"}).model_id == "fake-4"
    with pytest.raises(SystemExit):
        load_backend("fake_backends", {})


class FakeCH:
    def __init__(self, answers: Sequence[str] = ()) -> None:
        self.sent: list[tuple[str, dict[str, str]]] = []
        self.answers = list(answers)

    def query(self, sql: str, **params: str) -> str:
        self.sent.append((sql, params))
        return self.answers.pop(0) if self.answers else ""


def test_clickhouse_store_queries() -> None:
    raw = {"source_id": "x'1", "day": "2026-10-11", "content_hash": "h", "title": None, "text": "t", "payload": "{}"}
    ch = FakeCH([json.dumps(raw | {"published": None})])
    store = ClickHouseStore(ch)
    rows = store.raw_rows("habr", [("x'1", "2026-10-11", "h")])
    assert rows[0]["dt"] == "2026-10-11" and "day" not in rows[0] and rows[0]["published_at"] is None
    sql, params = ch.sent[-1]
    assert "('x\\'1', toDate('2026-10-11'), 'h')" in sql and params == {"source": "habr"}

    store.insert([{"source": "habr", "skills": ["Python"]}, {"source": "habr", "skills": []}])
    head, *lines = ch.sent[-1][0].rstrip("\n").split("\n")
    assert head == f"INSERT INTO {TABLE} FORMAT JSONEachRow" and len(lines) == 2

    store.telegram_posts(date(2026, 1, 1), 3, 16)
    sql, params = ch.sent[-1]
    assert "published_at >= {since:Date}" in sql and params == {"part": "3", "parts": "16", "since": "2026-01-01"}


def test_version_key_and_id() -> None:
    v = Version("telegram", "c/1", "h", "2026-10-08", None, "text")
    assert v.key == ("telegram", "c/1", "h") and v.id == "telegram:c/1"
