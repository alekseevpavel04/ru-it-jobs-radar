import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from radar.extract import corpus_drops, corpus_run
from radar.extract.corpus_drops import DDL, TABLE, dropped_rows, replace_check, rules_sha256
from radar.extract.telegram_corpus import channel_langs

POST = (
    "Ищем Python-разработчика в команду платежей. Удалённо, полный день, зарплата от 250 000 руб. "
    "Нужны Python, PostgreSQL, Kafka и опыт от 3 лет. Пишите в личные сообщения."
)
DIGEST = "Вакансии недели для аналитиков: смотрите подборку ниже, там много интересного и полезного для всех."
STUB = "Python-разработчик https://career.habr.com/vacancies/1000001 #python"
ENGLISH = (
    "We are hiring a senior Python engineer for the payments team. Remote, full time, salary from 5000 USD. "
    "Python, PostgreSQL and Kafka required, three years of experience."
)


def _post(source_id: str, text: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "dt": "2026-10-08",
        "content_hash": f"h-{source_id}",
        "published_at": "2026-10-01 09:00:00.000",
        "text": text,
        "payload": "{}",
        "is_vacancy": True,
        "multi_vacancy": False,
    }


class Posts:
    def __init__(self, posts: Sequence[dict[str, Any]]) -> None:
        self.posts = list(posts)
        self.parts: list[int] = []

    def telegram_posts(self, since: Any, part: int, parts: int) -> list[dict[str, Any]]:
        self.parts.append(part)
        return [p for i, p in enumerate(self.posts) if i % parts == part]


class FakeCH:
    def __init__(self, posts_body: str = "") -> None:
        self.sent: list[tuple[str, dict[str, str]]] = []
        self.posts_body = posts_body

    def query(self, sql: str, **params: str) -> str:
        self.sent.append((sql, params))
        if "FROM raw.vacancies FINAL" in sql and params.get("part") == "0":
            return self.posts_body
        return ""


POSTS = [
    _post("devjobs/1", POST),
    _post("devjobs/2", DIGEST),
    _post("devjobs/3", STUB),
    _post("remoteit/4", ENGLISH),
    _post("devjobs/5", ENGLISH),
]


def test_each_drop_rule_is_a_row_with_its_reason_and_kept_posts_are_counted_only() -> None:
    langs = {"remoteit": "ru"}
    store = Posts(POSTS)
    rows, outcomes = dropped_rows(store, 4, 100, langs, "20261013T100000", "2026-10-13 10:00:00.000")
    assert store.parts == [0, 1, 2, 3]
    reasons = {r["source_id"]: r["reason"] for r in rows}
    assert reasons == {"devjobs/2": "digest", "devjobs/3": "link_stub", "remoteit/4": "channel_lang"}
    assert outcomes == {"kept": 2, "digest": 1, "link_stub": 1, "channel_lang": 1}
    row = next(r for r in rows if r["source_id"] == "devjobs/3")
    assert row["source"] == "telegram" and row["content_hash"] == "h-devjobs/3" and row["dt"] == "2026-10-08"
    assert row["check_id"] == "20261013T100000" and row["rules_sha256"] == rules_sha256(100, langs)


def test_rules_hash_follows_the_threshold_and_languages() -> None:
    base = rules_sha256(100, {"remoteit": "ru"})
    assert base == rules_sha256(100, {"remoteit": "ru"})
    assert base != rules_sha256(120, {"remoteit": "ru"})
    assert base != rules_sha256(100, {})


def test_a_check_is_one_insert_then_older_checks_are_deleted() -> None:
    ch = FakeCH()
    replace_check(ch, [{"source": "telegram", "source_id": "a/1"}, {"source": "telegram", "source_id": "a/2"}], "c2")
    (insert, _), (delete, params) = ch.sent
    head, *lines = insert.rstrip("\n").split("\n")
    assert head == f"INSERT INTO {TABLE} FORMAT JSONEachRow" and len(lines) == 2
    assert delete.startswith(f"ALTER TABLE {TABLE} DELETE WHERE check_id != {{check:String}}")
    assert params == {"check": "c2"}

    empty = FakeCH()
    replace_check(empty, [], "c3")
    assert len(empty.sent) == 1 and "DELETE" in empty.sent[0][0]


def test_ddl_columns_are_the_row_the_check_writes() -> None:
    ddl = DDL.read_text(encoding="utf-8")
    body = ddl[ddl.index("extract.vacancy_dropped") :]
    columns = re.findall(r"^\s{4}([a-z_0-9]+) ", body, re.M)
    rows, _ = dropped_rows(Posts([_post("devjobs/2", DIGEST)]), 1, 100, {}, "c", "2026-10-13 10:00:00.000")
    assert set(columns) == set(rows[0]) | {"inserted_at"}


def test_dropped_subcommand_checks_every_slice_and_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def raw(post: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in post.items() if k not in ("dt", "published_at")}
        return out | {"day": post["dt"], "published": post["published_at"]}

    assert channel_langs().get("remoteit") == "ru"
    ch = FakeCH("\n".join(json.dumps(raw(p), ensure_ascii=False) for p in POSTS))
    monkeypatch.setattr("radar.stream.clickhouse.ClickHouse", lambda: _Context(ch))
    written: dict[str, Any] = {}

    def fake_write(relpath: str, metrics: dict[str, Any], source_script: str, **extra: Any) -> Path:
        written.update(relpath=relpath, metrics=metrics)
        return tmp_path / relpath

    monkeypatch.setattr(corpus_drops, "write_result", fake_write)
    assert corpus_run.main(["dropped"]) == 0

    reads = [p for sql, p in ch.sent if "FROM raw.vacancies FINAL" in sql]
    assert [p["part"] for p in reads] == [str(i) for i in range(corpus_run.TELEGRAM_SLICES)]
    assert all("since" not in p for p in reads)
    insert = next(sql for sql, _ in ch.sent if sql.startswith(f"INSERT INTO {TABLE}"))
    assert {json.loads(line)["reason"] for line in insert.strip().split("\n")[1:]} == {
        "digest",
        "link_stub",
        "channel_lang",
    }
    assert any(sql.startswith("CREATE TABLE IF NOT EXISTS extract.vacancy_dropped") for sql, _ in ch.sent)
    assert written["relpath"] == "extract/corpus_dropped.json"
    assert written["metrics"]["telegram_posts_checked"] == {"value": 5}
    assert written["metrics"]["telegram_kept"] == {"value": 0.4, "n": 2, "denominator": 5}
    assert "telegram_digest" in capsys.readouterr().out


def test_check_id_is_the_utc_moment() -> None:
    ch = FakeCH()
    corpus_drops.run(Posts([]), ch, 2, now=datetime(2026, 10, 13, 8, 30, 5, tzinfo=UTC))
    assert ch.sent[-1][1] == {"check": "20261013T083005"}


class _Context:
    def __init__(self, ch: FakeCH) -> None:
        self.ch = ch

    def __enter__(self) -> FakeCH:
        return self.ch

    def __exit__(self, *exc: object) -> None:
        return None
