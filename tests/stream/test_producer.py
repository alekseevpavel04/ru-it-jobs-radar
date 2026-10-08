import logging
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from confluent_kafka import KafkaError, KafkaException

from radar.collect.envelope import RawRecord, content_hash
from radar.collect.writer import PartitionWriter, write_records
from radar.stream import producer
from radar.stream.messages import Message
from radar.stream.producer import PublishError, publish_after_collect, publish_partition

DT = date(2026, 10, 8)


def _record(source_id: str) -> RawRecord:
    return RawRecord(
        source="habr",
        source_id=source_id,
        fetched_at=datetime(2026, 10, 8, 12, 0, tzinfo=UTC),
        published_at=None,
        modified_at=None,
        url=f"https://career.habr.com/vacancies/{source_id}",
        title="ML Engineer",
        text=None,
        payload={"id": int(source_id)},
        content_hash=content_hash("ML Engineer", None, {"id": int(source_id)}),
        pii_removed={},
    )


def test_publish_partition_sends_every_row_with_its_day(tmp_path: Path) -> None:
    write_records([_record("1"), _record("2"), _record("3")], "habr", DT, root=tmp_path)
    sent: list[Message] = []
    assert publish_partition("habr", DT, sent.append, root=tmp_path) == 3
    assert [m.key for m in sent] == [b"1", b"2", b"3"]
    assert {m.topic for m in sent} == {"raw.habr"}
    assert all(m.headers == [("dt", b"2026-10-08")] for m in sent)


def test_partition_without_marker_is_not_published(tmp_path: Path) -> None:
    writer = PartitionWriter("habr", DT, root=tmp_path)
    writer.add(_record("1"))
    writer.close(complete=False)
    sent: list[Message] = []
    with pytest.raises(PublishError, match="_SUCCESS"):
        publish_partition("habr", DT, sent.append, root=tmp_path)
    assert sent == []


def test_collector_survives_unconfigured_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    write_records([_record("1")], "habr", DT, root=tmp_path)
    monkeypatch.delenv("REDPANDA_BROKERS", raising=False)
    with caplog.at_level(logging.WARNING):
        assert publish_after_collect("habr", DT, root=tmp_path) is False
    assert "REDPANDA_BROKERS" in caplog.text
    assert "radar.stream.replay --source habr --dt 2026-10-08" in caplog.text


def test_collector_survives_broker_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def down(source: str, dt: date, root: Path | None = None) -> int:
        raise KafkaException(KafkaError(KafkaError._TRANSPORT))

    monkeypatch.setattr(producer, "publish", down)
    with caplog.at_level(logging.WARNING):
        assert publish_after_collect("habr", DT, root=tmp_path) is False
    assert "is on disk but not in the stream" in caplog.text
