import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

from radar.collect.envelope import RawRecord, content_hash, parse_source_time
from radar.collect.writer import partition_dir, write_records
from radar.stream.clickhouse import DDL_DIR
from radar.stream.messages import iter_records, record_from_row, to_message, topic_for

DT = date(2026, 6, 30)
FETCHED = datetime(2026, 10, 8, 10, 56, 5, 123456, tzinfo=UTC)


def _record(source_id: str, text: str | None = "Обязанности: писать код на Python.") -> RawRecord:
    payload = {"id": source_id, "salary": {"from": 150000, "to": None}, "skills": ["Python", "SQL"]}
    return RawRecord(
        source="trudvsem_snapshot",
        source_id=source_id,
        fetched_at=FETCHED,
        published_at=parse_source_time("2026-06-20T09:15:00+0300"),
        modified_at=None,
        url=None,
        title="Программист",
        text=text,
        payload=payload,
        content_hash=content_hash("Программист", text, payload),
        pii_removed={"phone": 2, "email": 1},
    )


def test_records_read_back_from_parquet_unchanged(tmp_path: Path) -> None:
    records = [_record("a"), _record("b", text=None)]
    write_records(records, "trudvsem_snapshot", DT, root=tmp_path)
    assert list(iter_records(partition_dir("trudvsem_snapshot", DT, tmp_path), batch_rows=1)) == records


def test_row_with_empty_pii_map() -> None:
    row = {**_record("a").model_dump(), "payload": "{}", "pii_removed": []}
    assert record_from_row(row).pii_removed == {}


def test_message_is_the_envelope_keyed_by_source_id() -> None:
    record = _record("2aa01ec0-00ea-11f1")
    message = to_message(record, DT)
    assert message.topic == "raw.trudvsem_snapshot" == topic_for(record.source)
    assert message.key == b"2aa01ec0-00ea-11f1"
    assert message.headers == [("dt", b"2026-06-30")]
    assert RawRecord.model_validate_json(message.value) == record


def test_message_json_shape() -> None:
    value = to_message(_record("a"), DT).value
    body = json.loads(value)
    assert body["fetched_at"] == "2026-10-08T10:56:05.123456Z"
    assert body["published_at"] == "2026-06-20T06:15:00Z"
    assert body["payload"] == {"id": "a", "salary": {"from": 150000, "to": None}, "skills": ["Python", "SQL"]}
    assert body["pii_removed"] == {"phone": 2, "email": 1}
    # Text goes as UTF-8, not as \u escapes that triple the size of Cyrillic.
    assert "Программист".encode() in value


def test_queue_table_reads_every_envelope_field() -> None:
    ddl = (DDL_DIR / "001_raw_vacancies.sql").read_text(encoding="utf-8")
    match = re.search(r"CREATE TABLE IF NOT EXISTS raw\.vacancies_queue\s*\((.*?)\n\)", ddl, re.S)
    assert match is not None
    columns = {line.split()[0] for line in match.group(1).splitlines() if line.strip()}
    assert columns == set(RawRecord.model_fields)
