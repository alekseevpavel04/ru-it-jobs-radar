"""End-to-end check of the stream on real partitions: rows in Parquet against rows in ClickHouse.

Each target `source[:dt]` (no date: the latest complete partition) is replayed `--replays` times;
then the check waits until the ClickHouse consumer group has committed up to the end of the topic
and counts the rows of that source and day with FINAL. Equal counts after more than one replay mean
nothing was lost and nothing doubled. Writes `results/stream/replay.json`.

    python -m radar.stream.verify trudvsem habr trudvsem_snapshot:2026-06-30 trudvsem_snapshot:2026-07-01
"""

import argparse
import logging
import sys
import time
from datetime import date
from typing import Any

import pyarrow.parquet as pq
from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient

from radar.collect import setup_logging
from radar.collect.writer import existing_keys, partition_dir
from radar.results import write_result
from radar.stream.clickhouse import ClickHouse, apply_schema, rows_final, rows_stored
from radar.stream.messages import topic_for
from radar.stream.producer import SOURCES, Publisher, ensure_topics, kafka_config, publish_partition
from radar.stream.replay import complete_dates

log = logging.getLogger("radar.stream.verify")

RESULT_PATH = "stream/replay.json"
# kafka_group_name of raw.vacancies_queue in infra/clickhouse/001_raw_vacancies.sql.
CLICKHOUSE_GROUP = "clickhouse_raw"
METADATA_TIMEOUT_S = 10.0
STALL_TIMEOUT_S = 180.0


class StreamCheckError(RuntimeError):
    pass


def group_consumer() -> Consumer:
    """A handle on the ClickHouse group that only reads its committed offsets; it never subscribes."""
    return Consumer({**kafka_config(), "group.id": CLICKHOUSE_GROUP, "enable.auto.commit": False})


def lag(consumer: Consumer, topic: str) -> int:
    partition = TopicPartition(topic, 0)
    (committed,) = consumer.committed([partition], timeout=METADATA_TIMEOUT_S)
    low, high = consumer.get_watermark_offsets(partition, timeout=METADATA_TIMEOUT_S)
    # A group that has committed nothing yet reports a negative offset.
    return int(high - max(committed.offset, low))


def wait_consumed(consumer: Consumer, topic: str) -> None:
    last, last_change = lag(consumer, topic), time.monotonic()
    while last > 0:
        time.sleep(2)
        current = lag(consumer, topic)
        if current != last:
            last, last_change = current, time.monotonic()
            log.info("%s: lag %d", topic, current)
        elif time.monotonic() - last_change > STALL_TIMEOUT_S:
            raise StreamCheckError(f"{topic}: lag stuck at {current} for {STALL_TIMEOUT_S:.0f} s")


def verify_target(
    ch: ClickHouse, publisher: Publisher, consumer: Consumer, source: str, dt: date, replays: int
) -> dict[str, Any]:
    topic = topic_for(source)
    directory = partition_dir(source, dt)
    first_offset = consumer.get_watermark_offsets(TopicPartition(topic, 0), timeout=METADATA_TIMEOUT_S)[1]
    started = time.monotonic()
    sent = 0
    for _ in range(replays):
        sent += publish_partition(source, dt, publisher.send)
        publisher.flush()
    produced_s = time.monotonic() - started
    wait_consumed(consumer, topic)
    consumed_s = time.monotonic() - started
    errors = ch.scalar(
        "SELECT count() FROM raw.vacancies_errors WHERE topic = {topic:String} AND kafka_offset >= {first:UInt64}",
        topic=topic,
        first=str(first_offset),
    )
    result: dict[str, Any] = {
        "source": source,
        "dt": dt.isoformat(),
        "replays": replays,
        "messages_sent": sent,
        "rows_parquet": sum(pq.ParquetFile(p).metadata.num_rows for p in directory.glob("part-*.parquet")),
        "distinct_keys_parquet": len(existing_keys(directory)),
        "rows_clickhouse_final": rows_final(ch, source, dt),
        "rows_clickhouse_stored": rows_stored(ch, source, dt),
        "error_messages": errors,
        "seconds_produce": round(produced_s, 1),
        "seconds_until_consumed": round(consumed_s, 1),
    }
    result["match"] = result["rows_parquet"] == result["rows_clickhouse_final"] and errors == 0
    log.info("%s/dt=%s: %s", source, dt, result)
    return result


def parse_target(text: str) -> tuple[str, date]:
    source, _, day = text.partition(":")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; one of {', '.join(SOURCES)}")
    if day:
        return source, date.fromisoformat(day)
    complete = complete_dates(source)
    if not complete:
        raise ValueError(f"no complete partitions of {source}")
    return source, complete[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("targets", nargs="+", help="source or source:YYYY-MM-DD")
    parser.add_argument("--replays", type=int, default=2, help="times each partition is sent")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        wanted = [parse_target(t) for t in args.targets]
    except ValueError as exc:
        parser.error(str(exc))

    config = kafka_config()
    ensure_topics(AdminClient(config))
    publisher = Publisher(config)
    consumer = group_consumer()
    try:
        with ClickHouse() as ch:
            apply_schema(ch)
            targets = [verify_target(ch, publisher, consumer, s, dt, args.replays) for s, dt in wanted]
    finally:
        consumer.close()
    matched = sum(t["match"] for t in targets)
    metrics: dict[str, dict[str, Any]] = {
        "targets_matched": {"value": matched, "n": matched, "denominator": len(targets)},
        "rows_parquet_total": {"value": sum(t["rows_parquet"] for t in targets)},
        "rows_clickhouse_final_total": {"value": sum(t["rows_clickhouse_final"] for t in targets)},
    }
    path = write_result(
        RESULT_PATH,
        metrics,
        source_script="src/radar/stream/verify.py",
        method=(
            "each partition replayed from Parquet to Redpanda `replays` times; after the ClickHouse consumer group "
            "committed up to the end of the topic, rows of the source and day counted in raw.vacancies with FINAL"
        ),
        targets=targets,
    )
    log.info("results: %s; %d of %d targets match", path, matched, len(targets))
    return 0 if matched == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
