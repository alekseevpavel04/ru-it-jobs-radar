"""One raw partition into ClickHouse through Redpanda: the task body of the radar_load DAG.

    /opt/radar/.venv/bin/python /opt/radar/airflow/include/radar_load.py --source habr --dt 2026-10-08 [--force]

Runs in the project's environment, not Airflow's. Exits with SKIP_EXIT_CODE, and the task is
marked skipped, when there is nothing to load: the partition has no `_SUCCESS`, or ClickHouse
already holds it in full (rows with FINAL equal the Parquet rows; --force replays it anyway). So a
rerun or a catch-up over loaded days costs two counts per source. ClickHouse holding more rows of
the day than the partition fails at once: a replay cannot fix that (the key includes content_hash,
so a partition rewritten after loading leaves its old rows behind) and the day needs cleaning by
hand. Otherwise the partition goes through `radar.stream.replay`, the task waits until the
ClickHouse consumer group has committed up to the end of the topic and fails unless the counts
match and no message of this replay landed in raw.vacancies_errors. The last stdout line of a
replay is a JSON summary (Airflow keeps it as the task's XCom; the DAG's hand-over reads it).
"""

import argparse
import json
import logging
import sys
import time
from datetime import date

import pyarrow.parquet as pq
from confluent_kafka import TopicPartition

from radar.collect import setup_logging
from radar.collect.writer import is_complete, partition_dir
from radar.stream import replay
from radar.stream.clickhouse import ClickHouse, rows_final
from radar.stream.messages import topic_for
from radar.stream.producer import SOURCES
from radar.stream.verify import METADATA_TIMEOUT_S, group_consumer, wait_consumed

log = logging.getLogger("radar.airflow.load")

SKIP_EXIT_CODE = 99


def rows_parquet(source: str, dt: date) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in partition_dir(source, dt).glob("part-*.parquet"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--dt", required=True, type=date.fromisoformat, help="partition date, YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="replay even if ClickHouse holds the partition")
    args = parser.parse_args(argv)
    setup_logging()
    source, dt = args.source, args.dt

    if not is_complete(source, dt):
        log.info("%s/dt=%s: no complete partition, skip", source, dt)
        return SKIP_EXIT_CODE

    summary: dict[str, object] = {"source": source, "dt": dt.isoformat(), "rows_parquet": rows_parquet(source, dt)}
    with ClickHouse() as ch:
        before = rows_final(ch, source, dt)
        summary["rows_clickhouse_before"] = before
        if before > summary["rows_parquet"]:
            log.error("%s/dt=%s: ClickHouse holds more rows than the partition: %s", source, dt, summary)
            return 1
        if before == summary["rows_parquet"] and not args.force:
            log.info("%s/dt=%s: ClickHouse already holds all %d rows, skip", source, dt, before)
            return SKIP_EXIT_CODE

        topic = topic_for(source)
        consumer = group_consumer()
        try:
            first_offset = consumer.get_watermark_offsets(TopicPartition(topic, 0), timeout=METADATA_TIMEOUT_S)[1]
            started = time.monotonic()
            if replay.main(["--source", source, "--dt", dt.isoformat()]) != 0:
                raise RuntimeError(f"replay of {source}/dt={dt} failed")
            wait_consumed(consumer, topic)
        finally:
            consumer.close()
        errors = ch.scalar(
            "SELECT count() FROM raw.vacancies_errors WHERE topic = {topic:String} AND kafka_offset >= {first:UInt64}",
            topic=topic,
            first=str(first_offset),
        )
        summary.update(
            replayed=True,
            seconds_until_consumed=round(time.monotonic() - started, 1),
            rows_clickhouse_final=rows_final(ch, source, dt),
            error_messages=errors,
        )

    print(json.dumps(summary))
    if summary["rows_clickhouse_final"] != summary["rows_parquet"] or errors:
        log.error("%s/dt=%s: ClickHouse does not match the partition: %s", source, dt, summary)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
