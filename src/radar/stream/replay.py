"""Replay raw partitions from Parquet into the topic `raw.<source>`.

The same path the collectors take after writing a partition; here it repairs a day that was
collected while Redpanda was down, or refills ClickHouse from scratch. Without `--dt` or `--all`
the latest complete partition is sent.

    python -m radar.stream.replay --source habr [--dt 2026-10-08 | --all]
"""

import argparse
import logging
import sys
import time
from datetime import date

from confluent_kafka.admin import AdminClient

from radar.collect import setup_logging
from radar.collect.summary import partition_dates
from radar.collect.writer import is_complete
from radar.stream.producer import SOURCES, Publisher, ensure_topics, kafka_config, publish_partition

log = logging.getLogger("radar.stream.replay")


def complete_dates(source: str) -> list[date]:
    return [d for d in partition_dates(source) if is_complete(source, d)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--source", required=True, choices=SOURCES)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--dt", type=date.fromisoformat, help="partition date, YYYY-MM-DD")
    which.add_argument("--all", action="store_true", help="every complete partition of the source")
    args = parser.parse_args(argv)
    setup_logging()

    complete = complete_dates(args.source)
    if args.all:
        dates = complete
    elif args.dt is not None:
        dates = [args.dt]
    else:
        dates = complete[-1:]
    if not dates:
        log.error("no complete partitions of %s", args.source)
        return 1

    config = kafka_config()
    ensure_topics(AdminClient(config))
    publisher = Publisher(config)
    for dt in dates:
        started = time.monotonic()
        count = publish_partition(args.source, dt, publisher.send)
        publisher.flush()
        log.info("%s/dt=%s: %d messages in %.1f s", args.source, dt, count, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
