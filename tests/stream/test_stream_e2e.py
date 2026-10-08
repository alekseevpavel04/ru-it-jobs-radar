"""Parquet to Redpanda to ClickHouse on real partitions: every row of a day arrives, a replay doubles none.

Needs the core stack (scripts/up.sh core) and data/raw, so it is marked slow and stays out of
scripts/check.sh: `pytest -m slow tests/stream`. The results file comes from
`python -m radar.stream.verify`, not from this test.
"""

from collections.abc import Iterator
from datetime import date

import pytest
from confluent_kafka import Consumer
from confluent_kafka.admin import AdminClient

from radar.config import load_dotenv
from radar.stream.clickhouse import ClickHouse, apply_schema
from radar.stream.producer import Publisher, ensure_topics, kafka_config
from radar.stream.replay import complete_dates
from radar.stream.verify import group_consumer, verify_target

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stream() -> Iterator[tuple[ClickHouse, Publisher, Consumer]]:
    load_dotenv()
    ensure_topics(AdminClient(kafka_config()))
    consumer = group_consumer()
    with ClickHouse() as ch:
        apply_schema(ch)
        yield ch, Publisher(), consumer
    consumer.close()


def _day(source: str, earliest: bool = False) -> date:
    days = complete_dates(source)
    if not days:
        pytest.skip(f"no complete partitions of {source}")
    return days[0] if earliest else days[-1]


@pytest.mark.parametrize(("source", "earliest"), [("trudvsem", False), ("habr", False), ("trudvsem_snapshot", True)])
def test_day_arrives_whole_after_two_replays(
    stream: tuple[ClickHouse, Publisher, Consumer], source: str, earliest: bool
) -> None:
    ch, publisher, consumer = stream
    result = verify_target(ch, publisher, consumer, source, _day(source, earliest), replays=2)
    assert result["error_messages"] == 0
    assert result["rows_parquet"] == result["distinct_keys_parquet"] == result["rows_clickhouse_final"]
