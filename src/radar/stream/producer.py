"""Publishing complete raw partitions to Redpanda.

Parquet stays the source of truth: a partition is published whole, as often as needed, and the
ClickHouse side keeps one row per key, so a replay never doubles anything. Only partitions with a
`_SUCCESS` marker are published; one still being written is someone else's file.
"""

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, get_args

from confluent_kafka import KafkaError, KafkaException, Producer
from confluent_kafka import Message as KafkaMessage
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic

from radar.collect.envelope import Source
from radar.collect.writer import is_complete, partition_dir
from radar.config import ConfigError, require
from radar.stream.messages import Message, iter_records, to_message, topic_for

log = logging.getLogger("radar.stream.producer")

SOURCES: tuple[str, ...] = get_args(Source)
METADATA_TIMEOUT_S = 10.0
FLUSH_TIMEOUT_S = 300.0


class PublishError(RuntimeError):
    pass


def kafka_config() -> dict[str, Any]:
    return {
        "bootstrap.servers": require("REDPANDA_BROKERS"),
        # The broker advertises `localhost`; from Windows only IPv4 is forwarded into WSL, and every
        # attempt at ::1 costs a two-second connect timeout.
        "broker.address.family": "v4",
        # librdkafka writes to stderr otherwise.
        "logger": logging.getLogger("radar.stream.kafka"),
    }


def ensure_topics(admin: AdminClient, sources: tuple[str, ...] = SOURCES) -> list[str]:
    """Create the missing `raw.<source>` topics; returns the ones created.

    One partition each: a day is at most a snapshot of about 550 thousand records, and a single
    partition keeps the order of a source's records.
    """
    existing = admin.list_topics(timeout=METADATA_TIMEOUT_S).topics
    missing = [NewTopic(topic_for(s), num_partitions=1, replication_factor=1) for s in sources]
    missing = [t for t in missing if t.topic not in existing]
    if not missing:
        return []
    for topic, future in admin.create_topics(missing).items():
        try:
            future.result()
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
        log.info("topic %s created", topic)
    return [t.topic for t in missing]


class Publisher:
    """An idempotent, compressed producer that fails loudly if any message was not delivered."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.producer = Producer(
            {
                **(config or kafka_config()),
                "client.id": "radar-publisher",
                "enable.idempotence": True,
                "compression.type": "zstd",
                "linger.ms": 100,
                "message.timeout.ms": 120_000,
            }
        )
        self.delivered = 0
        self.failed = 0
        self.first_error: KafkaError | None = None

    def _on_delivery(self, error: KafkaError | None, _message: KafkaMessage) -> None:
        if error is None:
            self.delivered += 1
            return
        self.failed += 1
        self.first_error = self.first_error or error

    def send(self, message: Message) -> None:
        while True:
            try:
                self.producer.produce(
                    message.topic,
                    value=message.value,
                    key=message.key,
                    headers=message.headers,
                    on_delivery=self._on_delivery,
                )
                break
            except BufferError:
                # The local queue is full: wait for deliveries to free it.
                self.producer.poll(0.5)
        self.producer.poll(0)

    def flush(self) -> None:
        left = self.producer.flush(FLUSH_TIMEOUT_S)
        if left or self.failed:
            raise PublishError(f"{self.failed} messages failed, {left} undelivered; first error: {self.first_error}")


def publish_partition(source: str, dt: date, send: Callable[[Message], None], root: Path | None = None) -> int:
    if not is_complete(source, dt, root):
        raise PublishError(f"{source}/dt={dt} has no _SUCCESS marker; only complete partitions are published")
    count = 0
    for record in iter_records(partition_dir(source, dt, root)):
        send(to_message(record, dt))
        count += 1
    return count


def publish(source: str, dt: date, root: Path | None = None) -> int:
    config = kafka_config()
    ensure_topics(AdminClient(config), (source,))
    publisher = Publisher(config)
    count = publish_partition(source, dt, publisher.send, root)
    publisher.flush()
    log.info("published %s/dt=%s: %d messages to %s", source, dt, count, topic_for(source))
    return count


def publish_after_collect(source: str, dt: date, root: Path | None = None) -> bool:
    """The collectors' last step. The partition is already on disk, so a stream that is down (or not
    configured, as in CI) costs a warning, not the run; a replay sends the partition later.
    """
    try:
        publish(source, dt, root)
    except (KafkaException, PublishError, ConfigError) as exc:
        log.warning(
            "%s/dt=%s is on disk but not in the stream (%s); later: python -m radar.stream.replay --source %s --dt %s",
            source,
            dt,
            exc,
            source,
            dt,
        )
        return False
    return True
