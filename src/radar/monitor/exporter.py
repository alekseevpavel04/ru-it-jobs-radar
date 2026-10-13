"""Prometheus exporter of the data itself: freshness per source, tables, extraction progress.

Every scrape runs a few aggregate queries on ClickHouse: 0.4 s together with 55 million raw rows on
a quiet server, 4.5 s while the corpus was loading (the raw query reads the 2-byte `dt` column and
nothing wider). Prometheus scrapes this once a minute with a 30 s timeout.

- `radar_raw_latest_partition_timestamp_seconds{source}`: the newest raw partition `dt` of a
  source, as midnight UTC of that day. Freshness is `time() - this`.
- `radar_raw_rows{source}`: rows stored in raw.vacancies (before ReplacingMergeTree folds replays,
  so a replay raises it until the merge).
- `radar_raw_error_rows{topic}`: messages the consumer could not parse (raw.vacancies_errors).
- `radar_kafka_consumer_messages_read_total{table}`, `radar_kafka_consumer_last_poll_timestamp_seconds`,
  `radar_kafka_consumer_exceptions`: the Kafka engine table reading Redpanda (system.kafka_consumers);
  the counter restarts with the server, which rate() absorbs.
- `radar_extract_rows{model_id,source}`, `radar_extract_valid_rows{...}`,
  `radar_extract_last_created_timestamp_seconds{model_id}`: progress of the corpus extraction
  (extract.vacancy_fields, rows stored, so a rerun counts again until the merge).
- `radar_table_rows{database,table}`, `radar_table_modified_timestamp_seconds{...}`: active parts of
  the layers; dbt rebuilds its tables on every run, so their modification time is the last build.
- `ClickHouseMetrics_*`, `ClickHouseProfileEvents_*_total`: system.metrics and system.events, with
  the prefixes of ClickHouse's own Prometheus endpoint, which is off: it needs a server config change.
- `radar_clickhouse_up`, `radar_exporter_query_seconds{query}`: whether the last scrape got every
  answer and what each query cost.

    python -m radar.monitor.exporter [--port 9188]
"""

import argparse
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from radar.collect import setup_logging
from radar.stream.clickhouse import ClickHouse, ClickHouseError

log = logging.getLogger("radar.monitor.exporter")

PORT = 9188
LAYERS = ("raw", "staging", "marts", "extract", "dedup")

QUERIES = {
    "raw": """
        SELECT source, toUnixTimestamp(toDateTime(max(dt), 'UTC')) AS latest_dt, count() AS n_rows
        FROM raw.vacancies GROUP BY source""",
    "raw_errors": "SELECT topic, count() AS n_rows FROM raw.vacancies_errors GROUP BY topic",
    "kafka": """
        SELECT concat(database, '.', table) AS table, sum(num_messages_read) AS messages,
               toUnixTimestamp(max(last_poll_time)) AS last_poll, sum(length(exceptions.text)) AS exceptions
        FROM system.kafka_consumers GROUP BY database, table""",
    "extract": """
        SELECT model_id, source, count() AS n_rows, countIf(valid) AS n_valid,
               toUnixTimestamp64Milli(max(created_at)) / 1000 AS last_created
        FROM extract.vacancy_fields GROUP BY model_id, source""",
    "tables": f"""
        SELECT database, table, sum(rows) AS n_rows, toUnixTimestamp(max(modification_time)) AS modified
        FROM system.parts
        WHERE active AND database IN ({", ".join(f"'{d}'" for d in LAYERS)})
        GROUP BY database, table""",
    "metrics": "SELECT metric AS name, value FROM system.metrics",
    "events": "SELECT event AS name, value FROM system.events",
}

Rows = list[dict[str, Any]]


def family(name: str, doc: str, labels: list[str], rows: Rows, value: str, counter: bool = False) -> Metric:
    out = CounterMetricFamily(name, doc, labels=labels) if counter else GaugeMetricFamily(name, doc, labels=labels)
    for row in rows:
        out.add_metric([str(row[k]) for k in labels], float(row[value]))
    return out


def families(results: dict[str, Rows]) -> Iterator[Metric]:
    raw, kafka, extract, tables = results["raw"], results["kafka"], results["extract"], results["tables"]
    yield family(
        "radar_raw_latest_partition_timestamp_seconds",
        "Newest raw partition dt, midnight UTC",
        ["source"],
        raw,
        "latest_dt",
    )
    yield family("radar_raw_rows", "Rows stored in raw.vacancies, before merges", ["source"], raw, "n_rows")
    yield family("radar_raw_error_rows", "Rows of raw.vacancies_errors", ["topic"], results["raw_errors"], "n_rows")
    yield family(
        "radar_kafka_consumer_messages_read",
        "Messages read by the Kafka engine",
        ["table"],
        kafka,
        "messages",
        counter=True,
    )
    yield family(
        "radar_kafka_consumer_last_poll_timestamp_seconds",
        "Last poll of the Kafka engine",
        ["table"],
        kafka,
        "last_poll",
    )
    yield family(
        "radar_kafka_consumer_exceptions", "Exceptions the consumer keeps in memory", ["table"], kafka, "exceptions"
    )
    yield family("radar_extract_rows", "Rows of extract.vacancy_fields", ["model_id", "source"], extract, "n_rows")
    yield family(
        "radar_extract_valid_rows", "Rows whose answer passed the schema", ["model_id", "source"], extract, "n_valid"
    )
    latest: dict[str, float] = {}
    for row in extract:
        latest[row["model_id"]] = max(latest.get(row["model_id"], 0.0), float(row["last_created"]))
    yield family(
        "radar_extract_last_created_timestamp_seconds",
        "Last extracted answer of the model",
        ["model_id"],
        [{"model_id": k, "value": v} for k, v in latest.items()],
        "value",
    )
    yield family("radar_table_rows", "Rows in active parts", ["database", "table"], tables, "n_rows")
    yield family(
        "radar_table_modified_timestamp_seconds", "Newest active part", ["database", "table"], tables, "modified"
    )
    for row in results["metrics"]:
        yield family(f"ClickHouseMetrics_{row['name']}", "system.metrics", [], [row], "value")
    for row in results["events"]:
        yield family(f"ClickHouseProfileEvents_{row['name']}", "system.events", [], [row], "value", counter=True)


class DataCollector(Collector):
    def __init__(self, query: Callable[[str], Rows]) -> None:
        self.query = query

    def collect(self) -> Iterator[Metric]:
        up = GaugeMetricFamily("radar_clickhouse_up", "1 if the last scrape got every query answered")
        seconds = GaugeMetricFamily("radar_exporter_query_seconds", "Duration of each query", labels=["query"])
        results: dict[str, Rows] = {}
        try:
            for name, sql in QUERIES.items():
                start = time.perf_counter()
                results[name] = self.query(sql)
                seconds.add_metric([name], time.perf_counter() - start)
        except (ClickHouseError, httpx.HTTPError) as e:
            # The scrape still answers, so the outage shows as up=0 rather than as a dead target.
            log.warning("scrape failed: %s", e)
            up.add_metric([], 0.0)
            yield up
            return
        up.add_metric([], 1.0)
        yield up
        yield seconds
        yield from families(results)


def json_rows(ch: ClickHouse) -> Callable[[str], Rows]:
    def query(sql: str) -> Rows:
        # 64-bit integers arrive as JSON strings; float() takes both.
        text = ch.query(sql + "\nFORMAT JSONEachRow")
        return [json.loads(line) for line in text.splitlines() if line]

    return query


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m radar.monitor.exporter", description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args(argv)
    setup_logging()
    registry = CollectorRegistry()
    with ClickHouse(timeout_s=30.0) as ch:
        registry.register(DataCollector(json_rows(ch)))
        start_http_server(args.port, registry=registry)
        log.info("serving on :%d", args.port)
        threading.Event().wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
