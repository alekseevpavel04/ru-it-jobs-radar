from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from radar.monitor.exporter import QUERIES, DataCollector
from radar.stream.clickhouse import ClickHouseError

ROWS: dict[str, list[dict[str, Any]]] = {
    "raw": [
        {"source": "habr", "latest_dt": 1791763200, "n_rows": "5710"},
        {"source": "telegram", "latest_dt": 1791676800, "n_rows": "363821"},
    ],
    "raw_errors": [],
    "kafka": [{"table": "raw.vacancies_queue", "messages": "482699", "last_poll": 1791877601, "exceptions": 0}],
    "extract": [
        {"model_id": "m-awq", "source": "habr", "n_rows": "1258", "n_valid": "1258", "last_created": 1791877000.5},
        {
            "model_id": "m-awq",
            "source": "telegram",
            "n_rows": "37236",
            "n_valid": "37161",
            "last_created": 1791877488.9,
        },
    ],
    "tables": [{"database": "marts", "table": "mart_salaries", "n_rows": "993", "modified": 1791874394}],
    "metrics": [{"name": "MemoryTracking", "value": "248312901"}],
    "events": [{"name": "Query", "value": "551"}],
}


def scrape(query: Any) -> dict[str, dict[tuple[tuple[str, str], ...], float]]:
    registry = CollectorRegistry()
    registry.register(DataCollector(query))
    out: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for family in text_string_to_metric_families(generate_latest(registry).decode()):
        for s in family.samples:
            out.setdefault(s.name, {})[tuple(sorted(s.labels.items()))] = s.value
    return out


def fake_query(sql: str) -> list[dict[str, Any]]:
    (name,) = [k for k, v in QUERIES.items() if v == sql]
    return ROWS[name]


def test_data_metrics_from_query_rows() -> None:
    got = scrape(fake_query)
    assert got["radar_clickhouse_up"] == {(): 1.0}
    assert got["radar_raw_latest_partition_timestamp_seconds"] == {
        (("source", "habr"),): 1791763200.0,
        (("source", "telegram"),): 1791676800.0,
    }
    assert got["radar_raw_rows"][(("source", "telegram"),)] == 363821.0
    assert got["radar_kafka_consumer_messages_read_total"] == {(("table", "raw.vacancies_queue"),): 482699.0}
    assert got["radar_extract_valid_rows"][(("model_id", "m-awq"), ("source", "telegram"))] == 37161.0
    # The newest answer over the sources of a model.
    assert got["radar_extract_last_created_timestamp_seconds"] == {(("model_id", "m-awq"),): 1791877488.9}
    assert got["radar_table_rows"] == {(("database", "marts"), ("table", "mart_salaries")): 993.0}
    assert got["ClickHouseMetrics_MemoryTracking"] == {(): 248312901.0}
    assert got["ClickHouseProfileEvents_Query_total"] == {(): 551.0}
    assert set(got["radar_exporter_query_seconds"]) == {(("query", name),) for name in QUERIES}
    assert "radar_raw_error_rows" not in got


def test_unreachable_server_reports_down_only() -> None:
    def failing(sql: str) -> list[dict[str, Any]]:
        raise ClickHouseError("Code: 516. Authentication failed")

    assert scrape(failing) == {"radar_clickhouse_up": {(): 0.0}}
