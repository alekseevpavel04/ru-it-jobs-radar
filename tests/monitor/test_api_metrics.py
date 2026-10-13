import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from radar.monitor.api_metrics import instrument


def make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/items/{item_id}")
    def item(item_id: int) -> dict[str, int]:
        if item_id == 0:
            raise HTTPException(status_code=404)
        return {"id": item_id}

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    instrument(app)
    return app


def samples(client: TestClient, name: str) -> dict[tuple[tuple[str, str], ...], float]:
    text = client.get("/metrics").text
    return {
        tuple(sorted(s.labels.items())): s.value
        for family in text_string_to_metric_families(text)
        for s in family.samples
        if s.name == name
    }


def test_requests_are_labelled_by_route_template_and_status() -> None:
    client = TestClient(make_app(), raise_server_exceptions=False)
    for item_id in (1, 2, 0):
        client.get(f"/items/{item_id}")
    client.get("/nowhere")
    assert client.get("/boom").status_code == 500

    counts = samples(client, "radar_api_requests_total")
    assert counts == {
        (("method", "GET"), ("route", "/items/{item_id}"), ("status", "200")): 2.0,
        (("method", "GET"), ("route", "/items/{item_id}"), ("status", "404")): 1.0,
        (("method", "GET"), ("route", "unmatched"), ("status", "404")): 1.0,
        (("method", "GET"), ("route", "/boom"), ("status", "500")): 1.0,
    }
    latency = samples(client, "radar_api_request_duration_seconds_count")
    assert latency[(("method", "GET"), ("route", "/items/{item_id}"))] == 3.0
    assert samples(client, "radar_api_requests_in_progress") == {(("method", "GET"),): 0.0}


def test_scrapes_are_not_counted_and_apps_do_not_share_metrics() -> None:
    first, second = TestClient(make_app()), TestClient(make_app())
    first.get("/items/1")
    for _ in range(3):
        first.get("/metrics")
    assert sum(samples(first, "radar_api_requests_total").values()) == 1.0
    assert samples(second, "radar_api_requests_total") == {}


def test_metrics_endpoint_serves_the_text_format() -> None:
    response = TestClient(make_app()).get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "/metrics" not in TestClient(make_app()).get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("path", ["/items/1", "/nowhere"])
def test_response_body_passes_through(path: str) -> None:
    plain = FastAPI()

    @plain.get("/items/{item_id}")
    def item(item_id: int) -> dict[str, int]:
        return {"id": item_id}

    expected = TestClient(plain).get(path)
    got = TestClient(make_app()).get(path)
    assert (got.status_code, got.json()) == (expected.status_code, expected.json())
