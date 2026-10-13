import json
from typing import Any

import httpx
import pandas as pd
import pytest

from radar.api import match_latency
from radar.api.match_latency import cgroup_memory, measure, sample, summary

QUERIES = pd.DataFrame(
    {
        "query_id": ["q1", "q2", "q3", "q4"],
        "split": ["dev", "test", "dev", "dev"],
        "resume": ["Python, SQL", "Go", "PyTorch", "React"],
    }
)


def test_sample_takes_dev_resumes_in_a_salted_order() -> None:
    picked = sample(QUERIES, 10, "salt")
    assert sorted(picked["query_id"]) == ["q1", "q3", "q4"]
    assert sample(QUERIES, 2, "salt")["query_id"].tolist() == picked["query_id"].tolist()[:2]


def test_measure_skips_the_warmup() -> None:
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"items": [{"rank": 1}], "model": "match-x"})

    client = httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle))
    calls = measure(client, QUERIES.head(3), k=5)
    assert [c["query_id"] for c in calls] == ["q2", "q3"] and calls[0]["items"] == 1
    assert bodies[0] == {"resume_text": "Python, SQL", "k": 5}


def test_a_failed_match_stops_the_run() -> None:
    client = httpx.Client(base_url="http://api", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        measure(client, QUERIES.head(2), k=5)


def test_cgroup_memory_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    out = "629145600\n734003200\n1073741824\nanon 524288000\nfile 104857600"
    monkeypatch.setattr(match_latency, "docker_host", lambda *command: out)
    memory = cgroup_memory("abc")
    assert memory == {"current": 629145600, "peak": 734003200, "limit": 1073741824, "anon": 524288000}
    metrics = summary([{"ms": float(i)} for i in range(1, 11)], memory, 2_180_000_000)
    assert metrics["match_latency_p50_ms"]["value"] == 5.0 and metrics["match_latency_p95_ms"]["value"] == 10.0
    assert metrics["container_memory_current_mib"]["value"] == 600.0
    assert metrics["container_memory_limit_mib"]["value"] == 1024.0 and metrics["image_size_mb"]["value"] == 2180.0
