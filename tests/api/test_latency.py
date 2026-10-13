import json
from typing import Any

import httpx
import pandas as pd
import pytest

from radar.api.latency import NotTimed, measure, nearest_rank, sample, student_service, summary


def test_nearest_rank() -> None:
    xs = [float(i) for i in range(1, 21)]
    assert nearest_rank(xs, 0.5) == 10.0 and nearest_rank(xs, 0.95) == 19.0 and nearest_rank([3.0], 0.95) == 3.0


def test_sample_keeps_train_pool_trudvsem_one_per_group() -> None:
    corpus = pd.DataFrame(
        {
            "id": ["trudvsem:a", "trudvsem_snapshot:a", "trudvsem:b", "habr:c", "trudvsem:d", "trudvsem:e"],
            "source": ["trudvsem", "trudvsem_snapshot", "trudvsem", "habr", "trudvsem", "trudvsem"],
            "group_id": ["g1", "g1", "g2", "g3", "g_dev", "g_held"],
            "input": ["x"] * 6,
        }
    )
    split = pd.DataFrame(
        {
            "group_id": ["g1", "g2", "g3", "g_dev", "g_held"],
            "split": ["train_pool", "train_pool", "train_pool", "dev", "withheld"],
        }
    )
    picked = sample(corpus, split, 10, "salt")
    assert sorted(picked["group_id"]) == ["g1", "g2"]
    assert sample(corpus, split, 1, "salt")["id"].tolist() == picked["id"].tolist()[:1]


def api(answers: list[dict[str, Any]]) -> httpx.Client:
    replies = iter(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(replies))

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle))


STUDENT = {"cached": False, "backend": "student", "fallback_reason": None, "truncated_input": False, "model": "m"}
RECORDS = pd.DataFrame({"id": ["w", "a", "b"], "input": ["warm", "aa", "bbb"]})


def test_measure_skips_the_warmup_and_summarizes() -> None:
    calls = measure(api([STUDENT, STUDENT, {**STUDENT, "truncated_input": True}]), RECORDS)
    assert [c["id"] for c in calls] == ["a", "b"] and calls[1]["input_chars"] == 3
    metrics = summary(calls)
    assert metrics["requests"]["value"] == 2
    assert metrics["truncated_inputs"] == {"value": 0.5, "n": 1, "denominator": 2}
    json.dumps(metrics)


@pytest.mark.parametrize(
    "answer",
    [{**STUDENT, "cached": True}, {**STUDENT, "backend": "regex", "fallback_reason": "/tokenize: ConnectError"}],
    ids=["cached", "regex"],
)
def test_an_answer_that_does_not_time_the_student_stops_the_run(answer: dict[str, Any]) -> None:
    with pytest.raises(NotTimed):
        measure(api([STUDENT, answer, STUDENT]), RECORDS)


def test_student_service_is_read_from_compose() -> None:
    service = student_service()
    assert service["image"].startswith("ghcr.io/ggml-org/llama.cpp:server-b")
    assert service["threads"] >= 1 and service["parallel"] >= 1 and service["alias"].endswith("Q4_K_M")
