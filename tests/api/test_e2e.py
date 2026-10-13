"""The main path against the running stack (scripts/e2e.sh brings it up): health, extraction of a
vacancy by the student, market aggregates, the salary estimate of the extracted fields, open vacancies
matched to a resume, and the same estimate as an MCP tool over streamable HTTP. Reads only; the
vacancy below is made up, and a code unique to the run keeps its extraction out of the Redis cache,
so the student really answers it.

    bash scripts/e2e.sh                      up the core and api profiles, then these tests
    pytest -m slow tests/api/test_e2e.py     against what is running
"""

import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

import anyio
import httpx
import pytest
from mcp import Client

pytestmark = pytest.mark.slow

API_URL = os.environ.get("RADAR_API_URL", "http://127.0.0.1:8000")
MCP_URL = os.environ.get("RADAR_MCP_URL", "http://127.0.0.1:8001/mcp")
VACANCY = """Junior ML-инженер (компьютерное зрение)

Зарплата от 120 000 до 160 000 руб. до вычета НДФЛ.
Москва, гибридный формат: два дня в офисе.
Опыт работы от 1 года. Python, PyTorch, OpenCV, SQL, Docker.
Обучаем модели детекции дефектов на производственной линии и выводим их в сервис."""


@pytest.fixture(scope="module")
def api() -> Iterator[httpx.Client]:
    with httpx.Client(base_url=API_URL, timeout=150) as client:
        yield client


@pytest.fixture(scope="module")
def extracted(api: httpx.Client) -> dict[str, Any]:
    text = f"{VACANCY}\nКод вакансии: {uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    response = api.post("/extract", json={"text": text})
    seconds = time.perf_counter() - started
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return {**body, "seconds": seconds}


def test_health_sees_clickhouse_redis_the_student_and_the_salary_model(api: httpx.Client) -> None:
    body = api.get("/health").json()
    assert body["status"] == "ok", body
    assert body["extract_backend"] == "student", body
    assert body["salary_model"].startswith("salary-")
    assert body["match_model"].startswith("match-")


def test_extract_is_answered_by_the_student(extracted: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert extracted["backend"] == "student", extracted["fallback_reason"]
    assert extracted["cached"] is False and extracted["fallback_reason"] is None
    with capsys.disabled():
        print(f"\n/extract: {extracted['model']} answered in {extracted['seconds']:.2f} s (uncached)")


def test_extract_reads_the_vacancy(extracted: dict[str, Any]) -> None:
    fields = extracted["fields"]
    assert extracted["schema_version"] == 1
    assert fields["salary"]["from"] == 120000 and fields["salary"]["to"] == 160000
    assert fields["grade"] == "junior"
    assert "Python" in fields["skills"]


@pytest.mark.parametrize(
    ("path", "params", "key"),
    [
        ("/stats/weekly", {"source": "trudvsem", "weeks": 4}, "weeks"),
        ("/stats/salaries", {"source": "trudvsem", "level": "group"}, "rows"),
        ("/stats/skills", {"source": "habr", "limit": 5}, "skills"),
    ],
)
def test_stats_answer_from_the_marts_then_from_the_cache(
    api: httpx.Client, path: str, params: dict[str, Any], key: str
) -> None:
    first = api.get(path, params=params)
    assert first.status_code == 200, first.text
    assert first.json()[key], f"{path} returned no {key}"
    assert api.get(path, params=params).json()["cached"] is True


def test_salary_estimates_the_extracted_fields(api: httpx.Client, extracted: dict[str, Any]) -> None:
    response = api.post("/salary", json={"fields": extracted["fields"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 < body["low"] <= body["point"] <= body["high"]
    assert body["n_similar"] >= 0 and body["model"].startswith("salary-")


def test_match_lists_open_vacancies_under_the_publication_rules(api: httpx.Client) -> None:
    resume = "Junior ML-инженер, опыт 1 год. Python, PyTorch, OpenCV, SQL, Docker. Москва, гибрид."
    response = api.post("/match", json={"resume_text": resume, "k": 10})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"].startswith("match-") and body["items"]
    assert [i["rank"] for i in body["items"]] == list(range(1, len(body["items"]) + 1))
    for item in body["items"]:
        if item["source"] != "trudvsem":
            assert item["title"] is None and item["employer_name"] is None, item["vacancy_entity_id"]
        if item["source"] == "superjob":
            assert item["url"] is None, item["vacancy_entity_id"]


def test_metrics_count_the_requests_of_this_run(api: httpx.Client) -> None:
    response = api.get("/metrics")
    assert response.status_code == 200
    assert 'radar_api_requests_total{method="GET",route="/health",status="200"}' in response.text
    assert 'radar_api_request_duration_seconds_count{method="POST",route="/extract"}' in response.text


def test_mcp_estimate_salary_matches_post_salary(api: httpx.Client) -> None:
    arguments = {"role": "data_scientist", "grade": "middle", "cities": ["Москва"], "skills": ["Python", "SQL"]}

    async def session() -> tuple[set[str], Any]:
        async with Client(MCP_URL, read_timeout_seconds=60) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            return tools, await client.call_tool("estimate_salary", arguments)

    tools, result = anyio.run(session)
    assert tools == {"ask_market", "estimate_salary", "find_vacancies"}
    assert not result.is_error, result.content
    fields = {
        "role": "data_scientist",
        "grade": "middle",
        "skills": ["Python", "SQL"],
        "salary": None,
        "work_format": None,
        "cities": ["Москва"],
        "experience_years": None,
        "employment_type": None,
        "lang": "ru",
    }
    assert result.structured_content == api.post("/salary", json={"fields": fields}).json()
