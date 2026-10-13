"""The main path against the running stack (scripts/e2e.sh brings it up): health, extraction of a
vacancy, market aggregates, the salary estimate of the extracted fields, and the same estimate as an
MCP tool over streamable HTTP. Reads only; the vacancy below is made up.

    bash scripts/e2e.sh                      up the core and api profiles, then these tests
    pytest -m slow tests/api/test_e2e.py     against what is running
"""

import os
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
    with httpx.Client(base_url=API_URL, timeout=60) as client:
        yield client


@pytest.fixture(scope="module")
def extracted(api: httpx.Client) -> dict[str, Any]:
    response = api.post("/extract", json={"text": VACANCY})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def test_health_sees_clickhouse_redis_and_the_salary_model(api: httpx.Client) -> None:
    body = api.get("/health").json()
    assert body["checks"]["clickhouse"] == "ok", body
    assert body["checks"]["redis"] == "ok", body
    assert body["checks"]["salary_model"] == "ok", body
    assert body["salary_model"].startswith("salary-")


def test_extract_reads_the_vacancy(extracted: dict[str, Any]) -> None:
    fields = extracted["fields"]
    assert extracted["backend"] in ("student", "regex") and extracted["schema_version"] == 1
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
