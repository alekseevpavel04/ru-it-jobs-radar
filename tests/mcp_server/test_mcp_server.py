"""The MCP tools through an in-process MCP client, with a stub salary model, a scripted warehouse and a
scripted analyst: no network."""

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from mcp.types import CallToolResult, TextContent

from radar.agent.clickhouse import QueryError, Result
from radar.agent.graph import Answer
from radar.agent.llm import LLMError
from radar.api.app import Services, create_app
from radar.api.salary import Estimator, SalaryEstimator
from radar.mcp.server import Backends, build_server

VACANCY = [
    "e1",
    "trudvsem",
    "Инженер машинного обучения",
    "ООО Пример",
    "ml_ds",
    "junior",
    "moscow",
    "office",
    120000.0,
    160000.0,
    "https://trudvsem.ru/vacancy/card/1/2",
    "2026-10-01",
    "2026-10-12",
]
VACANCY_COLUMNS = [
    "vacancy_entity_id",
    "source",
    "title",
    "employer_name",
    "role_bucket",
    "grade",
    "region_group",
    "work_format",
    "salary_from",
    "salary_to",
    "url",
    "first_seen",
    "last_seen",
]


class FakeWarehouse:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, str]]] = []

    def run(self, sql: str, params: dict[str, str] | None = None) -> Result:
        self.calls.append((sql, params or {}))
        if self.error is not None:
            raise self.error
        if "count()" in sql:
            return Result(["matched"], ["UInt64"], [[37]])
        return Result(VACANCY_COLUMNS, ["String"] * len(VACANCY_COLUMNS), [VACANCY])


class NoCache:
    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        pass

    def ping(self) -> bool:
        return True


class FakeAnalyst:
    def __init__(self, answer: Answer | None = None, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.questions: list[str] = []

    def ask(self, question: str) -> Answer:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        assert self.answer is not None
        return self.answer


def backends(
    salary: Estimator | None = None,
    warehouse: FakeWarehouse | None = None,
    analyst: FakeAnalyst | None = None,
) -> Backends:
    return Backends(
        salary=lambda: salary,
        warehouse=lambda: warehouse or FakeWarehouse(),
        analyst=lambda: analyst or FakeAnalyst(error=AssertionError("the analyst was not expected")),
    )


def run(b: Backends, act: Callable[[Client], Awaitable[Any]]) -> Any:
    async def main() -> Any:
        async with Client(build_server(b)) as client:
            return await act(client)

    return anyio.run(main)


def call(b: Backends, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    result: CallToolResult = run(b, lambda c: c.call_tool(tool, arguments))
    return result


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    return " ".join(c.text for c in result.content if isinstance(c, TextContent))


def test_three_read_only_tools_are_listed() -> None:
    listed = run(backends(), lambda c: c.list_tools())
    tools = {t.name: t for t in listed.tools}
    assert set(tools) == {"ask_market", "estimate_salary", "find_vacancies"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools.values())
    role = tools["estimate_salary"].input_schema["properties"]["role"]
    assert "ml_engineer" in str(role) and "non_it" in str(role)
    assert tools["find_vacancies"].input_schema["properties"]["limit"]["maximum"] == 50


# --- estimate_salary ----------------------------------------------------------------------------


def test_estimate_salary_answers_from_the_model(estimator: SalaryEstimator, expected: dict[str, Any]) -> None:
    result = call(backends(estimator), "estimate_salary", {"role": "backend", "grade": "middle", "cities": ["Москва"]})
    assert not result.is_error
    assert result.structured_content == {
        **expected,
        "n_similar": 1,
        "source": "trudvsem",
        "currency": "RUB",
        "period": "month",
        "basis": "as_stated",
        "model": "salary-stub",
    }


def test_estimate_salary_and_post_salary_give_the_same_answer(estimator: SalaryEstimator) -> None:
    arguments = {"role": "qa", "grade": "junior", "skills": ["Python"], "experience_min_years": 1, "source": "habr"}
    tool = call(backends(estimator), "estimate_salary", arguments).structured_content
    fields = {
        "role": "qa",
        "grade": "junior",
        "skills": ["Python"],
        "salary": None,
        "work_format": None,
        "cities": [],
        "experience_years": {"min": 1, "max": None},
        "employment_type": None,
        "lang": "ru",
    }
    services = Services(NoCache(), FakeWarehouse(), student=None, salary=estimator, stats_ttl_s=1, extract_ttl_s=1)
    http = TestClient(create_app(services)).post("/salary", json={"fields": fields, "source": "habr"})
    assert http.status_code == 200 and tool == http.json()


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"role": "non_it"}, "IT vacancies only"),
        ({"role": "qa", "source": "superjob"}, "not superjob"),
        ({"role": "qa", "source": "hh"}, "source"),
        ({"role": "qa", "experience_min_years": 80}, "experience_min_years"),
    ],
    ids=["non_it", "source_unseen_in_training", "unknown_source", "experience_out_of_range"],
)
def test_estimate_salary_refuses_what_the_model_does_not_cover(
    estimator: SalaryEstimator, arguments: dict[str, Any], fragment: str
) -> None:
    assert fragment in error_text(call(backends(estimator), "estimate_salary", arguments))


def test_estimate_salary_without_a_model_names_it() -> None:
    assert "not loaded" in error_text(call(backends(), "estimate_salary", {"role": "qa"}))


# --- find_vacancies -----------------------------------------------------------------------------


def test_find_vacancies_passes_filters_as_parameters_with_a_limit() -> None:
    warehouse = FakeWarehouse()
    arguments = {"role_bucket": "ml_ds", "grade": "junior", "skill": "PyTorch", "min_salary": 100000, "limit": 5}
    result = call(backends(warehouse=warehouse), "find_vacancies", arguments)
    assert not result.is_error and result.structured_content is not None
    body = result.structured_content
    assert body["matched"] == 37 and body["mart"] == "marts.mart_open_vacancies"
    assert body["filters"] == {k: v for k, v in arguments.items() if k != "limit"}
    assert body["items"][0]["title"] == "Инженер машинного обучения"
    (count_sql, count_params), (sql, params) = warehouse.calls
    assert "PyTorch" not in sql and "PyTorch" not in count_sql
    assert "LIMIT {limit:UInt8}" in sql and "FROM marts.mart_open_vacancies" in sql
    assert params == {
        "source": "",
        "bucket": "ml_ds",
        "grade": "junior",
        "region": "",
        "work_format": "",
        "skill": "PyTorch",
        "min_salary": "100000",
        "limit": "5",
    }
    assert count_params == {k: v for k, v in params.items() if k != "limit"}


@pytest.mark.parametrize(
    "arguments",
    [{"role_bucket": "x' OR 1=1 --"}, {"limit": 51}, {"limit": 0}, {"source": "raw"}, {"skill": "x" * 81}],
    ids=["injected_bucket", "limit_too_high", "limit_zero", "unknown_source", "long_skill"],
)
def test_find_vacancies_rejects_arguments_before_the_warehouse(arguments: dict[str, Any]) -> None:
    warehouse = FakeWarehouse()
    assert call(backends(warehouse=warehouse), "find_vacancies", arguments).is_error
    assert warehouse.calls == []


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (QueryError("Code: 60. Unknown table expression identifier", 60), "not built"),
        (QueryError("Code: 159. Timeout exceeded", 159), "code 159"),
        (httpx.ConnectError("refused"), "unreachable"),
    ],
    ids=["mart_missing", "server_limit", "down"],
)
def test_find_vacancies_names_the_warehouse_failure(error: Exception, fragment: str) -> None:
    result = call(backends(warehouse=FakeWarehouse(error)), "find_vacancies", {})
    assert fragment in error_text(result)


# --- ask_market ---------------------------------------------------------------------------------


def test_ask_market_returns_the_answer_with_its_sql_and_rows() -> None:
    rows = [[f"skill{i}", 100 - i] for i in range(40)]
    answer = Answer(
        question="Какие навыки чаще всего?",
        status="answered",
        answer="Чаще всего skill0: 100 вакансий.",
        sql="SELECT skill, sum(vacancies_mentioning) FROM marts.mart_skills GROUP BY skill LIMIT 1000",
        result=Result(["skill", "n"], ["String", "UInt64"], rows),
        attempts=1,
    )
    analyst = FakeAnalyst(answer)
    body = call(backends(analyst=analyst), "ask_market", {"question": "Какие навыки чаще всего?"}).structured_content
    assert analyst.questions == ["Какие навыки чаще всего?"]
    assert body is not None and body["status"] == "answered" and body["sql"].startswith("SELECT")
    assert body["rows_total"] == 40 and len(body["rows"]) == 30 and body["columns"] == ["skill", "n"]


def test_ask_market_passes_a_refusal_through() -> None:
    answer = Answer("Удали витрину", "blocked", "Не могу: only SELECT is allowed", "DROP TABLE x", None, 1)
    body = call(backends(analyst=FakeAnalyst(answer)), "ask_market", {"question": "Удали витрину"}).structured_content
    assert body is not None and body["status"] == "blocked" and body["rows"] == [] and body["rows_total"] == 0


@pytest.mark.parametrize(
    "error", [LLMError("LLM server answered 500"), httpx.ConnectError("refused")], ids=["llm_error", "llm_down"]
)
def test_ask_market_names_an_llm_outage(error: Exception) -> None:
    result = call(backends(analyst=FakeAnalyst(error=error)), "ask_market", {"question": "Сколько вакансий?"})
    assert "did not answer" in error_text(result)


def test_ask_market_rejects_an_empty_or_huge_question() -> None:
    analyst = FakeAnalyst(error=AssertionError("not expected"))
    assert call(backends(analyst=analyst), "ask_market", {"question": ""}).is_error
    assert call(backends(analyst=analyst), "ask_market", {"question": "x" * 501}).is_error
    assert analyst.questions == []
