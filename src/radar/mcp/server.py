"""The MCP tools. Each reads, none writes: the analyst and the vacancy search query ClickHouse as the
read-only account, and the salary estimate goes through the same function as POST /salary.

- ask_market: a question in Russian to the LangGraph analyst (radar.agent), answered with numbers
  and the SQL that produced them, by the local LLM at LLM_BASE_URL.
- estimate_salary: the salary model on extraction fields given as arguments.
- find_vacancies: open vacancies of marts.mart_open_vacancies by filters, with a LIMIT.

The ClickHouse client and the analyst are built on first use, so the server starts (and lists its
tools) without ClickHouse or the LLM, and a tool whose back end is down answers with an error naming
it. The salary model is loaded at start.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Protocol

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field, ValidationError

from radar.agent.clickhouse import QueryError, ReadOnlyClickHouse
from radar.agent.graph import Answer
from radar.agent.llm import LLMError
from radar.api import stats, vacancies
from radar.api.app import Warehouse, load_salary, salary_body
from radar.api.models import Grade, RegionGroup, SalaryRequest, SalaryResponse, SalarySource, WorkFormat
from radar.api.salary import Estimator, OutOfScope, UnknownSource
from radar.config import ConfigError
from radar.extract.schema import EmploymentType, Experience, Role, VacancyFields
from radar.extract.schema import Grade as SchemaGrade
from radar.extract.schema import WorkFormat as SchemaWorkFormat

ANSWER_ROWS = 30
INSTRUCTIONS = (
    "Data on IT vacancies in Russia (trudvsem, Habr Career, SuperJob, Telegram), collected daily. "
    "ask_market answers a question about the market from the marts with the SQL it ran; "
    "estimate_salary gives a monthly RUB estimate with an interval for a vacancy description; "
    "find_vacancies lists open vacancies by filters. Amounts are as the sources state them."
)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


class Analyst(Protocol):
    def ask(self, question: str) -> Answer: ...


@dataclass(frozen=True)
class Backends:
    salary: Callable[[], Estimator | None]
    warehouse: Callable[[], Warehouse]
    analyst: Callable[[], Analyst]


def live_analyst() -> Analyst:
    from radar.agent.evaluate import make_agent
    from radar.agent.llm import LocalChat
    from radar.agent.profiles import live

    chat = LocalChat()
    chat.check_model()
    return make_agent(chat, ReadOnlyClickHouse(timeout_s=30.0), live(), compose_answer=True)


def live_backends(salary_model: Path) -> Backends:
    # Loaded at start, as the API does, so a restart is when both pick up a retrained model.
    estimator = load_salary(salary_model)
    return Backends(
        salary=lambda: estimator,
        warehouse=functools.cache(lambda: ReadOnlyClickHouse(timeout_s=15.0)),
        analyst=functools.cache(live_analyst),
    )


def _warehouse_error(e: Exception) -> ToolError:
    if isinstance(e, QueryError):
        return ToolError(f"the warehouse refused the query (code {e.code})")
    return ToolError(f"the warehouse is unreachable ({type(e).__name__})")


def build_server(backends: Backends) -> MCPServer:
    server = MCPServer("radar", instructions=INSTRUCTIONS, version=version("radar"))

    @server.tool(annotations=READ_ONLY)
    def ask_market(
        question: Annotated[str, Field(min_length=3, max_length=500, description="a question in Russian")],
    ) -> dict[str, Any]:
        """Answer a question about the Russian IT job market from the marts: the analyst writes one
        SELECT over marts.*, runs it as a read-only account and answers with the numbers and the SQL.
        Refuses questions the marts cannot answer and anything but reading."""
        try:
            answer = backends.analyst().ask(question)
        except (LLMError, httpx.HTTPError) as e:
            raise ToolError(f"the local LLM or the warehouse did not answer ({type(e).__name__})") from e
        except QueryError as e:
            raise _warehouse_error(e) from e
        except ConfigError as e:
            raise ToolError(str(e)) from e
        result = answer.result
        return {
            "status": answer.status,
            "answer": answer.answer,
            "sql": answer.sql,
            "columns": result.columns if result else [],
            "rows": result.rows[:ANSWER_ROWS] if result else [],
            "rows_total": len(result.rows) if result else 0,
            "attempts": answer.attempts,
        }

    @server.tool(annotations=READ_ONLY)
    def estimate_salary(
        role: Role | None = None,
        grade: SchemaGrade | None = None,
        cities: Annotated[list[Annotated[str, Field(min_length=1, max_length=80)]], Field(max_length=30)] | None = None,
        skills: Annotated[list[Annotated[str, Field(min_length=1, max_length=80)]], Field(max_length=60)] | None = None,
        work_format: SchemaWorkFormat | None = None,
        experience_min_years: Annotated[float, Field(ge=0, le=50)] | None = None,
        employment_type: EmploymentType | None = None,
        source: SalarySource = "trudvsem",
    ) -> SalaryResponse:
        """Monthly salary estimate in RUB for an IT vacancy, with an interval at the model's nominal
        coverage and n_similar, the training vacancies of the same role, grade and region group.
        Fields not given are unknown, not guessed. Same model and code path as POST /salary."""
        estimator = backends.salary()
        if estimator is None:
            raise ToolError("the salary model is not loaded on this server")
        experience = None if experience_min_years is None else Experience(min=experience_min_years, max=None)
        try:
            fields = VacancyFields(
                role=role,
                grade=grade,
                skills=skills or [],
                salary=None,
                work_format=work_format,
                cities=cities or [],
                experience_years=experience,
                employment_type=employment_type,
                lang="ru",
            )
            return salary_body(estimator, SalaryRequest(fields=fields, source=source))
        except (OutOfScope, UnknownSource, ValidationError) as e:
            raise ToolError(str(e)) from e

    @server.tool(annotations=READ_ONLY)
    def find_vacancies(
        source: SalarySource | None = None,
        role_bucket: stats.RoleBucket | None = None,
        grade: Grade | None = None,
        region_group: RegionGroup | None = None,
        work_format: WorkFormat | None = None,
        skill: Annotated[str, Field(min_length=1, max_length=80, description="a skill as the marts name it")]
        | None = None,
        min_salary: Annotated[int, Field(ge=1, le=10_000_000, description="RUB per month")] | None = None,
        limit: Annotated[int, Field(ge=1, le=vacancies.MAX_LIMIT)] = 10,
    ) -> dict[str, Any]:
        """Open IT vacancies matching every given filter, the most recently seen first, with the number
        of open vacancies the filters match. Titles and employer names only where the source's terms
        allow publishing them (trudvsem); a link to the original otherwise."""
        try:
            return vacancies.find(
                backends.warehouse().run,
                source=source,
                role_bucket=role_bucket,
                grade=grade,
                region_group=region_group,
                work_format=work_format,
                skill=skill,
                min_salary=min_salary,
                limit=limit,
            )
        except vacancies.MartMissing as e:
            raise ToolError(str(e)) from e
        except (QueryError, httpx.TransportError) as e:
            raise _warehouse_error(e) from e
        except ConfigError as e:
            raise ToolError(str(e)) from e

    return server
