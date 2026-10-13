"""The API with an in-memory cache, a scripted warehouse and a mocked llama-server: no network."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import redis
from fastapi.testclient import TestClient

from radar.agent.clickhouse import QueryError, Result
from radar.api import settings as api_settings
from radar.api.app import Services, create_app, load_salary
from radar.api.cache import RedisCache
from radar.api.extractors import REGEX_MODEL, StudentExtractor, spans_from_pieces
from radar.api.salary import Estimator, OutOfScope, UnknownSource
from radar.config import ROOT, ConfigError
from radar.extract.schema import VacancyFields

TEXT = (
    "Python-разработчик (Middle)\n\nЗарплата от 150 000 до 200 000 руб. на руки. "
    "Удалённая работа, опыт от 3 лет. Стек: Python, PostgreSQL, Docker."
)
META = {"system_prompt": "Extract the fields.", "max_input_tokens": 1536, "prompt_sha256": "abc123"}
ANSWER = {
    "role": "backend",
    "grade": "middle",
    "skills": ["Python", "PostgreSQL", "Docker"],
    "salary": {"from": 150000, "to": 200000, "currency": "RUB", "gross_net": "net", "period": "month"},
    "work_format": "remote",
    "cities": [],
    "experience_years": {"min": 3, "max": None},
    "employment_type": None,
    "lang": "ru",
}


class MemoryCache:
    def __init__(self, up: bool = True) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}
        self.up = up

    def get(self, key: str) -> Any | None:
        return self.store.get(key)

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        self.store[key] = json.loads(json.dumps(value))
        self.ttls[key] = ttl_s

    def ping(self) -> bool:
        return self.up


class FakeWarehouse:
    """Answers by the mart a query reads; records every query with its parameters."""

    def __init__(self, results: dict[str, Result] | None = None, error: Exception | None = None) -> None:
        self.results = results or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, str]]] = []

    def run(self, sql: str, params: dict[str, str] | None = None) -> Result:
        self.calls.append((sql, params or {}))
        if self.error is not None:
            raise self.error
        if sql == "SELECT 1":
            return Result(["1"], ["UInt8"], [[1]])
        return next(r for mart, r in self.results.items() if f"marts.{mart}" in sql)


def llama_server(
    answer: str | None = None, finish: str = "stop", pieces: list[Any] | None = None
) -> tuple[httpx.MockTransport, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        seen.append({"path": request.url.path, **body})
        if request.url.path == "/tokenize":
            tokens = pieces if pieces is not None else [body["content"]]
            return httpx.Response(200, json={"tokens": [{"id": i, "piece": p} for i, p in enumerate(tokens)]})
        content = answer if answer is not None else json.dumps(ANSWER, ensure_ascii=False)
        return httpx.Response(
            200,
            json={
                "model": "qwen3-1.7b-radar-Q4_K_M",
                "choices": [{"message": {"content": content}, "finish_reason": finish}],
            },
        )

    return httpx.MockTransport(handle), seen


def student(transport: httpx.BaseTransport, meta: dict[str, Any] = META) -> StudentExtractor:
    return StudentExtractor("http://student", meta, timeout_s=5.0, transport=transport)


class FakeEstimator:
    version = "salary-test"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[VacancyFields, str]] = []

    def estimate(self, fields: VacancyFields, source: str) -> dict[str, Any]:
        self.calls.append((fields, source))
        if self.error is not None:
            raise self.error
        return {
            "point": 150000,
            "low": 100000,
            "high": 220000,
            "typical_from": 130000,
            "typical_to": 170000,
            "level": 0.8,
            "n_similar": 0,
            "source": source,
            "model_version": "salary-test",
        }


def client(
    cache: MemoryCache | None = None,
    warehouse: FakeWarehouse | None = None,
    extractor: StudentExtractor | None = None,
    salary: Estimator | None = None,
) -> TestClient:
    services = Services(
        cache=cache or MemoryCache(),
        warehouse=warehouse or FakeWarehouse(),
        student=extractor,
        salary=salary,
        stats_ttl_s=3600,
        extract_ttl_s=86400,
    )
    return TestClient(create_app(services), raise_server_exceptions=False)


# --- /extract -----------------------------------------------------------------------------------


def test_regex_answers_without_a_student_and_is_cached_by_hash() -> None:
    cache = MemoryCache()
    api = client(cache)
    first = api.post("/extract", json={"text": TEXT}).json()
    assert first["backend"] == "regex" and first["model"] == REGEX_MODEL and first["schema_version"] == 1
    assert first["fields"]["salary"] == {
        "from": 150000,
        "to": 200000,
        "currency": "RUB",
        "gross_net": "net",
        "period": None,
    }
    assert first["fallback_reason"] is None and first["cached"] is False
    second = api.post("/extract", json={"text": TEXT}).json()
    assert second == {**first, "cached": True}
    [(stored_key, ttl)] = cache.ttls.items()
    assert ttl == 86400 and "Python" not in stored_key and stored_key.startswith("radar:api:v1:extract:")


@pytest.mark.parametrize(
    "body",
    [{"text": "   "}, {"text": ""}, {"text": "x" * 30_001}, {"text": TEXT, "extra": 1}, {}],
    ids=["blank", "empty", "too_long", "extra_key", "missing"],
)
def test_bad_extract_input_is_rejected(body: dict[str, Any]) -> None:
    assert client().post("/extract", json=body).status_code == 422


def test_student_gets_the_training_prompt_and_answers() -> None:
    transport, seen = llama_server()
    answer = client(extractor=student(transport)).post("/extract", json={"text": TEXT}).json()
    assert answer["backend"] == "student" and answer["model"] == "qwen3-1.7b-radar-Q4_K_M"
    assert answer["prompt_sha256"] == "abc123" and answer["truncated_input"] is False
    assert answer["fields"]["experience_years"] == {"min": 3.0, "max": None}
    tokenize, chat = seen
    assert tokenize["path"] == "/tokenize" and tokenize["content"] == TEXT and tokenize["with_pieces"] is True
    assert chat["path"] == "/v1/chat/completions"
    assert chat["messages"] == [
        {"role": "system", "content": "Extract the fields."},
        {"role": "user", "content": TEXT},
    ]
    assert chat["temperature"] == 0 and chat["max_tokens"] == 512
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat["response_format"]["json_schema"]["schema"]["required"][0] == "role"


def test_student_input_is_cut_at_max_input_tokens() -> None:
    pieces = ["Про", "ект ", "на ", "Py", "thon"]
    transport, seen = llama_server(pieces=pieces)
    answer = client(extractor=student(transport, {**META, "max_input_tokens": 3})).post(
        "/extract", json={"text": "".join(pieces)}
    )
    assert answer.json()["truncated_input"] is True
    assert seen[1]["messages"][1]["content"] == "Проект на "


def test_pieces_that_split_a_letter_count_it_on_the_completing_token() -> None:
    ya = "я".encode()
    spans = spans_from_pieces(["ab", [ya[0]], [ya[1]], "c"])
    assert spans == [(0, 2), (2, 2), (2, 3), (3, 4)]


@pytest.mark.parametrize(
    ("answer", "finish", "reason"),
    [
        ('{"role": "backend"}', "stop", "broke the schema"),
        (json.dumps(ANSWER), "length", "cut at max_tokens"),
        ("not json", "stop", "broke the schema"),
    ],
)
def test_bad_student_answer_falls_back_to_regex_uncached(answer: str, finish: str, reason: str) -> None:
    transport, seen = llama_server(answer=answer, finish=finish)
    cache = MemoryCache()
    api = client(cache, extractor=student(transport))
    body = api.post("/extract", json={"text": TEXT}).json()
    assert body["backend"] == "regex" and reason in body["fallback_reason"]
    assert cache.store == {}
    api.post("/extract", json={"text": TEXT})
    assert len(seen) == 4  # the student is asked again


def test_unreachable_student_falls_back_to_regex() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    body = client(extractor=student(httpx.MockTransport(refuse))).post("/extract", json={"text": TEXT}).json()
    assert body["backend"] == "regex" and body["fallback_reason"] == "/tokenize: ConnectError"


def test_pieces_that_do_not_cover_the_input_are_not_trusted() -> None:
    transport, seen = llama_server(pieces=["Python"])
    body = client(extractor=student(transport)).post("/extract", json={"text": TEXT}).json()
    assert body["backend"] == "regex" and "do not add up" in body["fallback_reason"]
    assert [s["path"] for s in seen] == ["/tokenize"]


# --- /stats -------------------------------------------------------------------------------------

WEEKLY = Result(
    [
        "week",
        "days_collected",
        "complete_days",
        "is_first_collected_week",
        "vacancies_active",
        "vacancies_new",
        "vacancies_closed",
        "telegram_posts",
    ],
    ["String", "UInt8", "UInt8", "Bool", "UInt64", "UInt64", "Nullable(UInt64)", "Nullable(UInt64)"],
    [["2026-10-05", 7, 7, False, 5200, 410, 380, None], ["2026-09-28", 7, 6, False, 5100, 390, 350, None]],
)


def test_weekly_reads_the_mart_with_parameters_and_caches() -> None:
    warehouse = FakeWarehouse({"mart_weekly_dynamics": WEEKLY})
    api = client(warehouse=warehouse)
    first = api.get("/stats/weekly", params={"source": "habr", "role_bucket": "ml_ds", "weeks": 2}).json()
    assert [w["week"] for w in first["weeks"]] == ["2026-09-28", "2026-10-05"]
    assert first["weeks"][1]["vacancies_active"] == 5200 and first["cached"] is False
    [(sql, params)] = warehouse.calls
    assert "FROM marts.mart_weekly_dynamics" in sql and "{source:String}" in sql
    assert params == {"source": "habr", "bucket": "ml_ds", "weeks": "2"}
    second = api.get("/stats/weekly", params={"source": "habr", "role_bucket": "ml_ds", "weeks": 2}).json()
    assert second["cached"] is True and len(warehouse.calls) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"source": "hh"},
        {"source": "habr", "role_bucket": "x' OR 1=1 --"},
        {"source": "habr", "weeks": 0},
        {"source": "habr", "weeks": 1000},
        {},
    ],
    ids=["unknown_source", "injected_bucket", "zero_weeks", "too_many_weeks", "no_source"],
)
def test_stats_filters_outside_the_marts_values_never_reach_clickhouse(params: dict[str, Any]) -> None:
    warehouse = FakeWarehouse({"mart_weekly_dynamics": WEEKLY})
    assert client(warehouse=warehouse).get("/stats/weekly", params=params).status_code == 422
    assert warehouse.calls == []


def test_salaries_keep_their_n_and_period() -> None:
    columns = [
        "role_group",
        "grade",
        "vacancies",
        "with_salary_rub",
        "with_salary_other_currency",
        "with_from",
        "salary_from_p25",
        "salary_from_median",
        "salary_from_p75",
        "with_to",
        "salary_to_p25",
        "salary_to_median",
        "salary_to_p75",
        "period_from",
        "period_to",
    ]
    row = ["ml_ds_data", "junior", 120, 60, 2, 55, 80000.0, 100000.0, 130000.0, 40, 120000.0, 150000.0, None]
    result = Result(columns, ["String"] * len(columns), [[*row, "2026-07-01", "2026-10-11"]])
    warehouse = FakeWarehouse({"mart_salaries": result})
    body = client(warehouse=warehouse).get("/stats/salaries", params={"source": "habr"}).json()
    assert body["level"] == "group_grade" and body["period_to"] == "2026-10-11"
    assert body["rows"][0]["vacancies"] == 120 and body["rows"][0]["with_from"] == 55
    assert "period_from" not in body["rows"][0]
    assert warehouse.calls[0][1] == {"source": "habr", "level": "group_grade"}


def test_skills_take_the_denominator_from_the_bucket_sizes() -> None:
    skills = Result(
        ["skill", "category", "vacancies_mentioning", "period_from", "period_to"],
        ["String", "String", "UInt64", "String", "String"],
        [["Python", "language", 300, "2026-07-01", "2026-10-11"], ["SQL", "language", 250, "2026-07-01", "2026-10-11"]],
    )
    base = Result(["vacancies"], ["UInt64"], [[900]])
    warehouse = FakeWarehouse({"mart_skills": skills, "mart_junior": base})
    body = client(warehouse=warehouse).get("/stats/skills", params={"source": "telegram", "limit": 2}).json()
    assert body["vacancies"] == 900 and [s["skill"] for s in body["skills"]] == ["Python", "SQL"]
    assert body["period_from"] == "2026-07-01" and body["period_to"] == "2026-10-11"
    assert warehouse.calls[0][1] == {"source": "telegram", "bucket": "", "limit": "2"}


@pytest.mark.parametrize(
    ("error", "status"),
    [(QueryError("Code: 497. Not enough privileges", 497), 502), (httpx.ConnectError("refused"), 503)],
)
def test_warehouse_failures_map_to_gateway_errors(error: Exception, status: int) -> None:
    response = client(warehouse=FakeWarehouse(error=error)).get("/stats/salaries", params={"source": "trudvsem"})
    assert response.status_code == status and "warehouse" in response.json()["detail"]


# --- /salary, /match, health --------------------------------------------------------------------


def test_salary_estimates_extraction_fields_for_the_default_source() -> None:
    estimator = FakeEstimator()
    response = client(salary=estimator).post("/salary", json={"fields": ANSWER})
    assert response.status_code == 200
    assert response.json() == {
        "point": 150000,
        "low": 100000,
        "high": 220000,
        "level": 0.8,
        "typical_from": 130000,
        "typical_to": 170000,
        "n_similar": 0,
        "source": "trudvsem",
        "currency": "RUB",
        "period": "month",
        "basis": "as_stated",
        "model": "salary-test",
    }
    [(fields, source)] = estimator.calls
    assert source == "trudvsem" and fields.role == "backend" and fields.cities == []


def test_salary_takes_the_fields_extract_returns() -> None:
    estimator = FakeEstimator()
    api = client(salary=estimator)
    fields = api.post("/extract", json={"text": TEXT}).json()["fields"]
    response = api.post("/salary", json={"fields": fields, "source": "habr"})
    assert response.status_code == 200 and response.json()["source"] == "habr"
    assert estimator.calls[0][0].skills == fields["skills"]


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (OutOfScope("the salary model covers IT vacancies only"), "IT vacancies only"),
        (UnknownSource("the salary model knows the sources habr, trudvsem, not superjob"), "not superjob"),
    ],
)
def test_salary_outside_the_models_reach_is_422(error: Exception, fragment: str) -> None:
    response = client(salary=FakeEstimator(error)).post("/salary", json={"fields": ANSWER, "source": "superjob"})
    assert response.status_code == 422 and fragment in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [{"fields": {"role": "backend"}}, {"fields": ANSWER, "source": "hh"}, {"fields": ANSWER, "extra": 1}, {}],
    ids=["partial_fields", "unknown_source", "extra_key", "missing"],
)
def test_bad_salary_input_never_reaches_the_model(body: dict[str, Any]) -> None:
    estimator = FakeEstimator()
    assert client(salary=estimator).post("/salary", json=body).status_code == 422
    assert estimator.calls == []


def test_salary_without_a_model_is_503() -> None:
    response = client().post("/salary", json={"fields": ANSWER})
    assert response.status_code == 503 and "not loaded" in response.json()["detail"]


def test_a_missing_salary_model_file_leaves_the_rest_of_the_api_up(tmp_path: Path) -> None:
    assert load_salary(tmp_path / "salary.pkl") is None


def test_match_answers_501_after_validation() -> None:
    api = client()
    match = api.post("/match", json={"resume_text": "Python, SQL, 2 года", "k": 5})
    assert match.status_code == 501 and "7.4" in match.json()["detail"]
    assert api.post("/match", json={"resume_text": "Python", "k": 0}).status_code == 422


def test_health_reports_each_dependency() -> None:
    transport, _ = llama_server()
    ok = client(extractor=student(transport), salary=FakeEstimator()).get("/health").json()
    assert ok == {
        "status": "ok",
        "extract_backend": "student",
        "salary_model": "salary-test",
        "checks": {"redis": "ok", "clickhouse": "ok", "student": "ok", "salary_model": "ok"},
    }
    down = client(MemoryCache(up=False), FakeWarehouse(error=httpx.ConnectError("refused"))).get("/health")
    assert down.status_code == 200
    assert down.json() == {
        "status": "degraded",
        "extract_backend": "regex",
        "salary_model": None,
        "checks": {"redis": "down", "clickhouse": "down", "salary_model": "missing"},
    }


# --- cache and settings -------------------------------------------------------------------------


class BrokenRedis:
    def __getattr__(self, name: str) -> Callable[..., Any]:
        def fail(*args: Any, **kwargs: Any) -> Any:
            raise redis.ConnectionError("refused")

        return fail


def test_unreachable_redis_is_a_miss_not_an_error() -> None:
    cache = RedisCache("redis://localhost:6379/0")
    cache.client = BrokenRedis()  # type: ignore[assignment]
    assert cache.get("k") is None and cache.ping() is False
    cache.set("k", {"a": 1}, 60)


def test_settings_need_redis_and_the_read_only_account(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "REDIS_URL": "redis://localhost:6379/0",
        "CLICKHOUSE_URL": "http://localhost:8123",
        "CLICKHOUSE_RO_USER": "agent",
        "CLICKHOUSE_RO_PASSWORD": "secret",
        "STUDENT_TIMEOUT_S": "",
        "STUDENT_LLM_URL": "",
        "STUDENT_ADAPTER_DIR": "",
        "SALARY_MODEL_PATH": "",
    }.items():
        monkeypatch.setenv(name, value)
    loaded = api_settings.from_env()
    assert loaded.student_url is None and loaded.student_timeout_s == 60.0
    assert loaded.salary_model == ROOT / "models" / "salary" / "salary.pkl"
    monkeypatch.setenv("SALARY_MODEL_PATH", "elsewhere/salary.pkl")
    assert api_settings.from_env().salary_model == ROOT / "elsewhere" / "salary.pkl"
    monkeypatch.setenv("STUDENT_TIMEOUT_S", "soon")
    with pytest.raises(ConfigError, match="STUDENT_TIMEOUT_S"):
        api_settings.from_env()
    monkeypatch.setenv("CLICKHOUSE_RO_USER", "")
    with pytest.raises(ConfigError, match="CLICKHOUSE_RO_USER"):
        api_settings.from_env()
