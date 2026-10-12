"""The HTTP API: extraction, market aggregates of the marts, and the routes of the salary and
matching models, which answer 501 until those models exist.

    python -m radar.api                 # uvicorn on 127.0.0.1:8000, settings from .env
"""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from radar.agent.clickhouse import QueryError, ReadOnlyClickHouse, Result
from radar.api import stats
from radar.api.cache import Cache, RedisCache, key
from radar.api.extractors import Extraction, StudentExtractor, extract
from radar.api.models import (
    ExtractRequest,
    ExtractResponse,
    MatchRequest,
    MatchResponse,
    Problem,
    SalaryRequest,
    SalaryResponse,
)
from radar.api.settings import Settings
from radar.extract.schema import SCHEMA_VERSION, dump

log = logging.getLogger("radar.api")


class Warehouse(Protocol):
    def run(self, sql: str, params: dict[str, str] | None = None) -> Result: ...


@dataclass
class Services:
    cache: Cache
    warehouse: Warehouse
    student: StudentExtractor | None
    stats_ttl_s: int
    extract_ttl_s: int


def services_from(settings: Settings) -> Services:
    student = None
    if settings.student_url:
        student = StudentExtractor.from_meta_dir(
            settings.student_url, settings.student_adapter, settings.student_timeout_s
        )
    return Services(
        cache=RedisCache(settings.redis_url),
        warehouse=ReadOnlyClickHouse(timeout_s=15.0),
        student=student,
        stats_ttl_s=settings.stats_ttl_s,
        extract_ttl_s=settings.extract_ttl_s,
    )


class JsonFormatter(logging.Formatter):
    FIELDS = ("method", "path", "status", "ms", "backend", "fallback", "cached")

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        line |= {f: getattr(record, f) for f in self.FIELDS if hasattr(record, f)}
        return json.dumps(line, ensure_ascii=False)


def _extraction_body(e: Extraction, fallback: str | None, cached: bool) -> dict[str, Any]:
    return {
        "fields": dump(e.fields),
        "backend": e.backend,
        "model": e.model,
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": e.prompt_sha256,
        "truncated_input": e.truncated_input,
        "fallback_reason": fallback,
        "cached": cached,
    }


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="ru-it-jobs-radar API", version=version("radar"))
    query: stats.Query = services.warehouse.run

    @app.middleware("http")
    async def access_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # The place for the request counters and latency histograms of the monitoring task (9.3).
        started = time.perf_counter()
        response = await call_next(request)
        ms = round((time.perf_counter() - started) * 1000, 1)
        extra = {"method": request.method, "path": request.url.path, "status": response.status_code, "ms": ms}
        log.info("request", extra=extra)
        return response

    @app.exception_handler(QueryError)
    def clickhouse_refused(request: Request, exc: QueryError) -> JSONResponse:
        log.error("ClickHouse refused a stats query: code %s", exc.code)
        return JSONResponse({"detail": f"the warehouse refused the query (code {exc.code})"}, status_code=502)

    @app.exception_handler(httpx.TransportError)
    def clickhouse_down(request: Request, exc: httpx.TransportError) -> JSONResponse:
        log.error("ClickHouse unreachable: %s", type(exc).__name__)
        return JSONResponse({"detail": "the warehouse is unreachable"}, status_code=503)

    # The SQL is part of the key, so a changed query is not answered from the old one's cache.
    def cached(kind: str, parts: tuple[str, ...], compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        k = key(kind, *parts)
        hit = services.cache.get(k)
        if hit is not None:
            return {**hit, "cached": True}
        value = compute()
        services.cache.set(k, value, services.stats_ttl_s)
        return {**value, "cached": False}

    @app.get("/health")
    def health() -> dict[str, Any]:
        checks: dict[str, str] = {
            "redis": "ok" if services.cache.ping() else "down",
            "clickhouse": "ok" if _warehouse_up(services.warehouse) else "down",
        }
        if services.student is not None:
            checks["student"] = "ok" if services.student.ping() else "down"
        # Extraction works without any of them (the regex), so the service is up, maybe degraded.
        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return {"status": status, "extract_backend": "student" if services.student else "regex", "checks": checks}

    @app.post("/extract", response_model=ExtractResponse)
    def post_extract(body: ExtractRequest) -> dict[str, Any]:
        identity = services.student.identity() if services.student else "regex"
        k = key("extract", str(SCHEMA_VERSION), identity, body.text)
        hit = services.cache.get(k)
        if hit is not None:
            log.info("extract", extra={"backend": hit["backend"], "cached": True})
            return {**hit, "cached": True}
        extraction, fallback = extract(body.text, services.student)
        answer = _extraction_body(extraction, fallback, cached=False)
        if fallback is None:  # a stand-in answer is not kept: the student may be back on the next call
            services.cache.set(k, answer, services.extract_ttl_s)
        log.info("extract", extra={"backend": extraction.backend, "fallback": fallback, "cached": False})
        return answer

    @app.get("/stats/weekly")
    def get_weekly(
        source: stats.WeeklySource,
        role_bucket: stats.RoleBucket | None = None,
        weeks: int = Query(26, ge=1, le=260),
    ) -> dict[str, Any]:
        """Vacancies open, new and closed per week (Telegram: posts), summed over role buckets or one bucket."""
        return cached(
            "weekly",
            (stats.WEEKLY_SQL, source, role_bucket or "", str(weeks)),
            lambda: stats.weekly(query, source, role_bucket, weeks),
        )

    @app.get("/stats/salaries")
    def get_salaries(source: stats.SalarySource, level: stats.SalaryLevel = "group_grade") -> dict[str, Any]:
        """Stated salary quantiles by role group and grade (or a coarser level), each row with its n."""
        return cached("salaries", (stats.SALARIES_SQL, source, level), lambda: stats.salaries(query, source, level))

    @app.get("/stats/skills")
    def get_skills(
        source: stats.SkillSource,
        role_bucket: stats.RoleBucket | None = None,
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        """The most mentioned skills, each with the vacancies of the source (and bucket) as denominator."""
        return cached(
            "skills",
            (stats.SKILLS_SQL, stats.SKILLS_BASE_SQL, source, role_bucket or "", str(limit)),
            lambda: stats.skills(query, source, role_bucket, limit),
        )

    @app.post("/salary", response_model=SalaryResponse, responses={501: {"model": Problem}})
    def post_salary(body: SalaryRequest) -> SalaryResponse:
        raise HTTPException(501, "salary model is not served yet")

    @app.post("/match", response_model=MatchResponse, responses={501: {"model": Problem}})
    def post_match(body: MatchRequest) -> MatchResponse:
        raise HTTPException(501, "vacancy matching is not served yet")

    return app


def _warehouse_up(warehouse: Warehouse) -> bool:
    try:
        warehouse.run("SELECT 1")
    except (QueryError, httpx.HTTPError):
        return False
    return True
