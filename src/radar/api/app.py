"""The HTTP API: extraction, market aggregates of the marts, the salary estimate and open
vacancies matched to a resume.

    python -m radar.api                 # uvicorn on 127.0.0.1:8000, settings from .env
"""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from radar.agent.clickhouse import QueryError, ReadOnlyClickHouse, Result
from radar.api import stats, vacancies
from radar.api.cache import Cache, RedisCache, key
from radar.api.extractors import Extraction, StudentExtractor, extract
from radar.api.models import (
    ExtractRequest,
    ExtractResponse,
    MatchItem,
    MatchRequest,
    MatchResponse,
    Problem,
    SalaryRequest,
    SalaryResponse,
)
from radar.api.salary import Estimator, OutOfScope, SalaryEstimator, UnknownSource
from radar.api.settings import Settings
from radar.extract.schema import SCHEMA_VERSION, dump
from radar.models import matching, matching_onnx
from radar.models.matching import MatchIndex
from radar.monitor.api_metrics import instrument

log = logging.getLogger("radar.api")


class Warehouse(Protocol):
    def run(self, sql: str, params: dict[str, str] | None = None) -> Result: ...


@dataclass
class Services:
    cache: Cache
    warehouse: Warehouse
    student: StudentExtractor | None
    salary: Estimator | None
    stats_ttl_s: int
    extract_ttl_s: int
    match_index: MatchIndex | None = None


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
        salary=load_salary(settings.salary_model),
        stats_ttl_s=settings.stats_ttl_s,
        extract_ttl_s=settings.extract_ttl_s,
        match_index=load_match_index(settings.match_index),
    )


def load_salary(path: Path) -> SalaryEstimator | None:
    """The model artifact lives outside git (models/); without it the API serves everything else,
    /salary answers 503 and /health names the missing model."""
    if not path.exists():
        log.warning("no salary model at %s: /salary answers 503", path)
        return None
    return SalaryEstimator.load(path)


def salary_body(estimator: Estimator, request: SalaryRequest) -> SalaryResponse:
    """The estimate of POST /salary and of the MCP tool estimate_salary."""
    out = estimator.estimate(request.fields, request.source)
    return SalaryResponse(
        point=out["point"],
        low=out["low"],
        high=out["high"],
        level=out["level"],
        typical_from=out["typical_from"],
        typical_to=out["typical_to"],
        n_similar=out["n_similar"],
        source=request.source,
        model=out["model_version"],
    )


def load_match_index(path: Path) -> MatchIndex | None:
    """Outside git like the salary model: without it /match answers 503. A dense system's queries are
    encoded by the ONNX export next to the index (models/matching/onnx), not by torch, which the API
    does not carry. The encoder is loaded here rather than on the first request, so that request is
    not slow and the process's memory shows it."""
    if not path.exists():
        log.warning("no match index at %s: /match answers 503", path)
        return None
    index = matching.load_index(path)
    try:
        if index.model is not None:
            onnx_dir = path.parent / matching_onnx.DIRNAME
            index.use_encoder(matching_onnx.load_encoder(onnx_dir, index.model, index.max_length))
        index.scores("Python")
    except (OSError, ValueError) as e:  # the export is missing or of another model
        log.error("the encoder of %s does not load (%s): /match answers 503", index.system, e)
        return None
    return index


# The index marks the vacancies open on the day it was built, the mart those open now: candidates
# beyond k stand in for the ones closed since.
OVERFETCH = 3


def match_body(index: MatchIndex, query: stats.Query, resume_text: str, k: int) -> MatchResponse:
    """The answer of POST /match and of the MCP tool match_resume: the index's closest vacancies that
    are still open in marts.mart_open_vacancies, with the fields as that mart publishes them."""
    found = matching.match(resume_text, k=k * OVERFETCH, open_only=True, index=index)
    rows = vacancies.by_id(query, [item["vacancy_entity_id"] for item in found["items"]])
    kept = [(rows[i["vacancy_entity_id"]], i["score"]) for i in found["items"] if i["vacancy_entity_id"] in rows]
    return MatchResponse(
        items=[MatchItem(**row, rank=rank, score=score) for rank, (row, score) in enumerate(kept[:k], start=1)],
        model=found["model_version"],
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
        checks["salary_model"] = "ok" if services.salary is not None else "missing"
        checks["match_index"] = "ok" if services.match_index is not None else "missing"
        # Extraction works without any of them (the regex), so the service is up, maybe degraded.
        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return {
            "status": status,
            "extract_backend": "student" if services.student else "regex",
            "salary_model": services.salary.version if services.salary else None,
            "match_model": services.match_index.version if services.match_index else None,
            "checks": checks,
        }

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

    @app.post("/salary", response_model=SalaryResponse, responses={422: {"model": Problem}, 503: {"model": Problem}})
    def post_salary(body: SalaryRequest) -> SalaryResponse:
        """Monthly RUB estimate with its conformal interval for extraction fields; a stated salary in
        the fields is not read."""
        if services.salary is None:
            raise HTTPException(503, "the salary model is not loaded on this server")
        try:
            return salary_body(services.salary, body)
        except (OutOfScope, UnknownSource) as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/match", response_model=MatchResponse, responses={503: {"model": Problem}})
    def post_match(body: MatchRequest) -> MatchResponse:
        """Open vacancies closest to a resume text, the closest first. Titles, employers and links only
        where the source's terms allow publishing them, as in marts.mart_open_vacancies."""
        if services.match_index is None:
            raise HTTPException(503, "the match index is not loaded on this server")
        try:
            return match_body(services.match_index, query, body.resume_text, body.k)
        except vacancies.MartMissing as e:
            raise HTTPException(503, str(e)) from e

    instrument(app)  # GET /metrics: request counts and latency per route (9.3)
    return app


def _warehouse_up(warehouse: Warehouse) -> bool:
    try:
        warehouse.run("SELECT 1")
    except (QueryError, httpx.HTTPError):
        return False
    return True
