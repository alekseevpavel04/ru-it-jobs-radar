"""Prometheus metrics of the API: requests, latency and requests in flight, per route.

`instrument(app)` wraps the app in an ASGI middleware and adds `GET /metrics`. A request is labelled
by its route template (`/stats/weekly`, `/items/{id}`), not by the raw path, so the label set stays
bounded; a path no route matched counts as `unmatched`. Scrapes of /metrics are not counted.
The metrics live in a registry of their own per app, which also carries the process collector
(memory, CPU, open files; Linux only), so two apps in one test process do not collide.
"""

import time
from collections.abc import MutableMapping
from typing import Any

from fastapi import FastAPI
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

METRICS_PATH = "/metrics"
UNMATCHED = "unmatched"
# From cached stats (milliseconds) to an extraction through the LLM server (seconds).
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


class ApiMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.registry = registry
        self.requests = Counter(
            "radar_api_requests_total",
            "HTTP requests finished, by method, route template and status code",
            ["method", "route", "status"],
            registry=registry,
        )
        self.latency = Histogram(
            "radar_api_request_duration_seconds",
            "Time from the request to the end of the response, by method and route template",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self.in_progress = Gauge(
            "radar_api_requests_in_progress", "HTTP requests being served", ["method"], registry=registry
        )


def route_of(scope: MutableMapping[str, Any]) -> str:
    # FastAPI's router puts the matched route into the scope the middleware shares with it.
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else UNMATCHED


class MetricsMiddleware:
    def __init__(self, app: ASGIApp, metrics: ApiMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == METRICS_PATH:
            await self.app(scope, receive, send)
            return
        method = scope["method"]
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        start = time.perf_counter()
        self.metrics.in_progress.labels(method).inc()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = route_of(scope)
            self.metrics.in_progress.labels(method).dec()
            self.metrics.latency.labels(method, route).observe(time.perf_counter() - start)
            self.metrics.requests.labels(method, route, str(status)).inc()


def instrument(app: FastAPI) -> ApiMetrics:
    """Count every request of `app` and serve the metrics at /metrics; returns the metric objects."""
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    metrics = ApiMetrics(registry)

    async def serve_metrics(_: Request) -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    app.add_route(METRICS_PATH, serve_metrics, methods=["GET"], include_in_schema=False)
    app.add_middleware(MetricsMiddleware, metrics=metrics)
    return metrics
