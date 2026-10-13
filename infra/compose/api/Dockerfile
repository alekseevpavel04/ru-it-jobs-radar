# The HTTP API (python -m radar.api) and the MCP server (python -m radar.mcp), profile api.
# Built from the repo root (compose: context ../.., this file); Dockerfile.dockerignore keeps the
# context to the dependency lock, the code and the repo files the services read at run time (the
# skill dictionary, the marts' descriptions, configs). The salary model and the match index with its
# ONNX query encoder are not in git and are mounted.
FROM ghcr.io/astral-sh/uv:0.10.5 AS uv

FROM python:3.11.16-slim

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /opt/radar
# ORT_DISABLE_TELEMETRY: the Linux build of onnxruntime otherwise sends usage events to Microsoft.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/radar/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ORT_DISABLE_TELEMETRY=1 \
    PATH=/opt/radar/.venv/bin:$PATH

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# The project is installed editable: radar.config finds the repo root from its own path, so the
# code has to stay under /opt/radar/src rather than move into site-packages.
COPY src ./src
COPY configs ./configs
COPY dbt/seeds ./dbt/seeds
COPY dbt/models/marts/_marts.yml ./dbt/models/marts/_marts.yml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

USER nobody
EXPOSE 8000 8001
CMD ["python", "-m", "radar.api", "--host", "0.0.0.0", "--port", "8000"]
