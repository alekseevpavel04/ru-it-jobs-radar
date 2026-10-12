# Airflow plus a second Python environment for the project. Tasks run the project's commands with
# /opt/radar/.venv/bin/python, so Airflow's pinned dependencies and the project's never share an
# environment. The code itself is not baked in: compose mounts src/, scripts/, dbt/ and airflow/
# read-only under /opt/radar, and the .pth file below puts /opt/radar/src on the project's path.
# Built from the repo root (compose: context ../.., this file); Dockerfile.dockerignore keeps the
# build context to pyproject.toml and uv.lock.
FROM ghcr.io/astral-sh/uv:0.10.5 AS uv

FROM apache/airflow:3.3.2-python3.11

COPY --from=uv /uv /usr/local/bin/uv

# dbt/ is mounted read-only; its target/ and logs/ are volumes, which take owner and mode from these.
# Group root and group-writable: compose runs the services as the host user with group root.
USER root
RUN install -d -o airflow -g root -m 775 /opt/radar /opt/radar/dbt/target /opt/radar/dbt/logs
USER airflow

WORKDIR /opt/radar
COPY --chown=airflow:root pyproject.toml uv.lock ./
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/radar/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN --mount=type=cache,target=/home/airflow/.cache/uv,uid=50000 \
    uv sync --frozen --no-dev --group monitor --no-install-project --python /usr/python/bin/python3.11 \
    && echo /opt/radar/src > /opt/radar/.venv/lib/python3.11/site-packages/radar_src.pth

WORKDIR /opt/airflow
