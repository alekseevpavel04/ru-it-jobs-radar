# Superset with the ClickHouse driver: the official image ships no database drivers. The version stays
# inside the range Superset 6.1 declares for its clickhouse extra (>=0.13.0, <1.0).
FROM apache/superset:6.1.0
USER root
RUN uv pip install --python /app/.venv/bin/python --no-cache clickhouse-connect==0.15.1
USER superset
