FROM ghcr.io/mlflow/mlflow:v3.17.0

# The upstream image ships without a PostgreSQL driver, and the tracking store lives in Postgres.
RUN pip install --no-cache-dir psycopg2-binary==2.9.13
