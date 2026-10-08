#!/usr/bin/env bash
# Entrypoint wrapper for a Postgres whose password comes from .env. initdb applies POSTGRES_PASSWORD
# only to an empty volume; on every later start this sets it again over the local socket (trusted
# inside the container), so a password changed in .env reaches the server without a volume reset.
set -euo pipefail

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  exec docker-entrypoint.sh "$@"
fi

docker-entrypoint.sh "$@" &
server=$!
trap 'kill -TERM "$server"; wait "$server"' TERM INT

until pg_isready --quiet --host=/var/run/postgresql --username="$POSTGRES_USER"; do
  sleep 1
done
psql --no-psqlrc --quiet -v ON_ERROR_STOP=1 --host=/var/run/postgresql --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" --set=password="$POSTGRES_PASSWORD" <<'SQL'
ALTER ROLE CURRENT_USER PASSWORD :'password';
SQL
wait "$server"
