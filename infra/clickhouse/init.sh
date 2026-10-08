#!/usr/bin/env bash
# Databases of the storage layers and the read-only account of the analyst agent. Idempotent: the
# clickhouse-init service runs it on every `up`, so the password in .env is always the live one.
set -euo pipefail

: "${CLICKHOUSE_USER:?}" "${CLICKHOUSE_PASSWORD:?}" "${CLICKHOUSE_RO_USER:?}" "${CLICKHOUSE_RO_PASSWORD:?}"

# Both values are spliced into SQL text (CREATE USER takes no query parameters), so their alphabet
# is restricted instead of escaped.
if [[ ! $CLICKHOUSE_RO_USER =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "CLICKHOUSE_RO_USER must be a plain identifier (letters, digits, underscore)" >&2
  exit 1
fi
if [[ ! $CLICKHOUSE_RO_PASSWORD =~ ^[A-Za-z0-9_.~+/=-]+$ ]]; then
  echo "CLICKHOUSE_RO_PASSWORD may contain only letters, digits and _.~+/=-" >&2
  exit 1
fi
if [[ $CLICKHOUSE_RO_USER == "$CLICKHOUSE_USER" ]]; then
  echo "CLICKHOUSE_RO_USER must differ from CLICKHOUSE_USER" >&2
  exit 1
fi

clickhouse-client --host "${CLICKHOUSE_HOST:-clickhouse}" --user "$CLICKHOUSE_USER" \
  --password "$CLICKHOUSE_PASSWORD" --multiquery <<SQL
CREATE DATABASE IF NOT EXISTS raw;
CREATE DATABASE IF NOT EXISTS staging;
CREATE DATABASE IF NOT EXISTS marts;

-- readonly = 2 still lets HTTP clients send per-query settings, but forbids writes and DDL.
-- CONST keeps the account from lifting it; the limits bound the cost of a runaway generated query.
CREATE SETTINGS PROFILE OR REPLACE agent_readonly SETTINGS
    readonly = 2 CONST,
    allow_ddl = 0 CONST,
    max_execution_time = 30,
    max_memory_usage = 2000000000;

CREATE USER IF NOT EXISTS ${CLICKHOUSE_RO_USER};
ALTER USER ${CLICKHOUSE_RO_USER} IDENTIFIED WITH sha256_password BY '${CLICKHOUSE_RO_PASSWORD}'
    SETTINGS PROFILE 'agent_readonly';
REVOKE ALL ON *.* FROM ${CLICKHOUSE_RO_USER};
GRANT SELECT ON marts.* TO ${CLICKHOUSE_RO_USER};
SQL

echo "databases raw, staging, marts; ${CLICKHOUSE_RO_USER} can SELECT on marts only"

# Tables of the layers: numbered DDL files next to this script, every statement idempotent. They are
# applied only when the directory is mounted (not just this file); `python -m radar.stream.clickhouse`
# applies the same files from the host.
shopt -s nullglob
for ddl in "$(dirname "$0")"/*.sql; do
  clickhouse-client --host "${CLICKHOUSE_HOST:-clickhouse}" --user "$CLICKHOUSE_USER" \
    --password "$CLICKHOUSE_PASSWORD" --multiquery <"$ddl"
  echo "applied $(basename "$ddl")"
done
