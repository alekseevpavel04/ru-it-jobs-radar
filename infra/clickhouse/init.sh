#!/usr/bin/env bash
# Databases of the storage layers and the read-only account of the analyst agent. Idempotent: the
# clickhouse-init service runs it on every `up`, so the password in .env is always the live one.
set -euo pipefail

: "${CLICKHOUSE_USER:?}" "${CLICKHOUSE_PASSWORD:?}" "${CLICKHOUSE_RO_USER:?}" "${CLICKHOUSE_RO_PASSWORD:?}"

# Both values are spliced into SQL text (CREATE USER takes no query parameters), so their alphabet
# is restricted instead of escaped; radar.agent.access checks the same alphabet.
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

here=$(dirname "$0")
client() {
  clickhouse-client --host "${CLICKHOUSE_HOST:-clickhouse}" --user "$CLICKHOUSE_USER" \
    --password "$CLICKHOUSE_PASSWORD" --multiquery
}

client <<SQL
CREATE DATABASE IF NOT EXISTS raw;
CREATE DATABASE IF NOT EXISTS staging;
CREATE DATABASE IF NOT EXISTS marts;
SQL
echo "databases raw, staging, marts"

# The account itself (profile, password, grants) has one definition, agent/readonly_user.sql, which
# scripts/apply_agent_user.py applies from the host as well. | cannot occur in either value, and
# neither can & or \, the other characters sed treats specially in a replacement.
sed -e "s|{ro_user}|${CLICKHOUSE_RO_USER}|g" -e "s|{ro_password}|${CLICKHOUSE_RO_PASSWORD}|g" \
  "$here/agent/readonly_user.sql" | client
echo "applied agent/readonly_user.sql: ${CLICKHOUSE_RO_USER} can SELECT on marts and agent_snapshot only"

# Tables of the layers: numbered DDL files next to this script, every statement idempotent. They are
# applied only when the directory is mounted (not just this file); `python -m radar.stream.clickhouse`
# applies the same files from the host.
shopt -s nullglob
for ddl in "$here"/*.sql; do
  client <"$ddl"
  echo "applied $(basename "$ddl")"
done
