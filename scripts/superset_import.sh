#!/usr/bin/env bash
# Load the Superset assets of infra/superset/assets (database, datasets, charts, dashboards) into the
# local Superset. Idempotent: every object is matched by its uuid and overwritten, so rerun it after
# any change of an asset file. Edits made in the UI are overwritten too; to keep one, export the
# dashboard from the UI and copy the changed files back into infra/superset/assets.
#   scripts/superset_import.sh     start Superset if needed (profile bi, this service alone), import
# From the env file (.env, see .env.example): SUPERSET_SECRET_KEY and SUPERSET_ADMIN_PASSWORD for the
# service, CLICKHOUSE_RO_USER and CLICKHOUSE_RO_PASSWORD for the ClickHouse connection (marts only).
# ClickHouse itself (profile core) must be up for the check of the datasets at the end.
# Works from Git Bash on Windows (re-runs itself inside the WSL distro) and from Linux.
set -euo pipefail
. "$(dirname "$0")/../infra/compose/lib.sh"

reexec_in_wsl superset_import.sh "$@"
if [ $# -gt 0 ]; then
  echo "unknown argument $1; see the header of $0" >&2
  exit 2
fi

require_env_file
PROFILE_ARGS=(--profile bi)
require_profile_secrets
for name in CLICKHOUSE_RO_USER CLICKHOUSE_RO_PASSWORD; do
  [ -n "$(env_value "$name")" ] || { echo "set $name in $RADAR_ENV_FILE (see .env.example)" >&2; exit 1; }
done

superset_compose() { compose --profile bi "$@"; }

superset_compose up --detach --build --wait --wait-timeout "${RADAR_UP_TIMEOUT:-600}" superset

# `exec -e NAME` without a value passes the variable from this environment, so the password never
# appears in a command line.
CLICKHOUSE_RO_USER=$(env_value CLICKHOUSE_RO_USER)
CLICKHOUSE_RO_PASSWORD=$(env_value CLICKHOUSE_RO_PASSWORD)
export CLICKHOUSE_RO_USER CLICKHOUSE_RO_PASSWORD
superset_compose exec -T -e CLICKHOUSE_RO_USER -e CLICKHOUSE_RO_PASSWORD superset \
  python /app/radar/import_assets.py /app/radar/assets

echo
echo "Superset: http://localhost:8088 (account admin)"
