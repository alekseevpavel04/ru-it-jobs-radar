#!/usr/bin/env bash
# Screenshots of the Airflow UI into docs/img (infra/airflow/screenshot.mjs). The admin password comes
# from the env file through the environment of node, never on the command line or in the output.
#   scripts/airflow_screenshot.sh [--dag radar_dbt] [--out-dir docs/img]
# Runs on the Windows side (Playwright and its browsers live there); Airflow is localhost:8090.
set -euo pipefail
. "$(dirname "$0")/../infra/compose/lib.sh"

require_env_file
AIRFLOW_ADMIN_PASSWORD=$(env_value AIRFLOW_ADMIN_PASSWORD)
[ -n "$AIRFLOW_ADMIN_PASSWORD" ] || { echo "set AIRFLOW_ADMIN_PASSWORD in $RADAR_ENV_FILE" >&2; exit 1; }
export AIRFLOW_ADMIN_PASSWORD
exec "${NODE:-node}" "$RADAR_ROOT/infra/airflow/screenshot.mjs" "$@"
