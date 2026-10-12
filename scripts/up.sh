#!/usr/bin/env bash
# Bring the local stack up and wait until every service is healthy.
#   scripts/up.sh            all compose profiles
#   scripts/up.sh core ml    only these profiles
# Works from Git Bash on Windows (re-runs itself inside the WSL distro) and from Linux.
set -euo pipefail
. "$(dirname "$0")/../infra/compose/lib.sh"

reexec_in_wsl up.sh "$@"
require_env_file
select_profiles "$@"
require_profile_secrets
# Mount points of the Airflow worker's dbt target/ and logs/ volumes inside its read-only dbt/, and
# the bind-mounted output directories of radar_drift, owned by the user the worker runs as.
mkdir -p "$RADAR_ROOT/dbt/target" "$RADAR_ROOT/dbt/logs" "$RADAR_ROOT/results/monitor" "$RADAR_ROOT/reports/evidently"

services=$(compose "${PROFILE_ARGS[@]}" config --services)
# `up --wait` reports a one-shot container that exited as a failure, so the *-init services
# (idempotent bootstrap: databases, users, topics) run after the rest is healthy.
mapfile -t long_running < <(grep -v -- '-init$' <<<"$services")
mapfile -t init < <(grep -- '-init$' <<<"$services" || true)

compose "${PROFILE_ARGS[@]}" up --detach --build --wait --wait-timeout "${RADAR_UP_TIMEOUT:-300}" "${long_running[@]}"
for service in "${init[@]}"; do
  echo "== $service"
  compose "${PROFILE_ARGS[@]}" run --rm --no-TTY "$service"
done

echo
compose "${PROFILE_ARGS[@]}" ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'
echo
echo "Ports are bound to 127.0.0.1 inside WSL and reachable as localhost from Windows."
