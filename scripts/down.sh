#!/usr/bin/env bash
# Stop the local stack. Volumes stay: ClickHouse, Redpanda and MLflow keep their data.
#   scripts/down.sh          all compose profiles
#   scripts/down.sh ml       only these profiles
set -euo pipefail
. "$(dirname "$0")/../infra/compose/lib.sh"

reexec_in_wsl down.sh "$@"
require_env_file
select_profiles "$@"

compose "${PROFILE_ARGS[@]}" down
