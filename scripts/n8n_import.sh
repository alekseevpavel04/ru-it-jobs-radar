#!/usr/bin/env bash
# Load the workflows of infra/n8n/workflows into the local n8n and publish them. Idempotent: rerun it
# after any change of a workflow file. The files are the source; edits made in the UI are overwritten,
# so a change made there goes back into the file with `n8n export:workflow` before the next import.
#   scripts/n8n_import.sh                 start n8n if needed, set it up, import and publish
#   scripts/n8n_import.sh --run           then run the weekly digest once (latest complete week)
#   scripts/n8n_import.sh --reset-owner   N8N_OWNER_PASSWORD changed: recreate the owner account first
# From the env file (.env, see .env.example) it sets up:
#   the owner account owner@radar.local with N8N_OWNER_PASSWORD, for the UI on http://localhost:5678;
#   the credential "ClickHouse read-only" (HTTP basic auth) from CLICKHOUSE_RO_USER and
#   CLICKHOUSE_RO_PASSWORD, stored encrypted in the n8n volume.
# Works from Git Bash on Windows (re-runs itself inside the WSL distro) and from Linux.
set -euo pipefail
. "$(dirname "$0")/../infra/compose/lib.sh"

reexec_in_wsl n8n_import.sh "$@"

run_once=false
reset_owner=false
for arg in "$@"; do
  case $arg in
    --run) run_once=true ;;
    --reset-owner) reset_owner=true ;;
    *) echo "unknown argument $arg; see the header of $0" >&2; exit 2 ;;
  esac
done

require_env_file
PROFILE_ARGS=(--profile mlops)
require_profile_secrets

OWNER_EMAIL=owner@radar.local
N8N_URL=http://127.0.0.1:5678
WORKFLOWS=/opt/radar/infra/n8n/workflows
DIGEST_WORKFLOW=radarWeeklyDigest

ro_user=$(env_value CLICKHOUSE_RO_USER)
ro_password=$(env_value CLICKHOUSE_RO_PASSWORD)
owner_password=$(env_value N8N_OWNER_PASSWORD)
# The values are spliced into JSON, so their alphabet is checked instead of escaped (as in init.sh).
for name in CLICKHOUSE_RO_USER CLICKHOUSE_RO_PASSWORD N8N_OWNER_PASSWORD; do
  if [[ ! $(env_value "$name") =~ ^[A-Za-z0-9_.~+/=-]+$ ]]; then
    echo "$name must be set in $RADAR_ENV_FILE and contain only letters, digits and _.~+/=-" >&2
    exit 1
  fi
done

n8n_compose() { compose --profile mlops "$@"; }
n8n_cli() { n8n_compose exec -T n8n n8n "$@"; }

# The bind mount of the digest pages needs the directory on the host before the container starts.
mkdir -p "$RADAR_ROOT/data/digest"

if $reset_owner; then
  n8n_compose stop n8n
  n8n_compose run --rm --no-deps n8n user-management:reset
fi
n8n_compose up --detach --build --wait n8n

echo "== credential: ClickHouse read-only"
# Through a file in the container's /tmp, deleted right after: the CLI imports from files only.
n8n_compose exec -T n8n sh -c 'umask 077 && cat > /tmp/radar-credentials.json' <<JSON
[{"id": "radarChReadOnly1", "name": "ClickHouse read-only", "type": "httpBasicAuth",
  "data": {"user": "$ro_user", "password": "$ro_password"}}]
JSON
status=0
n8n_cli import:credentials --input=/tmp/radar-credentials.json || status=$?
n8n_compose exec -T n8n rm -f /tmp/radar-credentials.json
[ "$status" -eq 0 ] || exit "$status"

echo "== workflows"
n8n_cli import:workflow --separate --input="$WORKFLOWS"
mapfile -t ids < <(n8n_compose exec -T n8n python3 -c '
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    print(json.loads(path.read_text(encoding="utf-8"))["id"])
' "$WORKFLOWS" | tr -d '\r')
for id in "${ids[@]}"; do
  n8n_cli publish:workflow --id="$id"
done
# A running instance picks up workflows published from the CLI on its next start.
n8n_compose restart n8n
n8n_compose up --detach --wait n8n

echo "== owner account"
# /healthz answers before the REST API is mounted.
settings=
for _ in $(seq 60); do
  settings=$(curl --silent --fail "$N8N_URL/rest/settings") && break
  sleep 1
done
[ -n "$settings" ] || { echo "the n8n REST API did not answer within 60 s" >&2; exit 1; }
needs_setup=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["data"]["userManagement"]["showSetupOnFirstLoad"])' "$settings")
if [ "$needs_setup" = True ]; then
  curl --silent --fail --output /dev/null -H 'Content-Type: application/json' "$N8N_URL/rest/owner/setup" \
    --data "{\"email\": \"$OWNER_EMAIL\", \"firstName\": \"Radar\", \"lastName\": \"Owner\", \"password\": \"$owner_password\"}"
  echo "created $OWNER_EMAIL"
elif curl --silent --fail --output /dev/null -H 'Content-Type: application/json' "$N8N_URL/rest/login" \
  --data "{\"emailOrLdapLoginId\": \"$OWNER_EMAIL\", \"password\": \"$owner_password\"}"; then
  echo "$OWNER_EMAIL signs in with N8N_OWNER_PASSWORD"
else
  echo "the owner account does not accept N8N_OWNER_PASSWORD: rerun with --reset-owner" >&2
  exit 1
fi

if $run_once; then
  echo "== run $DIGEST_WORKFLOW"
  # A second n8n process next to the server: its task broker needs a port the server does not hold.
  # The CLI prints the data of every node, so the output goes to a log.
  run_log=$RADAR_ROOT/logs/n8n/run.log
  mkdir -p "$(dirname "$run_log")"
  if ! n8n_compose exec -T -e N8N_RUNNERS_BROKER_PORT=5690 n8n n8n execute --id="$DIGEST_WORKFLOW" >"$run_log" 2>&1; then
    tail -n 40 "$run_log" >&2
    exit 1
  fi
  echo "done, node output in logs/n8n/run.log; issues: $(ls "$RADAR_ROOT/data/digest/data" | tr '\n' ' ')"
fi
echo
echo "n8n: http://localhost:5678 ($OWNER_EMAIL); workflows published: ${ids[*]}"
