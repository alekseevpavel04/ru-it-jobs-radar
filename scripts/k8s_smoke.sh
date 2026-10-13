#!/usr/bin/env bash
# Smoke test of the Helm chart on a local kind cluster. The cluster name and the kube
# context are fixed here, so the script never deploys anywhere else. Steps: create the cluster if it
# is missing, attach its node to the compose network (pods reach clickhouse, redis and mlflow by the
# compose names), load the API image, create the ClickHouse secret from the env file, install the
# chart and wait for it, then call the API, its metrics and the exporter through port-forwards.
#   scripts/k8s_smoke.sh            the cluster stays for the next run
#   scripts/k8s_smoke.sh --delete   delete the cluster at the end
# Needs kind, kubectl and helm in the WSL distro (docs/monitoring.md, versions pinned there), the core
# and ml profiles up with the registry filled (python -m radar.monitor.registry) and the image
# radar/api:0.1.0 built by `scripts/up.sh api`.
set -euo pipefail
. "$(dirname "$0")/../infra/compose/lib.sh"

reexec_in_wsl k8s_smoke.sh "$@"
require_env_file
# Where docs/monitoring.md installs kind, kubectl and helm; `wsl --exec` starts no login shell.
PATH=$HOME/.local/bin:$PATH

CLUSTER=radar
CONTEXT=kind-radar
NAMESPACE=radar
RELEASE=radar
NETWORK=radar_default
IMAGE=radar/api:0.1.0
CHART=$RADAR_ROOT/deploy/helm/radar

delete=false
case "${1:-}" in
  "") ;;
  --delete) delete=true ;;
  *)
    echo "usage: scripts/k8s_smoke.sh [--delete]" >&2
    exit 2
    ;;
esac

for tool in docker kind kubectl helm curl; do
  command -v "$tool" >/dev/null || {
    echo "$tool is not installed in this distro; docs/monitoring.md lists the pinned install commands" >&2
    exit 1
  }
done

k() { kubectl --context "$CONTEXT" --namespace "$NAMESPACE" "$@"; }

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --wait 120s
fi
node=$CLUSTER-control-plane
if ! docker network inspect "$NETWORK" --format '{{range .Containers}}{{.Name}} {{end}}' | grep -qw "$node"; then
  docker network connect "$NETWORK" "$node"
fi
kind load docker-image "$IMAGE" --name "$CLUSTER"

kubectl --context "$CONTEXT" create namespace "$NAMESPACE" --dry-run=client --output yaml |
  kubectl --context "$CONTEXT" apply --filename -
# The values go through a process substitution filled by the printf builtin: no password in argv.
missing=()
for name in CLICKHOUSE_USER CLICKHOUSE_PASSWORD CLICKHOUSE_RO_USER CLICKHOUSE_RO_PASSWORD; do
  [ -n "$(env_value "$name")" ] || missing+=("$name")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "set ${missing[*]} in $RADAR_ENV_FILE" >&2
  exit 1
fi
k create secret generic radar-clickhouse --dry-run=client --output yaml --from-env-file=<(
  for name in CLICKHOUSE_USER CLICKHOUSE_PASSWORD CLICKHOUSE_RO_USER CLICKHOUSE_RO_PASSWORD; do
    printf '%s=%s\n' "$name" "$(env_value "$name")"
  done
) | k apply --filename -

helm upgrade --install "$RELEASE" "$CHART" --kube-context "$CONTEXT" --namespace "$NAMESPACE" \
  --wait --timeout 5m
k get pods --output wide

pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  if $delete; then kind delete cluster --name "$CLUSTER"; fi
}
trap cleanup EXIT

forward() {
  k port-forward "service/$RELEASE-$1" "$2:$3" >/dev/null &
  pids+=($!)
}
forward api 18000 8000
forward exporter 19188 9188
for _ in $(seq 30); do
  curl --silent --fail --output /dev/null http://127.0.0.1:18000/health && break
  sleep 1
done

failed=0
check() {
  local what=$1 pattern=$2 body=$3
  if grep -qE -- "$pattern" <<<"$body"; then
    echo "ok    $what"
  else
    echo "FAIL  $what: ${body:0:300}"
    failed=1
  fi
}

check "health: salary model from the registry" '"salary_model":"ok"' "$(curl --silent http://127.0.0.1:18000/health)"
check "health: ClickHouse through the read-only account" '"clickhouse":"ok"' "$(curl --silent http://127.0.0.1:18000/health)"
check "extract" '"role"' "$(curl --silent --header 'Content-Type: application/json' \
  --data '{"text": "Python-разработчик (middle), удалённо, 250 000 руб. на руки. Django, PostgreSQL."}' \
  http://127.0.0.1:18000/extract)"
check "api metrics" '^radar_api_requests_total\{' "$(curl --silent http://127.0.0.1:18000/metrics)"
check "exporter reaches ClickHouse" '^radar_clickhouse_up 1' "$(curl --silent http://127.0.0.1:19188/metrics)"

exit $failed
