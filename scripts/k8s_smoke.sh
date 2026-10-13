#!/usr/bin/env bash
# Smoke test of the Helm chart on a local kind cluster. The cluster name and the kube
# context are fixed here, so the script never deploys anywhere else. Steps: create the cluster if it
# is missing, attach its node to the compose network (pods reach clickhouse, redis and mlflow by the
# compose names), load the API image, create the ClickHouse secret from the env file, install the
# chart and wait for it, then call the API, its metrics and the exporter through port-forwards.
#   scripts/k8s_smoke.sh            the cluster stays for the next run
#   scripts/k8s_smoke.sh --delete   delete the cluster at the end
# Every run, passed or not, writes results/k8s/smoke.json: tool versions, the wall time of each step,
# the checks with their outcome and time, the pods and the memory of the kind node.
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
REPORT=$RADAR_ROOT/results/k8s/smoke.json

delete=false
case "${1:-}" in
  "") ;;
  --delete) delete=true ;;
  *)
    echo "usage: scripts/k8s_smoke.sh [--delete]" >&2
    exit 2
    ;;
esac

for tool in docker kind kubectl helm curl python3; do
  command -v "$tool" >/dev/null || {
    echo "$tool is not installed in this distro; docs/monitoring.md lists the pinned install commands" >&2
    exit 1
  }
done

k() { kubectl --context "$CONTEXT" --namespace "$NAMESPACE" "$@"; }

# Report state: steps.tsv (step, seconds) and checks.tsv (check, 1 or 0, seconds) fill as the run
# goes; the EXIT trap snapshots the pods and the node before an optional delete and writes the JSON.
work=$(mktemp -d)
: >"$work/steps.tsv"
: >"$work/checks.tsv"
echo '{"items": []}' >"$work/pods.json"
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cluster_created=false
server_version=""
node_memory=""
current_step=""
pids=()

now() { date +%s.%N; }
elapsed() { awk -v a="$1" -v b="$(now)" 'BEGIN { printf "%.3f", b - a }'; }

# timed NAME CMD...: runs CMD and records its wall time; a failing CMD leaves NAME as the failed step.
timed() {
  local name=$1 t0
  shift
  current_step=$name
  t0=$(now)
  "$@"
  printf '%s\t%s\n' "$name" "$(elapsed "$t0")" >>"$work/steps.tsv"
  current_step=""
}

write_report() {
  mkdir -p "$(dirname "$REPORT")"
  SMOKE_STATUS=$1 SMOKE_FAILED_STEP=$current_step SMOKE_STARTED_AT=$started_at \
    SMOKE_CLUSTER=$CLUSTER SMOKE_CONTEXT=$CONTEXT SMOKE_CREATED=$cluster_created SMOKE_DELETED=$delete \
    SMOKE_KIND=$(kind version | awk '{print $2}') \
    SMOKE_KUBECTL=$(kubectl version --client | awk '/Client Version/ {print $3}') \
    SMOKE_HELM=$(helm version --short) \
    SMOKE_SERVER=$server_version SMOKE_IMAGE=$IMAGE \
    SMOKE_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE") \
    SMOKE_NODE_MEMORY=$node_memory \
    python3 - "$REPORT" "$work" <<'PY'
import json
import os
import sys
from pathlib import Path

report, work = Path(sys.argv[1]), Path(sys.argv[2])


def rows(name):
    return [line.split("\t") for line in (work / name).read_text(encoding="utf-8").splitlines() if line]


def env(name):
    return os.environ[f"SMOKE_{name}"] or None


checks = [{"name": n, "ok": ok == "1", "time_s": float(t)} for n, ok, t in rows("checks.tsv")]
pods = json.loads((work / "pods.json").read_text(encoding="utf-8"))["items"]
report.write_text(json.dumps({
    "started_at": env("STARTED_AT"),
    "status": env("STATUS"),
    "failed_step": env("FAILED_STEP"),
    "cluster": {"name": env("CLUSTER"), "context": env("CONTEXT"),
                "created_by_this_run": env("CREATED") == "true", "deleted_at_end": env("DELETED") == "true"},
    "versions": {"kind": env("KIND"), "kubectl": env("KUBECTL"), "helm": env("HELM"),
                 "kubernetes_server": env("SERVER")},
    "image": {"name": env("IMAGE"), "id": env("IMAGE_ID")},
    "steps_s": {name: float(seconds) for name, seconds in rows("steps.tsv")},
    "checks": checks,
    "checks_passed": sum(c["ok"] for c in checks),
    "checks_total": len(checks),
    "pods": [{"name": p["metadata"]["name"], "phase": p["status"].get("phase"),
              "ready": all(c.get("ready") for c in p["status"].get("containerStatuses", [])),
              "restarts": sum(c.get("restartCount", 0) for c in p["status"].get("containerStatuses", []))}
             for p in pods],
    "node_memory_usage": env("NODE_MEMORY"),
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
  echo "report: $REPORT"
}

cleanup() {
  local code=$? status=passed
  set +e
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null; done
  [ "$code" -eq 0 ] || status=failed
  k get pods --output json >"$work/pods.json.new" 2>/dev/null && mv "$work/pods.json.new" "$work/pods.json"
  node_memory=$(docker stats --no-stream --format '{{.MemUsage}}' "$CLUSTER-control-plane" 2>/dev/null)
  if $delete; then timed delete_cluster kind delete cluster --name "$CLUSTER"; fi
  write_report "$status"
  rm -rf "$work"
  exit "$code"
}
trap cleanup EXIT

if ! kind get clusters | grep -qx "$CLUSTER"; then
  timed create_cluster kind create cluster --name "$CLUSTER" --wait 120s
  cluster_created=true
fi
server_version=$(kubectl --context "$CONTEXT" version --output json | python3 -c 'import json, sys; print(json.load(sys.stdin)["serverVersion"]["gitVersion"])')
node=$CLUSTER-control-plane
if ! docker network inspect "$NETWORK" --format '{{range .Containers}}{{.Name}} {{end}}' | grep -qw "$node"; then
  docker network connect "$NETWORK" "$node"
fi
timed load_image kind load docker-image "$IMAGE" --name "$CLUSTER"

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

timed helm_install_wait helm upgrade --install "$RELEASE" "$CHART" --kube-context "$CONTEXT" --namespace "$NAMESPACE" \
  --wait --timeout 5m
k get pods --output wide

# kubectl itself goes to the background, not the k function: $! must be its pid for the trap to kill.
forward() {
  kubectl --context "$CONTEXT" --namespace "$NAMESPACE" port-forward "service/$RELEASE-$1" "$2:$3" >/dev/null 2>&1 &
  pids+=($!)
}
forward api 18000 8000
forward exporter 19188 9188
wait_health() {
  for _ in $(seq 30); do
    curl --silent --fail --output /dev/null http://127.0.0.1:18000/health && return 0
    sleep 1
  done
  return 1
}
timed port_forward_ready wait_health

failed=0
# check NAME PATTERN CURL_ARGS...: one request, its body matched against PATTERN.
check() {
  local what=$1 pattern=$2 body t0 t
  shift 2
  t0=$(now)
  body=$(curl --silent --max-time 180 "$@" || true)
  t=$(elapsed "$t0")
  if grep -qE -- "$pattern" <<<"$body"; then
    echo "ok    $what (${t}s)"
    printf '%s\t1\t%s\n' "$what" "$t" >>"$work/checks.tsv"
  else
    echo "FAIL  $what: ${body:0:300}"
    printf '%s\t0\t%s\n' "$what" "$t" >>"$work/checks.tsv"
    failed=1
  fi
}

check "health: salary model from the registry" '"salary_model":"ok"' http://127.0.0.1:18000/health
check "health: ClickHouse through the read-only account" '"clickhouse":"ok"' http://127.0.0.1:18000/health
check "extract" '"role"' --header 'Content-Type: application/json' \
  --data '{"text": "Python-разработчик (middle), удалённо, 250 000 руб. на руки. Django, PostgreSQL."}' \
  http://127.0.0.1:18000/extract
check "api metrics" '^radar_api_requests_total\{' http://127.0.0.1:18000/metrics
check "exporter reaches ClickHouse" '^radar_clickhouse_up 1' http://127.0.0.1:19188/metrics

exit $failed
