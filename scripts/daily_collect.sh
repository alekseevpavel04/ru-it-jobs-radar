#!/usr/bin/env bash
# Daily collection until Airflow takes over. One source after another, never in parallel:
# the trudvsem API and the snapshot files live on the same host. A failed source does not stop the
# others; the script exits non-zero if any failed. Each run appends to logs/daily/<date>.log, and
# the last line for every source says how it ended. Logs older than RADAR_LOG_DAYS (60) are removed.
#   scripts/daily_collect.sh                 all sources
#   scripts/daily_collect.sh habr trudvsem   only these
# RADAR_SNAPSHOT_SINCE=YYYY-MM-DD passes --since to the snapshot collector: the daily server takes
# new versions only and leaves older ones to the backfill on another machine (docs/vps.md).
# The stack is brought up only where it is defined (infra/compose); the daily server runs the
# collectors alone, and they finish with a warning when the stream is not configured. Without
# SUPERJOB_APP_KEY in .env the SuperJob collector logs a warning and exits 0.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=.venv/bin/python
fi

sources=("$@")
[ ${#sources[@]} -gt 0 ] || sources=(trudvsem habr trudvsem_snapshot superjob)

mkdir -p logs/daily
day_log="logs/daily/$(date '+%Y-%m-%d').log"
lock=logs/daily/.lock
# A lock older than 20 hours was left by a run that died with the machine.
find logs/daily -maxdepth 1 -name .lock -mmin +1200 -exec rmdir {} \; 2>/dev/null
if ! mkdir "$lock" 2>/dev/null; then
  echo "$(date '+%F %T') another daily run holds $lock, exiting" | tee -a "$day_log"
  exit 1
fi
trap 'rmdir "$lock"' EXIT
find logs/daily -maxdepth 1 -type f -name '*.log' -mtime "+${RADAR_LOG_DAYS:-60}" -delete

# The WSL VM shuts down when no wsl.exe session is open, and with it Redpanda and ClickHouse.
# A sleeping session keeps the stack up for the run; collectors still finish if it is down.
keepalive=
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    MSYS_NO_PATHCONV=1 wsl.exe -d "${RADAR_WSL_DISTRO:-radar}" --exec sleep 14400 &
    keepalive=$!
    trap 'kill "$keepalive" 2>/dev/null; rmdir "$lock"' EXIT
    ;;
esac
if [ -f infra/compose/docker-compose.yml ]; then
  scripts/up.sh core >>"$day_log" 2>&1 || echo "$(date '+%F %T') stack is not up, partitions wait for a replay" | tee -a "$day_log"
fi

failed=()
for source in "${sources[@]}"; do
  args=()
  if [ "$source" = trudvsem_snapshot ] && [ -n "${RADAR_SNAPSHOT_SINCE:-}" ]; then
    args=(--since "$RADAR_SNAPSHOT_SINCE")
  fi
  echo "$(date '+%F %T') == $source start ${args[*]}" >>"$day_log"
  started=$SECONDS
  "$PY" -m "radar.collect.$source" "${args[@]}" >>"$day_log" 2>&1
  code=$?
  echo "$(date '+%F %T') == $source exit $code in $((SECONDS - started)) s" | tee -a "$day_log"
  [ "$code" -eq 0 ] || failed+=("$source")
done

# ClickHouse reads the topics on its own; a snapshot day took 150 s to land (results/stream/replay.json).
[ -z "$keepalive" ] || sleep 300

if [ ${#failed[@]} -gt 0 ]; then
  echo "$(date '+%F %T') failed: ${failed[*]}" | tee -a "$day_log"
  exit 1
fi
