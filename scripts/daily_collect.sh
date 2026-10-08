#!/usr/bin/env bash
# Daily collection until Airflow takes over. One source after another, never in parallel:
# the trudvsem API and the snapshot files live on the same host. A failed source does not stop the
# others; the script exits non-zero if any failed. Each run appends to logs/daily/<date>.log, and
# the last line for every source says how it ended.
#   scripts/daily_collect.sh                 all sources
#   scripts/daily_collect.sh habr trudvsem   only these
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=.venv/bin/python
fi

sources=("$@")
[ ${#sources[@]} -gt 0 ] || sources=(trudvsem habr trudvsem_snapshot)

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

failed=()
for source in "${sources[@]}"; do
  echo "$(date '+%F %T') == $source start" >>"$day_log"
  started=$SECONDS
  "$PY" -m "radar.collect.$source" >>"$day_log" 2>&1
  code=$?
  echo "$(date '+%F %T') == $source exit $code in $((SECONDS - started)) s" | tee -a "$day_log"
  [ "$code" -eq 0 ] || failed+=("$source")
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "$(date '+%F %T') failed: ${failed[*]}" | tee -a "$day_log"
  exit 1
fi
