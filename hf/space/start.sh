#!/usr/bin/env bash
# llama-server on 127.0.0.1:8081 and the app on 7860; the container stops when either stops.
set -euo pipefail

model="${GGUF_PATH:-}"
if [ -z "$model" ]; then
  model="$(python -m app.fetch)"
fi

# One slot: the free Space has two vCPUs, a second request in parallel would only halve the speed.
llama-server -m "$model" --host 127.0.0.1 --port 8081 --ctx-size 4096 --parallel 1 \
  --threads "${LLAMA_THREADS:-2}" --no-webui &
uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1 &

trap 'kill 0' TERM INT
wait -n
exit 1
