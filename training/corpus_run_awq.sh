#!/usr/bin/env bash
# the whole corpus through the AWQ student behind vLLM on 127.0.0.1:8095, in stages so the
# portal sources and recent Telegram posts land first. Inside WSL, from the repo root:
#   setsid nohup bash training/corpus_run_awq.sh >> logs/corpus_run_awq.log 2>&1 < /dev/null &
# Stop: kill -TERM -- -<pid of this script> (its process group; the runner finishes the requests in
# flight and loads what it wrote). Resume: the same line. A finished stage only re-enumerates.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HOME="${HF_HOME:-/mnt/d/VScode_projects/assets/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

stages=(
  "--sources trudvsem,trudvsem_snapshot,habr,superjob"
  "--sources telegram --telegram-since 2025-01-01"
  "--sources telegram --telegram-since 2023-01-01"
  "--sources telegram"
)
for stage in "${stages[@]}"; do
  echo "$(date '+%F %T') stage: $stage"
  # shellcheck disable=SC2086  # the stage is a list of arguments
  "$HOME/venvs/train/bin/python" -m radar.extract.corpus_run run \
    --backend radar.extract.student_server:backend --backend-opt base_url=http://127.0.0.1:8095 \
    --workers 8 $stage
done
echo "$(date '+%F %T') all stages done"
