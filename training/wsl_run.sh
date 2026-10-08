#!/usr/bin/env bash
# Run a training-side script inside the WSL distro with the right venv and caches.
# Usage (inside WSL): training/wsl_run.sh train|vllm script.py [args...]
# Venvs live on the distro's ext4 (~/venvs/train, ~/venvs/vllm): vLLM pins a different torch than Unsloth.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$HOME/venvs/$1"
SCRIPT="$2"
shift 2
export HF_HOME="${HF_HOME:-/mnt/d/VScode_projects/assets/huggingface}"
export UNSLOTH_COMPILE_LOCATION="$HOME/.cache/radar/unsloth_compiled_cache"
export PYTHONUNBUFFERED=1
exec "$VENV/bin/python" "training/$SCRIPT" "$@"
