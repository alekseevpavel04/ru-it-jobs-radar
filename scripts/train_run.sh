#!/usr/bin/env bash
# Run a radar.train module (dataset, sft, predict) in the WSL training venv ~/venvs/train.
# From Git Bash it re-runs itself inside the distro; inside WSL it runs directly.
#   bash scripts/train_run.sh dataset
#   bash scripts/train_run.sh sft --config configs/sft/qwen3_1p7b.toml --probe 50
#   bash scripts/train_run.sh predict --adapter models/sft/qwen3_1p7b_r16/adapter --split dev --system qwen3_1p7b_sft_r16
# Models come from the shared HF cache on D: and are never downloaded here (HF_HUB_OFFLINE).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    # shellcheck source=../infra/compose/lib.sh
    source "$ROOT/infra/compose/lib.sh"
    MSYS_NO_PATHCONV=1 exec wsl.exe -d "${RADAR_WSL_DISTRO:-radar}" --cd "$(to_wsl_path "$ROOT")" \
      --exec bash scripts/train_run.sh "$@"
    ;;
esac

module=$1
shift
export HF_HOME="${HF_HOME:-/mnt/d/VScode_projects/assets/huggingface}"
export HF_HUB_OFFLINE=1
export UNSLOTH_COMPILE_LOCATION="$HOME/.cache/radar/unsloth_compiled_cache"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec "$HOME/venvs/train/bin/python" -m "radar.train.$module" "$@"
