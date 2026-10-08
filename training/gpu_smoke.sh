#!/usr/bin/env bash
# GPU smoke test of the whole LLM stack on one 8 GB card, one heavy process at a time.
# Run from Git Bash on Windows: bash training/gpu_smoke.sh
# Teachers run in llama-server on Windows; QLoRA and vLLM run inside the WSL distro `radar`
# (venvs ~/venvs/train and ~/venvs/vllm, see training/requirements-*.txt). Each step writes
# logs/gpu_smoke/<tag>.json and logs/<tag>.log; a failed step is reported and the rest still run.
# Result: results/env/gpu_smoke.json.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_WSL="${REPO_WSL:-/mnt/d/VScode_projects/ru-it-jobs-radar}"
PY=.venv/Scripts/python
failed=()

step() {
  local tag="$1"
  shift
  echo "== $tag ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use before)"
  if "$@" > "logs/$tag.log" 2>&1; then
    tail -1 "logs/$tag.log"
  else
    echo "   FAILED, see logs/$tag.log"
    failed+=("$tag")
  fi
}

wsl_run() {
  MSYS_NO_PATHCONV=1 wsl.exe -d radar --exec bash -lc "cd $REPO_WSL && bash training/wsl_run.sh $*"
}

mkdir -p logs
step llama_qwen "$PY" training/smoke_llama.py qwen --requests 10
step llama_yandex "$PY" training/smoke_llama.py yandex --requests 1

for model in 1p7b:Qwen/Qwen3-1.7B 4b:Qwen/Qwen3-4B; do
  for backend in unsloth peft; do
    tag="qlora_${model%%:*}_$backend"
    step "$tag" wsl_run train smoke_qlora.py --model "${model#*:}" --backend "$backend" --batch 2 --grad-accum 8 --tag "$tag"
  done
done

step vllm_1p7b wsl_run vllm smoke_vllm.py --model Qwen/Qwen3-1.7B --gpu-util 0.85 --tag vllm_1p7b
step vllm_4b_awq wsl_run vllm smoke_vllm.py --model Qwen/Qwen3-4B-AWQ --gpu-util 0.85 --tag vllm_4b_awq
step vllm_4b_bnb wsl_run vllm smoke_vllm.py --model Qwen/Qwen3-4B --quantization bitsandbytes --gpu-util 0.85 --tag vllm_4b_bnb

"$PY" training/smoke_collect.py
[ ${#failed[@]} -eq 0 ] || echo "failed steps: ${failed[*]}"
