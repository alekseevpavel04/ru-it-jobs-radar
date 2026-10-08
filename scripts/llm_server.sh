#!/usr/bin/env bash
# Start llama.cpp server with a teacher model (Git Bash on Windows), OpenAI-compatible API.
# Usage: scripts/llm_server.sh qwen|yandex [parallel_slots]
# Env: LLAMA_PORT (default 8080), LLAMA_CTX_PER_SLOT (default 4096), LLAMA_DIR, LLM_MODELS_DIR.
# Qwen3 thinking is switched off per request: "chat_template_kwargs": {"enable_thinking": false}.
set -euo pipefail
LLAMA="${LLAMA_DIR:-/d/VScode_projects/assets/tools/llama.cpp/b11052}/llama-server.exe"
MODELS="${LLM_MODELS_DIR:-D:/VScode_projects/assets/llm}"
SLOTS="${2:-2}"
CTX_PER_SLOT="${LLAMA_CTX_PER_SLOT:-4096}"
case "${1:-}" in
  qwen)   MODEL="$MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"; ALIAS=Qwen3-8B-Q4_K_M ;;
  yandex) MODEL="$MODELS/yandexgpt5-lite-8b/YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"; ALIAS=YandexGPT-5-Lite-8B-instruct-Q4_K_M ;;
  *) echo "usage: $0 qwen|yandex [parallel_slots]" >&2; exit 1 ;;
esac
exec "$LLAMA" -m "$MODEL" --alias "$ALIAS" --host 127.0.0.1 --port "${LLAMA_PORT:-8080}" \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np "$SLOTS" -c $((SLOTS * CTX_PER_SLOT)) --jinja
