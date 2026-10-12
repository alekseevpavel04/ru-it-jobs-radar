#!/usr/bin/env bash
# Start llama.cpp server with a teacher model (Git Bash on Windows), OpenAI-compatible API.
# Usage: scripts/llm_server.sh qwen|yandex [parallel_slots]
#        STUDENT_GGUF=models/gguf/<name>/<name>-Q4_K_M.gguf scripts/llm_server.sh student [parallel_slots]
# Env: LLAMA_PORT (default 8080), LLAMA_CTX_PER_SLOT (default 4096), LLAMA_CACHE_RAM (MiB, default 0),
# LLAMA_DIR, LLM_MODELS_DIR, LLAMA_KV_TYPE (KV cache type, q8_0 for the teachers, f16 for the student).
# Qwen3 thinking is switched off per request: "chat_template_kwargs": {"enable_thinking": false}.
# Each slot has its own LLAMA_CTX_PER_SLOT tokens of KV cache (-no-kvu). The teacher prompt alone is
# about 5 300 tokens, so teacher labeling runs with 8192 per slot. A shared pool (-kvu) was tried for
# the teachers and failed: idle slots keep their cached prompts, the pool fills up and requests stop
# with "Context size has been exceeded".
# The host-RAM prompt cache (-cram, 8 GiB by default) is off unless LLAMA_CACHE_RAM sets it: every
# post is a new prompt and the shared system prompt stays in each slot's KV on the card, while the
# cache kept entries of about 340 MiB and took the server to 12.9 GB of RAM on 2026-10-11.
set -euo pipefail
LLAMA="${LLAMA_DIR:-/d/VScode_projects/assets/tools/llama.cpp/b11052}/llama-server.exe"
MODELS="${LLM_MODELS_DIR:-D:/VScode_projects/assets/llm}"
SLOTS="${2:-2}"
CTX_PER_SLOT="${LLAMA_CTX_PER_SLOT:-4096}"
KV="${LLAMA_KV_TYPE:-q8_0}"
case "${1:-}" in
  qwen)   MODEL="$MODELS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"; ALIAS=Qwen3-8B-Q4_K_M ;;
  yandex) MODEL="$MODELS/yandexgpt5-lite-8b/YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"; ALIAS=YandexGPT-5-Lite-8B-instruct-Q4_K_M ;;
  # the student's prompt is about 300 tokens and its input is cut to 1536, so 4096 per slot holds an answer
  # f16 KV: the variants are compared with the student under transformers, whose cache is not quantized
  student) MODEL="${STUDENT_GGUF:?set STUDENT_GGUF to the student .gguf (radar.train.quantize)}"; ALIAS=$(basename "$MODEL" .gguf)
           KV="${LLAMA_KV_TYPE:-f16}" ;;
  *) echo "usage: $0 qwen|yandex|student [parallel_slots]" >&2; exit 1 ;;
esac
exec "$LLAMA" -m "$MODEL" --alias "$ALIAS" --host 127.0.0.1 --port "${LLAMA_PORT:-8080}" \
  -ngl 99 -fa on -ctk "$KV" -ctv "$KV" -np "$SLOTS" -c $((SLOTS * CTX_PER_SLOT)) -no-kvu -cram "${LLAMA_CACHE_RAM:-0}" --jinja
