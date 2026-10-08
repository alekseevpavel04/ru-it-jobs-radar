#!/usr/bin/env bash
# Everything a change must pass before it is called done. Same command locally and in CI.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then
  BIN=.venv/Scripts
else
  BIN=.venv/bin
fi

echo "== ruff"
"$BIN/ruff" check .
"$BIN/ruff" format --check .
echo "== mypy"
"$BIN/mypy" src/radar
echo "== pytest"
"$BIN/pytest" -q
echo "== verify_numbers"
"$BIN/python" scripts/verify_numbers.py
