#!/usr/bin/env bash
# The main path against the live stack. Brings up the core and api profiles with scripts/up.sh
# (running services stay as they are), then walks health, /extract on a fixture vacancy, /stats,
# /salary on the extracted fields and an MCP tool call (tests/api/test_e2e.py). Exits non-zero at
# the first failure.
#   scripts/e2e.sh           up, then the path
#   scripts/e2e.sh --no-up   the path against what is running
# RADAR_API_URL and RADAR_MCP_URL override http://127.0.0.1:8000 and http://127.0.0.1:8001/mcp.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then
  BIN=.venv/Scripts
else
  BIN=.venv/bin
fi

case "${1:-}" in
  --no-up) ;;
  "") bash scripts/up.sh core api ;;
  *)
    echo "usage: scripts/e2e.sh [--no-up]" >&2
    exit 2
    ;;
esac

echo "== main path"
"$BIN/pytest" -m slow -p no:cacheprovider -v tests/api/test_e2e.py
