#!/usr/bin/env bash
# Telegram channel survey; what it measures is in analyze.py.
# `bash scripts/telegram_survey/run.sh --offline` repeats it from the page cache without calling t.me.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=.venv/bin/python
fi

"$PY" scripts/telegram_survey/shallow.py "$@"
"$PY" scripts/telegram_survey/deep.py "$@"
"$PY" scripts/telegram_survey/analyze.py
