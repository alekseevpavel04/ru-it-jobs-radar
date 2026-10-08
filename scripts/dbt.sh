#!/usr/bin/env bash
# dbt with the project's profile and the connection from .env (values are never printed).
#   scripts/dbt.sh build                  seeds, staging models and their tests on the running ClickHouse
#   scripts/dbt.sh source freshness       raw.vacancies per source
#   scripts/dbt.sh docs generate
#   scripts/dbt.sh report                 results/dbt/build.json from the last build and freshness check
# Works from Git Bash on Windows and from Linux. Needs the core stack (scripts/up.sh core).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=.venv/bin/python
fi

export DBT_PROJECT_DIR=dbt DBT_PROFILES_DIR=dbt PYTHONIOENCODING=utf-8

if [ "${1:-}" = report ]; then
  shift
  exec "$PY" dbt/report.py "$@"
fi

# radar.config.load_dotenv keeps variables already set in the shell, as the collectors do.
status=0
"$PY" -P -c '
import sys
from radar.config import load_dotenv
from dbt.cli.main import cli

load_dotenv()
sys.argv[0] = "dbt"
cli()
' "$@" || status=$?

# Every dbt command rewrites run_results.json; the report needs the one of the last build.
if [ "${1:-}" = build ] && [ -f dbt/target/run_results.json ]; then
  cp dbt/target/run_results.json dbt/target/build_results.json
fi
exit "$status"
