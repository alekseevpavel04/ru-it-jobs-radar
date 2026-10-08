#!/usr/bin/env bash
# Start the API server (UI and REST) with the single local account `admin`. The simple auth manager
# reads passwords from a JSON file; it is written from AIRFLOW_ADMIN_PASSWORD on every start, so the
# password in .env is the only place it lives.
set -euo pipefail
: "${AIRFLOW_ADMIN_PASSWORD:?set AIRFLOW_ADMIN_PASSWORD in .env}"
: "${AIRFLOW__API_AUTH__JWT_SECRET:?set AIRFLOW_JWT_SECRET in .env}"

passwords=$AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE
(
  umask 077
  python -c 'import json, os, sys; json.dump({"admin": os.environ["AIRFLOW_ADMIN_PASSWORD"]}, open(sys.argv[1], "w"))' \
    "$passwords"
)
exec airflow api-server --port 8080
