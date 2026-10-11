#!/usr/bin/env bash
# Entrypoint of the superset service: migrate the metadata database, sync roles and permissions, set the
# admin account from SUPERSET_ADMIN_PASSWORD, then serve. Each step is a no-op when already done, so it
# runs on every start and a password changed in .env reaches the account without a volume reset.
set -euo pipefail

if [ -z "${SUPERSET_ADMIN_PASSWORD:-}" ]; then
  echo "SUPERSET_ADMIN_PASSWORD is empty: set it in .env (see .env.example)" >&2
  exit 1
fi

superset db upgrade
superset init
python /app/radar/admin_account.py
exec /usr/bin/run-server.sh
