#!/usr/bin/env bash
# One topic per source. Safe to rerun: existing topics keep their data and settings.
# A single partition keeps per-source order; a day is a few thousand messages, a snapshot about 550 thousand.
set -euo pipefail

for topic in raw.trudvsem raw.trudvsem_snapshot raw.habr raw.telegram raw.superjob; do
  if rpk topic describe "$topic" >/dev/null 2>&1; then
    echo "exists: $topic"
  else
    rpk topic create "$topic" --partitions 1 --replicas 1
  fi
done
