#!/usr/bin/env bash
# E11 dry-run by default; --submit queues six dependency-ordered GPU jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs/e11
source scripts/artemis_env.sh
exec python -u -m carrepair.e11_submit "$@"
