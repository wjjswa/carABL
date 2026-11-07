#!/usr/bin/env bash
# Artemis login node: preview by default; add --submit to queue E8 jobs.
# Do not switch the checkout used by already-running E6/E7 jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs/e8
source scripts/artemis_env.sh
exec python -u -m carrepair.e8_submit "$@"
