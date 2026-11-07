#!/usr/bin/env bash
# Preview by default; --submit enqueues one zero-shot and six training jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs/e9
source scripts/artemis_env.sh
exec python -u -m carrepair.e9_submit "$@"
