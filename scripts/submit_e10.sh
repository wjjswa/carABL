#!/usr/bin/env bash
# Preview by default; --submit enqueues dependent GPU jobs for root, experts, and evaluation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs/e10
source scripts/artemis_env.sh
exec python -u -m carrepair.e10_submit "$@"
