#!/usr/bin/env bash
# Run on the LOGIN node: this only plans/submits jobs, never trains here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/artemis_env.sh
python -u -m carrepair.submit "$@"
