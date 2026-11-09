#!/usr/bin/env bash
# Compatibility entry point; use the same planner/environment as E1–E5.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/submit_all.sh" --experiments 6 7 "$@"
