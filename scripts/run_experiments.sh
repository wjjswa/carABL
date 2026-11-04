#!/usr/bin/env bash
set -euo pipefail
TASK_CONFIG="${TASK_CONFIG:-configs/default.yaml}"
# Prints the entire matrix first; --run executes in dependency order.
python -m carrepair matrix --config "$TASK_CONFIG"
python -m carrepair matrix --config "$TASK_CONFIG" --run
python -c 'import sys; from carrepair.common import read_config; from carrepair.evaluation import export_results; export_results(read_config(sys.argv[1])["output"])' "$TASK_CONFIG"
