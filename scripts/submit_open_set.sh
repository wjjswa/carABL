#!/usr/bin/env bash
# One allocation runs both initialisations for every configured seed and writes the comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/default.yaml}"
[[ -r "$config" ]] || { echo "Config missing: $config" >&2; exit 1; }
mkdir -p logs
sbatch --job-name=abl-open-set --output=logs/open-set-%j.log \
    scripts/slurm_job.sh open-set --config "$config" --resume
