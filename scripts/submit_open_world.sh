#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for method in e6 e7; do
    echo "Submitting ${method} seed ${seed}"
    sbatch --partition=gpu --time=1-00:00:00 --job-name="carABL-${method}-s${seed}" scripts/slurm_job.sh open-world --method "$method" --seed "$seed"
  done
done
echo "After all six jobs finish: python -m carrepair open-world --aggregate"
 