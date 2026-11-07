#!/usr/bin/env bash
#SBATCH --job-name=carABL-e13e14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/artemis_env.sh
python - <<'PY'
import torch
print("CUDA", torch.version.cuda, flush=True)
assert torch.cuda.is_available(), "Slurm GPU allocation required"
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY
exec python -u -m carrepair.e13e14 "$@"
