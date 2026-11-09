#!/usr/bin/env bash
#SBATCH --job-name=carABL-e15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/artemis_env.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python - <<'PY'
import torch
print("CUDA", torch.version.cuda, flush=True)
assert torch.cuda.is_available(), "GPU allocation required"
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY
exec python -u -m carrepair.e15 "$@"
