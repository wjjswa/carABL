#!/usr/bin/env bash
#SBATCH --job-name=carABL-e11
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/artemis_env.sh
python -c 'import torch; print("CUDA:",torch.version.cuda); assert torch.cuda.is_available(), "E11 needs allocated GPU"; print(torch.cuda.get_device_name(0))'
exec python -u -m carrepair.e11 "$@"
