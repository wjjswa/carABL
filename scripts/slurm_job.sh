#!/usr/bin/env bash
#SBATCH --job-name=carABL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# gpu allows up to 3 days. Smoke jobs should override --time=00:30:00.
#SBATCH --output=carABL-%j.log
# Resource/partition directives remain site-specific; adjust to your allocation.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Slurm spools this script elsewhere; use the submission directory in batch.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$PROJECT_ROOT"
source scripts/artemis_env.sh
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "No GPU available. Run this training script inside an allocated GPU job."; print("GPU:", torch.cuda.get_device_name(0))'
if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch scripts/slurm_job.sh supervised --experiment 2 --method full --fraction .2 --seed 0" >&2
    exit 2
fi
python -u -m carrepair "$@"
