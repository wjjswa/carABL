#!/usr/bin/env bash
#SBATCH --job-name=carABL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=carABL-%j.log
# TEMPLATE: confirm scheduler/partition/GPU directives with your platform.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# Activate your Conda environment before submission, or add its activation here.
python -m carrepair "$@"
