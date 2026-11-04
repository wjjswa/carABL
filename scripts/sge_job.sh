#!/usr/bin/env bash
#$ -N carABL
#$ -cwd
#$ -j y
# Legacy SGE template: add your site's GPU/memory/parallel resource requests.
set -euo pipefail
cd "${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source scripts/artemis_env.sh
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "No GPU available. Run this training script inside an allocated GPU job."; print("GPU:", torch.cuda.get_device_name(0))'
if [[ $# -eq 0 ]]; then
    echo "Usage: qsub scripts/sge_job.sh supervised --experiment 2 --method full --fraction .2 --seed 0" >&2
    exit 2
fi
python -m carrepair "$@"
