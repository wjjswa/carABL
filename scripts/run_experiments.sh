#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/artemis_env.sh
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(), "No GPU available. Run this training script inside an allocated GPU job."; print("GPU:", torch.cuda.get_device_name(0))'
TASK_CONFIG="${TASK_CONFIG:-configs/default.yaml}"
python -m carrepair matrix --config "$TASK_CONFIG"
python -m carrepair matrix --config "$TASK_CONFIG" --run
python -c 'import sys; from carrepair.common import read_config; from carrepair.evaluation import export_results; export_results(read_config(sys.argv[1])["output"])' "$TASK_CONFIG"
