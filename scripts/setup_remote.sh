#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/artemis_env.sh
# Environment must already exist; never modify the shared base installation.
python -m pip install -e '.[test]'
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
OMP_NUM_THREADS=1 python -m pytest -q
