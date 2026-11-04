#!/usr/bin/env bash
set -euo pipefail
# Activate a dedicated Conda environment first; install the CUDA wheel suitable
# for your allocated GPU/driver before this script, as described in README.md.
python -m pip install -e '.[test]'
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
python -m pytest -q
