#!/usr/bin/env bash
#$ -N carABL
#$ -cwd
#$ -j y
# Add your site's GPU, memory, time and parallel-environment resource requests.
# This is a template: use your platform's documented resource names.
set -euo pipefail
# Activate your carabl Conda environment here if the scheduler does not inherit it.
python -m carrepair "$@"
