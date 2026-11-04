#!/usr/bin/env bash
# Source this file from an Artemis shell or job script.
# Site module/Conda scripts may reference unset variables.
set +u
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
        if [[ -r "$init" ]]; then
            source "$init"
            break
        fi
    done
fi
if ! type module >/dev/null 2>&1; then
    echo "Artemis module command is unavailable; initialise your site's module shell first." >&2
    return 1
fi
module load CUDA/12.4.0
module load Miniconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
# sbatch may inherit CONDA_SHLVL/CONDA_PREFIX from an active login environment.
# Loading the module can reset PATH to base; reactivating the same name may then
# be a no-op. Switching via base forces Conda to rebuild the project PATH.
conda activate base
conda activate "${CARABL_CONDA_ENV:-carabl}"
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
set -u
export PYTHONNOUSERSITE=1
python -c 'import os,sys; print("Python:", sys.version.split()[0], "Executable:", sys.executable, "Environment:", sys.prefix); assert sys.version_info >= (3, 10), "carABL requires Python >= 3.10"; assert os.path.realpath(sys.prefix) == os.path.realpath(os.environ["CONDA_PREFIX"]), "Python does not match the activated Conda environment"'
