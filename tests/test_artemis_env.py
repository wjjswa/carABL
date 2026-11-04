"""Regression: an inherited active Conda name must not skip PATH rebuilding."""
import os
from pathlib import Path
import subprocess
import sys


def test_artemis_reactivates_inherited_environment(tmp_path):
    profile = tmp_path / 'etc/profile.d'
    profile.mkdir(parents=True)
    log = tmp_path / 'activation.log'
    (profile / 'conda.sh').write_text('''
conda() {
    if [[ "$1" == activate ]]; then
        printf '%s\\n' "$2" >> "$ACTIVATION_LOG"
        if [[ "$2" == "$CONDA_DEFAULT_ENV" ]]; then return 0; fi
        export CONDA_DEFAULT_ENV="$2"
        if [[ "$2" == base ]]; then
            export CONDA_PREFIX="$MOCK_BASE"
        else
            export CONDA_PREFIX="$REAL_PREFIX"
        fi
    fi
}
''')
    script = Path('scripts/artemis_env.sh').resolve()
    shell = '''set -euo pipefail
module() { :; }
conda() { printf '%s\\n' "$MOCK_BASE"; }
source "$ENV_SCRIPT"
[[ "$CONDA_DEFAULT_ENV" == carabl ]]
[[ "$CONDA_PREFIX" == "$REAL_PREFIX" ]]
'''
    env = {**os.environ, 'MOCK_BASE': str(tmp_path), 'ACTIVATION_LOG': str(log),
           'REAL_PREFIX': sys.prefix, 'ENV_SCRIPT': str(script),
           'CONDA_DEFAULT_ENV': 'carabl', 'CONDA_PREFIX': '/stale/inherited/prefix',
           'CONDA_SHLVL': '1'}
    subprocess.run(['bash', '-c', shell], env=env, check=True, capture_output=True, text=True)
    assert log.read_text().splitlines() == ['base', 'carabl']
