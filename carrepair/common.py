from pathlib import Path
import hashlib
import json
import random
import subprocess
import numpy as np
import torch
import yaml


def read_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def provenance(cfg, directory):
    import importlib.metadata
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        revision = 'uncommitted'
    versions = {}
    for p in ['torch', 'torchvision', 'ultralytics', 'numpy', 'scikit-learn', 'streamlit']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {
        'config': cfg, 'git_commit': revision, 'versions': versions,
        'manifest_sha256': file_hash(cfg['manifest']),
        'checkpoint_sha256': file_hash(cfg['checkpoint']) if Path(cfg['checkpoint']).exists() else None,
    }
    destination = Path(directory) / 'provenance.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        for key in ['config', 'manifest_sha256', 'checkpoint_sha256']:
            if previous.get(key) != record.get(key):
                raise ValueError(f'Resume provenance mismatch: {key}. Use a fresh output directory.')
        return
    dump_json(destination, record)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)
