"""Short real-data integration checks, isolated from research results."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import pandas as pd
import torch
import yaml
from .common import read_config, dump_json
from .data import read_manifest, open_image


def run(config_path, device=None, activegcd=False):
    cfg = read_config(config_path)
    cfg['device'] = device or cfg['device']
    if str(cfg['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No GPU available; submit this smoke test as a GPU job, or explicitly use --device cpu.')
    if not Path(cfg['checkpoint']).is_file():
        raise FileNotFoundError('Download the checkpoint first: python -m carrepair download-model')
    full = read_manifest(cfg['manifest'])
    pieces = []
    for split, count in [('train', 3), ('val', 1), ('test', 1)]:
        for label in sorted(full.label.unique()):
            part = full[(full.split == split) & (full.label == label)].sort_values('id')
            if len(part) < count:
                raise ValueError(f'Smoke test requires {count} {split} images for {label}')
            pieces.append(part.head(count))
    small = pd.concat(pieces, ignore_index=True)
    for path in small.path:
        open_image(Path(cfg['data_root']) / path)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    root = Path(str(cfg['output']).rstrip('/') + '_smoke') / stamp
    root.mkdir(parents=True)
    cfg.update(manifest=str((root / 'smoke_manifest.csv').resolve()),
               output=str((root / 'artifacts').resolve()), batch_size=4, workers=0,
               epochs=1, probe_epochs=1, seeds=[0])
    small.to_csv(cfg['manifest'], index=False)
    cfg['dino'].update(epochs=1, local_crops=1)
    cfg['activegcd'].update(base_epochs=1, round_epochs=1, rounds=1, queries=3,
                            batch_size=4, al_batch_size=3, labelled_known_fraction=.5)
    (root / 'smoke_config.yaml').write_text(yaml.safe_dump(cfg))
    report = {'purpose': 'Code execution check only; NOT benchmark results',
              'source_manifest': str(Path(read_config(config_path)['manifest']).resolve()),
              'device': str(cfg['device']), 'passed': [], 'activegcd_requested': activegcd}
    def check(name, fn):
        print(f'\n[SMOKE] {name}', flush=True)
        try:
            fn()
            report['passed'].append(name)
            dump_json(root / 'smoke_report.json', report)
        except Exception as error:
            report['failed'] = {'stage': name, 'error': str(error)}
            dump_json(root / 'smoke_report.json', report)
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    from .inference import direct
    from .training import supervised, dino_adapt, dino_probe
    from .evaluation import export_results
    check('Exp1 inference and image exports', lambda: direct(cfg))
    for method in ['full', 'lora', 'block', 'linear']:
        check(f'Exp2 {method}: one epoch + linear/kNN',
              lambda m=method: supervised(cfg, 2, 0, m, 1.))
    for method in ['full', 'lora', 'block']:
        check(f'Exp3 {method}: one DINO epoch', lambda m=method: dino_adapt(cfg, 0, m))
        check(f'Exp3 {method}: frozen probe + kNN', lambda m=method: dino_probe(cfg, 0, m, 1.))
    check('Exp4 scratch: one epoch + linear/kNN', lambda: supervised(cfg, 4, 0, 'full', 1.))
    if activegcd:
        from .activegcd import run as ag_run
        for strategy in ['random', 'adaptive']:
            check(f'Exp5B {strategy}: base + one annotation round', lambda s=strategy: ag_run(cfg, 0, s))
    check('Comparison export', lambda: export_results(cfg['output']))
    report['status'] = 'PASS'
    dump_json(root / 'smoke_report.json', report)
    print(f'\nSMOKE TEST PASSED: {root / "smoke_report.json"}', flush=True)
    print('Original split/config/results were not modified. Do not use smoke scores as research results.', flush=True)
    return root
