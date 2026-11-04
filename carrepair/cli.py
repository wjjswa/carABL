import argparse
from pathlib import Path
import json
import subprocess
import sys
from .common import read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description='carABL experiments 1–4 and 5B; no 5A')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--root', required=True); prep.add_argument('--out', default='data/prepared')
    prep.add_argument('--csv'); prep.add_argument('--path-column'); prep.add_argument('--label-column')
    prep.add_argument('--group-column'); prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--near-distance', type=int, default=2)
    download = sub.add_parser('download-model')
    download.add_argument('--out', default='weights/cardd-v2.pt')
    for name in ['direct', 'supervised', 'dino', 'probe', 'activegcd', 'matrix']:
        p = sub.add_parser(name)
        p.add_argument('--config', default='configs/default.yaml')
        p.add_argument('--device'); p.add_argument('--output'); p.add_argument('--resume', action='store_true')
        if name in ['supervised', 'dino', 'probe', 'activegcd']:
            p.add_argument('--seed', type=int, default=0)
        if name in ['supervised', 'dino', 'probe']:
            p.add_argument('--method', choices=['full', 'lora', 'block', 'linear'], default='full')
        if name in ['supervised', 'probe']:
            p.add_argument('--fraction', type=float, default=1.)
        if name == 'supervised':
            p.add_argument('--experiment', type=int, choices=[2, 4], default=2)
        if name == 'activegcd':
            p.add_argument('--strategy', choices=['adaptive', 'random'], default='adaptive')
        if name == 'direct':
            p.add_argument('--confidence', type=float, default=.25)
            p.add_argument('--image-size', type=int, default=1024)
        if name == 'matrix':
            p.add_argument('--run', action='store_true', help='Without this flag only print commands')
            p.add_argument('--experiments', nargs='+', default=['1', '2', '3', '4', '5b'])
    export = sub.add_parser('export')
    export.add_argument('--output', default='outputs')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .data import prepare
        print(prepare(args.root, args.out, args.csv, args.path_column, args.label_column, args.group_column, args.seed, args.near_distance).groupby(['split', 'label']).size())
        return
    if args.command == 'download-model':
        from huggingface_hub import hf_hub_download
        import shutil
        target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download('abdullahg7/cardd-yolov8s', 'v2.0/best.pt')
        shutil.copy2(path, target)
        from .common import dump_json, file_hash
        dump_json(target.with_suffix('.json'), {'source': 'abdullahg7/cardd-yolov8s/v2.0/best.pt', 'sha256': file_hash(target)})
        print(target)
        return
    if args.command == 'export':
        from .evaluation import export_results
        print(export_results(args.output))
        return
    cfg = read_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.output:
        cfg['output'] = args.output
    if args.command == 'matrix':
        commands = build_matrix(args.config, cfg, args.experiments)
        for command in commands:
            if args.device:
                command += ['--device', args.device]
            if args.output:
                command += ['--output', args.output]
            if args.resume and 'activegcd' not in command:
                command += ['--resume']
            import shlex
            print(shlex.join(command), flush=True)
            if args.run:
                subprocess.run(command, check=True)
        return
    if args.command == 'direct':
        from .inference import direct
        direct(cfg, args.confidence, args.image_size)
    elif args.command == 'supervised':
        from .training import supervised
        supervised(cfg, args.experiment, args.seed, args.method, args.fraction, args.resume)
    elif args.command == 'dino':
        from .training import dino_adapt
        dino_adapt(cfg, args.seed, args.method, args.resume)
    elif args.command == 'probe':
        from .training import dino_probe
        dino_probe(cfg, args.seed, args.method, args.fraction, args.resume)
    elif args.command == 'activegcd':
        from .activegcd import run
        run(cfg, args.seed, args.strategy, args.resume)


def build_matrix(config_path, cfg, experiments):
    commands = []
    def cmd(name, *args):
        return [sys.executable, '-m', 'carrepair', name, '--config', str(config_path), *map(str, args)]
    if '1' in experiments:
        commands.append(cmd('direct'))
    for seed in cfg['seeds']:
        if '2' in experiments:
            for method in cfg['methods']:
                for p in cfg['fractions']:
                    commands.append(cmd('supervised', '--experiment', 2, '--seed', seed, '--method', method, '--fraction', p))
        if '3' in experiments:
            for method in ['full', 'lora', 'block']:
                commands.append(cmd('dino', '--seed', seed, '--method', method))
                for p in cfg['fractions']:
                    commands.append(cmd('probe', '--seed', seed, '--method', method, '--fraction', p))
        if '4' in experiments:
            for p in cfg['fractions']:
                commands.append(cmd('supervised', '--experiment', 4, '--seed', seed, '--method', 'full', '--fraction', p))
        if '5b' in experiments:
            for strategy in ['random', 'adaptive']:
                commands.append(cmd('activegcd', '--seed', seed, '--strategy', strategy))
    return commands


if __name__ == '__main__':
    main()
