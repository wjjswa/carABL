"""Plan/submit separate Slurm jobs; train nothing on the login node.

Default is a dry run. --submit enqueues the plan, recording each returned job ID
before continuing. DINO probes use afterok on their matching adaptation job.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from .common import read_config, file_hash


@dataclass
class Job:
    key: str
    args: list
    directory: Path
    completion: Path
    parent: str | None = None


def build_jobs(cfg, config_path, experiments):
    output, jobs = Path(cfg['output']), []
    seeds, fractions = cfg['seeds'], cfg['fractions']
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds')
    if any(not 0 < f <= 1 for f in fractions) or len({int(f*100) for f in fractions}) != len(fractions):
        raise ValueError('Fractions must be positive, <=1, and have distinct output percentage names')
    if set(cfg['methods']) - {'full', 'lora', 'block', 'linear'}:
        raise ValueError('Unsupported supervised method')
    def add(key, command, directory, done='metrics.json', parent=None):
        jobs.append(Job(key, [*map(str, command), '--config', str(config_path)], directory, directory/done, parent))
    if '1' in experiments:
        add('e1-direct', ['direct'], output/'exp1')
    for seed in seeds:
        if '2' in experiments:
            for method in cfg['methods']:
                for fraction in fractions:
                    p = int(fraction*100)
                    add(f'e2-{method}-s{seed}-p{p:03d}',
                        ['supervised', '--experiment', 2, '--method', method, '--fraction', fraction, '--seed', seed],
                        output/'exp2'/f'{method}_seed{seed}_p{p:03d}')
        if '3' in experiments:
            for method in ['full', 'lora', 'block']:
                parent = f'e3-{method}-s{seed}-dino'
                base = output/'exp3'/f'{method}_seed{seed}'
                add(parent, ['dino', '--method', method, '--seed', seed], base, 'teacher_features.pt')
                for fraction in fractions:
                    p = int(fraction*100)
                    add(f'e3-{method}-s{seed}-probe{p:03d}',
                        ['probe', '--method', method, '--fraction', fraction, '--seed', seed],
                        base/f'probe_p{p:03d}', parent=parent)
        if '4' in experiments:
            for fraction in fractions:
                p = int(fraction*100)
                add(f'e4-full-s{seed}-p{p:03d}',
                    ['supervised', '--experiment', 4, '--method', 'full', '--fraction', fraction, '--seed', seed],
                    output/'exp4'/f'full_seed{seed}_p{p:03d}')
        if '5b' in experiments:
            for strategy in ['random', 'adaptive']:
                add(f'e5b-{strategy}-s{seed}', ['activegcd', '--strategy', strategy, '--seed', seed],
                    output/'exp5b'/f'{strategy}_seed{seed}', 'protocol.json')
        for experiment in ['6', '7']:
            if experiment in experiments:
                add(f'e{experiment}-s{seed}',
                    ['open-world', '--method', 'e'+experiment, '--seed', seed],
                    output/f'exp{experiment}'/f'seed{seed}')
    if len({j.key for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job keys')
    return jobs


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def sbatch_command(job, root, partition, time_limit, inference_time, dependency=None, resume=False):
    command = ['sbatch', '--parsable', '--partition='+partition,
               '--time='+(inference_time if job.key == 'e1-direct' else time_limit),
               '--chdir='+str(root), '--job-name=carABL-'+job.key,
               '--output='+str(root/'logs'/'%x-%j.log')]
    if dependency:
        command += ['--dependency=afterok:'+str(dependency)]
    command += [str(root/'scripts'/'slurm_job.sh'), *job.args]
    if resume and job.args[0] in ['supervised', 'dino', 'probe']:
        command += ['--resume']
    return command


def submit_jobs(jobs, root, registry_path, signature, submit=False, resume=False,
                partition='gpu', time_limit='1-00:00:00', inference_time='00:30:00'):
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'signature': signature, 'jobs': {}}
    if registry['signature'] != signature:
        raise ValueError('Submission registry belongs to a different config/data/checkpoint. Use a different registry and output root for a different experiment protocol.')
    existing = registry['jobs']
    # Check EVERY job before enqueuing the first one, to avoid partial submissions.
    problems = []
    for j in jobs:
        if j.completion.exists() or j.key in existing:
            continue
        if j.directory.exists() and any(j.directory.iterdir()):
            if j.args[0] == 'activegcd':
                problems.append(f'{j.key}: incomplete ActiveGCD output; mid-round resume is unsupported. Use a fresh output root for a deliberate restart.')
            elif j.args[0] == 'open-world':
                problems.append(f'{j.key}: incomplete E6/E7 output; resume is unsupported. Check the job state, then use a fresh output root and registry for a deliberate restart.')
            elif j.args[0] != 'direct' and not resume:
                problems.append(f'{j.key}: existing incomplete output; check for a manually running job, or use --resume after it stops.')
    if problems:
        raise ValueError('\n'.join(problems))
    (root/'logs').mkdir(parents=True, exist_ok=True)
    identifiers, rows = {}, []
    for j in jobs:
        if j.completion.exists():
            identifiers[j.key] = None
            rows.append({'key': j.key, 'state': 'SKIP_COMPLETED', 'parent': j.parent or '', 'job_id': '', 'command': ''})
            print(f'SKIP completed: {j.key}', flush=True)
            continue
        if j.key in existing:
            ident = existing[j.key]['job_id']
            identifiers[j.key] = ident
            rows.append({'key': j.key, 'state': 'ALREADY_RECORDED', 'parent': j.parent or '', 'job_id': ident, 'command': ''})
            print(f'REUSE recorded job {ident}: {j.key} (inspect its state with sacct)', flush=True)
            continue
        if j.parent is not None and j.parent not in identifiers:
            raise ValueError(f'Missing adaptation dependency for {j.key}')
        dependency = identifiers.get(j.parent) if j.parent else None
        command = sbatch_command(j, root, partition, time_limit, inference_time, dependency, resume)
        if submit:
            try:
                answer = subprocess.run(command, text=True, capture_output=True, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'Slurm rejected {j.key}: {error.stderr or error.stdout}. '
                                   f'Earlier submissions remain recorded in {registry_path}.') from error
            # --parsable may return ID;cluster for federated Slurm.
            ident = answer.stdout.strip().split(';')[0]
            if not re.fullmatch(r'\d+', ident):
                raise RuntimeError(f'Unexpected sbatch response {answer.stdout!r}; inspect the queue before retrying.')
            registry['jobs'][j.key] = {'job_id': ident, 'parent': j.parent,
                'command': command, 'completion': str(j.completion),
                'submitted_at': datetime.now(timezone.utc).isoformat()}
            atomic_json(registry_path, registry)
            print(f'SUBMITTED {ident}: {j.key}', flush=True)
        else:
            ident = 'DINO_JOB_ID_'+j.key if j.args[0] == 'dino' else 'DRY_RUN_'+j.key
            print(shlex.join(command), flush=True)
        identifiers[j.key] = ident
        rows.append({'key': j.key, 'state': 'SUBMITTED' if submit else 'PLANNED',
                     'parent': j.parent or '', 'job_id': ident if submit else '', 'command': shlex.join(command)})
    with open(root/'logs'/'experiment_plan.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['key', 'state', 'parent', 'job_id', 'command'])
        writer.writeheader(); writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--experiments', nargs='+', choices=['1', '2', '3', '4', '5b', '6', '7'], default=['1','2','3','4','5b'])
    parser.add_argument('--submit', action='store_true', help='Actually call sbatch; default only prints the plan')
    parser.add_argument('--resume', action='store_true', help='Resume incomplete Exp2/3/4 only; check that no manual jobs are running first')
    parser.add_argument('--registry', default='logs/submissions.json')
    parser.add_argument('--partition', default='gpu')
    parser.add_argument('--time', default='1-00:00:00')
    parser.add_argument('--inference-time', default='00:30:00')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    import os
    os.chdir(root)
    config_path = Path(args.config).resolve()
    cfg = read_config(config_path)
    if args.submit:
        required = ['manifest'] + (['checkpoint'] if set(args.experiments) & {'1','2','3'} else [])
        for key in required:
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(f'Missing {key}: {cfg[key]}; prepare data/download weights first')
        if not Path(cfg['data_root']).is_dir():
            raise FileNotFoundError(cfg['data_root'])
    material = {'config': cfg, **{k: file_hash(cfg[k]) if Path(cfg[k]).exists() else None for k in ['manifest', 'checkpoint']}}
    signature = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    jobs = build_jobs(cfg, config_path, args.experiments)
    # Never reuse an output from a different protocol just because its name matches.
    for job in jobs:
        recorded = job.directory/'provenance.json'
        if recorded.exists():
            previous = json.loads(recorded.read_text())
            previous_cfg = previous.get('config')
            if job.args[0] == 'open-world' and previous_cfg is not None:
                # Earlier E6/E7 provenance stored run identity inside config.
                previous_cfg = {k: v for k, v in previous_cfg.items()
                                if k not in {'open_world_method', 'run_seed'}}
            if previous_cfg != cfg:
                raise ValueError(f'{job.key}: output was created with a different config; use a fresh output root.')
            for key in ['manifest', 'checkpoint']:
                if previous.get(key+'_sha256') != material[key]:
                    raise ValueError(f'{job.key}: {key} changed since this output was created.')
    print(f'{len(jobs)} experiment jobs in this plan; dry run={not args.submit}', flush=True)
    registry_path = Path(args.registry).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    # Prevent two concurrent launchers from submitting the same registry entries.
    with open(str(registry_path)+'.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit_jobs(jobs, root, registry_path, signature, args.submit, args.resume,
                    args.partition, args.time, args.inference_time)
    print('Plan saved: logs/experiment_plan.csv', flush=True)
    if args.submit:
        print(f'Job registry: {registry_path}', flush=True)


if __name__ == '__main__':
    main()
