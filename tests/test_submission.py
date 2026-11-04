from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
from carrepair.common import read_config
from carrepair.submit import build_jobs, submit_jobs


def test_full_job_plan_and_dino_dependencies():
    cfg = read_config('configs/default.yaml')
    jobs = build_jobs(cfg, 'configs/default.yaml', ['1','2','3','4','5b'])
    assert len(jobs) == 136
    assert sum(j.key.startswith('e2-') for j in jobs) == 60
    assert sum(j.args[0] == 'dino' for j in jobs) == 9
    assert sum(j.args[0] == 'probe' for j in jobs) == 45
    assert sum(j.key.startswith('e4-') for j in jobs) == 15
    assert sum(j.key.startswith('e5b-') for j in jobs) == 6
    seen = set()
    for j in jobs:
        if j.parent:
            assert j.parent in seen
        seen.add(j.key)


def small_plan(tmp_path):
    cfg = read_config('configs/default.yaml')
    cfg.update(output=str(tmp_path/'results'), seeds=[0], fractions=[.2,1.])
    jobs = build_jobs(cfg, 'config.yaml', ['3'])
    return jobs


def test_dry_run_submits_nothing(tmp_path):
    jobs = small_plan(tmp_path)
    with patch('carrepair.submit.subprocess.run') as call:
        rows = submit_jobs(jobs, tmp_path, tmp_path/'registry.json', 'hash')
    call.assert_not_called()
    assert len(rows) == 9
    assert not (tmp_path/'registry.json').exists()
    assert (tmp_path/'logs/experiment_plan.csv').exists()


def test_submission_dependency_ids_and_repeat_is_idempotent(tmp_path):
    jobs = small_plan(tmp_path)
    counter = iter(range(100,200))
    def fake_sbatch(command, **kwargs):
        return SimpleNamespace(stdout=f'{next(counter)};testcluster\n')
    registry = tmp_path/'registry.json'
    with patch('carrepair.submit.subprocess.run', side_effect=fake_sbatch) as call:
        submit_jobs(jobs, tmp_path, registry, 'hash', submit=True)
        assert call.call_count == 9
        commands = [x.args[0] for x in call.call_args_list]
        assert '--dependency=afterok:100' in commands[1]
        assert '--dependency=afterok:100' in commands[2]
        assert '--dependency=afterok:103' in commands[4]
        submit_jobs(jobs, tmp_path, registry, 'hash', submit=True)
        assert call.call_count == 9  # Already recorded: no duplicate sbatch calls.
    assert len(json.loads(registry.read_text())['jobs']) == 9


def test_completed_adaptation_needs_no_dependency(tmp_path):
    jobs = small_plan(tmp_path)
    jobs[0].completion.parent.mkdir(parents=True)
    jobs[0].completion.write_text('fixture')
    with patch('carrepair.submit.subprocess.run') as call:
        rows = submit_jobs(jobs, tmp_path, tmp_path/'registry.json', 'hash')
    call.assert_not_called()
    assert rows[0]['state'] == 'SKIP_COMPLETED'
    assert '--dependency=' not in rows[1]['command']
