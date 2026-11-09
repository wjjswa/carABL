from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest
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


def test_open_world_uses_shared_plan_config_and_registry(tmp_path):
    from carrepair.cli import build_matrix
    cfg = read_config('configs/default.yaml')
    cfg.update(output=str(tmp_path/'results'), seeds=[4, 9])
    jobs = build_jobs(cfg, '/custom/config.yaml', ['6', '7'])
    assert [j.key for j in jobs] == ['e6-s4', 'e7-s4', 'e6-s9', 'e7-s9']
    assert all(j.parent is None and j.args[-2:] == ['--config', '/custom/config.yaml'] for j in jobs)
    assert jobs[0].completion == tmp_path/'results/exp6/seed4/metrics.json'
    matrix = build_matrix('/custom/config.yaml', cfg, ['6', '7'])
    assert len(matrix) == 4 and all('open-world' in command for command in matrix)
    with patch('carrepair.submit.subprocess.run', return_value=SimpleNamespace(stdout='321\n')) as call:
        submit_jobs(jobs, tmp_path, tmp_path/'registry.json', 'same', submit=True, resume=True)
        assert call.call_count == 4
        command = call.call_args_list[0].args[0]
        assert str(tmp_path/'scripts/slurm_job.sh') in command
        assert '--resume' not in command
        assert '--partition=gpu' in command and '--time=1-00:00:00' in command
        submit_jobs(jobs, tmp_path, tmp_path/'registry.json', 'same', submit=True)
        assert call.call_count == 4


def test_open_world_incomplete_output_blocks_even_with_resume(tmp_path):
    cfg = read_config('configs/default.yaml')
    cfg.update(output=str(tmp_path/'results'), seeds=[0])
    jobs = build_jobs(cfg, 'config.yaml', ['6', '7'])
    jobs[1].directory.mkdir(parents=True)
    (jobs[1].directory/'provenance.json').write_text('{}')
    with patch('carrepair.submit.subprocess.run') as call:
        with pytest.raises(ValueError, match='resume is unsupported'):
            submit_jobs(jobs, tmp_path, tmp_path/'registry.json', 'same', submit=True, resume=True)
    call.assert_not_called()


def test_adding_open_world_preserves_old_plan():
    cfg = read_config('configs/default.yaml')
    old = build_jobs(cfg, 'config.yaml', ['1', '2', '3', '4', '5b'])
    expanded = build_jobs(cfg, 'config.yaml', ['1', '2', '3', '4', '5b', '6', '7'])
    assert [j for j in expanded if j.args[0] != 'open-world'] == old
    assert len(expanded) == 142


def test_open_world_alias_preserves_quoted_arguments(tmp_path):
    import subprocess
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    (scripts/'submit_open_world.sh').write_text(Path('scripts/submit_open_world.sh').read_text())
    (scripts/'submit_all.sh').write_text('printf "%s\\n" "$@"\n')
    result = subprocess.run(['bash', str(scripts/'submit_open_world.sh'),
                             '--config', '/config with spaces.yaml', '--submit'],
                            cwd=tmp_path, check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == [
        '--experiments', '6', '7', '--config', '/config with spaces.yaml', '--submit']
