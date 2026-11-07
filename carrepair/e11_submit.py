"""Safe E11 Artemis launcher: 3 CarDD full-tune roots + 3 cascade evaluations.

Dry-run unless --submit. E9-B is a PREEXISTING frozen model, never submitted.
Each evaluation waits for same-seed E11 training with Slurm afterok.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import read_config, file_hash
from .e11 import train_dir, eval_dir, expert_path


def jobs(cfg):
    return [task for seed in cfg["seeds"] for task in (("train", int(seed)), ("evaluate", int(seed)))]


def dependencies(task, seed):
    if task == "train":
        return []
    if task == "evaluate":
        return [f"train-s{seed}"]
    raise ValueError(task)


def task_dir(cfg, task, seed):
    if task == "train":
        return train_dir(cfg, seed)
    if task == "evaluate":
        return eval_dir(cfg, seed)
    raise ValueError(task)


def required_inputs(cfg):
    return [
        Path(cfg["manifest"]), Path(cfg["data_root"]), Path(cfg["checkpoint"]),
        Path(cfg["expert"]["source_checkpoint"]),
        *(expert_path(cfg, seed) for seed in cfg["seeds"]),
        *(expert_path(cfg, seed).parent / "metrics.json" for seed in cfg["seeds"]),
        *(expert_path(cfg, seed).parent / "provenance.json" for seed in cfg["seeds"]),
        *(expert_path(cfg, seed).parent / "test_manifest.csv" for seed in cfg["seeds"]),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e11.yaml")
    parser.add_argument("--registry", default="logs/e11_submissions.json")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="3-00:00:00")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if int(cfg["epochs"]) != 300 or int(cfg["early_stopping"]["patience"]) != 30:
        raise ValueError("E11 expects max 300 epochs and patience=30")
    if cfg["early_stopping"]["monitor"] != "val_macro_f1":
        raise ValueError("E11 monitors validation Macro-F1")
    if sorted(set(cfg["seeds"])) != [0, 1, 2] or len(cfg["seeds"]) != 3:
        raise ValueError("E11 requires seeds 0,1,2")
    root = Path(__file__).resolve().parent.parent
    config = Path(args.config).resolve()
    wrapper = root / "scripts" / "e11_slurm_job.sh"
    if not config.exists() or not wrapper.exists():
        raise FileNotFoundError("E11 config or Slurm wrapper not found")
    if args.submit:
        for path in required_inputs(cfg):
            if not path.exists():
                raise FileNotFoundError(f"Missing E11 input: {path}")
        from .e11 import verify_train_config
        verify_train_config(cfg)
        actual = file_hash(cfg["expert"]["source_checkpoint"])
        if actual != cfg["expert"]["source_sha256"]:
            raise ValueError(f"E11 E9-B source checkpoint SHA256 mismatch: {actual}")
    registry = Path(args.registry)
    sha = file_hash(config)
    previous = json.loads(registry.read_text()) if registry.exists() else None
    if previous is None:
        record = {"config_sha256": sha, "jobs": {}}
    elif not isinstance(previous, dict) or not isinstance(previous.get("jobs"), dict):
        raise ValueError("Invalid E11 submission registry")
    elif previous.get("config_sha256") != sha:
        raise ValueError("Config changed from recorded jobs; choose new output/registry")
    else:
        record = previous
    submitted = {}
    print(f"E11: 3 full-fine-tune Root jobs + 3 evaluations; dry-run={not args.submit}")
    for task, seed in jobs(cfg):
        key = f"{task}-s{seed}"
        folder = task_dir(cfg, task, seed)
        marker = folder / "metrics.json"
        if marker.exists():
            submitted[key] = None
            print(f"COMPLETED {key}: {marker}")
            continue
        if key in record["jobs"]:
            submitted[key] = str(record["jobs"][key]["job_id"])
            print(f"RECORDED {key}: {submitted[key]} (not automatically retried)")
            continue
        if folder.exists():
            raise FileExistsError(
                f"E11 incomplete output: {folder}; investigate before resubmitting")
        dep_ids = []
        for dependent in dependencies(task, seed):
            if dependent not in submitted:
                raise RuntimeError(f"E11 prerequisite not scheduled: {dependent}")
            if submitted[dependent] is not None:
                dep_ids.append(submitted[dependent])
            elif not task_dir(cfg, "train", seed).joinpath("metrics.json").exists():
                raise RuntimeError("Skipped prerequisite lacks completion marker")
        command = [
            "sbatch", "--parsable", f"--partition={args.partition}",
            f"--time={args.time}", f"--chdir={root}",
            f"--job-name=carABL-e11-{key}",
            f"--output={root}/logs/e11/%x-%j.log",
        ]
        if dep_ids:
            command.append("--dependency=afterok:" + ":".join(dep_ids))
        command += [
            str(wrapper), task, "--seed", str(seed), "--config", str(config)
        ]
        print(" ".join(command), flush=True)
        if args.submit:
            response = subprocess.run(command, check=True, text=True, capture_output=True)
            job_id = response.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(
                    f"Unexpected sbatch response: {response.stdout} {response.stderr}")
            submitted[key] = job_id
            record["jobs"][key] = {
                "job_id": job_id, "command": command,
                "dependencies": dependencies(task, seed),
            }
            registry.parent.mkdir(parents=True, exist_ok=True)
            tmp = registry.with_suffix(registry.suffix + ".tmp")
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(registry)
            print(f"SUBMITTED {key} JobID={job_id}", flush=True)
        else:
            submitted[key] = key  # symbolic dry-run dependency
    print("After all six jobs complete:")
    print(f"  python -m carrepair.e11 aggregate --config {config}")
    if args.submit:
        print(f"Recorded job IDs: {registry}")


if __name__ == "__main__":
    main()
