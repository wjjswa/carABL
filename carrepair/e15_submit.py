"""Idempotent Slurm E15 submitter: dry-run default; --submit queues 6+6 jobs."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from .common import file_hash, read_config
from .e15 import VARIANTS, evaluation_dir, run_dir, verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e15_artemis.yaml")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--registry", default="logs/e15_submissions.json")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="3-00:00:00")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    verify(cfg, require_sources=args.submit)
    project = Path(__file__).resolve().parent.parent
    config = Path(args.config).resolve()
    script = project / "scripts" / "e15_slurm_job.sh"
    if not script.exists():
        raise FileNotFoundError(script)
    if args.submit:
        from .e15 import fixed_partitions
        fixed_partitions(cfg)
    registry = Path(args.registry)
    signature = file_hash(config)
    record = json.loads(registry.read_text()) if registry.exists() else {
        "config_sha256": signature, "jobs": {}}
    if record.get("config_sha256") != signature:
        raise ValueError("E15 config hash changed; choose a new registry/output")
    jobs = record["jobs"]
    logdir = project / "logs" / "e15"
    if args.submit:
        logdir.mkdir(parents=True, exist_ok=True)
    print("E15 dry-run=", not args.submit, flush=True)
    for variant in dict.fromkeys(args.variants):
        for seed in cfg["seeds"]:
            train_key = f"{variant}-train-s{seed}"
            eval_key = f"{variant}-evaluate-s{seed}"
            prior = None
            for action, key, folder, marker in (
                ("train", train_key, run_dir(cfg, variant, seed), "training_metrics.json"),
                ("evaluate", eval_key, evaluation_dir(cfg, variant, seed), "metrics.json"),
            ):
                if (folder / marker).exists():
                    print("COMPLETED", key, flush=True)
                    if action == "train":
                        prior = None
                    continue
                if key in jobs:
                    print("RECORDED", key, jobs[key]["job_id"], flush=True)
                    if action == "train":
                        prior = jobs[key]["job_id"]
                    continue
                if folder.exists() and not (args.resume and action == "train"):
                    raise FileExistsError(f"Existing incomplete E15 output {folder}; investigate before requeueing")
                if action == "evaluate" and not (
                    (run_dir(cfg, variant, seed) / "training_metrics.json").exists() or prior
                ):
                    raise RuntimeError(f"Missing train prerequisite for {key}")
                command = [
                    "sbatch", "--parsable", f"--partition={args.partition}",
                    f"--time={args.time}", f"--chdir={project}",
                    f"--job-name=carABL-{key}",
                    f"--output={logdir}/%x-%j.log",
                ]
                if action == "evaluate" and prior:
                    command.append(f"--dependency=afterok:{prior}")
                command += [str(script), action, "--variant", variant,
                            "--seed", str(seed), "--config", str(config)]
                if args.resume and action == "train":
                    command.append("--resume")
                print(" ".join(command), flush=True)
                if args.submit:
                    res = subprocess.run(command, capture_output=True, text=True, check=True)
                    job_id = res.stdout.strip().split(";")[0]
                    if not job_id.isdecimal():
                        raise RuntimeError(f"Unexpected sbatch response: {res.stdout!r}")
                    jobs[key] = {"job_id": job_id, "command": command}
                    registry.parent.mkdir(parents=True, exist_ok=True)
                    registry.with_suffix(".tmp").write_text(json.dumps(record, indent=2))
                    registry.with_suffix(".tmp").replace(registry)
                    if action == "train":
                        prior = job_id
                    print("SUBMITTED", key, job_id, flush=True)
                elif action == "train":
                    prior = train_key  # preview symbolic dependency
    print("Aggregate: python -m carrepair.e15 aggregate --config", args.config)


if __name__ == "__main__":
    main()
