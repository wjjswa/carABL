"""Submit E13/E14 to Artemis with same-seed E9-B expert and afterok DAG.

Dry-run by default. --submit persists Slurm job IDs to a JSON registry.
Select variants individually to avoid enqueuing all 36 jobs at once.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import file_hash, read_config
from .e13e14 import (VARIANTS, root_dir, cascade_dir,
                     verify_protocol)
from .e13e14_preflight import needed_inputs, validate_inputs


def job_order(variants, seeds):
    return [(variant, action, int(seed)) for variant in variants
            for seed in seeds for action in ("train", "evaluate")]


def job_folder(cfg, variant, action, seed):
    return root_dir(cfg, variant, seed) if action == "train" else cascade_dir(cfg, variant, seed)


def job_key(variant, action, seed):
    return f"{variant}-{action}-s{seed}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e13e14_artemis.yaml")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+",
                        default=list(VARIANTS))
    parser.add_argument("--registry", default="logs/e13e14_submissions.json")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="3-00:00:00")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Allow explicit restart of incomplete root job using last.pt")
    args = parser.parse_args(argv)
    variants = list(dict.fromkeys(args.variants))
    cfg = read_config(args.config)
    verify_protocol(cfg, require_sources=False)
    project = Path(__file__).resolve().parent.parent
    conf_path = Path(args.config).resolve()
    script = project / "scripts" / "e13e14_slurm_job.sh"
    if not (conf_path.is_file() and script.is_file()):
        raise FileNotFoundError("Missing Artemis config or Slurm job script")
    if args.submit:
        validate_inputs(cfg, variants)
    registry = Path(args.registry)
    signature = file_hash(conf_path)
    previous = json.loads(registry.read_text()) if registry.exists() else None
    if previous is None:
        record = {"config_sha256": signature, "jobs": {}}
    elif not isinstance(previous, dict) or not isinstance(previous.get("jobs"), dict):
        raise ValueError("Invalid submissions registry")
    elif previous.get("config_sha256") != signature:
        raise ValueError("Config changed since prior jobs; use another registry/output")
    else:
        record = previous
    submitted = {}
    log_dir = project / "logs" / "e13e14"
    if args.submit:
        log_dir.mkdir(parents=True, exist_ok=True)
    print(f"E13/E14: {len(variants)} variants, "
          f"{len(variants)*len(cfg['seeds'])*2} Slurm tasks, "
          f"dry-run={not args.submit}", flush=True)
    for variant, action, seed in job_order(variants, cfg["seeds"]):
        key = job_key(variant, action, seed)
        folder = job_folder(cfg, variant, action, seed)
        if (folder / "metrics.json").is_file():
            submitted[key] = None
            print(f"COMPLETED {key}", flush=True)
            continue
        if key in record["jobs"]:
            submitted[key] = str(record["jobs"][key]["job_id"])
            print(f"RECORDED {key}: {submitted[key]}", flush=True)
            continue
        if folder.exists() and not (args.resume and action == "train"):
            raise FileExistsError(
                f"Incomplete existing output {folder}; inspect or --resume train")
        prerequisite = job_key(variant, "train", seed)
        dependency = []
        if action == "evaluate":
            if prerequisite not in submitted:
                raise RuntimeError(f"Unscheduled prerequisite {prerequisite}")
            previous_id = submitted[prerequisite]
            if previous_id is not None:
                dependency.append(str(previous_id))
            elif not (root_dir(cfg, variant, seed) / "metrics.json").is_file():
                raise RuntimeError(f"Skipped root {prerequisite} has no completion marker")
        command = [
            "sbatch", "--parsable", f"--partition={args.partition}",
            f"--time={args.time}", f"--chdir={project}",
            f"--job-name=carABL-{key}",
            f"--output={log_dir}/%x-%j.log",
        ]
        if dependency:
            command.append("--dependency=afterok:" + ":".join(dependency))
        command += [
            str(script), action, "--variant", variant, "--seed", str(seed),
            "--config", str(conf_path)]
        if args.resume and action == "train":
            command.append("--resume")
        print(" ".join(command), flush=True)
        if args.submit:
            response = subprocess.run(command, capture_output=True, text=True,
                                      check=True)
            job_id = response.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"Unexpected sbatch response: {response.stdout}")
            submitted[key] = job_id
            record["jobs"][key] = {
                "job_id": job_id, "command": command,
                "depends_on": dependency}
            registry.parent.mkdir(parents=True, exist_ok=True)
            temp = registry.with_suffix(".tmp")
            temp.write_text(json.dumps(record, indent=2))
            temp.replace(registry)
            print(f"SUBMITTED {key}: {job_id}", flush=True)
        else:
            submitted[key] = key  # symbolic ID for preview dependency
    print("Aggregate completed variants:")
    print("  python -m carrepair.e13e14 aggregate "
          "--config configs/e13e14_artemis.yaml --variants " +
          " ".join(variants))
    if args.submit:
        print(f"Registry: {registry}")


if __name__ == "__main__":
    main()
