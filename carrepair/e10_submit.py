"""Artemis E10 DAG submission: root -> three experts -> cascade, for each seed.

Dry-run by default. Slurm --dependency=afterok prevents experts from
starting before their matching root checkpoint is complete, and evaluation
before all matching experts are complete. A canceled/failed registered job
is NEVER silently retried; inspect and use a new output/registry deliberately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .common import file_hash, read_config
from .e10 import EXPERTS, stage_dir


def jobs(cfg):
    """Submission order determines dependency availability."""
    out = []
    for seed in cfg["seeds"]:
        seed = int(seed)
        out.append(("root", seed))
        out.extend((expert, seed) for expert in EXPERTS)
        out.append(("evaluate", seed))
    return out


def key_for(stage, seed):
    return f"{stage}-s{seed}"


def dependencies(stage, seed):
    if stage == "root":
        return []
    if stage == "evaluate":
        return [key_for(expert, seed) for expert in EXPERTS]
    if stage in EXPERTS:
        return [key_for("root", seed)]
    raise ValueError(f"Unsupported E10 submission task: {stage}")


def output_dir(cfg, stage, seed):
    if stage == "evaluate":
        return Path(cfg["output"]) / "cascade" / f"seed{seed}"
    return stage_dir(cfg, stage, seed)


def command_args(stage, seed):
    if stage == "root":
        return ["train-root", "--seed", str(seed)]
    if stage == "evaluate":
        return ["evaluate", "--seed", str(seed)]
    if stage in EXPERTS:
        return ["train-expert", "--expert", stage, "--seed", str(seed)]
    raise ValueError(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e10.yaml")
    parser.add_argument("--registry", default="logs/e10_submissions.json")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="3-00:00:00")
    parser.add_argument("--submit", action="store_true",
                        help="Actually enqueue all 15 dependency-ordered GPU jobs")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if int(cfg["epochs"]) != 300 or int(cfg["early_stopping"]["patience"]) != 30:
        raise ValueError("E10 protocol requires 300 maximum epochs and patience=30")
    if cfg["early_stopping"].get("monitor") != "val_macro_f1":
        raise ValueError("E10 requires validation Macro-F1 monitoring")
    seeds = cfg["seeds"]
    if len(seeds) != len(set(seeds)) or sorted(seeds) != [0, 1, 2]:
        raise ValueError("E10 comparison requires unique seeds [0, 1, 2]")
    root = Path(__file__).resolve().parent.parent
    config = Path(args.config).resolve()
    script = root / "scripts" / "e10_slurm_job.sh"
    if not config.is_file() or not script.is_file():
        raise FileNotFoundError("E10 config or Slurm script missing")
    if args.submit:
        for target in (cfg["manifest"], cfg["data_root"],
                       cfg["pretrained"]["checkpoint"]):
            if not Path(target).exists():
                raise FileNotFoundError(f"Missing E10 input: {target}")
    registry = Path(args.registry)
    config_sha = file_hash(config)
    old = json.loads(registry.read_text()) if registry.exists() else None
    if old is None:
        record = {"config_sha256": config_sha, "jobs": {}}
    elif not isinstance(old, dict) or "jobs" not in old:
        raise ValueError("Invalid E10 registry; do not silently replace prior records")
    elif old["config_sha256"] != config_sha:
        raise ValueError("E10 config changed since submission; use a NEW registry/output")
    else:
        record = old
    ids = {}  # Skipped completed stages have None; scheduled stages have job IDs.
    print(f"E10: {len(jobs(cfg))} GPU jobs (3 roots + 9 experts + 3 evaluations), "
          f"dry-run={not args.submit}", flush=True)
    for stage, seed in jobs(cfg):
        key = key_for(stage, seed)
        folder = output_dir(cfg, stage, seed)
        marker = folder / "metrics.json"
        if marker.exists():
            ids[key] = None
            print(f"COMPLETE {key}: {marker}", flush=True)
            continue
        if key in record["jobs"]:
            ids[key] = str(record["jobs"][key]["job_id"])
            print(f"RECORDED {key}: {ids[key]} (not resubmitted; check sacct if cancelled)",
                  flush=True)
            continue
        if folder.exists():
            raise FileExistsError(
                f"E10 incomplete output already exists at {folder}; inspect before submission")
        dep_ids = []
        for dependency in dependencies(stage, seed):
            if dependency not in ids:
                raise RuntimeError(f"E10 missing submission dependency: {dependency}")
            if ids[dependency] is not None:
                dep_ids.append(ids[dependency])
            elif not output_dir(cfg, dependency.split("-s")[0], seed).joinpath(
                    "metrics.json").exists():
                raise RuntimeError(f"E10 upstream {dependency} is not completed")
        cmd = [
            "sbatch", "--parsable",
            f"--partition={args.partition}", f"--time={args.time}",
            f"--chdir={root}", f"--job-name=carABL-e10-{key}",
            f"--output={root}/logs/e10/%x-%j.log",
        ]
        if dep_ids:
            cmd.append("--dependency=afterok:" + ":".join(dep_ids))
        cmd += [str(script)] + command_args(stage, seed) + ["--config", str(config)]
        print(" ".join(cmd), flush=True)
        if args.submit:
            response = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            job_id = response.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"Unexpected Slurm response: {response.stdout} {response.stderr}")
            ids[key] = job_id
            record["jobs"][key] = {
                "job_id": job_id, "command": cmd,
                "dependencies": dependencies(stage, seed),
            }
            registry.parent.mkdir(parents=True, exist_ok=True)
            tmp = registry.with_suffix(registry.suffix + ".tmp")
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(registry)
            print(f"SUBMITTED {job_id} {key}", flush=True)
        else:
            ids[key] = key  # symbolic placeholder for useful dry-run dependencies
    print("Final aggregation runs manually after all 15 jobs complete:")
    print(f"  python -m carrepair.e10 aggregate --config {config}", flush=True)
    if args.submit:
        print(f"Job IDs saved to {registry}", flush=True)


if __name__ == "__main__":
    main()
