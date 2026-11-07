"""Preview/submit E9 zero-shot job + six 300-epoch GPU training jobs.

Dry-run by default. A recorded job is not automatically retried.
"""
import argparse
import json
from pathlib import Path
import subprocess

from .common import read_config
from .e9 import TASKS


def jobs(cfg):
    return [("zero-shot", None)] + [
        (task, int(seed)) for task in TASKS for seed in cfg["seeds"]
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/e9.yaml")
    p.add_argument("--registry", default="logs/e9_submissions.json")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--time", default="3-00:00:00")
    p.add_argument("--submit", action="store_true")
    args = p.parse_args(argv)
    cfg = read_config(args.config)
    if int(cfg["epochs"]) != 300:
        raise ValueError("E9 requires 300 epochs")
    if len(cfg["seeds"]) != len(set(cfg["seeds"])):
        raise ValueError("Duplicate seeds")
    root = Path(__file__).resolve().parent.parent
    config = Path(args.config).resolve()
    wrapper = root / "scripts" / "e9_slurm_job.sh"
    if not config.is_file() or not wrapper.is_file():
        raise FileNotFoundError("Missing config or GPU wrapper")
    if args.submit:
        for path in (cfg["manifest"], cfg["data_root"], cfg["pretrained"]["checkpoint"]):
            if not Path(path).exists():
                raise FileNotFoundError(f"E9 required input does not exist: {path}")
    record = Path(args.registry)
    previous = json.loads(record.read_text()) if record.exists() else {}
    if not isinstance(previous, dict):
        raise ValueError("Invalid job registry")
    print(f"E9: {len(jobs(cfg))} jobs; dry-run={not args.submit}", flush=True)
    for task, seed in jobs(cfg):
        key = "zero-shot" if task == "zero-shot" else f"{task}-s{seed}"
        if key in previous:
            print(f"RECORDED {key}: {previous[key]['job_id']} (skip)")
            continue
        target = (Path(cfg["output"]) / "zero_shot" if task == "zero-shot"
                  else Path(cfg["output"]) / task / f"seed{seed}")
        if target.exists():
            print(f"EXISTS {target}: inspect before resubmission", flush=True)
            continue
        command = [
            "sbatch", "--parsable", f"--partition={args.partition}",
            f"--time={args.time}", f"--chdir={root}",
            f"--job-name=carABL-e9-{key}",
            f"--output={root}/logs/e9/%x-%j.log",
            str(wrapper), "zero-shot" if task == "zero-shot" else "train",
            "--config", str(config),
        ]
        if task != "zero-shot":
            command += ["--task", task, "--seed", str(seed)]
        print(" ".join(command), flush=True)
        if args.submit:
            job = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = job.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"Unexpected sbatch response: {job.stdout} {job.stderr}")
            previous[key] = {"job_id": job_id, "command": command}
            record.parent.mkdir(parents=True, exist_ok=True)
            temporary = record.with_suffix(record.suffix + ".tmp")
            temporary.write_text(json.dumps(previous, indent=2))
            temporary.replace(record)
            print(f"SUBMITTED {job_id}: {key}", flush=True)
    if args.submit:
        print("Registry:", record, flush=True)


if __name__ == "__main__":
    main()
