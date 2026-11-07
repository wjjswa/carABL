"""Dry-run by default; explicitly submit six independent E8 Artemis GPU jobs."""
import argparse
import json
from pathlib import Path
import subprocess
from .common import read_config
from .e8 import VARIANTS

def jobs(cfg):
    return [(variant, seed) for variant in VARIANTS for seed in cfg["seeds"]]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/e8.yaml")
    p.add_argument("--registry", default="logs/e8_submissions.json")
    p.add_argument("--partition", default="gpu")
    p.add_argument("--time", default="3-00:00:00")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Use last.pt if an interrupted run exists")
    args = p.parse_args(argv)
    cfg = read_config(args.config)
    root = Path(__file__).resolve().parent.parent
    config = Path(args.config).resolve()
    slurm = root / "scripts" / "e8_slurm_job.sh"
    if not config.is_file() or not slurm.is_file():
        raise FileNotFoundError("Missing E8 config or Slurm wrapper")
    if args.submit:
        for path in (cfg["manifest"], cfg["pretrained"]["checkpoint"]):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing E8 input {path}; download weights first")
    record = Path(args.registry)
    existing = json.loads(record.read_text()) if record.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError("Unexpected E8 submission registry format")
    print(f"E8: {len(jobs(cfg))} tasks, 300 epochs each, dry run={not args.submit}", flush=True)
    for variant, seed in jobs(cfg):
        key = f"{variant}-s{seed}"
        if key in existing:
            print(f"RECORDED {existing[key]['job_id']}: {key} (skip duplicate)", flush=True)
            continue
        target = Path(cfg["output"]) / variant / f"seed{seed}"
        if target.exists() and not args.resume:
            print(f"SKIP existing output {target}; inspect before resubmitting", flush=True)
            continue
        cmd = [
            "sbatch", "--parsable", f"--partition={args.partition}", f"--time={args.time}",
            f"--chdir={root}", f"--job-name=carABL-e8-{variant}-s{seed}",
            f"--output={root}/logs/e8/%x-%j.log", str(slurm),
            "train", "--config", str(config), "--variant", variant, "--seed", str(seed),
        ]
        if args.resume:
            cmd.append("--resume")
        print(" ".join(cmd), flush=True)
        if args.submit:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"Unrecognized sbatch ID: {result.stdout}")
            existing[key] = {
                "job_id": job_id, "variant": variant, "seed": seed, "command": cmd,
                "config": str(config),
            }
            record.parent.mkdir(parents=True, exist_ok=True)
            tmp = record.with_suffix(record.suffix + ".tmp")
            tmp.write_text(json.dumps(existing, indent=2))
            tmp.replace(record)
            print(f"SUBMITTED {job_id}: {key}", flush=True)
    if args.submit:
        print(f"Registry: {record}", flush=True)

if __name__ == "__main__":
    main()
