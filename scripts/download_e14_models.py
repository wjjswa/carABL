"""Download the official pretrained DINOv3/SigLIP2 weights outside GPU jobs.

DINOv3 official weights are gated. Accept Meta's licence on Hugging Face,
then run 'hf auth login' first. No token is included in configs or commits.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from carrepair.common import read_config
from carrepair.e13e14 import VARIANTS, variant_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e13e14_artemis.yaml")
    parser.add_argument("--variants", nargs="+", default=["e14a", "e14c", "e14d"],
                        choices=[v for v in VARIANTS if v.startswith("e14")])
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    cfg = read_config(args.config)
    unique = {}
    for variant in args.variants:
        v = variant_config(cfg, variant)
        unique[str(v["pretrained_dir"])] = v["pretrained_repo"]
    for location, repo in unique.items():
        path = Path(location)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo} -> {path}", flush=True)
        snapshot_download(
            repo_id=repo, local_dir=str(path),
            allow_patterns=["*.safetensors", "*.json", "*.txt",
                            "*.model", "README.md", "LICENSE*"],
        )
        if not (path / "config.json").exists() or not list(path.glob("*.safetensors")):
            raise RuntimeError(f"Download incomplete: {path}")
        print(f"Downloaded {repo} to {path}", flush=True)


if __name__ == "__main__":
    main()
