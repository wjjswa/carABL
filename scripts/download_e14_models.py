"""Download official E14 vision weights outside GPU jobs.

The e14_ungated template uses public DINOv2 and SigLIP2; use --anonymous.
The original DINOv3 template still requires approved access.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# A direct script invocation puts scripts/, not the worktree root, on sys.path.
# Always prefer THIS worktree over an editable install of an older E11/main.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carrepair.common import read_config
from carrepair.e13e14 import VARIANTS, variant_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e13e14_artemis.yaml")
    parser.add_argument("--variants", nargs="+", default=["e14a", "e14c", "e14d"],
                        choices=[v for v in VARIANTS if v.startswith("e14")])
    parser.add_argument("--anonymous", action="store_true",
                        help="Use public models without any stored Hugging Face token")
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download
    cfg = read_config(args.config)
    unique = {}
    for variant in args.variants:
        v = variant_config(cfg, variant)
        location = str(v["pretrained_dir"])
        source = (v["pretrained_repo"], v.get("pretrained_revision"))
        if location in unique and unique[location] != source:
            raise ValueError(f"Conflicting model repositories/revisions for {location}")
        unique[location] = source
    for location, (repo, revision) in unique.items():
        path = Path(location)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo} -> {path}", flush=True)
        snapshot_download(
            repo_id=repo, revision=revision, local_dir=str(path),
            token=False if args.anonymous else None,
            allow_patterns=["*.safetensors", "*.json", "*.txt",
                            "*.model", "README.md", "LICENSE*"],
        )
        if not (path / "config.json").exists() or not list(path.glob("*.safetensors")):
            raise RuntimeError(f"Download incomplete: {path}")
        print(f"Downloaded {repo} to {path}", flush=True)


if __name__ == "__main__":
    main()
