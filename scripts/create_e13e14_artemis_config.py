"""Build Artemis-local config by borrowing verified E11 manifests and expert."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e11", default="configs/e11_artemis.yaml")
    p.add_argument("--template", default="configs/e13e14.yaml")
    p.add_argument("--output", default="configs/e13e14_artemis.yaml")
    args = p.parse_args()
    previous = yaml.safe_load(Path(args.e11).read_text())
    cfg = yaml.safe_load(Path(args.template).read_text())
    for name in ("manifest", "data_root", "checkpoint", "expert", "device", "workers"):
        cfg[name] = previous[name]
    # These are the same dataset, splits, sources and matching E9-B expert.
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Created {root}")
    for label in ("manifest", "data_root", "checkpoint"):
        target = Path(cfg[label])
        print(("FOUND" if target.exists() else "MISSING"), label, target)
    source = Path(cfg["expert"]["source_checkpoint"])
    print(("FOUND" if source.exists() else "MISSING"), "E9-B source", source)
    for seed in cfg["seeds"]:
        target = Path(cfg["expert"]["directory"]) / f"seed{seed}" / "best.pt"
        print(("FOUND" if target.exists() else "MISSING"), f"E9-B seed{seed}", target)
    print("HF checkpoints need to be downloaded before using E14 (see docs/e13e14.md).")


if __name__ == "__main__":
    main()
