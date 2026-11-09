"""Build Artemis-local config by borrowing verified E11 manifests and expert."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import yaml


def absolute_path(value, project):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else project / path).resolve())


def build_config(previous, template, e11_project, project):
    cfg = deepcopy(template)
    for name in ("manifest", "data_root", "checkpoint"):
        cfg[name] = absolute_path(previous[name], e11_project)
    cfg["expert"] = deepcopy(previous["expert"])
    for name in ("directory", "source_checkpoint"):
        cfg["expert"][name] = absolute_path(cfg["expert"][name], e11_project)
    for name in ("device", "workers"):
        cfg[name] = previous[name]
    cfg["reference_e11"] = absolute_path(previous["output"], e11_project)
    cfg["output"] = absolute_path(template["output"], project)
    if Path(cfg["output"]) == Path(cfg["reference_e11"]):
        raise ValueError("E13/E14 output must differ from E11 output")
    for setting in cfg["variants"].values():
        if setting["kind"] == "hf":
            setting["pretrained_dir"] = absolute_path(setting["pretrained_dir"], project)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e11", default="../carABL-e11/configs/e11_artemis.yaml")
    p.add_argument("--template", default="configs/e13e14.yaml")
    p.add_argument("--output", default="configs/e13e14_artemis.yaml")
    p.add_argument("--e11-project", help="E11 worktree root if YAML is outside its configs directory")
    args = p.parse_args(argv)
    source = Path(args.e11).resolve()
    previous = yaml.safe_load(source.read_text())
    template = yaml.safe_load(Path(args.template).read_text())
    e11_project = Path(args.e11_project).resolve() if args.e11_project else source.parent.parent
    cfg = build_config(previous, template, e11_project, Path.cwd())
    # These are the same dataset, splits, sources and matching E9-B expert.
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    # Never leave an apparently valid config pointing at missing E11 inputs.
    required = [Path(cfg[k]) for k in ("manifest", "data_root", "checkpoint")]
    required.append(Path(cfg["expert"]["source_checkpoint"]))
    required.extend(Path(cfg["expert"]["directory"]) / f"seed{s}" / "best.pt"
                    for s in cfg["seeds"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing E11 inputs; config was not written:\n" + "\n".join(missing))
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
    print("Completed E11 reference:", cfg["reference_e11"])


if __name__ == "__main__":
    main()
