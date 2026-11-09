"""Generate E15 config from the existing E14 or E11 Artemis setup."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def absolute(path, root):
    p = Path(path).expanduser()
    return str((p if p.is_absolute() else root / p).resolve())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", default="configs/e14_ungated_artemis.yaml",
                   help="Existing E14 config; may instead specify E11 config")
    p.add_argument("--template", default="configs/e15.yaml")
    p.add_argument("--output", default="configs/e15_artemis.yaml")
    p.add_argument("--previous-project",
                   help="Directory containing the previous project (default parent of config's directory)")
    args = p.parse_args(argv)
    source = Path(args.previous).resolve()
    original = yaml.safe_load(source.read_text())
    template = yaml.safe_load(Path(args.template).read_text())
    prior_root = Path(args.previous_project).resolve() if args.previous_project else source.parent.parent
    project = Path.cwd().resolve()
    for key in ("manifest", "data_root"):
        template[key] = absolute(original[key], prior_root)
    template["output"] = absolute(template["output"], project)
    if Path(template["output"]).exists():
        print("INFO: Existing E15 output path; submission will protect completed files.")
    model = original.get("variants", {}).get("e14d", {})
    if model.get("family") == "siglip2":
        location = absolute(model["pretrained_dir"], prior_root)
    else:
        location = absolute(template["variants"]["e15a"]["pretrained_dir"], project)
    if Path(template["output"]) == Path(original.get("output", "")):
        raise ValueError("E15 output must not reuse E14 output")
    for variant in template["variants"].values():
        variant["pretrained_dir"] = location
    missing = [x for x in (Path(template["manifest"]), Path(template["data_root"]),
                         Path(location) / "config.json",
                         Path(location) / "preprocessor_config.json") if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing Artemis inputs: " + ", ".join(map(str, missing)))
    if not list(Path(location).glob("*.safetensors")):
        raise FileNotFoundError(f"No pretrained safetensors in {location}; download E14-D first")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; reuse existing config")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(template, sort_keys=False))
    print(f"Created {output}; SigLIP2 snapshot: {location}", flush=True)


if __name__ == "__main__":
    main()
