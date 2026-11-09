"""Read-only Artemis preflight using E11 inputs and optional CPU model smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from .common import file_hash, read_config, seed_all
from .e11 import BINARY, expert_path, load_expert
from .e13e14 import (
    VARIANTS, ROOT, RootDataset, _count, _move, batches, build_model,
    fixed_partitions, source_signature, variant_config, verify_protocol,
)


def needed_inputs(cfg, variants):
    paths = [Path(cfg["manifest"]), Path(cfg["data_root"]),
             Path(cfg["expert"]["source_checkpoint"])]
    for seed in cfg["seeds"]:
        checkpoint = expert_path(cfg, seed)
        paths.extend([checkpoint, checkpoint.parent / "metrics.json",
                      checkpoint.parent / "provenance.json",
                      checkpoint.parent / "test_manifest.csv"])
    for variant in variants:
        setting = variant_config(cfg, variant)
        if setting["kind"] == "cardd":
            paths.append(Path(cfg["checkpoint"]))
        else:
            location = Path(setting["pretrained_dir"])
            paths.extend([location / "config.json", location / "preprocessor_config.json"])
            paths.extend(sorted(location.glob("*.safetensors")))
    return list(dict.fromkeys(paths))


def _json(path):
    return json.loads(Path(path).read_text())


def _same_samples(saved, expected, label):
    columns = ["id", "path", "label", "group", "split"]
    left = pd.read_csv(saved, dtype=str)[columns].sort_values("id").reset_index(drop=True)
    right = expected[columns].astype(str).sort_values("id").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"{label}: original E11 image IDs/labels/groups/splits differ")


def validate_inputs(cfg, variants):
    """Fail before sbatch on missing inputs, wrong experts, or changed E11 splits."""
    verify_protocol(cfg, require_sources=False)
    for path in needed_inputs(cfg, variants):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
    frames = fixed_partitions(cfg)
    for frame in frames:
        for image in frame.path:
            path = Path(cfg["data_root"]) / image
            if not path.is_file():
                raise FileNotFoundError(f"Missing original dataset image: {path}")
    manifest_hash = file_hash(cfg["manifest"])
    source_hash = cfg["expert"]["source_sha256"]
    if file_hash(cfg["expert"]["source_checkpoint"]) != source_hash:
        raise ValueError("E9-B source SHA256 mismatch; wrong model")
    test = frames[2]
    expected_expert = test[test.label.isin(
        ("bumper_dent", "bumper_scratch", "door_dent", "door_scratch"))]
    for seed in cfg["seeds"]:
        directory = expert_path(cfg, seed).parent
        metrics = _json(directory / "metrics.json")
        provenance = _json(directory / "provenance.json")
        if (metrics.get("task") != "binary" or metrics.get("seed") != seed
                or metrics.get("classes") != BINARY):
            raise ValueError(f"E9-B seed{seed} completion metadata mismatch")
        if (provenance.get("manifest_sha256") != manifest_hash
                or provenance.get("pretrained_sha256") != source_hash):
            raise ValueError(f"E9-B seed{seed} source/manifest provenance mismatch")
        _same_samples(directory / "test_manifest.csv", expected_expert,
                      f"E9-B seed{seed}")
    reference = cfg.get("reference_e11")
    if reference:
        cardd_hash = file_hash(cfg["checkpoint"])
        for seed in cfg["seeds"]:
            root = Path(reference) / "six_class" / f"seed{seed}"
            metrics = _json(root / "metrics.json")
            cascade = _json(Path(reference) / "cascade" / f"seed{seed}" / "metrics.json")
            provenance = _json(root / "provenance.json")
            if metrics.get("seed") != seed or cascade.get("seed") != seed:
                raise ValueError(f"E11 seed{seed} completion metadata mismatch")
            if (provenance.get("manifest_sha256") != manifest_hash
                    or provenance.get("cardd_v2_sha256") != cardd_hash):
                raise ValueError(f"E11 seed{seed} CarDD/manifest provenance mismatch")
            for name, frame in zip(("train", "val", "test"), frames):
                _same_samples(root / f"{name}_manifest.csv", frame,
                              f"E11 seed{seed} {name}")
    for variant in variants:
        source_signature(cfg, variant)
    print("PASS original E11 splits/images, same-seed E9-B metadata, source hashes",
          flush=True)
    return frames


def model_smoke(cfg, variants, frames):
    """Exercise actual local weights/processors, backward, and expert loading on CPU.

    No optimiser step, output checkpoint, network download, or GPU allocation.
    This checks compatibility, not prediction accuracy or GPU memory usage.
    """
    cpu = dict(cfg, device="cpu", workers=0, batch_size=2)
    seed_all(0)
    for variant in variants:
        setting = variant_config(cpu, variant)
        model = build_model(cpu, variant)
        train = not (setting["kind"] == "hf" and setting["mode"] == "frozen")
        dataset = RootDataset(frames[0].iloc[:2], cpu, variant, train=train)
        images, target, _ = next(iter(batches(dataset, cpu)))
        pixels = images["pixel_values"] if isinstance(images, dict) else images
        size = int(setting["image_size"])
        if tuple(pixels.shape) != (2, 3, size, size):
            raise ValueError(f"{variant}: processor returned {tuple(pixels.shape)}, expected 2x3x{size}x{size}")
        model.train()
        logits = model(_move(images, "cpu"))
        if logits.shape != (2, len(ROOT)) or not torch.isfinite(logits).all():
            raise ValueError(f"{variant}: invalid six-class logits")
        F.cross_entropy(logits, target).backward()
        if model.head.weight.grad is None:
            raise RuntimeError(f"{variant}: classifier head has no gradients")
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"{variant}: non-finite gradients in {name}")
            if not parameter.requires_grad and parameter.grad is not None:
                raise RuntimeError(f"{variant}: frozen parameter received a gradient: {name}")
        print(f"PASS {variant}: actual local model/processor forward+backward {_count(model)}",
              flush=True)
        del model, images, logits, dataset
    for seed in cpu["seeds"]:
        expert, _ = load_expert(cpu, seed, frames[2])
        print(f"PASS E9-B seed{seed}: original trained checkpoint loads strictly", flush=True)
        del expert


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e13e14_artemis.yaml")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--model-smoke", action="store_true",
                        help="Also run a small CPU forward/backward using actual local weights")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    variants = list(dict.fromkeys(args.variants))
    frames = validate_inputs(cfg, variants)
    if args.model_smoke:
        model_smoke(cfg, variants, frames)
    print("PREFLIGHT PASSED; no jobs submitted", flush=True)


if __name__ == "__main__":
    main()
