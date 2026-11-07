"""E8: full fine-tuning from Majorburn's Carparts-Seg YOLO11 segmentation backbone.

Two separate experiments: eight image-level damage classes including 'unknown',
and seven named classes with 'unknown' completely withheld. Same immutable
manifest, all eligible training images, 300 epochs, three seeds, no early stopping.
Run only on an allocated Artemis GPU. This does not run part-mask inference:
it transfers learned part-aware backbone weights to an image classifier.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import atomic_save, dump_json, file_hash, read_config, seed_all
from .data import open_image, read_manifest

SEVEN = (
    "bumper_dent", "bumper_scratch", "door_dent", "door_scratch",
    "glass_shatter", "head_lamp", "tail_lamp",
)
VARIANTS = ("8class", "7class")


def class_names(variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError("variant must be 8class or 7class")
    return list(SEVEN) + (["unknown"] if variant == "8class" else [])


def select_data(manifest: pd.DataFrame, variant: str):
    """Keep original splits/groups, and never use unknown for the 7-class run."""
    labels = class_names(variant)
    expected_all = set(SEVEN) | {"unknown"}
    if set(manifest.label) != expected_all:
        raise ValueError(f"E8 expects eight existing manifest labels: {sorted(expected_all)}; "
                         f"found {sorted(set(manifest.label))}")
    filtered = manifest[manifest.label.isin(labels)]
    parts = tuple(filtered[filtered.split == s].copy().reset_index(drop=True)
                  for s in ("train", "val", "test"))
    for name, frame in zip(("train", "val", "test"), parts):
        if set(frame.label) != set(labels):
            raise ValueError(f"{name} split is missing classes: {set(labels)-set(frame.label)}")
        if frame.empty:
            raise ValueError(f"No {name} images")
    if any(set(a.id) & set(b.id) or set(a.group) & set(b.group)
           for a, b in ((parts[0], parts[1]), (parts[0], parts[2]), (parts[1], parts[2]))):
        raise ValueError("E8 detected train/val/test leakage")
    return parts, labels


def padded_transforms(size: int, train: bool):
    """No aggressive cropping: keep bumper/door context in view."""
    if size < 64:
        raise ValueError("image_size must be >=64")
    steps = [
        T.Lambda(lambda img: ImageOps.pad(
            img, (size, size), method=Image.Resampling.BILINEAR,
            color=(114, 114, 114)))
    ]
    if train:
        steps += [T.RandomHorizontalFlip(),
                  T.RandomApply([T.ColorJitter(brightness=.15, contrast=.15,
                                               saturation=.1, hue=.025)], p=.4)]
    steps += [T.ToTensor()]  # Ultralytics YOLO expects float images in [0, 1].
    return T.Compose(steps)


class DamageImages(Dataset):
    def __init__(self, frame, root, labels, image_size, train):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.lookup = {s: i for i, s in enumerate(labels)}
        self.transform = padded_transforms(image_size, train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.transform(open_image(self.root / row.path))
        return image, self.lookup[row.label], index


class PartPretrainedClassifier(nn.Module):
    """Transfer ALL YOLO11 encoder tensors through C2PSA, not the segment head.

    The segment checkpoint's segmentation neck/head cannot be relabeled as a
    classification head. This intentionally extracts the sequential YOLO11
    backbone (Conv/C3k2/SPPF/C2PSA) and adds a new global-pool classifier.
    """
    def __init__(self, segmentation_checkpoint, n_classes, image_size=384, expected_parts=23):
        super().__init__()
        from ultralytics import YOLO
        ckpt = Path(segmentation_checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing E8 pretrained checkpoint: {ckpt}. "
                                    "Run: python -m carrepair.e8 download")
        source = YOLO(str(ckpt))  # WARNING: third-party PyTorch pickle; trust source first.
        if source.task != "segment":
            raise ValueError(f"Expected part-segmentation weights, got task={source.task}")
        if len(source.names) != expected_parts:
            raise ValueError(f"Expected {expected_parts} pretrained parts, "
                             f"found {len(source.names)}; check checkpoint")
        src_layers = list(source.model.model.children())
        sppf = [i for i, layer in enumerate(src_layers) if type(layer).__name__ == "SPPF"]
        c2psa = [i for i, layer in enumerate(src_layers) if type(layer).__name__ == "C2PSA"]
        if len(sppf) != 1 or len(c2psa) != 1 or c2psa[0] != sppf[0] + 1:
            raise ValueError("Unexpected YOLO11 architecture; refusing partial/misordered transfer")
        last = c2psa[0]
        if last + 1 >= len(src_layers):
            raise ValueError("Segmentation checkpoint lacks the expected head")
        self.backbone = nn.Sequential(*deepcopy(src_layers[:last + 1]))
        self.backbone.eval()
        with torch.no_grad():
            z = self.backbone(torch.zeros(1, 3, image_size, image_size))
        if not isinstance(z, torch.Tensor) or z.ndim != 4:
            raise ValueError("Transferred YOLO11 backbone does not return [N,C,H,W]")
        self.features_dim = z.shape[1]
        self.head = nn.Linear(self.features_dim, n_classes)
        # Full fine-tuning: not frozen, no LoRA, no linear probing.
        self.backbone.requires_grad_(True)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1))


def loader(frame, cfg, labels, train=False, seed=0):
    ds = DamageImages(frame, cfg["data_root"], labels, int(cfg["image_size"]), train)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=train,
                      num_workers=int(cfg["workers"]), pin_memory=str(cfg["device"]).startswith("cuda"),
                      generator=generator, drop_last=False)


@torch.no_grad()
def evaluate(model, data_loader, device, labels, include_predictions=False):
    model.eval()
    losses, all_true, all_pred, all_score = [], [], [], []
    for images, targets, _ in data_loader:
        logits = model(images.to(device))
        losses.append((float(F.cross_entropy(logits, targets.to(device)).item()), len(images)))
        probabilities = logits.softmax(dim=-1).cpu().numpy()
        all_score.append(probabilities)
        all_true.extend(targets.numpy().tolist())
        all_pred.extend(probabilities.argmax(1).tolist())
    if not all_true:
        raise ValueError("Empty evaluation split")
    truth = np.asarray(all_true, dtype=np.int64)
    prediction = np.asarray(all_pred, dtype=np.int64)
    result = {
        "loss": sum(loss * n for loss, n in losses) / len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=range(len(labels)),
                                   average="macro", zero_division=0)),
    }
    if include_predictions:
        result["report"] = classification_report(
            truth, prediction, labels=range(len(labels)), target_names=labels,
            zero_division=0, output_dict=True)
        result["confusion_matrix"] = confusion_matrix(
            truth, prediction, labels=range(len(labels))).tolist()
        return result, truth, prediction, np.concatenate(all_score, axis=0)
    return result


def detect_overfitting(history, config):
    """Flag sustained train/validation divergence without ending training."""
    p = config["overfitting"]
    window = int(p["window"])
    if len(history) <= window:
        return False
    earlier, last = history[-window - 1], history[-1]
    training_improves = last["train_eval_loss"] <= earlier["train_eval_loss"] * (
        1 - float(p["min_train_loss_improvement"]))
    val_worsens = last["val_loss"] >= earlier["val_loss"] * (
        1 + float(p["min_val_loss_worsening"]))
    ratio = last["val_loss"] / max(last["train_eval_loss"], 1e-8)
    acc_gap = last["train_eval_accuracy"] - last["val_accuracy"]
    return bool(
        (training_improves and val_worsens and
         ratio >= float(p["min_val_train_loss_ratio"]))
        or (ratio >= float(p["min_val_train_loss_ratio"]) and
            acc_gap >= float(p["min_accuracy_gap"]))
    )


def _run_provenance(cfg, variant, seed):
    return {
        "variant": variant, "seed": int(seed), "config": cfg,
        "manifest_sha256": file_hash(cfg["manifest"]),
        "pretrained_sha256": file_hash(cfg["pretrained"]["checkpoint"]),
        "pretrained_repository": cfg["pretrained"]["repository"],
        "pretrained_revision": cfg["pretrained"]["revision"],
    }


def _save_curves(directory, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame.epoch, frame.train_eval_loss, label="train loss (eval transform)")
    ax.plot(frame.epoch, frame.val_loss, label="validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="E8 overfitting check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "loss_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame.epoch, frame.train_eval_accuracy, label="train accuracy")
    ax.plot(frame.epoch, frame.val_accuracy, label="validation accuracy")
    ax.set(xlabel="Epoch", ylabel="Accuracy", title="E8 generalization gap", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "accuracy_curve.png", dpi=160)
    plt.close(fig)


def train_one(cfg, variant, seed, resume=False):
    if seed not in cfg["seeds"]:
        raise ValueError(f"Seed {seed} not in config.seeds: {cfg['seeds']}")
    if int(cfg["epochs"]) != 300:
        raise ValueError("E8 protocol requires exactly 300 epochs")
    if file_hash(cfg["pretrained"]["checkpoint"]) != cfg["pretrained"]["sha256"]:
        raise ValueError("E8 pretrained checkpoint SHA256 mismatch")
    seed_all(seed)
    partitions, labels = select_data(read_manifest(cfg["manifest"]), variant)
    train_frame, val_frame, test_frame = partitions
    directory = Path(cfg["output"]) / variant / f"seed{seed}"
    complete = directory / "metrics.json"
    if directory.exists() and not resume:
        raise FileExistsError(f"{directory} already exists; use --resume or a new output")
    directory.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(cfg, variant, seed)
    prov_path = directory / "provenance.json"
    if prov_path.exists():
        if json.loads(prov_path.read_text()) != provenance:
            raise ValueError("E8 config/manifest/pretrained checkpoint changed; do not resume")
    else:
        dump_json(prov_path, provenance)
    if complete.exists():
        print("Already completed:", directory, flush=True)
        return json.loads(complete.read_text())
    for split, frame in zip(("train", "val", "test"), partitions):
        frame[["id", "path", "label", "group", "split"]].to_csv(
            directory / f"{split}_manifest.csv", index=False)
    model = PartPretrainedClassifier(
        cfg["pretrained"]["checkpoint"], len(labels), int(cfg["image_size"]),
        int(cfg["pretrained"]["expected_classes"])).to(cfg["device"])
    # All encoder + newly initialized head weights are optimized.
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(cfg["lr"])},
        {"params": model.head.parameters(), "lr": float(cfg["head_lr"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(cfg["epochs"]))
    history, start, best_score, best_epoch = [], 0, -1.0, None
    last_path = directory / "last.pt"
    best_path = directory / "best.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["variant"] != variant or state["seed"] != seed:
            raise ValueError("E8 checkpoint does not match variant/seed")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start = int(state["epoch"])
        best_score = float(state["best_score"])
        best_epoch = state["best_epoch"]
        print(f"Resuming {variant} seed={seed} from epoch {start}", flush=True)
    train_eval = loader(train_frame, cfg, labels)
    validation = loader(val_frame, cfg, labels)
    test_eval = loader(test_frame, cfg, labels)
    print(f"E8 {variant} seed{seed}: {len(train_frame)} train, {len(val_frame)} val, "
          f"{len(test_frame)} test, {len(labels)} labels; 300 epochs", flush=True)
    for epoch in range(start, int(cfg["epochs"])):
        # Set augmentation, dropout and sampler RNG by (run seed, epoch), so
        # a resumed run reproduces the same subsequent epoch as an uninterrupted run.
        seed_all(seed * 100000 + epoch + 1)
        training = loader(train_frame, cfg, labels, train=True, seed=seed * 1000 + epoch)
        model.train()
        running, count = 0.0, 0
        for images, targets, _ in training:
            if len(images) == 1:
                for mod in model.modules():
                    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                        mod.eval()
            images, targets = images.to(cfg["device"]), targets.to(cfg["device"])
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch+1}")
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(images)
            count += len(images)
        scheduler.step()
        t = evaluate(model, train_eval, cfg["device"], labels)
        v = evaluate(model, validation, cfg["device"], labels)
        item = {
            "epoch": epoch+1, "augmented_train_loss": running / count,
            "train_eval_loss": t["loss"], "val_loss": v["loss"],
            "train_eval_accuracy": t["accuracy"], "val_accuracy": v["accuracy"],
            "train_eval_macro_f1": t["macro_f1"], "val_macro_f1": v["macro_f1"],
            "accuracy_gap": t["accuracy"] - v["accuracy"],
            "val_to_train_loss_ratio": v["loss"] / max(t["loss"], 1e-8),
            "lr_backbone": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[1]["lr"],
        }
        history.append(item)
        item["overfitting_warning"] = detect_overfitting(history, cfg)
        if v["macro_f1"] > best_score:
            best_score, best_epoch = v["macro_f1"], epoch+1
            atomic_save({
                "model": model.state_dict(), "variant": variant, "seed": seed,
                "labels": labels, "epoch": best_epoch,
                "val_macro_f1": best_score, "pretrained_sha256": provenance["pretrained_sha256"],
            }, best_path)
        atomic_save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(), "history": history,
            "epoch": epoch+1, "best_score": best_score,
            "best_epoch": best_epoch, "variant": variant, "seed": seed,
        }, last_path)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        tag = " WARNING:OVERFITTING" if item["overfitting_warning"] else ""
        print(f"E8 {variant} seed{seed} epoch {epoch+1}/300: "
              f"train loss={t['loss']:.4f}, val loss={v['loss']:.4f}, "
              f"train acc={t['accuracy']:.3f}, val acc={v['accuracy']:.3f}, "
              f"val macroF1={v['macro_f1']:.3f}{tag}", flush=True)
    if not best_path.is_file():
        raise RuntimeError("Best validation checkpoint not found")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    result, truth, prediction, scores = evaluate(
        model, test_eval, cfg["device"], labels, include_predictions=True)
    predictions = test_frame[["id", "path", "label"]].copy().reset_index(drop=True)
    predictions["predicted"] = [labels[int(k)] for k in prediction]
    predictions["confidence"] = scores.max(axis=1)
    predictions["correct"] = truth == prediction
    predictions.to_csv(directory / "test_predictions.csv", index=False)
    np.savez_compressed(directory / "test_scores.npz", scores=scores, targets=truth)
    pd.DataFrame([{"label": label, **result["report"][label]}
                  for label in labels]).to_csv(directory / "per_class.csv", index=False)
    dump_json(directory / "confusion_matrix.json", {
        "labels": labels, "matrix": result["confusion_matrix"]})
    _save_curves(directory, history)
    warnings = [r["epoch"] for r in history if r["overfitting_warning"]]
    overfit = {
        "epoch_count": len(history), "best_validation_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "warning_count": len(warnings), "first_warning_epoch": warnings[0] if warnings else None,
        "warning_epochs": warnings,
        "last_train_val_accuracy_gap": history[-1]["accuracy_gap"],
        "notes": "Monitoring only; no early stop; test evaluated after selection by validation.",
    }
    dump_json(directory / "overfitting_summary.json", overfit)
    summary = {
        "variant": variant, "seed": seed, "epochs_completed": len(history),
        "train_count": len(train_frame), "val_count": len(val_frame),
        "test_count": len(test_frame), "classes": labels,
        "pretrained_repository": provenance["pretrained_repository"],
        "pretrained_revision": provenance["pretrained_revision"],
        "pretrained_sha256": provenance["pretrained_sha256"],
        "backbone_fully_trainable": True, "selection": "best validation macro-F1",
        "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
        "test": result, "overfitting": overfit,
    }
    # Completion marker LAST. Never silently overwrite a partially completed run.
    dump_json(directory / "metrics.json", summary)
    print(f"E8 results saved: {directory}", flush=True)
    return summary


def aggregate(cfg):
    """Require three completed seeds per variant; never mix protocols/checkpoints."""
    root = Path(cfg["output"])
    runs, per_class = [], []
    hashes = set()
    for variant in VARIANTS:
        for seed in cfg["seeds"]:
            path = root / variant / f"seed{seed}"
            meta_path = path / "metrics.json"
            prov_path = path / "provenance.json"
            if not meta_path.is_file() or not prov_path.is_file():
                raise FileNotFoundError(f"Missing completed E8 run: {path}")
            m = json.loads(meta_path.read_text())
            if m["variant"] != variant or m["seed"] != seed or m["epochs_completed"] != cfg["epochs"]:
                raise ValueError(f"E8 results mismatch in {path}")
            if json.loads(prov_path.read_text()) != _run_provenance(cfg, variant, seed):
                raise ValueError(f"E8 provenance changed: {path}")
            hashes.add(m["pretrained_sha256"])
            runs.append({
                "variant": variant, "seed": seed, "test_accuracy": m["test"]["accuracy"],
                "test_macro_f1": m["test"]["macro_f1"],
                "best_validation_macro_f1": m["best_validation_macro_f1"],
                "best_epoch": m["best_epoch"],
                "overfitting_warning_count": m["overfitting"]["warning_count"],
            })
            for label in m["classes"]:
                detail = m["test"]["report"][label]
                per_class.append({
                    "variant": variant, "seed": seed, "class": label,
                    "precision": detail["precision"], "recall": detail["recall"],
                    "f1": detail["f1-score"], "support": detail["support"],
                })
    if len(hashes) != 1:
        raise ValueError("E8 runs do not use the same pretrained checkpoint")
    root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(runs)
    cats = pd.DataFrame(per_class)
    summary.to_csv(root / "per_seed.csv", index=False)
    cats.to_csv(root / "per_class_per_seed.csv", index=False)
    stats = summary.groupby("variant")[["test_accuracy", "test_macro_f1",
                                         "best_validation_macro_f1"]].agg(["mean", "std"])
    stats.columns = ["_".join(name) for name in stats.columns]
    stats["n_seeds"] = summary.groupby("variant").seed.nunique()
    stats.to_csv(root / "mean_std.csv")
    cat_stats = cats.groupby(["variant", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    cat_stats.columns = ["_".join(name) for name in cat_stats.columns]
    cat_stats.to_csv(root / "per_class_mean_std.csv")
    print(stats.to_string(), flush=True)
    print(f"Results: {root}", flush=True)


def download(cfg):
    from huggingface_hub import hf_hub_download
    p = cfg["pretrained"]
    destination = Path(p["checkpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = file_hash(destination)
        if actual != p["sha256"]:
            raise ValueError(f"Unexpected pretrained checkpoint SHA256: {actual}")
        print(f"Using verified checkpoint: {destination} SHA256 {actual}")
        return
    upstream = hf_hub_download(
        repo_id=p["repository"], filename=p["filename"], revision=p["revision"])
    if file_hash(upstream) != p["sha256"]:
        raise ValueError("Downloaded pretrained weights failed SHA256 verification")
    shutil.copyfile(upstream, destination)
    dump_json(destination.with_suffix(".json"), {
        "repository": p["repository"], "revision": p["revision"],
        "filename": p["filename"], "sha256": file_hash(destination),
        "warning": "External PyTorch .pt checkpoint may execute pickle code when loaded; review source.",
    })
    print(f"Downloaded {destination} SHA256 {file_hash(destination)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "train", "aggregate"])
    parser.add_argument("--config", default="configs/e8.yaml")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.action == "download":
        return download(cfg)
    if args.action == "aggregate":
        return aggregate(cfg)
    if args.variant is None or args.seed is None:
        parser.error("train requires both --variant and --seed")
    return train_one(cfg, args.variant, args.seed, args.resume)


if __name__ == "__main__":
    main()
