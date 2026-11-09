"""E15-A/B: direct eight-class SigLIP2 classification (no E9-B second stage).

E15-A: last four SigLIP2 vision blocks + final norm + eight-class head.
E15-B: entire pretrained SigLIP2 vision encoder + eight-class head.

Train/validation/test image identities are identical to E11/E14, and test
labels are accessed only in the separate evaluate command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from .common import atomic_save, dump_json, file_hash, read_config, seed_all
from .data import open_image
from .e9 import EIGHT, early_stopping_step
from .e13e14 import (
    HFRoot, _count, _load_trained_state, _move, _trainable_state,
    batches, fixed_partitions, save_curves, source_signature,
)

VARIANTS = ("e15a", "e15b")
LABELS = tuple(EIGHT)


def setting_for(cfg, variant):
    if variant not in VARIANTS or variant not in cfg.get("variants", {}):
        raise ValueError(f"Unknown E15 variant: {variant}")
    setting = cfg["variants"][variant]
    expected = "partial" if variant == "e15a" else "full"
    if (setting.get("family") != "siglip2" or setting.get("mode") != expected
            or int(setting.get("image_size", -1)) != 224):
        raise ValueError(f"{variant} requires SigLIP2-224 and mode={expected}")
    if variant == "e15a" and int(setting.get("unfreeze_last_blocks", 0)) != 4:
        raise ValueError("E15-A must fine-tune exactly the last four blocks")
    if not setting.get("pretrained_dir"):
        raise ValueError("Missing local SigLIP2 checkpoint path")
    return setting


def run_dir(cfg, variant, seed):
    return Path(cfg["output"]) / variant / f"seed{int(seed)}"


def evaluation_dir(cfg, variant, seed):
    return run_dir(cfg, variant, seed) / "evaluation"


def source_hash(cfg, variant):
    # Reuse the exact E14 checkpoint fingerprinting, including safetensors.
    e14_cfg = {"variants": {"e14d": setting_for(cfg, variant)}}
    return source_signature(e14_cfg, "e14d")


def verify(cfg, variant=None, require_sources=False):
    if list(cfg["seeds"]) != [0, 1, 2]:
        raise ValueError("E15 seeds must be [0,1,2]")
    if int(cfg["epochs"]) != 300 or int(cfg["batch_size"]) != 16:
        raise ValueError("E15 requires 300 max epochs and batch size 16")
    rule = cfg["early_stopping"]
    if (rule["monitor"] != "val_macro_f1" or int(rule["patience"]) != 30):
        raise ValueError("E15 selection: val_macro_f1, patience=30")
    if float(cfg["head_lr"]) != 0.001:
        raise ValueError("E15 head learning rate must be 1e-3")
    check = [variant] if variant else VARIANTS
    for name in check:
        s = setting_for(cfg, name)
        if float(s["backbone_lr"]) != 1e-5:
            raise ValueError("E15 backbone learning rate must be 1e-5")
    if cfg["variants"]["e15a"]["pretrained_dir"] != cfg["variants"]["e15b"]["pretrained_dir"]:
        raise ValueError("E15-A/B must use same pretrained snapshot")
    if require_sources:
        for key in ("manifest", "data_root"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {cfg[key]}")
        for name in check:
            source_hash(cfg, name)


class DirectImages(Dataset):
    """E14 SigLIP2 processor/augmentations, but retain all eight labels."""
    def __init__(self, frame, cfg, variant, train=False):
        from transformers import AutoImageProcessor
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cfg["data_root"])
        self.processor = AutoImageProcessor.from_pretrained(
            str(setting_for(cfg, variant)["pretrained_dir"]), local_files_only=True)
        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.35),
        ]) if train else None
        self.lookup = {name: i for i, name in enumerate(LABELS)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = open_image(self.root / row.path).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        processed = self.processor(images=image, return_tensors="pt")
        values = {key: tensor.squeeze(0) for key, tensor in processed.items()
                  if isinstance(tensor, torch.Tensor)}
        if "pixel_values" not in values or values["pixel_values"].shape != (3, 224, 224):
            raise ValueError("Expected SigLIP2 224px image processor")
        return values, self.lookup[str(row.label)], int(index)


class DirectSigLIP2(HFRoot):
    """Reuse E14's exact pretrained vision encoder and pooling behaviour."""
    def __init__(self, setting):
        mode = setting["mode"]
        parent_setting = dict(setting, mode=("partial" if mode == "partial" else "frozen"))
        super().__init__(parent_setting)
        self.head = nn.Linear(int(self.encoder.config.hidden_size), len(LABELS))
        if mode == "full":
            self.encoder.requires_grad_(True)
            self.mode = "full"
        if mode == "partial":
            from .e13e14 import _vision_layers_and_norm
            blocks, norm = _vision_layers_and_norm(self.encoder)
            assert len(blocks) >= 4
            assert any(p.requires_grad for p in norm.parameters())
            assert all(not p.requires_grad for p in list(blocks)[:-4] for _ in [0])
        if mode == "full" and not all(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Full fine-tuning did not unfreeze all parameters")


def make_model(cfg, variant):
    return DirectSigLIP2(setting_for(cfg, variant))


@torch.no_grad()
def score(model, loader, cfg, detailed=False):
    model.eval()
    y_all, pred_all, prob_all, loss_sum = [], [], [], 0.0
    for x, y, _ in loader:
        logits = model(_move(x, cfg["device"]))
        probabilities = logits.softmax(-1).cpu().numpy()
        loss_sum += float(F.cross_entropy(logits, y.to(cfg["device"])).item()) * len(y)
        y_all.extend(y.tolist())
        pred_all.extend(probabilities.argmax(axis=1).tolist())
        prob_all.append(probabilities)
    if not y_all:
        raise ValueError("Empty evaluation loader")
    truth, predicted = np.asarray(y_all), np.asarray(pred_all)
    labels = list(range(len(LABELS)))
    result = {
        "loss": loss_sum / len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
    }
    if detailed:
        result["per_class"] = classification_report(
            truth, predicted, labels=labels, target_names=list(LABELS),
            output_dict=True, zero_division=0)
        result["confusion_matrix"] = confusion_matrix(truth, predicted, labels=labels).tolist()
        return result, truth, predicted, np.concatenate(prob_all)
    return result


def provenance(cfg, variant, seed, hash_value):
    return {
        "protocol": "E15-direct-eight-SigLIP2-v1",
        "variant": variant, "seed": seed,
        "manifest_sha256": file_hash(cfg["manifest"]),
        "source_sha256": hash_value,
        "config": cfg,
        "classes": list(LABELS),
    }


def train_one(cfg, variant, seed, resume=False):
    verify(cfg, variant, require_sources=True)
    if seed not in cfg["seeds"]:
        raise ValueError("Unsupported E15 seed")
    seed_all(seed)
    train, val, test = fixed_partitions(cfg)
    if any(set(frame.label) != set(LABELS) for frame in (train, val, test)):
        raise ValueError("Original eight-class E11 splits changed")
    fingerprint = provenance(cfg, variant, seed, source_hash(cfg, variant))
    out = run_dir(cfg, variant, seed)
    if out.exists() and not resume:
        raise FileExistsError(f"Existing E15 run: {out}; use --resume to continue")
    out.mkdir(parents=True, exist_ok=True)
    previous = out / "provenance.json"
    if previous.exists() and json.loads(previous.read_text()) != fingerprint:
        raise ValueError("Resume provenance mismatch")
    if not previous.exists():
        dump_json(previous, fingerprint)
    if (out / "training_metrics.json").exists():
        print(f"COMPLETED {variant} seed{seed}", flush=True)
        return
    for name, frame in zip(("train", "val", "test"), (train, val, test)):
        frame[["id", "path", "label", "group", "split"]].to_csv(
            out / f"{name}_manifest.csv", index=False)

    model = make_model(cfg, variant).to(cfg["device"])
    train_data = DirectImages(train, cfg, variant, train=True)
    train_eval = batches(DirectImages(train, cfg, variant), cfg)
    val_eval = batches(DirectImages(val, cfg, variant), cfg)
    optim = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad],
         "lr": float(setting_for(cfg, variant)["backbone_lr"])},
        {"params": model.head.parameters(), "lr": float(cfg["head_lr"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(cfg["epochs"]))
    history, start, best_score, best_epoch, early = [], 0, -1.0, None, None
    last = out / "last.pt"
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if (state["seed"] != seed or state["variant"] != variant
                or state["manifest_sha256"] != fingerprint["manifest_sha256"]
                or state["source_sha256"] != fingerprint["source_sha256"]):
            raise ValueError("Invalid E15 resume checkpoint")
        _load_trained_state(model, state)
        optim.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start = state["history"], int(state["epoch"])
        best_score, best_epoch, early = state["best_score"], state["best_epoch"], state["early"]
        if early["bad_epochs"] >= int(cfg["early_stopping"]["patience"]):
            start = int(cfg["epochs"])
    for epoch in range(start, int(cfg["epochs"])):
        seed_all(seed * 100000 + epoch + 1)
        model.train()
        total_loss, count = 0.0, 0
        for x, y, _ in batches(train_data, cfg, shuffle=True, seed=seed * 1000 + epoch):
            optim.zero_grad(set_to_none=True)
            logits = model(_move(x, cfg["device"]))
            loss = F.cross_entropy(logits, y.to(cfg["device"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite E15 training loss")
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * len(y)
            count += len(y)
        scheduler.step()
        tr, va = score(model, train_eval, cfg), score(model, val_eval, cfg)
        row = {
            "epoch": epoch + 1, "augmented_train_loss": total_loss / count,
            "train_eval_loss": tr["loss"], "val_loss": va["loss"],
            "train_eval_accuracy": tr["accuracy"], "val_accuracy": va["accuracy"],
            "train_eval_macro_f1": tr["macro_f1"], "val_macro_f1": va["macro_f1"],
            "accuracy_gap": tr["accuracy"] - va["accuracy"],
            "lr_backbone": optim.param_groups[0]["lr"],
            "lr_head": optim.param_groups[1]["lr"],
        }
        history.append(row)
        early, stop = early_stopping_step(va["macro_f1"], early, cfg, epoch + 1)
        row["bad_epochs"] = early["bad_epochs"]
        if va["macro_f1"] > best_score:
            best_score, best_epoch = va["macro_f1"], epoch + 1
            weights, kind = _trainable_state(model, "hf")
            atomic_save({
                "state": weights, "state_kind": kind,
                "variant": variant, "seed": seed, "epoch": best_epoch,
                "classes": list(LABELS),
                "manifest_sha256": fingerprint["manifest_sha256"],
                "source_sha256": fingerprint["source_sha256"],
            }, out / "best.pt")
        weights, kind = _trainable_state(model, "hf")
        atomic_save({
            "state": weights, "state_kind": kind,
            "optimizer": optim.state_dict(), "scheduler": scheduler.state_dict(),
            "variant": variant, "seed": seed, "epoch": epoch + 1,
            "manifest_sha256": fingerprint["manifest_sha256"],
            "source_sha256": fingerprint["source_sha256"],
            "history": history, "best_score": best_score,
            "best_epoch": best_epoch, "early": early,
        }, last)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"{variant} seed{seed} epoch {epoch+1}: "
              f"train_acc={tr['accuracy']:.4f} val_acc={va['accuracy']:.4f} "
              f"val_macro_f1={va['macro_f1']:.4f} bad={early['bad_epochs']}", flush=True)
        if stop:
            break
    if best_epoch is None:
        raise RuntimeError("No E15 best checkpoint")
    save_curves(out, history, variant)
    dump_json(out / "training_metrics.json", {
        "variant": variant, "seed": seed, "task": "direct_eight_class",
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_validation_macro_f1": best_score, "selection": "validation macro-F1",
        "parameter_counts": _count(model),
        "split_sizes": [len(train), len(val), len(test)],
        "test_evaluated_during_training": False,
    })


def evaluate_one(cfg, variant, seed):
    verify(cfg, variant, require_sources=True)
    if seed not in cfg["seeds"]:
        raise ValueError("Unsupported E15 seed")
    out = evaluation_dir(cfg, variant, seed)
    if (out / "metrics.json").exists():
        print(f"COMPLETED {variant} seed{seed} evaluation", flush=True)
        return
    if out.exists():
        raise FileExistsError(f"Incomplete E15 evaluation: {out}")
    training_path = run_dir(cfg, variant, seed) / "training_metrics.json"
    best_path = run_dir(cfg, variant, seed) / "best.pt"
    if not training_path.is_file() or not best_path.is_file():
        raise FileNotFoundError("E15 training not complete; no test evaluation")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if (checkpoint["variant"] != variant or checkpoint["seed"] != seed
            or checkpoint["classes"] != list(LABELS)
            or checkpoint["manifest_sha256"] != file_hash(cfg["manifest"])
            or checkpoint["source_sha256"] != source_hash(cfg, variant)):
        raise ValueError("E15 checkpoint provenance/labels mismatch")
    seed_all(seed)
    _, _, test = fixed_partitions(cfg)
    model = make_model(cfg, variant).to(cfg["device"])
    _load_trained_state(model, checkpoint)
    report, truth, predicted, probs = score(
        model, batches(DirectImages(test, cfg, variant), cfg), cfg, detailed=True)
    out.mkdir(parents=True, exist_ok=False)
    trace = test[["id", "path", "label"]].copy().reset_index(drop=True)
    trace["variant"] = variant
    trace["seed"] = seed
    trace["prediction"] = [LABELS[int(i)] for i in predicted]
    trace["confidence"] = probs.max(axis=1)
    trace["correct"] = predicted == truth
    for i, label in enumerate(LABELS):
        trace[f"p_{label}"] = probs[:, i]
    trace.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "probabilities.npz", probabilities=probs, target=truth)
    pd.DataFrame([{"class": label, **report["per_class"][label]} for label in LABELS]
                 ).to_csv(out / "per_class.csv", index=False)
    dump_json(out / "confusion_matrix.json", {
        "classes": list(LABELS), "matrix": report["confusion_matrix"]})
    dump_json(out / "metrics.json", {
        "variant": variant, "seed": seed, "test_count": len(test),
        "direct": report, "source_sha256": checkpoint["source_sha256"],
        "best_checkpoint_sha256": file_hash(best_path),
        "note": "226-image historical test set; exploratory comparison, "
                "not a new independent generalisation benchmark.",
    })
    print(f"{variant} seed{seed}: accuracy={report['accuracy']:.4f} "
          f"macro_f1={report['macro_f1']:.4f}", flush=True)


def aggregate(cfg, selected=VARIANTS, reference_e14=None):
    records, classes, comparisons = [], [], []
    for variant in selected:
        for seed in cfg["seeds"]:
            directory = evaluation_dir(cfg, variant, seed)
            path = directory / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing E15 evaluation {path}")
            payload = json.loads(path.read_text())
            if payload["variant"] != variant or payload["seed"] != seed:
                raise ValueError(f"Wrong seed/variant in {path}")
            metrics = payload["direct"]
            records.append({
                "variant": variant, "seed": seed,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "correct_n": int(round(metrics["accuracy"] * payload["test_count"])),
            })
            for label in LABELS:
                entry = metrics["per_class"][label]
                classes.append({
                    "variant": variant, "seed": seed, "class": label,
                    "precision": entry["precision"], "recall": entry["recall"],
                    "f1": entry["f1-score"], "support": entry["support"],
                })
            if reference_e14:
                base = Path(reference_e14) / "e14d" / "cascade" / f"seed{seed}" / "test_predictions.csv"
                if not base.is_file():
                    raise FileNotFoundError(f"E14-D reference missing: {base}")
                earlier = pd.read_csv(base).set_index("id").sort_index()
                current = pd.read_csv(directory / "test_predictions.csv").set_index("id").sort_index()
                if not earlier.index.equals(current.index) or not earlier.label.equals(current.label):
                    raise ValueError("E14-D and E15 test IDs/labels differ")
                for mode in ("hard", "soft"):
                    past = earlier[f"{mode}_prediction"] == earlier.label
                    new = current.prediction == current.label
                    comparisons.append({
                        "variant": variant, "seed": seed, "baseline": f"e14d_{mode}",
                        "both_correct": int((past & new).sum()),
                        "fixed_from_e14": int((~past & new).sum()),
                        "new_errors_vs_e14": int((past & ~new).sum()),
                        "both_wrong": int((~past & ~new).sum()),
                        "net_correct_gain": int(new.sum() - past.sum()),
                        "n_images": len(new),
                    })
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "per_seed.csv", index=False)
    summary = table.groupby("variant")[["accuracy", "macro_f1", "correct_n"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary["n_seeds"] = table.groupby("variant").seed.nunique()
    summary.to_csv(out / "mean_std.csv")
    per_class = pd.DataFrame(classes)
    per_class.to_csv(out / "per_class_per_seed.csv", index=False)
    class_summary = per_class.groupby(["variant", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    class_summary.columns = ["_".join(col) for col in class_summary.columns]
    class_summary["support_per_seed"] = per_class.groupby(["variant", "class"]).support.first()
    class_summary.to_csv(out / "per_class_mean_std.csv")
    if comparisons:
        pd.DataFrame(comparisons).to_csv(out / "paired_vs_e14d.csv", index=False)
    print(summary.to_string(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "train", "evaluate", "aggregate"])
    parser.add_argument("--config", default="configs/e15_artemis.yaml")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-smoke", action="store_true",
                        help="Load actual checkpoint and check one CPU forward (no backward)")
    parser.add_argument("--reference-e14",
                        help="E14 output directory for paired comparison against E14-D")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.action == "aggregate":
        return aggregate(cfg, args.variants or VARIANTS, args.reference_e14)
    if args.action == "preflight":
        verify(cfg, require_sources=True)
        frames = fixed_partitions(cfg)
        for frame in frames:
            for filename in frame.path:
                path = Path(cfg["data_root"]) / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing image: {path}")
        for v in VARIANTS:
            print(f"PASS {v}: {setting_for(cfg, v)['mode']}, source={source_hash(cfg, v)}", flush=True)
        print("PASS E11 image IDs/splits and complete eight-class labels", flush=True)
        if args.model_smoke:
            cpu = {**cfg, "device": "cpu", "workers": 0, "batch_size": 2}
            for v in VARIANTS:
                model = make_model(cpu, v).eval()
                x, y, _ = next(iter(batches(DirectImages(frames[0].iloc[:1], cpu, v), cpu)))
                with torch.no_grad():
                    z = model(_move(x, "cpu"))
                if z.shape != (1, len(LABELS)) or not torch.isfinite(z).all():
                    raise ValueError(f"Smoke forward failed {v}")
                print(f"PASS {v}: eight class forward; parameters={_count(model)}", flush=True)
                del model, z
        print("E15 PREFLIGHT PASSED", flush=True)
        return
    if args.variant is None:
        parser.error("--variant required for train/evaluate")
    if args.action == "train":
        return train_one(cfg, args.variant, args.seed, resume=args.resume)
    if args.resume:
        parser.error("--resume is only allowed for train")
    return evaluate_one(cfg, args.variant, args.seed)


if __name__ == "__main__":
    main()
