"""E11: damage-first six-way CarDD YOLOv8s-seg v2 full fine-tuning.

Stage 1: fully train original pretrained YOLOv8s backbone + new six-way
image classification head (dent, scratch, glass, head lamp, tail lamp,
unknown). Stage 2: REUSE, without retraining, the matching seed E9-B binary
Bumper/Door best.pt YOLO11 expert when Stage 1 predicts dent/scratch.
Compute hard and soft eight-way cascade predictions on original test items.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from .common import atomic_save, dump_json, file_hash, read_config, seed_all
from .data import read_manifest, open_image, transforms
from .models import yolo_features, Classifier, parameter_counts
from .e8 import select_data, padded_transforms, PartPretrainedClassifier, detect_overfitting
from .e9 import EIGHT, BINARY, early_stopping_step

ROOT = ("dent", "scratch", "glass_shatter", "head_lamp", "tail_lamp", "unknown")
BRANCHED = ("dent", "scratch")
DIRECT = ("glass_shatter", "head_lamp", "tail_lamp", "unknown")

# The ordering of the eight classes in E9/E10 is fixed and must be preserved.
E11_LABELS = {
    "bumper_dent": "dent",
    "bumper_scratch": "scratch",
    "door_dent": "dent",
    "door_scratch": "scratch",
    "glass_shatter": "glass_shatter",
    "head_lamp": "head_lamp",
    "tail_lamp": "tail_lamp",
    "unknown": "unknown",
}
assert set(E11_LABELS) == set(EIGHT)
assert set(E11_LABELS.values()) == set(ROOT)


def root_label(label: str) -> str:
    try:
        return E11_LABELS[label]
    except KeyError as e:
        raise ValueError(f"E11 unexpected original label: {label}") from e


def partitions(cfg):
    """Always preserve E8/E9 group-disjoint original eight-class manifest."""
    split, _ = select_data(read_manifest(cfg["manifest"]), "8class")
    return split


def train_dir(cfg, seed):
    return Path(cfg["output"]) / "six_class" / f"seed{seed}"


def eval_dir(cfg, seed):
    return Path(cfg["output"]) / "cascade" / f"seed{seed}"


def expert_path(cfg, seed):
    return Path(cfg["expert"]["directory"]) / f"seed{seed}" / "best.pt"


class RootImages(Dataset):
    def __init__(self, frame, cfg, train=False):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cfg["data_root"])
        # Exactly the previous Experiment 2 image transforms: 224px and
        # train-time random resized crop/flip/jitter; no ImageNet norm.
        self.transform = transforms(int(cfg["image_size"]), train)
        self.index = {label: i for i, label in enumerate(ROOT)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        img = self.transform(open_image(self.root / row.path))
        return img, self.index[root_label(row.label)], index


class ExpertImages(Dataset):
    """E9-B's original 384px aspect-preserving eval preprocessing."""
    def __init__(self, frame, cfg):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cfg["data_root"])
        self.transform = padded_transforms(int(cfg["expert"]["image_size"]), False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.transform(open_image(self.root / row.path)), index


def loader(dataset, cfg, train=False, seed=0):
    return DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=train,
        num_workers=int(cfg["workers"]), drop_last=False,
        generator=torch.Generator().manual_seed(int(seed)),
        pin_memory=str(cfg["device"]).startswith("cuda"))


def verify_train_config(cfg):
    if int(cfg["epochs"]) != 300:
        raise ValueError("E11 protocol requires a maximum of 300 epochs")
    if int(cfg["early_stopping"]["patience"]) != 30:
        raise ValueError("E11 protocol requires patience=30")
    if cfg["early_stopping"].get("monitor") != "val_macro_f1":
        raise ValueError("E11 requires validation Macro-F1 for early stopping")
    if sorted(set(cfg["seeds"])) != [0, 1, 2]:
        raise ValueError("E11 seeds must be exactly 0,1,2")
    if not Path(cfg["checkpoint"]).is_file():
        raise FileNotFoundError(
            f"Missing original CarDD YOLOv8s-seg v2 weights: {cfg['checkpoint']}")
    if int(cfg["image_size"]) != 224:
        raise ValueError("E11 Stage 1 uses the prior E2 224x224 transform")


def build_root(cfg):
    # Explicit 'full' method trains EVERY CarDD backbone parameter, not linear probe.
    features = yolo_features(cfg, "full")
    model = Classifier(features, len(ROOT))
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("E11 root is not fully trainable")
    return model


@torch.no_grad()
def measure(model, batches, device, detailed=False):
    model.eval()
    truth, predictions, probs, loss_sum = [], [], [], 0.0
    for x, y, _ in batches:
        logits = model(x.to(device))
        scores = logits.softmax(-1).cpu().numpy()
        loss_sum += float(F.cross_entropy(logits, y.to(device)).item()) * len(y)
        truth.extend(y.tolist())
        predictions.extend(scores.argmax(1).tolist())
        probs.append(scores)
    if not truth:
        raise ValueError("E11 empty evaluation partition")
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    metrics = {
        "loss": loss_sum / len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(
            truth, predicted, labels=range(len(ROOT)),
            average="macro", zero_division=0)),
    }
    if detailed:
        metrics["per_class"] = classification_report(
            truth, predicted, labels=list(range(len(ROOT))),
            target_names=list(ROOT), output_dict=True, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(
            truth, predicted, labels=range(len(ROOT))).tolist()
        return metrics, truth, predicted, np.concatenate(probs)
    return metrics


def provenance(cfg, seed):
    return {
        "protocol": "E11-CarDDv2-six-class-full-v1",
        "seed": int(seed),
        "config": cfg,
        "manifest_sha256": file_hash(cfg["manifest"]),
        "cardd_v2_sha256": file_hash(cfg["checkpoint"]),
    }


def save_curves(out, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    for filename, train_col, val_col, ylabel in (
        ("loss_curve.png", "train_eval_loss", "val_loss", "Cross-entropy"),
        ("accuracy_curve.png", "train_eval_accuracy", "val_accuracy", "Accuracy"),
        ("f1_curve.png", "train_eval_macro_f1", "val_macro_f1", "Macro-F1"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(frame.epoch, frame[train_col], label="Train")
        ax.plot(frame.epoch, frame[val_col], label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"E11 Stage 1 {ylabel}")
        if ylabel != "Cross-entropy":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def train_root(cfg, seed, resume=False):
    verify_train_config(cfg)
    if seed not in cfg["seeds"]:
        raise ValueError(f"E11 unsupported seed: {seed}")
    seed_all(seed)
    train, val, test = partitions(cfg)
    for part in (train, val, test):
        if {root_label(x) for x in part.label} != set(ROOT):
            raise ValueError("E11 split lacks one or more root categories")
    out = train_dir(cfg, seed)
    if out.exists() and not resume:
        raise FileExistsError(f"E11 existing output: {out}; use --resume")
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = provenance(cfg, seed)
    if (out / "provenance.json").exists():
        if json.loads((out / "provenance.json").read_text()) != fingerprint:
            raise ValueError("E11 provenance mismatch: refuse unsafe resume")
    else:
        dump_json(out / "provenance.json", fingerprint)
    if (out / "metrics.json").exists():
        print(f"E11 already completed: {out}", flush=True)
        return
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame[["id", "path", "label", "group", "split"]].to_csv(
            out / f"{name}_manifest.csv", index=False)
    model = build_root(cfg).to(cfg["device"])
    counts = parameter_counts(model)
    optimiser = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": float(cfg["lr"])},
        {"params": model.head.parameters(), "lr": float(cfg["head_lr"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, int(cfg["epochs"]))
    best = out / "best.pt"
    last = out / "last.pt"
    history, start, best_score, best_epoch, early = [], 0, -1.0, None, None
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state.get("seed") != seed:
            raise ValueError("E11 last.pt seed mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimiser.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start = state["history"], int(state["epoch"])
        best_score, best_epoch = float(state["best_score"]), int(state["best_epoch"])
        early = state["early_stopping_state"]
        if early["bad_epochs"] >= cfg["early_stopping"]["patience"]:
            start = int(cfg["epochs"])
    train_eval = loader(RootImages(train, cfg), cfg)
    val_eval = loader(RootImages(val, cfg), cfg)
    test_eval = loader(RootImages(test, cfg), cfg)
    for epoch in range(start, int(cfg["epochs"])):
        seed_all(seed * 100000 + epoch + 1)
        batches = loader(RootImages(train, cfg, train=True), cfg, train=True,
                         seed=seed * 1000 + epoch)
        model.train()
        total, count = 0.0, 0
        for x, y, _ in batches:
            if len(y) == 1:
                for layer in model.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite E11 root training loss")
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        scheduler.step()
        tr = measure(model, train_eval, cfg["device"])
        va = measure(model, val_eval, cfg["device"])
        row = {
            "epoch": epoch + 1,
            "augmented_train_loss": total / count,
            "train_eval_loss": tr["loss"], "val_loss": va["loss"],
            "train_eval_accuracy": tr["accuracy"], "val_accuracy": va["accuracy"],
            "train_eval_macro_f1": tr["macro_f1"], "val_macro_f1": va["macro_f1"],
            "accuracy_gap": tr["accuracy"] - va["accuracy"],
            "val_to_train_loss_ratio": va["loss"] / max(tr["loss"], 1e-8),
            "lr_backbone": optimiser.param_groups[0]["lr"],
            "lr_head": optimiser.param_groups[1]["lr"],
        }
        history.append(row)
        row["overfitting_warning"] = detect_overfitting(history, cfg)
        early, stop = early_stopping_step(va["macro_f1"], early, cfg, epoch + 1)
        row["early_stop_bad_epochs"] = early["bad_epochs"]
        row["early_stop_triggered"] = stop
        if va["macro_f1"] > best_score:
            best_score, best_epoch = va["macro_f1"], epoch + 1
            atomic_save({
                "model": model.state_dict(), "classes": list(ROOT),
                "seed": seed, "epoch": best_epoch,
                "val_macro_f1": best_score,
                "manifest_sha256": fingerprint["manifest_sha256"],
                "cardd_v2_sha256": fingerprint["cardd_v2_sha256"],
            }, best)
        atomic_save({
            "model": model.state_dict(),
            "optimizer": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "epoch": epoch + 1,
            "best_score": best_score,
            "best_epoch": best_epoch, "seed": seed,
            "early_stopping_state": early,
        }, last)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"E11 full seed{seed} epoch {epoch+1}/{cfg['epochs']}: "
              f"train_acc={tr['accuracy']:.4f} val_acc={va['accuracy']:.4f} "
              f"val_macro_f1={va['macro_f1']:.4f} bad_epochs={early['bad_epochs']}",
              flush=True)
        if stop:
            print(f"E11 early stop seed{seed} at {epoch+1}, best_epoch={best_epoch}",
                  flush=True)
            break
    if not best.exists():
        raise FileNotFoundError(f"E11 missing best checkpoint: {best}")
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    measured, truth, pred, probs = measure(
        model, test_eval, cfg["device"], detailed=True)
    rows = test[["id", "path", "label"]].copy().reset_index(drop=True)
    rows["root_truth"] = [ROOT[i] for i in truth]
    rows["root_prediction"] = [ROOT[i] for i in pred]
    rows["root_confidence"] = probs.max(axis=1)
    rows["correct"] = pred == truth
    rows.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "test_probabilities.npz", scores=probs, targets=truth)
    pd.DataFrame([{"class": label, **measured["per_class"][label]}
                  for label in ROOT]).to_csv(out / "per_class.csv", index=False)
    dump_json(out / "confusion_matrix.json", {
        "labels": list(ROOT), "matrix": measured["confusion_matrix"]})
    save_curves(out, history)
    warnings = [h["epoch"] for h in history if h["overfitting_warning"]]
    dump_json(out / "metrics.json", {
        "task": "six_class_root", "seed": seed, "classes": list(ROOT),
        "train_count": len(train), "validation_count": len(val),
        "test_count": len(test), "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
        "test": measured,
        "parameter_counts": counts,
        "overfitting_warning_epochs": warnings,
        "early_stopping": {
            "monitor": "val_macro_f1", "patience": 30,
            "min_delta": float(cfg["early_stopping"]["min_delta"]),
            "stopped_early": bool(history[-1]["early_stop_triggered"]),
            "last_significant_epoch": early["last_significant_epoch"],
        },
        "initialisation": "CarDD YOLOv8s-seg v2 backbone; full fine-tuning",
    })
    print(f"E11 root complete seed{seed}: accuracy={measured['accuracy']:.4f} "
          f"macro_f1={measured['macro_f1']:.4f}", flush=True)


def joint_probabilities(root_prob, part_prob):
    """Eight properly normalised class probabilities, ordered as EIGHT."""
    rp = np.asarray(root_prob, dtype=float)
    pp = np.asarray(part_prob, dtype=float)
    if rp.ndim != 2 or rp.shape[1] != 6:
        raise ValueError(f"E11 expects [N,6] root probabilities, got {rp.shape}")
    if pp.shape != (len(rp), 2):
        raise ValueError(f"E11 expects [N,2] Bumper/Door scores, got {pp.shape}")
    if not (np.isfinite(rp).all() and np.isfinite(pp).all()):
        raise ValueError("Non-finite E11 scores")
    if np.any(rp < 0) or np.any(pp < 0):
        raise ValueError("Negative E11 probabilities")
    if not (np.allclose(rp.sum(1), 1, atol=1e-5)
            and np.allclose(pp.sum(1), 1, atol=1e-5)):
        raise ValueError("E11 root or expert probabilities not normalised")
    final = np.zeros((len(rp), len(EIGHT)), dtype=float)
    for name in EIGHT:
        i = EIGHT.index(name)
        if name in DIRECT:
            final[:, i] = rp[:, ROOT.index(name)]
        else:
            damage = root_label(name)
            part = "bumper" if name.startswith("bumper_") else "door"
            final[:, i] = rp[:, ROOT.index(damage)] * pp[:, BINARY.index(part)]
    if not np.allclose(final.sum(axis=1), 1, atol=1e-5):
        raise AssertionError("E11 eight-class probabilities must sum to one")
    return final


def hard_route(root_prob, part_prob):
    """Damage-first: the top root wins; only Dent/Scratch asks Bumper/Door."""
    top = np.asarray(root_prob).argmax(axis=1)
    part = np.asarray(part_prob).argmax(axis=1)
    predicted = []
    for i, root_index in enumerate(top):
        damage = ROOT[int(root_index)]
        if damage in DIRECT:
            label = damage
        else:
            label = f"{BINARY[int(part[i])]}_{damage}"
        predicted.append(EIGHT.index(label))
    return np.asarray(predicted, dtype=int)


def score_predictions(truth, predicted):
    ix = list(range(len(EIGHT)))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=ix,
                                 average="macro", zero_division=0)),
        "per_class": classification_report(
            truth, predicted, labels=ix, target_names=EIGHT,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=ix).tolist(),
    }


def error_analysis(y_true, root_pred, part_pred, hard_pred):
    root_truth = np.asarray([ROOT.index(root_label(EIGHT[int(y)]))
                             for y in y_true], dtype=int)
    root_wrong = root_pred != root_truth
    part_wrong = (~root_wrong) & (hard_pred != y_true)
    if int(root_wrong.sum() + part_wrong.sum()) != int((hard_pred != y_true).sum()):
        raise AssertionError("E11 hard cascade error decomposition is inconsistent")
    # Label-based oracle uses test ground truth solely for diagnosis, NOT inference.
    oracle = []
    for i, y in enumerate(y_true):
        actual_root = ROOT[root_truth[i]]
        if actual_root in DIRECT:
            oracle.append(EIGHT.index(actual_root))
        else:
            oracle.append(EIGHT.index(f"{BINARY[int(part_pred[i])]}_{actual_root}"))
    oracle = np.asarray(oracle, dtype=int)
    return {
        "root_wrong": int(root_wrong.sum()),
        "correct_root_wrong_expert": int(part_wrong.sum()),
        "hard_correct": int((hard_pred == y_true).sum()),
        "total": len(y_true),
        "root_accuracy": float((~root_wrong).mean()),
        "oracle_root_accuracy_diagnostic_only": float((oracle == y_true).mean()),
    }


def load_expert(cfg, seed, target_test):
    """Load same-seed E9-B weights strictly; do NOT retrain this expert."""
    root = expert_path(cfg, seed)
    if not root.exists():
        raise FileNotFoundError(
            f"E11 needs original E9-B seed{seed} best.pt at {root}. "
            "Old Drive exports excluded *.pt; locate Artemis checkpoints.")
    src = Path(cfg["expert"]["source_checkpoint"])
    if not src.is_file() or file_hash(src) != cfg["expert"]["source_sha256"]:
        raise ValueError("Missing or SHA256-mismatched Majorburn source for E9-B model")
    metrics_path = root.parent / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"E9-B completion marker missing: {metrics_path}")
    meta = json.loads(metrics_path.read_text())
    if meta.get("task") != "binary" or int(meta.get("seed", -1)) != seed:
        raise ValueError("E9-B metrics task/seed mismatch")
    if meta.get("classes") != BINARY:
        raise ValueError("E9-B binary class order mismatch")
    provenance_path = root.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing E9-B provenance: {provenance_path}")
    original = json.loads(provenance_path.read_text())
    if original.get("manifest_sha256") != file_hash(cfg["manifest"]):
        raise ValueError("E9-B was trained on a DIFFERENT manifest; refuse cascade")
    if original.get("pretrained_sha256") != cfg["expert"]["source_sha256"]:
        raise ValueError("E9-B pretrained source hash mismatch")
    # Verify E9-B really used the same 83 Bumper/Door test samples, not
    # merely the same number of samples on a different split.
    e9_manifest = root.parent / "test_manifest.csv"
    if not e9_manifest.exists():
        raise FileNotFoundError(f"E9-B test manifest missing: {e9_manifest}")
    e9_test = pd.read_csv(e9_manifest)
    expected = target_test[target_test.label.isin(
        ("bumper_dent", "bumper_scratch", "door_dent", "door_scratch"))]
    if set(e9_test.id) != set(expected.id) or len(e9_test) != len(expected):
        raise ValueError("E9-B test IDs differ from E11's fixed test manifest")
    ck = torch.load(root, map_location="cpu", weights_only=False)
    if ck.get("task") != "binary" or int(ck.get("seed", -1)) != seed:
        raise ValueError("E9-B best.pt task/seed mismatch")
    if ck.get("classes") != BINARY:
        raise ValueError("E9-B checkpoint class order mismatch")
    model = PartPretrainedClassifier(
        src, len(BINARY), int(cfg["expert"]["image_size"]),
        int(cfg["expert"]["expected_parts"]))
    model.load_state_dict(ck["model"], strict=True)
    model.requires_grad_(False)
    return model.to(cfg["device"]).eval(), file_hash(root)


def evaluate_cascade(cfg, seed):
    verify_train_config(cfg)
    if seed not in cfg["seeds"]:
        raise ValueError("E11 cascade seed not configured")
    out = eval_dir(cfg, seed)
    if out.exists():
        raise FileExistsError(f"E11 cascade output exists: {out}")
    root_dir = train_dir(cfg, seed)
    if not (root_dir / "metrics.json").exists():
        raise FileNotFoundError(f"E11 root not completed: {root_dir}")
    root_checkpoint = torch.load(
        root_dir / "best.pt", map_location="cpu", weights_only=False)
    if (root_checkpoint.get("classes") != list(ROOT)
            or root_checkpoint.get("seed") != seed
            or root_checkpoint.get("manifest_sha256") != file_hash(cfg["manifest"])
            or root_checkpoint.get("cardd_v2_sha256") != file_hash(cfg["checkpoint"])):
        raise ValueError("E11 root checkpoint provenance or classes mismatch")
    _, _, test = partitions(cfg)
    seed_all(seed)
    root_model = build_root(cfg)
    root_model.load_state_dict(root_checkpoint["model"], strict=True)
    root_model = root_model.to(cfg["device"]).eval()
    expert_model, expert_sha = load_expert(cfg, seed, test)
    root_scores, root_order, part_scores, part_order = [], [], [], []
    for images, _, indices in loader(RootImages(test, cfg), cfg):
        with torch.no_grad():
            root_scores.append(root_model(images.to(cfg["device"])).softmax(-1).cpu().numpy())
        root_order.extend(indices.tolist())
    for images, indices in loader(ExpertImages(test, cfg), cfg):
        with torch.no_grad():
            part_scores.append(expert_model(images.to(cfg["device"])).softmax(-1).cpu().numpy())
        part_order.extend(indices.tolist())
    if root_order != list(range(len(test))) or part_order != root_order:
        raise AssertionError("E11 Root/E9-B test sample order mismatch")
    rp = np.concatenate(root_scores)
    pp = np.concatenate(part_scores)
    truth = np.asarray([EIGHT.index(label) for label in test.label], dtype=int)
    root_pred = rp.argmax(axis=1)
    expert_pred = pp.argmax(axis=1)
    eight_prob = joint_probabilities(rp, pp)
    hard = hard_route(rp, pp)
    soft = eight_prob.argmax(axis=1)
    decomposition = error_analysis(truth, root_pred, expert_pred, hard)
    hard_metrics = score_predictions(truth, hard)
    soft_metrics = score_predictions(truth, soft)
    trace = test[["id", "path", "label"]].copy().reset_index(drop=True)
    trace["actual_root"] = [root_label(label) for label in test.label]
    trace["predicted_root"] = [ROOT[i] for i in root_pred]
    trace["predicted_part"] = [BINARY[i] for i in expert_pred]
    trace["root_confidence"] = rp.max(axis=1)
    trace["hard_prediction"] = [EIGHT[i] for i in hard]
    trace["soft_prediction"] = [EIGHT[i] for i in soft]
    trace["hard_correct"] = hard == truth
    trace["soft_correct"] = soft == truth
    trace["error_source"] = np.where(
        trace.actual_root.to_numpy() != trace.predicted_root.to_numpy(),
        "root_error", np.where(hard != truth, "expert_error", "correct"))
    for i, name in enumerate(ROOT):
        trace[f"p_{name}"] = rp[:, i]
    trace["p_bumper"] = pp[:, BINARY.index("bumper")]
    trace["p_door"] = pp[:, BINARY.index("door")]
    out.mkdir(parents=True, exist_ok=False)
    trace.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(
        out / "probabilities.npz", root=rp, bumper_door=pp,
        eight_class=eight_prob, target=truth)
    for name, report in (("hard", hard_metrics), ("soft", soft_metrics)):
        pd.DataFrame([{"class": label, **report["per_class"][label]}
                      for label in EIGHT]).to_csv(out / f"{name}_per_class.csv",
                                                  index=False)
        dump_json(out / f"{name}_confusion_matrix.json", {
            "classes": EIGHT, "matrix": report["confusion_matrix"]})
    root_truth = np.asarray(
        [ROOT.index(root_label(label)) for label in test.label], dtype=int)
    dump_json(out / "root_confusion_matrix.json", {
        "classes": list(ROOT), "matrix": confusion_matrix(
            root_truth, root_pred, labels=range(len(ROOT))).tolist()})
    dump_json(out / "metrics.json", {
        "task": "damage_first_eight_class", "seed": seed, "n_test": len(test),
        "eight_classes": EIGHT, "root_classes": list(ROOT),
        "root_accuracy": decomposition["root_accuracy"],
        "root_macro_f1": float(f1_score(
            root_truth, root_pred, labels=range(len(ROOT)),
            average="macro", zero_division=0)),
        "hard": hard_metrics, "soft": soft_metrics,
        "error_decomposition": decomposition,
        "source_checkpoint_sha256": {
            "cardd_root_best": file_hash(root_dir / "best.pt"),
            "e9_binary_best": expert_sha,
            "e9_source_model": cfg["expert"]["source_sha256"],
        },
        "warning": "Unknown is supervised, not unseen OOD. Test labels were "
                   "used only for metrics and diagnostic oracle, never adaptation. "
                   "Soft routing runs E9-B on all test images including out-of-domain "
                   "ones, but weights scores by Root dent/scratch probability.",
    })
    print(f"E11 seed{seed} eight-way: HARD acc={hard_metrics['accuracy']:.4f} "
          f"macroF1={hard_metrics['macro_f1']:.4f}; SOFT "
          f"acc={soft_metrics['accuracy']:.4f} macroF1={soft_metrics['macro_f1']:.4f}",
          flush=True)


def aggregate(cfg):
    output = Path(cfg["output"])
    records, categories = [], []
    for seed in cfg["seeds"]:
        first = train_dir(cfg, seed) / "metrics.json"
        second = eval_dir(cfg, seed) / "metrics.json"
        if not first.exists() or not second.exists():
            raise FileNotFoundError(f"E11 seed {seed} incomplete: {first} or {second}")
        root = json.loads(first.read_text())
        end = json.loads(second.read_text())
        if root.get("seed") != seed or end.get("seed") != seed:
            raise ValueError("E11 aggregation seed mismatch")
        for stage, result, classes in (
            ("root_six", root["test"], ROOT),
            ("cascade_hard", end["hard"], EIGHT),
            ("cascade_soft", end["soft"], EIGHT),
        ):
            records.append({
                "stage": stage, "seed": seed, "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "best_epoch": root["best_epoch"] if stage == "root_six" else None,
                "epochs_completed": root["epochs_completed"] if stage == "root_six" else None,
            })
            for label in classes:
                s = result["per_class"][label]
                categories.append({
                    "stage": stage, "seed": seed, "class": label,
                    "precision": s["precision"], "recall": s["recall"],
                    "f1": s["f1-score"], "support": s["support"],
                })
    df = pd.DataFrame(records)
    df.to_csv(output / "per_seed.csv", index=False)
    summary = df.groupby("stage")[["accuracy", "macro_f1",
                                    "best_epoch", "epochs_completed"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary["n_seeds"] = df.groupby("stage").seed.nunique()
    summary.to_csv(output / "mean_std.csv")
    rows = pd.DataFrame(categories)
    rows.to_csv(output / "per_class_per_seed.csv", index=False)
    per_class = rows.groupby(["stage", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    per_class.columns = ["_".join(col) for col in per_class.columns]
    per_class["support_per_seed"] = rows.groupby(["stage", "class"]).support.first()
    per_class.to_csv(output / "per_class_mean_std.csv")
    print(summary.to_string(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate", "aggregate"])
    parser.add_argument("--config", default="configs/e11.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.action == "aggregate":
        return aggregate(cfg)
    if args.seed is None:
        parser.error("train and evaluate require --seed")
    if args.action == "train":
        return train_root(cfg, args.seed, args.resume)
    return evaluate_cascade(cfg, args.seed)


if __name__ == "__main__":
    main()
