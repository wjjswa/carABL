"""E10: five-class root-guided car damage cascade (separate expert models).

Protocol: YOLO11 23-part pretrained backbone -> five-class root full fine-tuning
-> three independently full-fine-tuned two-class experts initialised from the
best *same-seed* root backbone -> hard/soft eight-class test-time routing.

The root model is NOT overwritten by expert training. Original immutable
group-disjoint carABL splits are preserved; test labels never tune a model.
'unknown' is a supervised catch-all, NOT unseen open-set rejection.
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import atomic_save, dump_json, file_hash, read_config, seed_all
from .data import open_image, read_manifest
from .e8 import PartPretrainedClassifier, padded_transforms, detect_overfitting, select_data, download
from .e9 import EIGHT, early_stopping_step

ROOT = ("bumper", "door", "lamp", "glass", "unknown")
EXPERTS = {
    "bumper": ("bumper_dent", "bumper_scratch"),
    "door": ("door_dent", "door_scratch"),
    "lamp": ("head_lamp", "tail_lamp"),
}
DIRECT = {"glass": "glass_shatter", "unknown": "unknown"}
ALL_LABELS = {v for labels in EXPERTS.values() for v in labels} | set(DIRECT.values())
if ALL_LABELS != set(EIGHT):
    raise RuntimeError("E10 labels must cover the original eight classes exactly")


def root_label(label: str) -> str:
    for expert, classes in EXPERTS.items():
        if label in classes:
            return expert
    for name, original in DIRECT.items():
        if label == original:
            return name
    raise ValueError(f"E10 unexpected original class: {label}")


def stage_names(stage: str) -> tuple[str, ...]:
    if stage == "root":
        return ROOT
    if stage in EXPERTS:
        return EXPERTS[stage]
    raise ValueError(f"Invalid E10 training stage: {stage}")


def stage_dir(cfg, stage: str, seed: int) -> Path:
    if stage == "root":
        return Path(cfg["output"]) / "root" / f"seed{seed}"
    if stage in EXPERTS:
        return Path(cfg["output"]) / "experts" / stage / f"seed{seed}"
    raise ValueError(stage)


def root_best_path(cfg, seed: int) -> Path:
    return stage_dir(cfg, "root", seed) / "best_root.pt"


def stage_best_path(cfg, stage: str, seed: int) -> Path:
    return root_best_path(cfg, seed) if stage == "root" else stage_dir(cfg, stage, seed) / "best.pt"


def partitions(cfg, stage="root"):
    # select_data validates the immutable manifest and all eight labels/splits.
    frames, _ = select_data(read_manifest(cfg["manifest"]), "8class")
    if stage == "root":
        return frames
    classes = set(stage_names(stage))
    return tuple(frame[frame.label.isin(classes)].copy().reset_index(drop=True)
                 for frame in frames)


class E10Images(Dataset):
    def __init__(self, frame, cfg, stage, train=False):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cfg["data_root"])
        self.stage = stage
        self.transform = padded_transforms(int(cfg["image_size"]), train)
        self.class_lookup = {name: index for index, name in enumerate(stage_names(stage))}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        item = self.frame.iloc[index]
        label = root_label(item.label) if self.stage == "root" else item.label
        return self.transform(open_image(self.root / item.path)), self.class_lookup[label], index


def loader(frame, cfg, stage, train=False, seed=0):
    return DataLoader(E10Images(frame, cfg, stage, train),
        batch_size=int(cfg["batch_size"]), shuffle=train,
        num_workers=int(cfg["workers"]), drop_last=False,
        pin_memory=str(cfg["device"]).startswith("cuda"),
        generator=torch.Generator().manual_seed(int(seed)))


def root_backbone_from_checkpoint(model: nn.Module, checkpoint: dict, seed: int):
    """Strictly transfer trained root encoder; new expert head is not copied."""
    if checkpoint.get("stage") != "root" or int(checkpoint.get("seed", -1)) != int(seed):
        raise ValueError("E10 expert root checkpoint has wrong stage or seed")
    state = {
        key[len("backbone."):]: value
        for key, value in checkpoint["model"].items()
        if key.startswith("backbone.")
    }
    if not state:
        raise ValueError("E10 root checkpoint has no backbone weights")
    model.backbone.load_state_dict(state, strict=True)


def build_model(cfg, stage, seed):
    source = Path(cfg["pretrained"]["checkpoint"])
    if not source.is_file() or file_hash(source) != cfg["pretrained"]["sha256"]:
        raise ValueError("Missing or SHA256-mismatched YOLO11 segmentation checkpoint")
    model = PartPretrainedClassifier(
        source, len(stage_names(stage)), int(cfg["image_size"]),
        int(cfg["pretrained"]["expected_classes"]))
    if stage != "root":
        path = root_best_path(cfg, seed)
        if not path.is_file():
            raise FileNotFoundError(f"Train E10 root seed{seed} first: {path}")
        root_state = torch.load(path, map_location="cpu", weights_only=False)
        root_backbone_from_checkpoint(model, root_state, seed)
    return model


@torch.no_grad()
def evaluate_classifier(model, batches, device, labels, with_predictions=False):
    model.eval()
    ys, pred, probs, loss_sum = [], [], [], 0.
    for x, y, _ in batches:
        logits = model(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        p = logits.softmax(-1).cpu().numpy()
        probs.append(p)
        ys.extend(y.tolist())
        pred.extend(p.argmax(-1).tolist())
        loss_sum += float(loss.item()) * len(y)
    if not ys:
        raise ValueError("Empty E10 evaluation split")
    truth = np.asarray(ys, dtype=np.int64)
    prediction = np.asarray(pred, dtype=np.int64)
    result = {
        "loss": loss_sum / len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(
            truth, prediction, labels=range(len(labels)), average="macro",
            zero_division=0)),
    }
    if with_predictions:
        result["per_class"] = classification_report(
            truth, prediction, labels=list(range(len(labels))), target_names=list(labels),
            output_dict=True, zero_division=0)
        result["confusion_matrix"] = confusion_matrix(
            truth, prediction, labels=list(range(len(labels)))).tolist()
        return result, truth, prediction, np.concatenate(probs, axis=0)
    return result


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unavailable"


def run_provenance(cfg, stage, seed):
    record = {
        "protocol": "E10-five-root-three-experts-v1",
        "stage": stage, "seed": int(seed), "config": cfg,
        "manifest_sha256": file_hash(cfg["manifest"]),
        "pretrained_sha256": file_hash(cfg["pretrained"]["checkpoint"]),
        "git_commit": _git_revision(),
    }
    if stage != "root":
        path = root_best_path(cfg, seed)
        record["root_best_sha256"] = file_hash(path)
    return record


def save_curves(out, history, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    for filename, train_col, val_col, title, ylabel in (
        ("loss_curve.png", "train_eval_loss", "val_loss", "Cross-entropy", "Loss"),
        ("accuracy_curve.png", "train_eval_accuracy", "val_accuracy", "Accuracy", "Accuracy"),
        ("f1_curve.png", "train_eval_macro_f1", "val_macro_f1", "Macro-F1", "Macro-F1"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(frame.epoch, frame[train_col], label="Training")
        ax.plot(frame.epoch, frame[val_col], label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"E10 {stage} · {title}")
        if ylabel != "Loss":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def train(cfg, stage, seed, resume=False):
    if int(seed) not in cfg["seeds"]:
        raise ValueError(f"E10 seed {seed} not in {cfg['seeds']}")
    if int(cfg["epochs"]) < 1:
        raise ValueError("E10 epochs must be positive")
    if stage != "root" and stage not in EXPERTS:
        raise ValueError(f"Unsupported E10 stage {stage}")
    if cfg["early_stopping"].get("monitor") != "val_macro_f1":
        raise ValueError("E10 early stopping must monitor validation Macro-F1")
    seed_all(int(seed))
    train_frame, val_frame, test_frame = partitions(cfg, stage)
    labels = stage_names(stage)
    for part in (train_frame, val_frame, test_frame):
        observed = {root_label(x) for x in part.label} if stage == "root" else set(part.label)
        if observed != set(labels):
            raise ValueError(f"E10 {stage} split missing classes {set(labels) - observed}")
    out = stage_dir(cfg, stage, seed)
    if out.exists() and not resume:
        raise FileExistsError(f"E10 output exists: {out}; use --resume or a new output root")
    out.mkdir(parents=True, exist_ok=True)
    expected = run_provenance(cfg, stage, seed)
    provenance_path = out / "provenance.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if old != expected:
            raise ValueError("E10 output provenance mismatch; do not resume with modified inputs")
    else:
        dump_json(provenance_path, expected)
    final = out / "metrics.json"
    if final.exists():
        print(f"E10 already completed: {out}", flush=True)
        return json.loads(final.read_text())
    for split, frame in zip(("train", "val", "test"),
                            (train_frame, val_frame, test_frame)):
        frame[["id", "path", "label", "group", "split"]].to_csv(
            out / f"{split}_manifest.csv", index=False)
    model = build_model(cfg, stage, seed).to(cfg["device"])
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(cfg["lr"])},
        {"params": model.head.parameters(), "lr": float(cfg["head_lr"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(cfg["epochs"]))
    last_path = out / "last.pt"
    best_path = stage_best_path(cfg, stage, seed)
    history, start, best_score, best_epoch, early_state = [], 0, -1.0, None, None
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("stage") != stage or saved.get("seed") != seed:
            raise ValueError("E10 resume checkpoint stage/seed mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, start = saved["history"], int(saved["epoch"])
        best_score, best_epoch = float(saved["best_score"]), int(saved["best_epoch"])
        early_state = saved.get("early_stopping_state")
        if start and early_state is None:
            raise ValueError("Resume checkpoint lacks E10 early stopping state")
        if early_state["bad_epochs"] >= cfg["early_stopping"]["patience"]:
            print("E10 early stop already triggered; finalising held-out evaluation", flush=True)
            start = int(cfg["epochs"])
    train_eval = loader(train_frame, cfg, stage)
    val_eval = loader(val_frame, cfg, stage)
    test_eval = loader(test_frame, cfg, stage)
    print(f"E10 {stage} seed{seed}: train={len(train_frame)}, val={len(val_frame)}, "
          f"test={len(test_frame)}, classes={labels}, max_epochs={cfg['epochs']}", flush=True)
    for epoch in range(start, int(cfg["epochs"])):
        seed_all(seed * 100000 + epoch + 1)
        batches = loader(train_frame, cfg, stage, train=True,
                         seed=seed * 1000 + epoch)
        model.train()
        total_loss, count = 0., 0
        for images, targets, _ in batches:
            if len(images) == 1:
                for mod in model.modules():
                    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                        mod.eval()
            images = images.to(cfg["device"])
            targets = targets.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"E10 non-finite loss: {stage}, epoch={epoch+1}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            count += len(targets)
        scheduler.step()
        tr = evaluate_classifier(model, train_eval, cfg["device"], labels)
        va = evaluate_classifier(model, val_eval, cfg["device"], labels)
        row = {
            "epoch": epoch + 1, "augmented_train_loss": total_loss / count,
            "train_eval_loss": tr["loss"], "val_loss": va["loss"],
            "train_eval_accuracy": tr["accuracy"], "val_accuracy": va["accuracy"],
            "train_eval_macro_f1": tr["macro_f1"], "val_macro_f1": va["macro_f1"],
            "accuracy_gap": tr["accuracy"] - va["accuracy"],
            "val_to_train_loss_ratio": va["loss"] / max(tr["loss"], 1e-8),
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        row["overfitting_warning"] = detect_overfitting(history, cfg)
        early_state, stop = early_stopping_step(
            va["macro_f1"], early_state, cfg, epoch + 1)
        row["early_stop_bad_epochs"] = early_state["bad_epochs"]
        row["early_stop_triggered"] = stop
        if va["macro_f1"] > best_score:
            best_score, best_epoch = va["macro_f1"], epoch + 1
            atomic_save({
                "stage": stage, "seed": seed,
                "model": model.state_dict(), "classes": list(labels),
                "epoch": best_epoch, "val_macro_f1": best_score,
                "manifest_sha256": expected["manifest_sha256"],
                "pretrained_sha256": expected["pretrained_sha256"],
                "root_best_sha256": expected.get("root_best_sha256"),
            }, best_path)
        atomic_save({
            "stage": stage, "seed": seed, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "history": history,
            "best_score": best_score, "best_epoch": best_epoch,
            "early_stopping_state": early_state,
        }, last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        flag = " OVERFITTING_WARNING" if row["overfitting_warning"] else ""
        print(f"E10 {stage} seed{seed} epoch {epoch+1}/{cfg['epochs']}: "
              f"train_acc={tr['accuracy']:.4f}, val_acc={va['accuracy']:.4f}, "
              f"val_macro_f1={va['macro_f1']:.4f}, bad_epochs={early_state['bad_epochs']}"
              f"{flag}", flush=True)
        if stop:
            print(f"E10 EARLY STOP {stage} seed{seed}: epoch={epoch+1}, "
                  f"best_epoch={best_epoch}, best_val_macro_f1={best_score:.4f}",
                  flush=True)
            break
    if not best_path.is_file():
        raise FileNotFoundError(f"E10 best checkpoint was not produced: {best_path}")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"], strict=True)
    test, truth, predicted, scores = evaluate_classifier(
        model, test_eval, cfg["device"], labels, with_predictions=True)
    items = test_frame[["id", "path", "label"]].copy().reset_index(drop=True)
    items["expected"] = [labels[i] for i in truth]
    items["predicted"] = [labels[i] for i in predicted]
    items["confidence"] = scores.max(1)
    items["correct"] = truth == predicted
    items.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame([
        {"class": cls, **test["per_class"][cls]} for cls in labels
    ]).to_csv(out / "per_class.csv", index=False)
    dump_json(out / "confusion_matrix.json", {
        "labels": list(labels), "matrix": test["confusion_matrix"]})
    save_curves(out, history, stage)
    warnings = [int(r["epoch"]) for r in history if r["overfitting_warning"]]
    stopped_early = bool(history[-1]["early_stop_triggered"])
    summary = {
        "stage": stage, "seed": seed,
        "classes": list(labels),
        "train_count": len(train_frame), "validation_count": len(val_frame),
        "test_count": len(test_frame),
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "early_stopping": {
            "monitor": cfg["early_stopping"]["monitor"],
            "patience": int(cfg["early_stopping"]["patience"]),
            "min_delta": float(cfg["early_stopping"]["min_delta"]),
            "stopped_early": stopped_early,
            "stop_epoch": len(history) if stopped_early else None,
            "last_significant_epoch": early_state["last_significant_epoch"],
            "reason": "patience_exhausted" if stopped_early else "max_epochs",
        },
        "overfitting_warning_count": len(warnings),
        "overfitting_warning_epochs": warnings,
        "test": test,
        "initialisation": (
            "Majorburn YOLO11 original part-segmentation backbone"
            if stage == "root" else
            f"E10 same-seed root best backbone, randomly initialised {stage} head"
        ),
        "root_best_sha256": expected.get("root_best_sha256"),
    }
    # Write completion marker last: interrupted/cancelled training is not complete.
    dump_json(final, summary)
    print(f"E10 completed {stage} seed{seed}: test accuracy={test['accuracy']:.4f}, "
          f"test macroF1={test['macro_f1']:.4f}", flush=True)
    return summary


def build_joint_probabilities(root_prob: np.ndarray, expert_prob: dict) -> np.ndarray:
    """Fully normalised eight-class probabilities for soft routing."""
    rp = np.asarray(root_prob, dtype=float)
    if rp.ndim != 2 or rp.shape[1] != len(ROOT):
        raise ValueError(f"Expected root probs with [N,5], got {rp.shape}")
    if not np.all(np.isfinite(rp)) or np.any(rp < -1e-7):
        raise ValueError("Invalid root probabilities")
    result = np.zeros((len(rp), len(EIGHT)), dtype=float)
    for expert, labels in EXPERTS.items():
        pp = np.asarray(expert_prob[expert], dtype=float)
        if pp.shape != (len(rp), len(labels)):
            raise ValueError(f"Expert {expert} wrong probability shape: {pp.shape}")
        for i, label in enumerate(labels):
            result[:, EIGHT.index(label)] = rp[:, ROOT.index(expert)] * pp[:, i]
    for root_name, label in DIRECT.items():
        result[:, EIGHT.index(label)] = rp[:, ROOT.index(root_name)]
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("E10 soft probabilities are not normalised")
    return result


def hard_route(root_prob: np.ndarray, expert_prob: dict) -> np.ndarray:
    """One winning root, one matching expert; direct glass/unknown outputs."""
    root_choice = np.asarray(root_prob).argmax(axis=1)
    predictions = np.empty(len(root_choice), dtype=int)
    for index, parent in enumerate(root_choice):
        root_name = ROOT[int(parent)]
        if root_name in DIRECT:
            label = DIRECT[root_name]
        else:
            subtype = int(np.asarray(expert_prob[root_name])[index].argmax())
            label = EXPERTS[root_name][subtype]
        predictions[index] = EIGHT.index(label)
    return predictions


def prediction_metrics(truth, pred):
    labels = list(EIGHT)
    indices = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(
            truth, pred, labels=indices, average="macro", zero_division=0)),
        "per_class": classification_report(
            truth, pred, labels=indices, target_names=labels,
            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, pred, labels=indices).tolist(),
    }


def error_decomposition(y_true, root_pred, hard_pred):
    true_parent = np.asarray([ROOT.index(root_label(EIGHT[int(y)])) for y in y_true])
    wrong_root = root_pred != true_parent
    wrong_expert = (~wrong_root) & (hard_pred != y_true)
    if int(wrong_root.sum() + wrong_expert.sum()) != int((hard_pred != y_true).sum()):
        raise AssertionError("E10 hard routing errors must be fully decomposable")
    return {
        "total": len(y_true),
        "root_wrong_count": int(wrong_root.sum()),
        "correct_root_wrong_expert_count": int(wrong_expert.sum()),
        "hard_correct_count": int((hard_pred == y_true).sum()),
        "root_accuracy": float((~wrong_root).mean()),
    }


def load_best_model(cfg, stage, seed):
    path = stage_best_path(cfg, stage, seed)
    if not path.is_file():
        raise FileNotFoundError(f"E10 requires completed {stage} seed{seed}: {path}")
    marker = stage_dir(cfg, stage, seed) / "metrics.json"
    if not marker.exists():
        raise FileNotFoundError(f"E10 {stage} seed{seed} has no completion marker: {marker}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != stage or checkpoint.get("seed") != seed:
        raise ValueError("E10 selected checkpoint stage/seed mismatch")
    model = build_model(cfg, stage, seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(cfg["device"]).eval()


def evaluate_cascade(cfg, seed):
    if seed not in cfg["seeds"]:
        raise ValueError("Unsupported E10 seed")
    out = Path(cfg["output"]) / "cascade" / f"seed{seed}"
    if out.exists():
        raise FileExistsError(f"E10 evaluation output exists: {out}")
    # Prevent partial or mismatched model families being evaluated together.
    root_metrics = json.loads((stage_dir(cfg, "root", seed) / "metrics.json").read_text())
    expert_metrics = {
        name: json.loads((stage_dir(cfg, name, seed) / "metrics.json").read_text())
        for name in EXPERTS
    }
    root_best_sha = file_hash(root_best_path(cfg, seed))
    for name, metrics in expert_metrics.items():
        if metrics.get("root_best_sha256") != root_best_sha:
            raise ValueError(f"E10 {name} seed{seed} not trained from selected root best")
    if root_metrics["seed"] != seed or any(m["seed"] != seed for m in expert_metrics.values()):
        raise ValueError("E10 cascade mixes seeds")
    _, _, test = partitions(cfg, "root")
    root_model = load_best_model(cfg, "root", seed)
    expert_models = {name: load_best_model(cfg, name, seed) for name in EXPERTS}
    ys, roots, expert_rows = [], [], {name: [] for name in EXPERTS}
    for x, _, indices in loader(test, cfg, "root"):
        x = x.to(cfg["device"])
        with torch.no_grad():
            roots.append(root_model(x).softmax(-1).cpu().numpy())
            for name, model in expert_models.items():
                expert_rows[name].append(model(x).softmax(-1).cpu().numpy())
        ys.extend(indices.tolist())
    # Iterate in the original test manifest order for clean per-image traceability.
    if ys != list(range(len(test))):
        raise AssertionError("E10 test evaluation order must match the immutable manifest")
    rp = np.concatenate(roots, axis=0)
    ep = {name: np.concatenate(chunks, axis=0) for name, chunks in expert_rows.items()}
    y_true = np.asarray([EIGHT.index(label) for label in test.label], dtype=int)
    parent_pred = rp.argmax(axis=1)
    joint = build_joint_probabilities(rp, ep)
    soft_pred = joint.argmax(axis=1)
    hard_pred = hard_route(rp, ep)
    decomposition = error_decomposition(y_true, parent_pred, hard_pred)
    hard = prediction_metrics(y_true, hard_pred)
    soft = prediction_metrics(y_true, soft_pred)
    parent_true = np.asarray([ROOT.index(root_label(label)) for label in test.label])
    trace = test[["id", "path", "label"]].copy().reset_index(drop=True)
    trace["true_root"] = [ROOT[i] for i in parent_true]
    trace["predicted_root"] = [ROOT[i] for i in parent_pred]
    trace["root_confidence"] = rp.max(axis=1)
    trace["hard_prediction"] = [EIGHT[i] for i in hard_pred]
    trace["soft_prediction"] = [EIGHT[i] for i in soft_pred]
    trace["hard_correct"] = hard_pred == y_true
    trace["soft_correct"] = soft_pred == y_true
    trace["hard_joint_confidence"] = joint[np.arange(len(test)), hard_pred]
    trace["soft_joint_confidence"] = joint.max(axis=1)
    trace["error_source"] = np.where(
        parent_pred != parent_true, "root_error",
        np.where(hard_pred != y_true, "expert_error", "correct"))
    for index, name in enumerate(ROOT):
        trace[f"p_root_{name}"] = rp[:, index]
    trace["root_margin"] = np.sort(rp, axis=1)[:, -1] - np.sort(rp, axis=1)[:, -2]
    out.mkdir(parents=True, exist_ok=False)
    trace.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "probabilities.npz", root=rp, soft=joint, true=y_true)
    for mode, report in (("hard", hard), ("soft", soft)):
        pd.DataFrame([
            {"class": label, **report["per_class"][label]} for label in EIGHT
        ]).to_csv(out / f"{mode}_per_class.csv", index=False)
        dump_json(out / f"{mode}_confusion_matrix.json", {
            "labels": list(EIGHT), "matrix": report["confusion_matrix"]})
    dump_json(out / "root_confusion_matrix.json", {
        "labels": list(ROOT), "matrix": confusion_matrix(
            parent_true, parent_pred, labels=list(range(len(ROOT)))).tolist()})
    dump_json(out / "metrics.json", {
        "seed": seed, "n_test": len(test),
        "classes": list(EIGHT), "root_classes": list(ROOT),
        "root_accuracy": decomposition["root_accuracy"],
        "root_macro_f1": float(f1_score(
            parent_true, parent_pred, labels=list(range(len(ROOT))),
            average="macro", zero_division=0)),
        "root_confusion_matrix": confusion_matrix(
            parent_true, parent_pred, labels=list(range(len(ROOT)))).tolist(),
        "hard": hard, "soft": soft,
        "error_decomposition": decomposition,
        "checkpoint_sha256": {
            "root": root_best_sha,
            **{name: file_hash(stage_best_path(cfg, name, seed))
               for name in EXPERTS},
        },
        "warning": (
            "Root labels come from image-level damage labels, not annotated damaged-part "
            "locations. Unknown is supervised, not unseen OOD. Hard routing selects one "
            "expert; soft routing scores all three, so their inference costs differ."
        ),
    })
    print(f"E10 cascade seed{seed}: hard acc={hard['accuracy']:.4f}, "
          f"macroF1={hard['macro_f1']:.4f}; soft acc={soft['accuracy']:.4f}, "
          f"macroF1={soft['macro_f1']:.4f}", flush=True)


def aggregate(cfg):
    root = Path(cfg["output"])
    overall, per_class = [], []
    for seed in cfg["seeds"]:
        for stage in ("root", *EXPERTS):
            path = stage_dir(cfg, stage, seed) / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing completed E10 run: {path}")
            m = json.loads(path.read_text())
            if (m.get("seed") != seed or m.get("stage") != stage
                    or m.get("early_stopping", {}).get("patience")
                    != cfg["early_stopping"]["patience"]):
                raise ValueError(f"E10 protocol mismatch: {path}")
            overall.append({
                "task": f"stage-{stage}", "seed": seed,
                "accuracy": m["test"]["accuracy"],
                "macro_f1": m["test"]["macro_f1"],
                "best_epoch": m["best_epoch"],
                "epochs_completed": m["epochs_completed"],
            })
            for label in m["classes"]:
                scores = m["test"]["per_class"][label]
                per_class.append({
                    "task": f"stage-{stage}", "seed": seed,
                    "class": label, "precision": scores["precision"],
                    "recall": scores["recall"], "f1": scores["f1-score"],
                    "support": scores["support"],
                })
        path = root / "cascade" / f"seed{seed}" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing E10 cascade evaluation: {path}")
        m = json.loads(path.read_text())
        for mode in ("hard", "soft"):
            overall.append({
                "task": f"cascade-{mode}", "seed": seed,
                "accuracy": m[mode]["accuracy"], "macro_f1": m[mode]["macro_f1"],
                "best_epoch": None, "epochs_completed": None,
            })
            for label in EIGHT:
                scores = m[mode]["per_class"][label]
                per_class.append({
                    "task": f"cascade-{mode}", "seed": seed,
                    "class": label, "precision": scores["precision"],
                    "recall": scores["recall"], "f1": scores["f1-score"],
                    "support": scores["support"],
                })
    table = pd.DataFrame(overall)
    table.to_csv(root / "per_seed.csv", index=False)
    summary = table.groupby("task")[["accuracy", "macro_f1",
                                      "best_epoch", "epochs_completed"]].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary["n_seeds"] = table.groupby("task").seed.nunique()
    summary.to_csv(root / "mean_std.csv")
    p = pd.DataFrame(per_class)
    p.to_csv(root / "per_class_per_seed.csv", index=False)
    class_summary = p.groupby(["task", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    class_summary.columns = ["_".join(col) for col in class_summary.columns]
    class_summary["support_per_seed"] = p.groupby(["task", "class"]).support.first()
    class_summary.to_csv(root / "per_class_mean_std.csv")
    print(summary.to_string(), flush=True)
    print(f"E10 aggregated results written to {root}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[
        "download", "train-root", "train-expert", "evaluate", "aggregate"])
    parser.add_argument("--config", default="configs/e10.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expert", choices=tuple(EXPERTS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.action == "download":
        return download(cfg)
    if args.action == "aggregate":
        return aggregate(cfg)
    if args.seed is None:
        parser.error(f"{args.action} requires --seed")
    if args.action == "train-root":
        return train(cfg, "root", args.seed, args.resume)
    if args.action == "train-expert":
        if not args.expert:
            parser.error("train-expert requires --expert")
        return train(cfg, args.expert, args.seed, args.resume)
    if args.action == "evaluate":
        return evaluate_cascade(cfg, args.seed)


if __name__ == "__main__":
    main()
