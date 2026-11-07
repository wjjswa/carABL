"""E9: zero-shot car-part inference, binary part fine-tuning, hierarchical damage model.

Uses only the fixed group-disjoint carABL manifest. E9-A is an untrained
part-segmentation inference heuristic, not damage localisation. E9-B (bumper/
door) and E9-C (root part -> conditional damage subtype) fine-tune all YOLO11
backbone weights for 300 epochs. Test labels are read for evaluation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from .common import read_config, seed_all, file_hash, dump_json, atomic_save
from .data import read_manifest, open_image
from .e8 import (PartPretrainedClassifier, padded_transforms, detect_overfitting,
                 _save_curves, class_names, select_data, download)

EIGHT = class_names("8class")
BINARY = ["bumper", "door"]
PARENTS = ["bumper", "door", "other"]
DAMAGE = ["dent", "scratch"]
OTHER = ["glass_shatter", "head_lamp", "tail_lamp", "unknown"]
TASKS = ("binary", "hierarchical")
BUMPER_SEG = {"back_bumper", "front_bumper"}
DOOR_SEG = {"back_door", "back_left_door", "back_right_door",
            "front_door", "front_left_door", "front_right_door"}


def part_label(label: str) -> str:
    if label.startswith("bumper_"):
        return "bumper"
    if label.startswith("door_"):
        return "door"
    return "other"


def datasets(manifest, task):
    if task not in TASKS:
        raise ValueError(f"Invalid E9 training task: {task}")
    # E8's guard validates all eight original labels and group-disjoint splits.
    partitions, _ = select_data(manifest, "8class")
    if task == "binary":
        parts = tuple(p[p.label.map(part_label).isin(BINARY)].copy().reset_index(drop=True)
                      for p in partitions)
        if any(set(p.label.map(part_label)) != set(BINARY) for p in parts):
            raise ValueError("E9 binary requires bumper and door data in every split")
        return parts, BINARY
    return partitions, EIGHT


class E9Images(Dataset):
    def __init__(self, frame, cfg, task, train):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cfg["data_root"])
        self.task = task
        self.transform = padded_transforms(int(cfg["image_size"]), train)
        self.lookup = {v: i for i, v in enumerate(BINARY if task == "binary" else EIGHT)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        label = part_label(row.label) if self.task == "binary" else row.label
        return self.transform(open_image(self.root / row.path)), self.lookup[label], i


def make_loader(frame, cfg, task, train=False, seed=0):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(E9Images(frame, cfg, task, train),
                      batch_size=int(cfg["batch_size"]), shuffle=train,
                      num_workers=int(cfg["workers"]), generator=generator,
                      pin_memory=str(cfg["device"]).startswith("cuda"),
                      drop_last=False)


class HierarchicalClassifier(nn.Module):
    """Shared fully-trainable part-pretrained encoder; root and conditional heads."""
    def __init__(self, checkpoint, image_size, expected_parts):
        super().__init__()
        base = PartPretrainedClassifier(checkpoint, 8, image_size, expected_parts)
        self.backbone = base.backbone
        dim = base.features_dim
        self.root_head = nn.Linear(dim, len(PARENTS))
        # A shared dent/scratch branch for Bumper and Door.
        self.damage_head = nn.Linear(dim, len(DAMAGE))
        self.other_head = nn.Linear(dim, len(OTHER))

    def forward(self, x):
        z = F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)
        return self.root_head(z), self.damage_head(z), self.other_head(z)

    @staticmethod
    def joint_log_probs(logits):
        root, damage, other = [z.log_softmax(dim=-1) for z in logits]
        return torch.stack([
            root[:, 0] + damage[:, 0],  # bumper_dent
            root[:, 0] + damage[:, 1],  # bumper_scratch
            root[:, 1] + damage[:, 0],  # door_dent
            root[:, 1] + damage[:, 1],  # door_scratch
            root[:, 2] + other[:, 0],   # glass_shatter
            root[:, 2] + other[:, 1],   # head_lamp
            root[:, 2] + other[:, 2],   # tail_lamp
            root[:, 2] + other[:, 3],   # unknown
        ], dim=-1)


def logits_and_logp(model, x, task):
    if task == "binary":
        logits = model(x)
        return logits.log_softmax(dim=-1), None
    logits = model(x)
    return model.joint_log_probs(logits), logits


@torch.no_grad()
def evaluate(model, batches, task, device, details=False):
    model.eval()
    y_all, pred_all, score_all, loss_sum, n = [], [], [], 0., 0
    root_predictions, damage_predictions, other_predictions = [], [], []
    for images, labels, _ in batches:
        labels = labels.to(device)
        logp, logits = logits_and_logp(model, images.to(device), task)
        loss = F.nll_loss(logp, labels)
        prob = logp.exp()
        y_all.extend(labels.cpu().tolist())
        pred_all.extend(prob.argmax(-1).cpu().tolist())
        if details:
            score_all.append(prob.cpu().numpy())
            if task == "hierarchical":
                root_predictions.extend(logits[0].argmax(-1).cpu().tolist())
                damage_predictions.extend(logits[1].argmax(-1).cpu().tolist())
                other_predictions.extend(logits[2].argmax(-1).cpu().tolist())
        loss_sum += float(loss) * len(labels)
        n += len(labels)
    if not n:
        raise ValueError("E9 evaluation split is empty")
    names = BINARY if task == "binary" else EIGHT
    true, predicted = np.array(y_all), np.array(pred_all)
    result = {
        "loss": loss_sum / n,
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=list(range(len(names))),
                                    average="macro", zero_division=0)),
    }
    if not details:
        return result
    result["per_class"] = classification_report(true, predicted,
        labels=list(range(len(names))), target_names=names,
        output_dict=True, zero_division=0)
    result["confusion_matrix"] = confusion_matrix(true, predicted,
        labels=list(range(len(names)))).tolist()
    if task == "hierarchical":
        true_root = np.array([PARENTS.index(part_label(EIGHT[i])) for i in true])
        root_pred = np.array(root_predictions)
        result["root_accuracy"] = float(accuracy_score(true_root, root_pred))
        result["root_confusion_matrix"] = confusion_matrix(true_root, root_pred,
                                                           labels=[0, 1, 2]).tolist()
        # Comparable with E9-A/B: original four bumper/door-damage test labels.
        bd = true_root != 2
        result["root_bumper_door_accuracy"] = float(accuracy_score(true_root[bd], root_pred[bd]))
        # An incorrect routing to 'other' counts as an error in this 3-column matrix.
        result["root_bumper_door_confusion_with_other"] = confusion_matrix(
            true_root[bd], root_pred[bd], labels=[0, 1, 2]).tolist()
        # Oracle-parent conditional head scores are *diagnostics*, not deployed 8-class scores.
        masks = {
            "bumper": true_root == 0,
            "door": true_root == 1,
            "other": true_root == 2,
        }
        for part in ("bumper", "door"):
            mask = masks[part]
            y_damage = np.array([DAMAGE.index(EIGHT[i].split("_")[-1])
                                  for i in true[mask]])
            result[f"{part}_damage_accuracy_given_true_parent"] = float(
                accuracy_score(y_damage, np.array(damage_predictions)[mask]))
            result[f"{part}_damage_confusion_given_true_parent"] = confusion_matrix(
                y_damage, np.array(damage_predictions)[mask], labels=[0, 1]).tolist()
        mask = masks["other"]
        y_other = np.array([OTHER.index(EIGHT[i]) for i in true[mask]])
        result["other_subtype_accuracy_given_true_parent"] = float(
            accuracy_score(y_other, np.array(other_predictions)[mask]))
    return result, true, predicted, np.concatenate(score_all, axis=0)


def model_from_pretrained(cfg, task):
    p = cfg["pretrained"]
    checkpoint = Path(p["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}; run E9 download first")
    actual = file_hash(checkpoint)
    if actual != p["sha256"]:
        raise ValueError(f"Pretrained weight SHA256 mismatch: {actual}")
    if task == "binary":
        return PartPretrainedClassifier(checkpoint, 2, int(cfg["image_size"]),
                                        int(p["expected_classes"]))
    return HierarchicalClassifier(checkpoint, int(cfg["image_size"]),
                                  int(p["expected_classes"]))


def early_stopping_step(score, state, cfg, epoch):
    """Track validation Macro-F1 patience separately from exact-max best.pt."""
    rule = cfg["early_stopping"]
    patience = int(rule["patience"])
    min_delta = float(rule.get("min_delta", 0.0))
    if patience < 1 or min_delta < 0:
        raise ValueError("Invalid early stopping patience or min_delta")
    if state is None:
        state = {"reference_score": None, "bad_epochs": 0, "last_significant_epoch": None}
    reference = state["reference_score"]
    if reference is None or float(score) > reference + min_delta:
        state = {"reference_score": float(score), "bad_epochs": 0,
                 "last_significant_epoch": int(epoch)}
    else:
        state = {**state, "bad_epochs": int(state["bad_epochs"]) + 1}
    return state, state["bad_epochs"] >= patience


def source_provenance(cfg, task, seed):
    return {
        "config": cfg, "task": task, "seed": int(seed),
        "manifest_sha256": file_hash(cfg["manifest"]),
        "pretrained_sha256": file_hash(cfg["pretrained"]["checkpoint"]),
        "pretrained_repo": cfg["pretrained"]["repository"],
        "pretrained_revision": cfg["pretrained"]["revision"],
    }


def train(cfg, task, seed, resume=False):
    if task not in TASKS or seed not in cfg["seeds"]:
        raise ValueError("Unsupported E9 task/seed")
    if int(cfg["epochs"]) != 300:
        raise ValueError("E9 protocol requires 300 epochs")
    seed_all(seed)
    split, names = datasets(read_manifest(cfg["manifest"]), task)
    train_frame, val_frame, test_frame = split
    output = Path(cfg["output"]) / task / f"seed{seed}"
    if output.exists() and not resume:
        raise FileExistsError(f"Existing E9 output {output}; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    meta = source_provenance(cfg, task, seed)
    if (output / "provenance.json").exists():
        if json.loads((output / "provenance.json").read_text()) != meta:
            raise ValueError("E9 provenance/config mismatch; use a new output root")
    else:
        dump_json(output / "provenance.json", meta)
    if (output / "metrics.json").exists():
        print(f"Already finished: {output}", flush=True)
        return
    for partition, frame in zip(("train", "val", "test"), split):
        frame[["id", "path", "label", "group", "split"]].to_csv(
            output / f"{partition}_manifest.csv", index=False)
    model = model_from_pretrained(cfg, task).to(cfg["device"])
    head_modules = ([model.head] if task == "binary"
                    else [model.root_head, model.damage_head, model.other_head])
    optim = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(cfg["lr"])},
        {"params": [p for module in head_modules for p in module.parameters()],
         "lr": float(cfg["head_lr"])},
    ], weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(cfg["epochs"]))
    last, best = output / "last.pt", output / "best.pt"
    history, start, best_score, best_epoch = [], 0, -1., None
    early_state = None
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if (state["task"], state["seed"]) != (task, seed):
            raise ValueError("E9 resume checkpoint incompatible with task and seed")
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, start = state["history"], int(state["epoch"])
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        early_state = state.get("early_stopping_state")
        if start and early_state is None:
            raise ValueError("Checkpoint lacks early-stopping state; use fresh outputs")
        if early_state["bad_epochs"] >= cfg["early_stopping"]["patience"]:
            print("E9 early stopping was already reached; finalizing test only.", flush=True)
            start = int(cfg["epochs"])
    train_eval = make_loader(train_frame, cfg, task)
    val_eval = make_loader(val_frame, cfg, task)
    test_eval = make_loader(test_frame, cfg, task)
    for epoch in range(start, int(cfg["epochs"])):
        # Deterministic epoch-level reseeding supports resume without RNG leakage.
        seed_all(seed * 100000 + epoch + 1)
        batches = make_loader(train_frame, cfg, task, train=True, seed=seed*1000+epoch)
        model.train()
        total, count = 0., 0
        for x, y, _ in batches:
            if len(x) == 1:
                for m in model.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optim.zero_grad(set_to_none=True)
            logp, _ = logits_and_logp(model, x, task)
            loss = F.nll_loss(logp, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"E9 nonfinite loss, epoch {epoch+1}")
            loss.backward()
            optim.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        scheduler.step()
        t = evaluate(model, train_eval, task, cfg["device"])
        v = evaluate(model, val_eval, task, cfg["device"])
        row = {
            "epoch": epoch + 1, "augmented_train_loss": total/count,
            "train_eval_loss": t["loss"], "val_loss": v["loss"],
            "train_eval_accuracy": t["accuracy"], "val_accuracy": v["accuracy"],
            "train_eval_macro_f1": t["macro_f1"], "val_macro_f1": v["macro_f1"],
            "accuracy_gap": t["accuracy"] - v["accuracy"],
            "val_to_train_loss_ratio": v["loss"] / max(t["loss"], 1e-8),
            "lr_backbone": optim.param_groups[0]["lr"],
            "lr_head": optim.param_groups[1]["lr"],
        }
        history.append(row)
        row["overfitting_warning"] = detect_overfitting(history, cfg)
        early_state, should_stop = early_stopping_step(
            v["macro_f1"], early_state, cfg, epoch+1)
        row["early_stop_bad_epochs"] = early_state["bad_epochs"]
        row["early_stop_triggered"] = should_stop
        if v["macro_f1"] > best_score:
            best_score, best_epoch = v["macro_f1"], epoch+1
            atomic_save({"model": model.state_dict(), "task": task, "seed": seed,
                         "epoch": best_epoch, "classes": names}, best)
        atomic_save({"model": model.state_dict(), "optimizer": optim.state_dict(),
                     "scheduler": scheduler.state_dict(), "history": history,
                     "epoch": epoch+1, "best_score": best_score,
                     "best_epoch": best_epoch, "task": task, "seed": seed,
                     "early_stopping_state": early_state}, last)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        alert = " OVERFITTING_WARNING" if row["overfitting_warning"] else ""
        print(f"E9 {task} seed{seed} epoch {epoch+1}/300 "
              f"train_loss={t['loss']:.4f} val_loss={v['loss']:.4f} "
              f"train_acc={t['accuracy']:.4f} val_f1={v['macro_f1']:.4f}{alert}",
              flush=True)
        if should_stop:
            print(f"E9 EARLY STOP {task} seed{seed} after epoch {epoch+1}; "
                  f"patience={cfg['early_stopping']['patience']}, "
                  f"best_epoch={best_epoch}, best_macro_F1={best_score:.4f}", flush=True)
            break
    model.load_state_dict(torch.load(best, map_location="cpu", weights_only=False)["model"])
    result, truth, pred, scores = evaluate(model, test_eval, task, cfg["device"], details=True)
    photos = test_frame[["id", "path", "label"]].copy().reset_index(drop=True)
    photos["expected"] = [names[i] for i in truth]
    photos["predicted"] = [names[i] for i in pred]
    photos["confidence"] = scores.max(axis=1)
    photos["correct"] = truth == pred
    photos.to_csv(output / "test_predictions.csv", index=False)
    np.savez_compressed(output / "test_scores.npz", scores=scores, truth=truth)
    pd.DataFrame([{"label": label, **result["per_class"][label]}
                  for label in names]).to_csv(output / "per_class.csv", index=False)
    dump_json(output / "confusion_matrix.json",
              {"labels": names, "matrix": result["confusion_matrix"]})
    if task == "hierarchical":
        dump_json(output / "root_confusion_matrix.json",
                  {"labels": PARENTS, "matrix": result["root_confusion_matrix"]})
    _save_curves(output, history)
    warnings = [h["epoch"] for h in history if h["overfitting_warning"]]
    dump_json(output / "overfitting_summary.json",
              {"warning_epochs": warnings, "best_epoch": best_epoch,
               "epochs_completed": len(history), "early_stopping": True,
                "patience": cfg["early_stopping"]["patience"],
                "stopped_early": len(history) < cfg["epochs"]})
    # Completion marker is written last.
    dump_json(output / "metrics.json", {
        "task": task, "seed": seed, "classes": names, "epochs_completed": len(history),
        "early_stopping": {
            "enabled": True, "patience": int(cfg["early_stopping"]["patience"]),
            "min_delta": float(cfg["early_stopping"].get("min_delta", 0.0)),
            "stopped_early": len(history) < int(cfg["epochs"]),
            "stop_epoch": len(history) if len(history) < int(cfg["epochs"]) else None,
            "bad_epochs": early_state["bad_epochs"],
            "last_significant_epoch": early_state["last_significant_epoch"],
            "reason": "patience_exhausted" if len(history) < int(cfg["epochs"]) else "max_epochs",
        },
        "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
        "train_count": len(train_frame), "validation_count": len(val_frame),
        "test_count": len(test_frame), "test": result, "overfitting_warning_epochs": warnings,
        "initialisation": "Majorburn pretrained YOLO11 part segmentation backbone, full fine-tuning",
    })
    print(f"E9 completed: {output}", flush=True)


def zero_shot_decision(result, names):
    """Pick visible part with maximum (mask area × instance confidence).

    Image-level heuristic, NOT localisation of the damage. Return None if
    no bumper/door is segmented. This fixed decision rule uses no test labels.
    """
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        return None, [], 0.
    names = {int(k): str(v).lower() for k, v in dict(names).items()}
    candidates = []
    for cls, conf, mask in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                               masks.data):
        name = names.get(int(cls), "")
        group = ("bumper" if name in BUMPER_SEG
                 else "door" if name in DOOR_SEG else None)
        if group is None:
            continue
        area = float((mask > 0.5).sum().item())
        score = area * float(conf)
        if area > 0 and score > 0:
            candidates.append({"part": group, "source_class": name,
                               "mask_area": area, "confidence": float(conf),
                               "score": score})
    if not candidates:
        return None, [], 0.
    candidates.sort(key=lambda x: x["score"], reverse=True)
    by_part = {p: max((x["score"] for x in candidates if x["part"] == p),
                       default=0.) for p in BINARY}
    best = max(by_part, key=by_part.get)
    if by_part["bumper"] == by_part["door"]:
        return None, candidates, 0.
    return best, candidates, float(by_part[best])


def zero_shot(cfg):
    """Use pretrained segmentation model directly on Bumper/Door test samples."""
    from ultralytics import YOLO
    split, _ = datasets(read_manifest(cfg["manifest"]), "binary")
    test = split[2]
    out = Path(cfg["output"]) / "zero_shot"
    if out.exists():
        raise FileExistsError(f"Zero-shot results already exist: {out}")
    ckpt = Path(cfg["pretrained"]["checkpoint"])
    if file_hash(ckpt) != cfg["pretrained"]["sha256"]:
        raise ValueError("Pretrained checkpoint SHA256 mismatch")
    out.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(ckpt))
    if model.task != "segment" or len(model.names) != 23:
        raise ValueError("Expected 23-class Carparts-Seg segmentation model")
    rows = []
    device = 0 if str(cfg["device"]).startswith("cuda") else "cpu"
    for _, item in test.iterrows():
        image = Path(cfg["data_root"]) / item.path
        result = model.predict(source=str(image), imgsz=int(cfg["zero_shot"]["image_size"]),
                               conf=float(cfg["zero_shot"]["confidence"]),
                               device=device, verbose=False)[0]
        choice, detected, score = zero_shot_decision(result, model.names)
        rows.append({"id": item.id, "path": item.path, "original_label": item.label,
                     "true_part": part_label(item.label),
                     "predicted_part": choice if choice else "abstain",
                     "confidence_weighted_mask_area": score,
                     "both_part_types_detected": len({x["part"] for x in detected}) == 2,
                     "all_part_detections": json.dumps(detected)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "part_predictions.csv", index=False)
    identified = frame.predicted_part.isin(BINARY)
    truth = frame.true_part.to_numpy()
    prediction = frame.predicted_part.to_numpy()
    # Abstentions count as errors for strict accuracy; no cherry-picked subset.
    correct = truth == prediction
    result = {
        "n_test": int(len(frame)), "covered": int(identified.sum()),
        "abstained": int((~identified).sum()), "coverage": float(identified.mean()),
        "strict_accuracy": float(correct.mean()),
        "conditional_accuracy": float(correct[identified].mean()) if identified.any() else None,
        "bumper_recall_including_abstentions": float(correct[truth == "bumper"].mean()),
        "door_recall_including_abstentions": float(correct[truth == "door"].mean()),
        "confusion_labels": ["bumper", "door", "abstain"],
        "confusion_matrix": confusion_matrix(truth, prediction,
                        labels=["bumper", "door", "abstain"]).tolist(),
        "heuristic": "largest confidence-weighted visible bumper/door mask",
        "warning": "Part presence does not prove the damaged part; multiple parts may be visible",
        "checkpoint_sha256": file_hash(ckpt),
    }
    dump_json(out / "metrics.json", result)
    dump_json(out / "provenance.json", {
        "model": cfg["pretrained"], "manifest_sha256": file_hash(cfg["manifest"]),
        "zero_shot": cfg["zero_shot"],
    })
    print(json.dumps(result, indent=2), flush=True)


def aggregate(cfg):
    root = Path(cfg["output"])
    rows, per_cat = [], []
    for task in TASKS:
        for seed in cfg["seeds"]:
            output = root / task / f"seed{seed}"
            p = output / "metrics.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing finished E9 run: {p}")
            meta = json.loads((output / "provenance.json").read_text())
            if meta != source_provenance(cfg, task, seed):
                raise ValueError(f"E9 provenance mismatch: {output}")
            record = json.loads(p.read_text())
            if (record["task"] != task or record["seed"] != seed or
                not 1 <= record["epochs_completed"] <= cfg["epochs"] or
                record.get("early_stopping", {}).get("patience") != cfg["early_stopping"]["patience"]):
                raise ValueError(f"E9 run incomplete or incompatible: {output}")
            report = record["test"]
            rows.append({"task": task, "seed": seed, "test_accuracy": report["accuracy"],
                         "test_macro_f1": report["macro_f1"],
                         "best_epoch": record["best_epoch"],
                         "overfitting_warning_count": len(record["overfitting_warning_epochs"]),
                         "epochs_completed": record["epochs_completed"],
                         "stopped_early": record["early_stopping"]["stopped_early"],
                         "root_accuracy": report.get("root_accuracy"),
                         "root_bumper_door_accuracy": report.get("root_bumper_door_accuracy")})
            for label in record["classes"]:
                m = report["per_class"][label]
                per_cat.append({"task": task, "seed": seed, "class": label,
                                "precision": m["precision"], "recall": m["recall"],
                                "f1": m["f1-score"], "support": m["support"]})
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "trained_per_seed.csv", index=False)
    summary = frame.groupby("task")[["test_accuracy", "test_macro_f1", "root_accuracy",
                                     "root_bumper_door_accuracy", "epochs_completed",
                                     "best_epoch"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_seeds"] = frame.groupby("task").seed.nunique()
    summary.to_csv(root / "trained_mean_std.csv")
    categories = pd.DataFrame(per_cat)
    categories.to_csv(root / "per_class_per_seed.csv", index=False)
    per_class_summary = categories.groupby(["task", "class"])[["precision", "recall", "f1"]].agg(["mean", "std"])
    per_class_summary.columns = ["_".join(c) for c in per_class_summary.columns]
    per_class_summary.to_csv(root / "per_class_mean_std.csv")
    if (root / "zero_shot/metrics.json").exists():
        print("Zero-shot:", json.loads((root / "zero_shot/metrics.json").read_text()))
    print(summary.to_string(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "zero-shot", "train", "aggregate"])
    parser.add_argument("--config", default="configs/e9.yaml")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.action == "download":
        return download(cfg)
    if args.action == "zero-shot":
        return zero_shot(cfg)
    if args.action == "aggregate":
        return aggregate(cfg)
    if args.task is None or args.seed is None:
        parser.error("train requires --task and --seed")
    return train(cfg, args.task, args.seed, args.resume)


if __name__ == "__main__":
    main()
