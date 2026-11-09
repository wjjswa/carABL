"""E13/E14: E11-compatible damage-first root ablations and foundation encoders.

E13-B/C: CarDD v2 YOLOv8s backbone at 384px with aspect-preserving padding.
E14-A: frozen DINOv3-S/16 + learned linear six-way head (cached features).
E14-B/C: DINOv3-S/16 or B/16, last four transformer blocks fine-tuned.
E14-D: SigLIP2 Base vision encoder, last four blocks fine-tuned.

No re-split, no test-time tuning, same-seed frozen E9-B expert for eight-class
hard/soft routing. Validation macro-F1 selects checkpoints. Test used only
after model selection; historical 226-image benchmark is exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T

from .common import atomic_save, dump_json, file_hash, read_config, seed_all
from .data import open_image
from .e9 import early_stopping_step
from .e11 import (
    ROOT, DIRECT, EIGHT, BINARY, ExpertImages, build_root, error_analysis,
    hard_route, joint_probabilities, load_expert, partitions, root_label,
    score_predictions,
)

VARIANTS = ("e13b", "e13c", "e14a", "e14b", "e14c", "e14d")


def variant_config(cfg, variant):
    if variant not in VARIANTS or variant not in cfg["variants"]:
        raise ValueError(f"Unknown experiment variant {variant!r}")
    setting = cfg["variants"][variant]
    if setting["kind"] not in ("cardd", "hf"):
        raise ValueError("Expected 'cardd' or 'hf' encoder kind")
    if variant == "e13b" and setting.get("augmentation") != "pad":
        raise ValueError("E13-B must use pad augmentation")
    if variant == "e13c" and setting.get("augmentation") != "mild":
        raise ValueError("E13-C must use mild augmentation")
    if variant.startswith("e14") and not setting.get("pretrained_dir"):
        raise ValueError("E14 requires a local Hugging Face checkpoint directory")
    return setting


def root_dir(cfg, variant, seed):
    return Path(cfg["output"]) / variant / "root" / f"seed{int(seed)}"


def cascade_dir(cfg, variant, seed):
    return Path(cfg["output"]) / variant / "cascade" / f"seed{int(seed)}"


def _padded(image, size, fill=(114, 114, 114)):
    return ImageOps.pad(image.convert("RGB"), (size, size),
                        method=Image.Resampling.BILINEAR,
                        color=fill, centering=(0.5, 0.5))


class PadSquare:
    """Pickle-safe resizing callable for multiprocess DataLoader workers."""
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, image):
        return _padded(image, self.size)


def cardd_transform(size, train=False, augmentation="pad"):
    """Never crops away a damage region; E13-C adds gentle geometric jitter."""
    ops = [PadSquare(size)]
    if train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
        if augmentation == "mild":
            ops.extend([
                T.RandomAffine(degrees=4, translate=(0.04, 0.04),
                               scale=(0.96, 1.04), fill=(114, 114, 114)),
                T.RandomApply([T.ColorJitter(0.10, 0.10, 0.10, 0.02)], p=0.4),
            ])
    ops.append(T.ToTensor())  # match CarDD/E11: no ImageNet normalization
    return T.Compose(ops)


class RootDataset(Dataset):
    def __init__(self, frame, cfg, variant, train=False):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(cfg["data_root"])
        self.variant = variant
        setting = variant_config(cfg, variant)
        self.kind = setting["kind"]
        self.train = bool(train)
        if self.kind == "cardd":
            self.transform = cardd_transform(
                int(setting["image_size"]), train,
                setting["augmentation"])
        else:
            from transformers import AutoImageProcessor
            location = str(setting["pretrained_dir"])
            self.processor = AutoImageProcessor.from_pretrained(
                location, local_files_only=True)
            # Frozen E14-A uses exactly the same deterministic preprocessing
            # for feature caching and validation.
            self.augment = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.35),
            ]) if train else None
        self.labels = {name: index for index, name in enumerate(ROOT)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        sample = self.frame.iloc[index]
        image = open_image(self.root / sample.path).convert("RGB")
        if self.kind == "cardd":
            values = self.transform(image)
        else:
            if self.augment is not None:
                image = self.augment(image)
            processed = self.processor(images=image, return_tensors="pt")
            values = {key: tensor.squeeze(0) for key, tensor in processed.items()
                      if isinstance(tensor, torch.Tensor)}
            if "pixel_values" not in values:
                raise RuntimeError("Hugging Face processor did not return pixel_values")
        return values, self.labels[root_label(sample.label)], index


def batches(dataset, cfg, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=int(cfg["batch_size"]),
                      shuffle=shuffle, num_workers=int(cfg["workers"]),
                      generator=torch.Generator().manual_seed(int(seed)),
                      pin_memory=str(cfg["device"]).startswith("cuda"))


def _move(images, device):
    if isinstance(images, dict):
        return {key: value.to(device) for key, value in images.items()}
    return images.to(device)


def _vision_layers_and_norm(module):
    """Resolve actual Transformers DINOv3 and SigLIP2 vision layouts."""
    if getattr(module.config, "model_type", None) == "dinov3_vit":
        return module.layer, module.norm
    if getattr(module.config, "model_type", None) == "siglip_vision_model":
        vision = module.vision_model
        return vision.encoder.layers, vision.post_layernorm
    raise TypeError(f"Unsupported HF vision layout: {type(module).__name__}")


def _transformer_blocks(module):
    return _vision_layers_and_norm(module)[0]


class HFRoot(nn.Module):
    def __init__(self, setting):
        super().__init__()
        from transformers import AutoConfig, AutoModel, SiglipVisionModel
        family = setting["family"]
        location = str(setting["pretrained_dir"])
        source_config = AutoConfig.from_pretrained(location, local_files_only=True)
        if family == "dinov3":
            if source_config.model_type != "dinov3_vit":
                raise ValueError("DINOv3 variant requires a DINOv3 ViT snapshot")
            self.encoder, loading = AutoModel.from_pretrained(
                location, local_files_only=True, output_loading_info=True)
        elif family == "siglip2":
            # The fixed-resolution SigLIP2 checkpoint is published as 'siglip'.
            # Siglip2VisionModel is the NaFlex architecture and is incompatible.
            if source_config.model_type not in ("siglip", "siglip_vision_model"):
                raise ValueError("E14-D requires the fixed-resolution SigLIP2 SiglipVisionModel snapshot")
            self.encoder, loading = SiglipVisionModel.from_pretrained(
                location, local_files_only=True, output_loading_info=True)
        else:
            raise ValueError(f"Unsupported HF family: {family}")
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Pretrained vision weights did not load completely: {loading}")
        dimension = int(self.encoder.config.hidden_size)
        self.head = nn.Linear(dimension, len(ROOT))
        self.mode = setting["mode"]
        self.encoder.requires_grad_(False)
        if self.mode == "partial":
            blocks, norm = _vision_layers_and_norm(self.encoder)
            n = int(setting["unfreeze_last_blocks"])
            if not 0 < n <= len(blocks):
                raise ValueError(f"Bad unfreeze_last_blocks={n}, total={len(blocks)}")
            for block in list(blocks)[-n:]:
                block.requires_grad_(True)
            # Keep final normalization trainable for feature-space adaptation.
            norm.requires_grad_(True)
        elif self.mode != "frozen":
            raise ValueError("HF modes must be 'frozen' or 'partial'")
        if not any(p.requires_grad for p in self.head.parameters()):
            raise RuntimeError("Classification head must be trainable")

    def embedding(self, inputs):
        output = self.encoder(**inputs) if isinstance(inputs, dict) else self.encoder(pixel_values=inputs)
        representation = getattr(output, "pooler_output", None)
        if representation is None:
            representation = output.last_hidden_state[:, 0]
        if representation.ndim != 2:
            raise ValueError("Expected pooled feature matrix [batch, channels]")
        return representation

    def forward(self, inputs):
        return self.head(self.embedding(inputs))

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "frozen":
            self.encoder.eval()
        return self


def build_model(cfg, variant):
    setting = variant_config(cfg, variant)
    if setting["kind"] == "cardd":
        config = dict(cfg)
        config["image_size"] = setting["image_size"]
        return build_root(config)
    return HFRoot(setting)


def _source_paths(cfg, variant):
    setting = variant_config(cfg, variant)
    if setting["kind"] == "cardd":
        return [Path(cfg["checkpoint"])]
    path = Path(setting["pretrained_dir"])
    files = [path / "config.json", path / "preprocessor_config.json"]
    files += sorted(path.glob("*.safetensors"))
    if len(files) <= 2:
        raise FileNotFoundError(f"No safetensors in local pretrained snapshot: {path}")
    return files


def source_signature(cfg, variant):
    h = hashlib.sha256()
    for p in _source_paths(cfg, variant):
        if not p.is_file():
            raise FileNotFoundError(f"Missing pretrained input {p}")
        h.update(p.name.encode("utf-8"))
        h.update(file_hash(p).encode("ascii"))
    return h.hexdigest()


def verify_protocol(cfg, variant=None, require_sources=True):
    if list(cfg["seeds"]) != [0, 1, 2] or len(cfg["seeds"]) != 3:
        raise ValueError("Seeds must be exactly 0,1,2")
    if int(cfg["epochs"]) != 300:
        raise ValueError("Max epochs must be 300")
    es = cfg["early_stopping"]
    if int(es["patience"]) != 30 or es["monitor"] != "val_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro-F1, patience30")
    if int(cfg["batch_size"]) < 2:
        raise ValueError("Batch size must be >=2")
    if variant is not None:
        variant_config(cfg, variant)
    if require_sources:
        for key in ("manifest", "data_root"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {cfg[key]}")
        for v in ([variant] if variant is not None else VARIANTS):
            source_signature(cfg, v)


def _count(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


@torch.no_grad()
def frozen_features(model, dataset, cfg):
    model.eval()
    vectors, labels, indices = [], [], []
    for images, y, ids in batches(dataset, cfg):
        vectors.append(model.embedding(_move(images, cfg["device"])).cpu())
        labels.append(y)
        indices.append(ids)
    if not vectors:
        raise ValueError("No images for feature cache")
    features = torch.cat(vectors)
    labels = torch.cat(labels)
    indices = torch.cat(indices)
    if not torch.equal(indices, torch.arange(len(dataset))):
        raise AssertionError("Frozen feature extraction changed row order")
    return TensorDataset(features, labels, indices)


@torch.no_grad()
def measure(model, dataset_or_batches, cfg, detailed=False):
    model.eval()
    truth, predictions, probabilities, running_loss = [], [], [], 0.0
    for images, y, _ in dataset_or_batches:
        logits = model(_move(images, cfg["device"]))
        y_cuda = y.to(cfg["device"])
        running_loss += float(F.cross_entropy(logits, y_cuda).item()) * len(y)
        pred = logits.softmax(-1).cpu().numpy()
        probabilities.append(pred)
        predictions.extend(pred.argmax(axis=1).tolist())
        truth.extend(y.tolist())
    if not truth:
        raise ValueError("Cannot measure an empty split")
    labels = np.asarray(truth, dtype=int)
    guesses = np.asarray(predictions, dtype=int)
    result = {
        "accuracy": float(accuracy_score(labels, guesses)),
        "macro_f1": float(f1_score(labels, guesses,
                                 labels=list(range(len(ROOT))),
                                 average="macro", zero_division=0)),
        "loss": running_loss / len(labels),
    }
    if detailed:
        result["per_class"] = classification_report(
            labels, guesses, labels=list(range(len(ROOT))),
            target_names=list(ROOT), output_dict=True, zero_division=0)
        result["confusion_matrix"] = confusion_matrix(
            labels, guesses, labels=list(range(len(ROOT)))).tolist()
        return result, labels, guesses, np.concatenate(probabilities)
    return result


class HeadOnly(nn.Module):
    """Train a linear probe on one-time cached deterministic frozen features."""
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        return self.head(features)


def _trainable_state(model, kind):
    if kind == "cardd":
        return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, "full"
    params = dict(model.named_parameters())
    return {k: p.detach().cpu().clone() for k, p in params.items()
            if p.requires_grad}, "trainable"


def _load_trained_state(model, checkpoint):
    if checkpoint["state_kind"] == "full":
        model.load_state_dict(checkpoint["state"], strict=True)
        return
    params = dict(model.named_parameters())
    expected = {k for k, p in params.items() if p.requires_grad}
    state = checkpoint["state"]
    if set(state) != expected:
        raise ValueError("Trainable parameter names changed in HF checkpoint")
    with torch.no_grad():
        for name, weight in state.items():
            if params[name].shape != weight.shape:
                raise ValueError(f"Shape mismatch {name}")
            params[name].copy_(weight)


def _provenance(cfg, variant, seed, source_sha):
    return {
        "protocol": "E13E14-damage-first-v1",
        "variant": variant, "seed": int(seed),
        "manifest_sha256": file_hash(cfg["manifest"]),
        "source_sha256": source_sha,
        "settings": variant_config(cfg, variant),
        "image_size": variant_config(cfg, variant)["image_size"],
        "config": cfg,
    }


def fixed_partitions(cfg):
    """Require E11's exact original 1050/229/226 train/val/test protocol."""
    frames = partitions(cfg)
    sizes = tuple(len(frame) for frame in frames)
    if sizes != (1050, 229, 226):
        raise ValueError(f"Wrong split sizes {sizes}; expected (1050, 229, 226)")
    for frame in frames:
        if frame.id.duplicated().any():
            raise ValueError("Duplicate image ID inside an E11 split")
    if len(set().union(*(set(frame.id) for frame in frames))) != sum(sizes):
        raise ValueError("E11 Train/Val/Test image IDs overlap")
    return frames


def save_curves(out, history, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    for filename, train_column, val_column, ylabel in (
        ("loss_curve.png", "train_eval_loss", "val_loss", "Cross-entropy"),
        ("accuracy_curve.png", "train_eval_accuracy", "val_accuracy", "Accuracy"),
        ("f1_curve.png", "train_eval_macro_f1", "val_macro_f1", "Macro-F1"),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(frame.epoch, frame[train_column], label="Train")
        axis.plot(frame.epoch, frame[val_column], label="Validation")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=f"{variant.upper()} {ylabel}")
        if ylabel != "Cross-entropy":
            axis.set_ylim(0, 1)
        axis.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def train_one(cfg, variant, seed, resume=False):
    verify_protocol(cfg, variant)
    if seed not in cfg["seeds"]:
        raise ValueError(f"Unknown seed {seed}")
    seed_all(seed)
    source_sha = source_signature(cfg, variant)
    fingerprint = _provenance(cfg, variant, seed, source_sha)
    train_frame, val_frame, test_frame = fixed_partitions(cfg)
    assert all(set(ROOT) == set(map(root_label, frame.label))
               for frame in (train_frame, val_frame, test_frame))
    out = root_dir(cfg, variant, seed)
    if out.exists() and not resume:
        raise FileExistsError(f"Existing run {out}; use --resume")
    out.mkdir(parents=True, exist_ok=True)
    prior = out / "provenance.json"
    if prior.exists():
        if json.loads(prior.read_text()) != fingerprint:
            raise ValueError("Resume would mix different data/model/configuration")
    else:
        dump_json(prior, fingerprint)
    if (out / "metrics.json").exists():
        print(f"COMPLETED {variant} seed{seed}", flush=True)
        return
    for split, frame in (("train", train_frame), ("val", val_frame),
                         ("test", test_frame)):
        frame[["id", "path", "label", "group", "split"]].to_csv(
            out / f"{split}_manifest.csv", index=False)
    model = build_model(cfg, variant).to(cfg["device"])
    setting = variant_config(cfg, variant)
    frozen = setting["kind"] == "hf" and setting["mode"] == "frozen"
    if frozen:
        train_data = frozen_features(
            model, RootDataset(train_frame, cfg, variant), cfg)
        val_data = frozen_features(
            model, RootDataset(val_frame, cfg, variant), cfg)
        active_model = HeadOnly(model.head)
        train_eval = batches(train_data, cfg)
        val_eval = batches(val_data, cfg)
    else:
        active_model = model
        train_data = RootDataset(train_frame, cfg, variant, train=True)
        train_eval = batches(RootDataset(train_frame, cfg, variant), cfg)
        val_eval = batches(RootDataset(val_frame, cfg, variant), cfg)
    trainable_backbone = [p for p in model.encoder.parameters() if p.requires_grad] if setting["kind"] == "hf" else list(model.features.parameters())
    groups = [{"params": model.head.parameters(), "lr": float(cfg["head_lr"])}]
    if trainable_backbone:
        groups.insert(0, {"params": trainable_backbone,
                          "lr": float(setting.get("backbone_lr", cfg["lr"]))})
    optimiser = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, int(cfg["epochs"]))
    history, start, best_score, best_epoch, bad_epochs = [], 0, -1.0, None, 0
    early_state = None
    last = out / "last.pt"
    if resume and last.is_file():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["seed"] != seed or checkpoint["variant"] != variant:
            raise ValueError("Resume checkpoint seed/variant mismatch")
        _load_trained_state(model, checkpoint)
        optimiser.load_state_dict(checkpoint["optimizer"])
        schedule.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        start = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_score"])
        best_epoch = checkpoint["best_epoch"]
        bad_epochs = int(checkpoint["bad_epochs"])
        early_state = checkpoint.get("early_stopping_state")
        if early_state is None:
            # Recover the E11 patience rule from older E13/E14 checkpoints.
            for row in history:
                early_state, _ = early_stopping_step(
                    row["val_macro_f1"], early_state, cfg, row["epoch"])
            bad_epochs = early_state["bad_epochs"]
        if bad_epochs >= int(cfg["early_stopping"]["patience"]):
            start = int(cfg["epochs"])  # already early-stopped; only finalize metrics
    for epoch in range(start, int(cfg["epochs"])):
        seed_all(seed * 100000 + epoch + 1)
        if frozen:
            iterator = batches(train_data, cfg, shuffle=True,
                               seed=seed * 1000 + epoch)
        else:
            iterator = batches(train_data, cfg, shuffle=True,
                               seed=seed * 1000 + epoch)
        active_model.train()
        total_loss, total_count = 0.0, 0
        for images, y, _ in iterator:
            # BatchNorm for the CarDD YOLO backbone cannot train on a batch of 1.
            if not frozen and len(y) == 1:
                for layer in model.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
            optimiser.zero_grad(set_to_none=True)
            logits = active_model(_move(images, cfg["device"]))
            loss = F.cross_entropy(logits, y.to(cfg["device"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(y)
            total_count += len(y)
        schedule.step()
        training = measure(active_model, train_eval, cfg)
        validation = measure(active_model, val_eval, cfg)
        score = validation["macro_f1"]
        early_state, stop = early_stopping_step(score, early_state, cfg, epoch + 1)
        bad_epochs = early_state["bad_epochs"]
        row = {
            "epoch": epoch + 1,
            "augmented_train_loss": total_loss / total_count,
            "train_eval_loss": training["loss"],
            "val_loss": validation["loss"],
            "train_eval_accuracy": training["accuracy"],
            "val_accuracy": validation["accuracy"],
            "train_eval_macro_f1": training["macro_f1"],
            "val_macro_f1": score,
            "bad_epochs": bad_epochs,
            "accuracy_gap": training["accuracy"] - validation["accuracy"],
        }
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            state, kind = _trainable_state(model, setting["kind"])
            atomic_save({
                "state": state, "state_kind": kind,
                "seed": seed, "variant": variant, "epoch": best_epoch,
                "classes": list(ROOT),
                "manifest_sha256": fingerprint["manifest_sha256"],
                "source_sha256": source_sha,
            }, out / "best.pt")
        state, kind = _trainable_state(model, setting["kind"])
        atomic_save({
            "state": state, "state_kind": kind,
            "optimizer": optimiser.state_dict(),
            "scheduler": schedule.state_dict(),
            "seed": seed, "variant": variant, "epoch": epoch + 1,
            "history": history, "best_score": best_score,
            "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "early_stopping_state": early_state,
        }, last)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"{variant} seed{seed} epoch {epoch+1}/{cfg['epochs']} "
              f"train_acc={training['accuracy']:.4f} "
              f"val_acc={validation['accuracy']:.4f} "
              f"val_macro_f1={score:.4f} bad={bad_epochs}", flush=True)
        if stop:
            print(f"{variant} seed{seed} early stop; best epoch={best_epoch}",
                  flush=True)
            break
    if best_epoch is None:
        raise RuntimeError("Training finished without a best checkpoint")
    save_curves(out, history, variant)
    dump_json(out / "metrics.json", {
        "variant": variant, "seed": seed,
        "task": "six_class_root_training",
        "train_count": len(train_frame), "val_count": len(val_frame),
        "test_count": len(test_frame),
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_validation_macro_f1": best_score,
        "selection": "validation macro-F1",
        "parameter_counts": _count(model),
        "test_evaluated_during_training": False,
        "early_stopping_state": early_state,
    })


@torch.no_grad()
def evaluate_one(cfg, variant, seed):
    verify_protocol(cfg, variant)
    out = cascade_dir(cfg, variant, seed)
    if out.exists():
        if (out / "metrics.json").exists():
            print(f"COMPLETED {variant} cascade seed{seed}", flush=True)
            return
        raise FileExistsError(f"Incomplete E13/E14 evaluation {out}")
    checkpoint_path = root_dir(cfg, variant, seed) / "best.pt"
    if not (root_dir(cfg, variant, seed) / "metrics.json").exists():
        raise FileNotFoundError(f"Root is not finished: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_sha = source_signature(cfg, variant)
    if (checkpoint["seed"] != seed or checkpoint["variant"] != variant
            or checkpoint["classes"] != list(ROOT)
            or checkpoint["manifest_sha256"] != file_hash(cfg["manifest"])
            or checkpoint["source_sha256"] != expected_sha):
        raise ValueError("Trained root checkpoint does not match input manifest/model")
    seed_all(seed)
    _, _, test = fixed_partitions(cfg)
    root_model = build_model(cfg, variant).to(cfg["device"]).eval()
    _load_trained_state(root_model, checkpoint)
    measured, truth_root, predicted_root, root_probs = measure(
        root_model, batches(RootDataset(test, cfg, variant), cfg),
        cfg, detailed=True)
    expert, expert_hash = load_expert(cfg, seed, test)
    expert_probs, order = [], []
    for images, idx in batches(ExpertImages(test, cfg), cfg):
        expert_probs.append(expert(images.to(cfg["device"])).softmax(-1).cpu().numpy())
        order.extend(idx.tolist())
    if order != list(range(len(test))):
        raise AssertionError("Expert prediction IDs were shuffled")
    part_probs = np.concatenate(expert_probs)
    final_probs = joint_probabilities(root_probs, part_probs)
    hard = hard_route(root_probs, part_probs)
    soft = final_probs.argmax(axis=1)
    true_eight = np.asarray([EIGHT.index(label) for label in test.label])
    hard_report = score_predictions(true_eight, hard)
    soft_report = score_predictions(true_eight, soft)
    diagnostic = error_analysis(true_eight, predicted_root,
                                part_probs.argmax(axis=1), hard)
    out.mkdir(parents=True, exist_ok=False)
    trace = test[["id", "path", "label"]].copy().reset_index(drop=True)
    trace["variant"] = variant
    trace["seed"] = seed
    trace["actual_root"] = [root_label(n) for n in test.label]
    trace["predicted_root"] = [ROOT[i] for i in predicted_root]
    trace["root_confidence"] = root_probs.max(axis=1)
    trace["predicted_part"] = [BINARY[i] for i in part_probs.argmax(axis=1)]
    trace["hard_prediction"] = [EIGHT[i] for i in hard]
    trace["soft_prediction"] = [EIGHT[i] for i in soft]
    trace["hard_correct"] = hard == true_eight
    trace["soft_correct"] = soft == true_eight
    trace["error_source"] = np.where(
        trace.actual_root != trace.predicted_root, "root_error",
        np.where(hard != true_eight, "expert_error", "correct"))
    for i, name in enumerate(ROOT):
        trace[f"p_{name}"] = root_probs[:, i]
    trace["p_bumper"] = part_probs[:, BINARY.index("bumper")]
    trace["p_door"] = part_probs[:, BINARY.index("door")]
    trace.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "probabilities.npz",
                        root=root_probs, bumper_door=part_probs,
                        eight_class=final_probs, target=true_eight)
    for title, report in (("root", measured), ("hard", hard_report),
                          ("soft", soft_report)):
        classes = ROOT if title == "root" else EIGHT
        pd.DataFrame([{"class": k, **report["per_class"][k]} for k in classes]
                     ).to_csv(out / f"{title}_per_class.csv", index=False)
        dump_json(out / f"{title}_confusion_matrix.json", {
            "classes": list(classes), "matrix": report["confusion_matrix"]})
    dump_json(out / "metrics.json", {
        "variant": variant, "seed": seed, "test_count": len(test),
        "root": measured, "hard": hard_report, "soft": soft_report,
        "error_decomposition": diagnostic,
        "source_sha256": expected_sha,
        "root_checkpoint_sha256": file_hash(checkpoint_path),
        "e9_binary_checkpoint_sha256": expert_hash,
        "note": "Existing 226-image benchmark was used in earlier model development. "
                "Treat these scores as exploratory; use independent test data "
                "for unbiased final generalisation claims.",
    })
    print(f"{variant} seed{seed}: root={measured['accuracy']:.4f} "
          f"hard={hard_report['accuracy']:.4f} "
          f"soft={soft_report['accuracy']:.4f}", flush=True)


def aggregate(cfg, selected=VARIANTS, reference_e11=None):
    output = Path(cfg["output"])
    records, categories, comparisons = [], [], []
    for variant in selected:
        for seed in cfg["seeds"]:
            filename = cascade_dir(cfg, variant, seed) / "metrics.json"
            if not filename.is_file():
                raise FileNotFoundError(f"Missing completed cascade metrics: {filename}")
            report = json.loads(filename.read_text())
            if report["variant"] != variant or report["seed"] != seed:
                raise ValueError(f"Wrong variant/seed in {filename}")
            for stage, classes in (("root", ROOT), ("hard", EIGHT),
                                   ("soft", EIGHT)):
                stage_metrics = report[stage]
                records.append({
                    "variant": variant, "seed": seed, "stage": stage,
                    "accuracy": stage_metrics["accuracy"],
                    "macro_f1": stage_metrics["macro_f1"],
                    "correct_n": int(round(stage_metrics["accuracy"] *
                                           stage_metrics["per_class"]["macro avg"]["support"])),
                })
                for label in classes:
                    s = stage_metrics["per_class"][label]
                    categories.append({
                        "variant": variant, "seed": seed,
                        "stage": stage, "class": label,
                        "precision": s["precision"], "recall": s["recall"],
                        "f1": s["f1-score"], "support": s["support"],
                    })
            if reference_e11 is not None:
                ref = Path(reference_e11) / "cascade" / f"seed{seed}" / "test_predictions.csv"
                new = cascade_dir(cfg, variant, seed) / "test_predictions.csv"
                if not ref.exists():
                    raise FileNotFoundError(f"Missing E11 reference predictions: {ref}")
                left = pd.read_csv(ref)
                right = pd.read_csv(new)
                if (left.id.duplicated().any() or right.id.duplicated().any()
                        or set(left.id) != set(right.id)):
                    raise ValueError("E11 and new model have different test image IDs")
                x = left.set_index("id").sort_index()
                y = right.set_index("id").sort_index()
                if not x.label.equals(y.label):
                    raise ValueError("E11 and new model use different test labels")
                for mode in ("hard", "soft"):
                    was = x[f"{mode}_prediction"] == x.label
                    now = y[f"{mode}_prediction"] == y.label
                    comparisons.append({
                        "variant": variant, "seed": seed, "routing": mode,
                        "both_correct": int((was & now).sum()),
                        "fixed_from_e11": int((~was & now).sum()),
                        "new_errors_vs_e11": int((was & ~now).sum()),
                        "both_wrong": int((~was & ~now).sum()),
                        "net_correct_gain": int(now.sum() - was.sum()),
                        "n_images": len(x),
                    })
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(output / "per_seed.csv", index=False)
    summary = table.groupby(["variant", "stage"])[
        ["accuracy", "macro_f1", "correct_n"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_seeds"] = table.groupby(["variant", "stage"]).seed.nunique()
    summary.to_csv(output / "mean_std.csv")
    cls = pd.DataFrame(categories)
    cls.to_csv(output / "per_class_per_seed.csv", index=False)
    class_summary = cls.groupby(["variant", "stage", "class"])[
        ["precision", "recall", "f1"]].agg(["mean", "std"])
    class_summary.columns = ["_".join(c) for c in class_summary.columns]
    class_summary["support_per_seed"] = cls.groupby(
        ["variant", "stage", "class"]).support.first()
    class_summary.to_csv(output / "per_class_mean_std.csv")
    if comparisons:
        pd.DataFrame(comparisons).to_csv(
            output / "paired_vs_e11.csv", index=False)
    print(summary.to_string(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate", "aggregate"])
    parser.add_argument("--config", default="configs/e13e14.yaml")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-e11", default=None)
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.action == "aggregate":
        verify_protocol(cfg, require_sources=False)
        return aggregate(cfg, selected=args.variants or VARIANTS,
                         reference_e11=args.reference_e11 or cfg.get("reference_e11"))
    if args.variant is None or args.seed is None:
        parser.error("train/evaluate require --variant and --seed")
    if args.action == "train":
        return train_one(cfg, args.variant, args.seed, args.resume)
    return evaluate_one(cfg, args.variant, args.seed)


if __name__ == "__main__":
    main()
