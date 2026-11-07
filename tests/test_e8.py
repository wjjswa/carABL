"""CPU-only E8 protocol and transfer tests. No third-party downloads or GPU."""
import sys
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from torch import nn

from carrepair.common import read_config
from carrepair.e8 import class_names, select_data, detect_overfitting, early_stopping_step, PartPretrainedClassifier
from carrepair.e8_submit import jobs


def sample_manifest():
    rows = []
    labels = class_names("8class")
    for split in ("train", "val", "test"):
        for label in labels:
            for i in range(2):
                rows.append(dict(id=f"{split}-{label}-{i}", group=f"{split}-{label}-{i}",
                                 path=f"{split}/{label}/{i}.jpg", split=split, label=label))
    return pd.DataFrame(rows)


def test_seven_class_never_receives_unknown():
    train, val, test = select_data(sample_manifest(), "7class")[0]
    assert all("unknown" not in set(f.label) for f in (train, val, test))
    assert len(train) == 14 and len(val) == 14 and len(test) == 14


def test_eight_class_uses_all_images():
    parts, names = select_data(sample_manifest(), "8class")
    assert len(names) == 8
    assert all(len(f) == 16 for f in parts)
    assert "unknown" in names


def test_data_leak_rejected():
    manifest = sample_manifest()
    manifest.loc[manifest.split == "test", "group"] = "train-bumper_dent-0"
    with pytest.raises(ValueError, match="leak"):
        select_data(manifest, "8class")


def test_overfitting_warning_is_observational():
    cfg = read_config("configs/e8.yaml")
    baseline = {
        "train_eval_loss": 1.0, "val_loss": 0.8,
        "train_eval_accuracy": 0.5, "val_accuracy": .5,
    }
    end = {"train_eval_loss": .3, "val_loss": 1.8,
           "train_eval_accuracy": .98, "val_accuracy": .5}
    history = [baseline.copy() for _ in range(15)] + [end.copy()]
    assert detect_overfitting(history, cfg) is True
    assert detect_overfitting([baseline] * 15, cfg) is False


def test_pinned_pretrained_full_finetuning_configuration():
    cfg = read_config("configs/e8.yaml")
    assert cfg["epochs"] == 300 and cfg["seeds"] == [0, 1, 2]
    assert cfg["early_stopping"]["patience"] == 30
    assert cfg["early_stopping"]["monitor"] == "val_macro_f1"
    assert cfg["output"] == "outputs/e8"
    assert cfg["pretrained"]["repository"] == "Majorburn/yolov11-carparts-seg"
    assert len(cfg["pretrained"]["revision"]) == 40
    assert len(jobs(cfg)) == 6


def test_part_backbone_transfer_and_new_head(monkeypatch, tmp_path):
    class SPPF(nn.Module):
        def forward(self, x):
            return x
    class C2PSA(nn.Module):
        def forward(self, x):
            return x
    class DummyModel:
        task = "segment"
        names = {i: f"part{i}" for i in range(23)}
        model = SimpleNamespace(model=nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            SPPF(), C2PSA(), nn.Identity(),
        ))
    monkeypatch.setitem(sys.modules, "ultralytics",
                        SimpleNamespace(YOLO=lambda checkpoint: DummyModel()))
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"dummy")
    model = PartPretrainedClassifier(checkpoint, 7, image_size=64)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 7)
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert len(list(model.backbone.children())) == 3


def test_early_stopping_patience_30_and_resume_state():
    cfg = read_config("configs/e8.yaml")
    state, stopped = early_stopping_step(.70, None, cfg, 1)
    assert not stopped and state["bad_epochs"] == 0
    for epoch in range(2, 31):
        state, stopped = early_stopping_step(.70, state, cfg, epoch)
        assert not stopped
    state, stopped = early_stopping_step(.70, state, cfg, 31)
    assert stopped and state["bad_epochs"] == 30
    assert state["last_significant_epoch"] == 1
    # Improvement beyond min_delta resets the exact same persisted state.
    state, stopped = early_stopping_step(.705, state, cfg, 32)
    assert not stopped and state["bad_epochs"] == 0
    assert state["last_significant_epoch"] == 32


def test_early_stopping_insignificant_gain_does_not_reset():
    cfg = read_config("configs/e8.yaml")
    state, _ = early_stopping_step(.700, None, cfg, 1)
    state, stopped = early_stopping_step(.7005, state, cfg, 2)
    assert not stopped and state["bad_epochs"] == 1
