"""CPU-only checks for E13/E14 configuration, labels, routing, and checkpointing."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from carrepair import e13e14 as ex
from carrepair.e13e14_submit import job_order, job_key, needed_inputs


@pytest.fixture
def config(tmp_path):
    return {
        "manifest": str(tmp_path / "manifest.csv"),
        "data_root": str(tmp_path),
        "output": str(tmp_path / "outputs"),
        "checkpoint": str(tmp_path / "cardd.pt"),
        "device": "cpu", "batch_size": 2, "workers": 0,
        "seeds": [0, 1, 2], "epochs": 300, "head_lr": 0.001,
        "lr": 1e-4, "weight_decay": 1e-4,
        "early_stopping": {"patience": 30, "min_delta": 0.001,
                           "monitor": "val_macro_f1"},
        "expert": {
            "directory": str(tmp_path / "binary"),
            "source_checkpoint": str(tmp_path / "source.pt"),
            "source_sha256": "dummy", "image_size": 384,
            "expected_parts": 23,
        },
        "variants": {
            "e13b": {"kind": "cardd", "image_size": 384, "augmentation": "pad"},
            "e13c": {"kind": "cardd", "image_size": 384, "augmentation": "mild"},
            "e14a": {"kind": "hf", "family": "dinov3", "mode": "frozen",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "facebook/model", "unfreeze_last_blocks": 0},
            "e14b": {"kind": "hf", "family": "dinov3", "mode": "partial",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "facebook/model", "unfreeze_last_blocks": 4},
            "e14c": {"kind": "hf", "family": "dinov3", "mode": "partial",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "facebook/model", "unfreeze_last_blocks": 4},
            "e14d": {"kind": "hf", "family": "siglip2", "mode": "partial",
                     "image_size": 224, "pretrained_dir": str(tmp_path / "hf"),
                     "pretrained_repo": "google/model", "unfreeze_last_blocks": 4},
        },
    }


def test_all_six_variants_and_fixed_seeds(config):
    assert set(ex.VARIANTS) == set(config["variants"])
    for v in ex.VARIANTS:
        ex.verify_protocol(config, v, require_sources=False)
    bad = dict(config, seeds=[0, 1])
    with pytest.raises(ValueError, match="Seeds"):
        ex.verify_protocol(bad, require_sources=False)
    bad = dict(config, epochs=100)
    with pytest.raises(ValueError, match="300"):
        ex.verify_protocol(bad, require_sources=False)


def test_e13_pad_preserves_aspect_and_is_deterministic():
    image = Image.new("RGB", (120, 45), color=(25, 180, 10))
    f = ex.cardd_transform(384, train=False)
    a, b = f(image), f(image)
    assert a.shape == (3, 384, 384)
    assert torch.equal(a, b)
    assert tuple((a[:, 0, 0] * 255).round().int().tolist()) == (114, 114, 114)
    # Centered car colour retained rather than cropped out.
    assert tuple((a[:, 192, 192] * 255).round().int().tolist()) == (25, 180, 10)


def test_e13_mild_transform_still_returns_384_tensor():
    im = Image.new("RGB", (50, 200))
    x = ex.cardd_transform(384, train=True, augmentation="mild")(im)
    assert x.shape == (3, 384, 384)
    assert torch.isfinite(x).all()


def test_root_label_and_data_ids(config, tmp_path):
    Image.new("RGB", (60, 40)).save(tmp_path / "a.jpeg")
    frame = pd.DataFrame([{
        "id": "same-img-id", "path": "a.jpeg", "label": "bumper_scratch",
        "group": "g0", "split": "test"
    }])
    dataset = ex.RootDataset(frame, config, "e13b", train=False)
    image, label, row = dataset[0]
    assert label == ex.ROOT.index("scratch")
    assert image.shape == (3, 384, 384)
    assert row == 0


def test_checkpoint_contains_only_trainable_hf_weights():
    class Mock(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.head = nn.Linear(4, 6)
            self.encoder.requires_grad_(False)
    a, b = Mock(), Mock()
    with torch.no_grad():
        a.head.weight.fill_(3.5)
    state, kind = ex._trainable_state(a, "hf")
    assert kind == "trainable"
    assert set(state) == {"head.weight", "head.bias"}
    previous_encoder_weight = b.encoder.weight.clone()
    ex._load_trained_state(b, {"state": state, "state_kind": kind})
    assert torch.equal(b.head.weight, a.head.weight)
    assert torch.equal(b.encoder.weight, previous_encoder_weight)


def test_slurm_job_dag_and_e9_expert_same_seed(config):
    work = job_order(["e13b", "e14a"], [0, 1, 2])
    assert len(work) == 12
    assert work[0] == ("e13b", "train", 0)
    assert work[1] == ("e13b", "evaluate", 0)
    assert job_key("e14a", "evaluate", 2) == "e14a-evaluate-s2"
    paths = needed_inputs(config, ["e13b"])
    assert Path(config["checkpoint"]) in paths
    for seed in [0, 1, 2]:
        assert Path(config["expert"]["directory"]) / f"seed{seed}" / "best.pt" in paths


def test_normalized_soft_joint_and_hard_prediction():
    rp = np.array([[0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
                   [0.10, 0.75, 0.05, 0.03, 0.04, 0.03]])
    pp = np.array([[0.70, 0.30], [0.20, 0.80]])
    probs = ex.joint_probabilities(rp, pp)
    hard = ex.hard_route(rp, pp)
    assert np.allclose(probs.sum(1), 1)
    assert ex.EIGHT[hard[0]] == "bumper_dent"
    assert ex.EIGHT[hard[1]] == "door_scratch"


def test_source_signature_fails_closed(config):
    with pytest.raises(FileNotFoundError):
        ex.source_signature(config, "e14a")
    with pytest.raises(FileNotFoundError):
        ex.source_signature(config, "e13b")


def test_fixed_partitions_reject_wrong_sizes(monkeypatch, config):
    three = [pd.DataFrame({"id": ["a"], "label": ["unknown"]}),
             pd.DataFrame({"id": ["b"], "label": ["unknown"]}),
             pd.DataFrame({"id": ["c"], "label": ["unknown"]})]
    monkeypatch.setattr(ex, "partitions", lambda cfg: tuple(three))
    with pytest.raises(ValueError, match="split sizes"):
        ex.fixed_partitions(config)


def test_cached_frozen_features_preserve_original_ids(config):
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5
        def __getitem__(self, ix):
            return torch.full((3, 4, 4), float(ix)), ix % 6, ix
    class MockModel(nn.Module):
        def embedding(self, x):
            return x.flatten(1).mean(1, keepdim=True)
    cached = ex.frozen_features(MockModel(), MockDataset(), config)
    assert len(cached) == 5
    assert [int(cached[i][2]) for i in range(5)] == list(range(5))
    assert cached[4][0].item() == pytest.approx(4.0)
