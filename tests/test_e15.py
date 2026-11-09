"""E15 protocol and model-gradient boundary regression tests."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from carrepair.common import read_config
from carrepair.e9 import EIGHT
from carrepair.e15 import LABELS, VARIANTS, setting_for, verify, run_dir, evaluation_dir


def config():
    return read_config(Path(__file__).resolve().parents[1] / "configs" / "e15.yaml")


def test_eight_way_direct_protocol():
    cfg = config()
    verify(cfg)
    assert VARIANTS == ("e15a", "e15b")
    assert list(LABELS) == list(EIGHT)
    assert len(LABELS) == 8
    assert setting_for(cfg, "e15a")["mode"] == "partial"
    assert setting_for(cfg, "e15b")["mode"] == "full"
    assert cfg["variants"]["e15a"]["pretrained_dir"] == cfg["variants"]["e15b"]["pretrained_dir"]
    assert run_dir(cfg, "e15a", 0) != run_dir(cfg, "e15b", 0)
    assert evaluation_dir(cfg, "e15a", 0) != run_dir(cfg, "e15a", 0)


@pytest.mark.parametrize("variant,field,value", [
    ("e15a", "mode", "full"), ("e15a", "unfreeze_last_blocks", 3),
    ("e15b", "mode", "partial"), ("e15b", "family", "dinov2"),
    ("e15a", "image_size", 384), ("e15b", "backbone_lr", 0.0001),
])
def test_reject_mismatched_experimental_setting(variant, field, value):
    cfg = config()
    cfg["variants"][variant][field] = value
    with pytest.raises(ValueError):
        verify(cfg)


def test_split_and_training_constraints():
    for key, value in [("seeds", [0]), ("epochs", 100), ("batch_size", 8)]:
        cfg = config()
        cfg[key] = value
        with pytest.raises(ValueError):
            verify(cfg)


def test_checkpoint_trainable_state_roundtrip():
    from carrepair.e13e14 import _trainable_state, _load_trained_state
    model = nn.Linear(4, 8)
    state, kind = _trainable_state(model, "hf")
    assert kind == "trainable"
    expected = {key: tensor.clone() for key, tensor in state.items()}
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    _load_trained_state(model, {"state": state, "state_kind": kind})
    for key, tensor in model.named_parameters():
        assert torch.equal(tensor.cpu(), expected[key])


def test_source_hash_accepts_e15_settings_without_kind(tmp_path):
    """E15 configs do not need an E14-only 'kind' field."""
    from carrepair.e15 import source_hash
    cfg = config()
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "siglip"}')
    (model / "preprocessor_config.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b"fixture-only-hash")
    for variant in cfg["variants"].values():
        variant["pretrained_dir"] = str(model)
    first = source_hash(cfg, "e15a")
    second = source_hash(cfg, "e15b")
    assert len(first) == 64
    assert first == second
