"""E11 CPU-only regression tests: zero downloads, checkpoints, or GPU training."""
import numpy as np
import pandas as pd
import pytest

from carrepair.common import read_config
from carrepair.e9 import EIGHT, BINARY
from carrepair.e11 import (
    ROOT, BRANCHED, DIRECT, root_label, partitions,
    joint_probabilities, hard_route, error_analysis, train_dir, eval_dir,
    expert_path,
)
from carrepair.e11_submit import jobs, dependencies, task_dir, required_inputs


def fake_manifest():
    return pd.DataFrame([
        {
            "id": f"{split}-{label}-{i}", "label": label,
            "path": f"{split}/{label}/{i}.jpg",
            "group": f"group-{split}-{label}-{i}", "split": split,
        }
        for split in ("train", "val", "test")
        for label in EIGHT
        for i in range(2)
    ])


def test_six_damage_classes_cover_existing_eight():
    assert ROOT == (
        "dent", "scratch", "glass_shatter",
        "head_lamp", "tail_lamp", "unknown")
    assert BRANCHED == ("dent", "scratch")
    assert DIRECT == ("glass_shatter", "head_lamp", "tail_lamp", "unknown")
    assert BINARY == ["bumper", "door"]
    mapping = {
        "bumper_dent": "dent", "door_dent": "dent",
        "bumper_scratch": "scratch", "door_scratch": "scratch",
        "glass_shatter": "glass_shatter", "head_lamp": "head_lamp",
        "tail_lamp": "tail_lamp", "unknown": "unknown",
    }
    assert {c: root_label(c) for c in EIGHT} == mapping
    with pytest.raises(ValueError):
        root_label("unexpected")


def test_manifest_groups_and_split_not_changed(monkeypatch):
    import carrepair.e11 as mod
    monkeypatch.setattr(mod, "read_manifest", lambda _: fake_manifest())
    frames = partitions({"manifest": "unused"})
    assert [len(f) for f in frames] == [16, 16, 16]
    assert all(set(f.label) == set(EIGHT) for f in frames)
    assert len(set(frames[0].group) & set(frames[1].group)) == 0
    assert len(set(frames[1].group) & set(frames[2].group)) == 0


def test_eight_class_soft_probabilities_normalised_and_correct():
    root = np.array([
        [.7, .1, .05, .05, .05, .05],
        [.1, .7, .05, .05, .05, .05],
        [.01, .01, .80, .06, .06, .06],
        [.01, .01, .02, .02, .02, .92],
    ])
    expert = np.array([
        [.2, .8],
        [.8, .2],
        [.2, .8],
        [.5, .5],
    ])
    p = joint_probabilities(root, expert)
    assert p.shape == (4, 8)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-8)
    assert p[0, EIGHT.index("door_dent")] == pytest.approx(.7 * .8)
    assert p[1, EIGHT.index("bumper_scratch")] == pytest.approx(.7 * .8)
    assert p[2, EIGHT.index("glass_shatter")] == pytest.approx(.8)
    assert p[3, EIGHT.index("unknown")] == pytest.approx(.92)
    with pytest.raises(ValueError):
        joint_probabilities(np.ones((4, 5)), expert)
    with pytest.raises(ValueError):
        joint_probabilities(root, np.ones((4, 3)))
    with pytest.raises(ValueError):
        joint_probabilities(root, expert * .5)


def test_hard_routing_dent_scratch_expert_and_four_direct():
    root = np.array([
        [.9, .02, .02, .02, .02, .02],
        [.02, .9, .02, .02, .02, .02],
        [.02, .02, .9, .02, .02, .02],
        [.02, .02, .02, .9, .02, .02],
        [.02, .02, .02, .02, .9, .02],
        [.02, .02, .02, .02, .02, .9],
    ])
    expert = np.array([
        [.6, .4],
        [.2, .8],
        [.5, .5], [.5, .5], [.5, .5], [.5, .5],
    ])
    labels = [EIGHT[i] for i in hard_route(root, expert)]
    assert labels == [
        "bumper_dent", "door_scratch", "glass_shatter",
        "head_lamp", "tail_lamp", "unknown"]


def test_root_vs_expert_error_decomposition_and_oracle():
    y = np.asarray([EIGHT.index(name) for name in [
        "bumper_dent", "door_scratch", "glass_shatter", "head_lamp"]])
    root_predict = np.asarray([
        ROOT.index("scratch"), ROOT.index("scratch"),
        ROOT.index("glass_shatter"), ROOT.index("head_lamp")])
    part_predict = np.asarray([0, 0, 0, 1])
    hard = np.asarray([EIGHT.index(name) for name in [
        "bumper_scratch", "bumper_scratch", "glass_shatter", "head_lamp"]])
    dec = error_analysis(y, root_predict, part_predict, hard)
    assert dec["total"] == 4
    assert dec["root_wrong"] == 1
    assert dec["correct_root_wrong_expert"] == 1
    assert dec["hard_correct"] == 2
    # Correct root would rescue the first mistake; the second remains.
    assert dec["oracle_root_accuracy_diagnostic_only"] == pytest.approx(.75)


def test_e11_config_and_artemis_six_job_dag():
    cfg = read_config("configs/e11.yaml")
    assert cfg["checkpoint"] == "weights/cardd-v2.pt"
    assert cfg["image_size"] == 224 and cfg["batch_size"] == 32
    assert cfg["epochs"] == 300 and cfg["early_stopping"]["patience"] == 30
    assert cfg["early_stopping"]["monitor"] == "val_macro_f1"
    assert cfg["seeds"] == [0, 1, 2]
    assert len(jobs(cfg)) == 6
    assert jobs(cfg)[:2] == [("train", 0), ("evaluate", 0)]
    assert dependencies("train", 0) == []
    assert dependencies("evaluate", 1) == ["train-s1"]
    assert str(expert_path(cfg, 2)).endswith("e9/binary/seed2/best.pt")
    assert str(train_dir(cfg, 2)).endswith("six_class/seed2")
    assert str(eval_dir(cfg, 1)).endswith("cascade/seed1")
    assert task_dir(cfg, "evaluate", 1) == eval_dir(cfg, 1)
    assert len(required_inputs(cfg)) == 16


def test_e11_stage1_is_full_finetuning_not_linear_probe(monkeypatch):
    import carrepair.e11 as mod
    class FakeModel:
        def __init__(self):
            import torch
            from torch import nn
            self.net = nn.Linear(4, 6)
        def parameters(self):
            return self.net.parameters()
    passed = {}
    def fake_yolo_features(cfg, method):
        passed["method"] = method
        import torch
        from torch import nn
        net = nn.Linear(3, 4)
        net.dim = 4
        return net
    monkeypatch.setattr(mod, "yolo_features", fake_yolo_features)
    model = mod.build_root({"checkpoint": "unused"})
    assert passed["method"] == "full"
    assert all(p.requires_grad for p in model.parameters())


def test_root_source_is_cardd_while_expert_is_e9_binary():
    cfg = read_config("configs/e11.yaml")
    assert "cardd" in cfg["checkpoint"]
    assert "e9/binary" in cfg["expert"]["directory"]
    assert cfg["expert"]["image_size"] == 384
    assert len(cfg["expert"]["source_sha256"]) == 64
