"""CPU-only E10 protocol checks. No YOLO weights, network or GPU required."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from carrepair.common import read_config
from carrepair.e9 import EIGHT
from carrepair.e10 import (
    ROOT, EXPERTS, DIRECT, E10Images, root_label, stage_names, stage_dir,
    root_best_path, partitions, build_joint_probabilities, hard_route,
    error_decomposition, root_backbone_from_checkpoint, prediction_metrics,
)
from carrepair.e10_submit import jobs, dependencies, command_args, output_dir


def fake_manifest():
    rows = []
    for split in ("train", "val", "test"):
        for label in EIGHT:
            for i in range(2):
                rows.append({
                    "id": f"{split}-{label}-{i}",
                    "path": f"{split}/{label}/{i}.jpg", "label": label,
                    "group": f"{split}-{label}-{i}", "split": split,
                })
    return pd.DataFrame(rows)


def test_five_roots_cover_all_eight_original_classes():
    assert ROOT == ("bumper", "door", "lamp", "glass", "unknown")
    assert set(DIRECT) == {"glass", "unknown"}
    assert stage_names("bumper") == ("bumper_dent", "bumper_scratch")
    assert stage_names("door") == ("door_dent", "door_scratch")
    assert stage_names("lamp") == ("head_lamp", "tail_lamp")
    assert len({name for labels in EXPERTS.values() for name in labels} |
               set(DIRECT.values())) == 8
    expected = {
        "bumper_dent": "bumper", "bumper_scratch": "bumper",
        "door_dent": "door", "door_scratch": "door",
        "head_lamp": "lamp", "tail_lamp": "lamp",
        "glass_shatter": "glass", "unknown": "unknown",
    }
    assert {k: root_label(k) for k in EIGHT} == expected
    with pytest.raises(ValueError):
        root_label("unexpected_label")
    with pytest.raises(ValueError):
        stage_names("glass")


def test_all_splits_remain_group_disjoint_and_expert_only_sees_its_classes(
        monkeypatch):
    import carrepair.e10 as mod
    monkeypatch.setattr(mod, "read_manifest", lambda unused: fake_manifest())
    cfg = {"manifest": "unused"}
    root = partitions(cfg)
    assert all(len(part) == 16 for part in root)
    assert [set(part.group).isdisjoint(root[1].group) for part in root[:1]] == [True]
    for expert in EXPERTS:
        frames = partitions(cfg, expert)
        assert all(len(part) == 4 for part in frames)
        assert all(set(part.label) == set(EXPERTS[expert]) for part in frames)


def test_soft_joint_probabilities_normalise_and_map_all_classes():
    root = np.array([
        [.6, .2, .1, .07, .03],
        [.1, .1, .7, .05, .05],
        [.1, .1, .1, .65, .05],
    ])
    experts = {
        "bumper": np.array([[.25, .75], [.5, .5], [.5, .5]]),
        "door": np.array([[.2, .8], [.4, .6], [.5, .5]]),
        "lamp": np.array([[.7, .3], [.2, .8], [.5, .5]]),
    }
    result = build_joint_probabilities(root, experts)
    assert result.shape == (3, 8)
    np.testing.assert_allclose(result.sum(axis=1), np.ones(3))
    assert result[0, EIGHT.index("bumper_scratch")] == pytest.approx(.6 * .75)
    assert result[1, EIGHT.index("tail_lamp")] == pytest.approx(.7 * .8)
    assert result[2, EIGHT.index("glass_shatter")] == pytest.approx(.65)
    assert result[2, EIGHT.index("unknown")] == pytest.approx(.05)
    with pytest.raises(ValueError):
        build_joint_probabilities(np.ones((2, 3)), experts)
    with pytest.raises(ValueError):
        build_joint_probabilities(root, {**experts, "door": np.ones((3, 3))})


def test_hard_routes_only_to_selected_expert_or_direct_output():
    roots = np.array([
        [.8, .1, .05, .03, .02],
        [.1, .7, .1, .06, .04],
        [.05, .05, .8, .05, .05],
        [.02, .02, .02, .9, .04],
        [.02, .02, .02, .04, .9],
    ])
    expert = {
        "bumper": np.tile([.1, .9], (5, 1)),
        "door": np.tile([.8, .2], (5, 1)),
        "lamp": np.tile([.2, .8], (5, 1)),
    }
    got = [EIGHT[i] for i in hard_route(roots, expert)]
    assert got == ["bumper_scratch", "door_dent", "tail_lamp",
                   "glass_shatter", "unknown"]


def test_hard_errors_separate_bad_root_from_bad_expert():
    y_true = np.array([EIGHT.index(x) for x in (
        "bumper_dent", "door_scratch", "head_lamp", "glass_shatter")])
    roots = np.array([ROOT.index(x) for x in ("door", "door", "lamp", "glass")])
    hard = np.array([EIGHT.index(x) for x in (
        "door_dent", "door_dent", "head_lamp", "glass_shatter")])
    result = error_decomposition(y_true, roots, hard)
    assert result["total"] == 4
    assert result["root_wrong_count"] == 1
    assert result["correct_root_wrong_expert_count"] == 1
    assert result["hard_correct_count"] == 2
    assert result["root_accuracy"] == pytest.approx(.75)
    metrics = prediction_metrics(y_true, hard)
    assert metrics["accuracy"] == pytest.approx(.5)
    assert metrics["confusion_matrix"][EIGHT.index("door_scratch")][
        EIGHT.index("door_dent")] == 1


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)


def test_root_backbone_transfer_is_strict_and_seed_matched():
    model = _ToyModel()
    state = model.state_dict()
    expected = {
        "backbone.weight": torch.full_like(state["backbone.weight"], .4),
        "backbone.bias": torch.full_like(state["backbone.bias"], .6),
        "head.weight": torch.randn(5, 3),
    }
    root_backbone_from_checkpoint(
        model, {"stage": "root", "seed": 2, "model": expected}, seed=2)
    torch.testing.assert_close(model.backbone.weight, expected["backbone.weight"])
    torch.testing.assert_close(model.backbone.bias, expected["backbone.bias"])
    with pytest.raises(ValueError):
        root_backbone_from_checkpoint(
            model, {"stage": "root", "seed": 1, "model": expected}, seed=2)
    with pytest.raises(ValueError):
        root_backbone_from_checkpoint(
            model, {"stage": "root", "seed": 2, "model": {}}, seed=2)


def test_submission_dag_has_correct_dependencies_and_checkpoints():
    cfg = read_config("configs/e10.yaml")
    assert cfg["epochs"] == 300 and cfg["early_stopping"]["patience"] == 30
    assert cfg["early_stopping"]["monitor"] == "val_macro_f1"
    assert cfg["seeds"] == [0, 1, 2]
    assert cfg["pretrained"]["repository"] == "Majorburn/yolov11-carparts-seg"
    assert [k for k, seed in jobs(cfg) if seed == 0] == [
        "root", "bumper", "door", "lamp", "evaluate"]
    assert len(jobs(cfg)) == 15
    assert dependencies("root", 1) == []
    assert dependencies("lamp", 1) == ["root-s1"]
    assert dependencies("evaluate", 1) == ["bumper-s1", "door-s1", "lamp-s1"]
    assert command_args("root", 1) == ["train-root", "--seed", "1"]
    assert command_args("bumper", 2) == [
        "train-expert", "--expert", "bumper", "--seed", "2"]
    assert command_args("evaluate", 0) == ["evaluate", "--seed", "0"]
    assert str(root_best_path(cfg, 0)).endswith("root/seed0/best_root.pt")
    assert str(stage_dir(cfg, "lamp", 1)).endswith("experts/lamp/seed1")
    assert str(output_dir(cfg, "evaluate", 2)).endswith("cascade/seed2")


def test_early_stop_persisted_logic_patience_30():
    from carrepair.e9 import early_stopping_step
    cfg = read_config("configs/e10.yaml")
    state = None
    for epoch in range(1, 31):
        state, stop = early_stopping_step(.8, state, cfg, epoch)
        assert not stop
    state, stop = early_stopping_step(.8, state, cfg, 31)
    assert stop and state["bad_epochs"] == 30
    state, stop = early_stopping_step(.803, state, cfg, 32)
    assert not stop and state["bad_epochs"] == 0
