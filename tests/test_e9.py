"""E9 CPU-only protocol tests; never download weights or run GPU training."""
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from carrepair.common import read_config
from carrepair.e9 import (
    EIGHT, BINARY, TASKS, PARENTS, HierarchicalClassifier,
    datasets, part_label, zero_shot_decision,
)
from carrepair.e9_submit import jobs


def fake_manifest():
    rows = []
    for split in ("train", "val", "test"):
        for label in EIGHT:
            for i in range(2):
                rows.append({
                    "id": f"{split}-{label}-{i}", "label": label,
                    "group": f"{split}-{label}-{i}", "split": split,
                    "path": f"{split}/{label}/{i}.jpg",
                })
    return pd.DataFrame(rows)


def test_binary_merges_dent_and_scratch_without_mixing_splits():
    part, names = datasets(fake_manifest(), "binary")
    assert names == ["bumper", "door"]
    assert all(len(x) == 8 for x in part)
    assert all(set(x.label) == {"bumper_dent", "bumper_scratch", "door_dent", "door_scratch"}
               for x in part)
    assert len(set(part[0].id) & set(part[1].id)) == 0


def test_hierarchical_preserves_all_original_eight():
    parts, names = datasets(fake_manifest(), "hierarchical")
    assert names == EIGHT
    assert all(len(p) == 16 for p in parts)
    assert part_label("door_dent") == "door"
    assert part_label("bumper_scratch") == "bumper"
    assert part_label("glass_shatter") == "other"
    assert part_label("unknown") == "other"


def test_hierarchical_joint_probabilities_normalise():
    logits = (torch.randn(6, 3), torch.randn(6, 2), torch.randn(6, 4))
    logp = HierarchicalClassifier.joint_log_probs(logits)
    assert logp.shape == (6, 8)
    torch.testing.assert_close(logp.exp().sum(1), torch.ones(6))


def test_zero_shot_uses_weighted_mask_area_and_abstains():
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=torch.tensor([8, 11]), conf=torch.tensor([.9, .8])),
        masks=SimpleNamespace(
            data=torch.stack([
                torch.ones(10, 10),
                torch.cat([torch.ones(2, 10), torch.zeros(8, 10)]),
            ])),
    )
    prediction, detected, score = zero_shot_decision(
        result, {8: "front_bumper", 11: "front_left_door"})
    assert prediction == "bumper" and len(detected) == 2 and score > 0
    no_mask = SimpleNamespace(boxes=result.boxes, masks=None)
    assert zero_shot_decision(no_mask, {8: "front_bumper"})[0] is None


def test_e9_job_matrix_and_epoch_policy():
    cfg = read_config("configs/e9.yaml")
    assert cfg["output"] == "outputs/e9"
    assert cfg["epochs"] == 300
    assert cfg["seeds"] == [0, 1, 2]
    assert cfg["pretrained"]["repository"] == "Majorburn/yolov11-carparts-seg"
    assert TASKS == ("binary", "hierarchical")
    matrix = jobs(cfg)
    assert len(matrix) == 7 and matrix[0] == ("zero-shot", None)
    assert set(matrix[1:]) == {(t, s) for t in TASKS for s in cfg["seeds"]}
