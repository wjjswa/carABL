import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from carrepair.data import subset, read_manifest, prepare, Images
from carrepair.models import LoRAConv, ResidualExpansion, YoloFeatures, ema_update
from carrepair.training import DINOLoss
from carrepair.evaluation import knn_predict, select_knn
from carrepair.activegcd import adaptive_query, label_mapping
from carrepair.cli import build_matrix


def test_nested_group_subsets_and_no_test_access():
    frame = pd.DataFrame([{'id': f'{c}{i}', 'path': f'{c}{i}.jpg', 'label': c, 'group': f'{c}{i}', 'split': 'train'}
                          for c in ['a', 'b'] for i in range(20)])
    sets = [set(subset(frame, p, 3).id) for p in [.2, .4, .6, .8, 1.]]
    assert all(a <= b for a, b in zip(sets, sets[1:]))
    assert sets[0] == set(subset(frame, .2, 3).id)
    with pytest.raises(ValueError):
        subset(frame.assign(split='test'), .2, 0)


def test_manifest_rejects_group_leakage(tmp_path):
    pd.DataFrame([{'id': 'a', 'path': 'a.jpg', 'label': 'x', 'group': 'same', 'split': 'train'},
                  {'id': 'b', 'path': 'b.jpg', 'label': 'x', 'group': 'same', 'split': 'test'}]).to_csv(tmp_path/'m.csv', index=False)
    with pytest.raises(ValueError, match='leakage'):
        read_manifest(tmp_path/'m.csv')


def test_prepare_deduplicates_and_preserves_groups(tmp_path):
    from PIL import Image
    root = tmp_path/'images'; root.mkdir()
    rows = []
    rng = np.random.default_rng(41)
    for c in ['a', 'b']:
        for i in range(30):
            name = f'{c}{i}.png'
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root/name)
            rows.append({'path': name, 'classes': c, 'vehicle': f'{c}{i//2}'})
    csv = tmp_path/'labels.csv'; pd.DataFrame(rows).to_csv(csv, index=False)
    frame = prepare(root, tmp_path/'prepared', csv, group_column='vehicle', near_distance=0)
    assert frame.groupby('group').split.nunique().max() == 1
    assert set(frame.split) == {'train', 'val', 'test'}
    assert all(set(frame[frame.split == s].label) == {'a', 'b'} for s in frame.split.unique())
    with pytest.raises(FileExistsError):
        prepare(root, tmp_path/'prepared', csv)


def test_lora_identity_then_updates_only_adapter():
    torch.manual_seed(0)
    base = nn.Conv2d(8, 12, 1)
    model = LoRAConv(copy.deepcopy(base), rank=3)
    x = torch.randn(4, 8, 5, 5)
    assert torch.equal(base(x), model(x))
    original = model.base.weight.detach().clone()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    model(x).square().mean().backward(); opt.step()
    assert torch.equal(original, model.base.weight)
    assert not torch.equal(base(x), model(x))


def test_expansion_starts_at_identity_and_frozen_bn_stays_frozen():
    x = torch.randn(2, 8, 4, 4)
    block = ResidualExpansion(8, 3)
    assert torch.equal(block(x), x)
    bn = nn.BatchNorm2d(8)
    model = YoloFeatures(nn.Sequential(nn.Conv2d(8, 8, 1), bn), 8, 'linear')
    model.train(); before = bn.running_mean.clone(); model(x)
    assert not bn.training
    assert torch.equal(before, bn.running_mean)


def test_dino_teacher_no_gradient_and_centre_update():
    criterion = DINOLoss(8)
    s = [torch.randn(3, 8, requires_grad=True) for _ in range(4)]
    t = [torch.randn(3, 8, requires_grad=True) for _ in range(2)]
    loss = criterion(s, t, .04)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(x.grad is not None for x in s)
    assert all(x.grad is None for x in t)
    assert criterion.centre.abs().sum() > 0


def test_label_free_dataset_never_reads_label(tmp_path):
    from PIL import Image
    Image.new('RGB', (16, 16)).save(tmp_path/'a.png')
    frame = pd.DataFrame([{'path': 'a.png'}])  # There is deliberately no label column.
    ds = Images(frame, tmp_path, [], lambda x: torch.zeros(3, 16, 16), label_free=True)
    assert ds[0].shape == (3, 16, 16)


def test_knn_uses_bank_and_validation_selection():
    bank = np.array([[1., 0.], [0., 1.]])
    y = np.array([0, 1]); val = np.array([[.9, .1], [.1, .9]])
    chosen = select_knn(bank, y, val, y, 2, [1, 2, 100], [.07])
    pred, scores, indexes = knn_predict(bank, y, val, 2, chosen['k'], chosen['temperature'])
    assert np.array_equal(pred, y)
    assert np.allclose(scores.sum(1), 1)
    assert indexes.max() < len(bank)


def test_adaptive_query_has_no_oracle_and_covers_novel_slots():
    probs = np.array([[.05, .9, .05], [.1, .7, .2], [.02, .03, .95], [.8, .1, .1]])
    selected, fallback = adaptive_query(probs, 2, 1, 1)
    assert set(selected) == {0, 2}
    assert fallback == 0
    selected, fallback = adaptive_query(probs[:1], 12, 1, 1)
    assert selected.tolist() == [0]


def test_mapping_only_returns_observed_labels():
    mapping = label_mapping([2, 2, 0, 0], [0, 0, 1, 1], 4)
    assert mapping == {0: 2, 1: 0}
    assert 2 not in mapping and 3 not in mapping


def test_matrix_preserves_agreed_scope():
    cfg = {'seeds': [0], 'fractions': [.2, .4, .6, .8, 1.], 'methods': ['full', 'lora', 'block', 'linear']}
    matrix = build_matrix('config.yaml', cfg, ['1', '2', '3', '4', '5b'])
    assert len(matrix) == 46  # 1 + 20 + 3 DINO + 15 probes + 5 scratch + 2 ActiveGCD
    assert all('5a' not in cmd for cmd in matrix)
    assert sum('dino' in cmd for cmd in matrix) == 3
