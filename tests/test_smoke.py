"""Real YOLO module forward/backward smoke; no checkpoints downloaded."""
import torch
from carrepair.models import yolo_features, Classifier
from carrepair.common import read_config


def test_yolo_scratch_forward_backward():
    torch.set_num_threads(1)
    cfg = read_config('configs/default.yaml')
    features = yolo_features(cfg, 'full', scratch=True)
    model = Classifier(features, 8)
    y = model(torch.rand(2, 3, 64, 64))
    assert y.shape == (2, 8)
    y.sum().backward()
    assert model.head.weight.grad is not None
