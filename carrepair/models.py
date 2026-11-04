"""YOLO feature extraction and explicit CNN-compatible PEFT modules."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


class LoRAConv(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        if base.kernel_size != (1, 1) or base.groups != 1:
            raise ValueError('This LoRA implementation targets ungrouped 1x1 convolutions only')
        self.base = base
        self.base.requires_grad_(False)
        rank = min(rank, base.in_channels, base.out_channels)
        self.down = nn.Conv2d(base.in_channels, rank, 1, stride=base.stride, padding=base.padding, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


class ResidualExpansion(nn.Module):
    """Identity-initialised CNN block expansion, not ViT block duplication."""
    def __init__(self, channels, bottleneck=64):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


def inject_lora(module, rank, alpha):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and child.groups == 1:
            setattr(module, name, LoRAConv(child, rank, alpha))
            count += 1
        else:
            count += inject_lora(child, rank, alpha)
    return count


class YoloFeatures(nn.Module):
    def __init__(self, backbone, channels, method='full', cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.backbone, self.dim, self.method = backbone, channels, method
        self.expansion = nn.Identity()
        self.backbone.requires_grad_(method == 'full')
        if method == 'lora':
            if not inject_lora(self.backbone, cfg.get('lora_rank', 8), cfg.get('lora_alpha', 8)):
                raise ValueError('No LoRA targets found')
        elif method == 'block':
            self.expansion = ResidualExpansion(channels, cfg.get('block_bottleneck', 64))
        elif method not in ['full', 'linear']:
            raise ValueError(method)

    def train(self, mode=True):
        super().train(mode)
        # Freezing gradients alone would still update BatchNorm running statistics.
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self

    def forward(self, x):
        x = self.expansion(self.backbone(x))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def yolo_features(cfg, method, scratch=False):
    from ultralytics import YOLO
    # Trusted checkpoint downloaded from the specified model author's repository.
    base = YOLO('yolov8s-seg.yaml' if scratch else cfg['checkpoint']).model
    parts = list(base.model.children())
    if len(parts) < 10 or type(parts[9]).__name__ != 'SPPF':
        raise ValueError('Checkpoint architecture differs from standard YOLOv8s; inspect backbone before proceeding')
    backbone = nn.Sequential(*deepcopy(parts[:10]))
    channels = backbone[-1].cv2.conv.out_channels
    return YoloFeatures(backbone, channels, method, cfg)


class Classifier(nn.Module):
    def __init__(self, features, n_classes):
        super().__init__()
        self.features = features
        self.head = nn.Linear(features.dim, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


def parameter_counts(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


@torch.no_grad()
def ema_update(teacher, student, momentum):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(momentum).add_(s, alpha=1 - momentum)
    for t, s in zip(teacher.buffers(), student.buffers()):
        if t.is_floating_point():
            t.mul_(momentum).add_(s, alpha=1 - momentum)
        else:
            t.copy_(s)
