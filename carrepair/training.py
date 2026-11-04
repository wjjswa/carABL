from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .common import seed_all, dump_json, atomic_save, provenance
from .data import read_manifest, subset, loader, MultiCrop
from .models import yolo_features, Classifier, parameter_counts, ema_update
from .evaluation import classification_eval


def validation_score(model, data_loader, device, n_classes):
    from sklearn.metrics import f1_score
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for x, y, _ in data_loader:
            pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
            true.extend(y.tolist())
    return float(f1_score(true, pred, labels=list(range(n_classes)), average='macro', zero_division=0))


def fit_classifier(model, cfg, train_frame, val_frame, classes, directory, resume=False, probe=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.to(cfg['device'])
    opt = torch.optim.AdamW([
        {'params': [p for p in model.features.parameters() if p.requires_grad], 'lr': cfg['lr']},
        {'params': model.head.parameters(), 'lr': cfg['head_lr']},
    ], weight_decay=cfg['weight_decay'])
    epochs = cfg['probe_epochs'] if probe else cfg['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(epochs, 1))
    start, best, history = 0, -1., []
    if resume and (directory / 'last.pt').exists():
        state = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, history = state['epoch'], state['best'], state['history']
        torch.set_rng_state(state['torch_rng'])
    train_loader = loader(train_frame, cfg, classes, train=True)
    val_loader = loader(val_frame, cfg, classes)
    begin = time.monotonic()
    for epoch in range(start, epochs):
        model.train()
        if probe:
            model.features.eval()
        loss_sum, n = 0., 0
        for x, y, _ in train_loader:
            # A single last image cannot train the deepest BatchNorm in a tiny smoke run.
            if len(x) == 1:
                for m in model.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(cfg['device'])), y.to(cfg['device']))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite supervised loss')
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(x)
            n += len(x)
        score = validation_score(model, val_loader, cfg['device'], len(classes))
        scheduler.step()
        history.append({'epoch': epoch + 1, 'loss': loss_sum / max(n, 1), 'validation_macro_f1': score})
        if score > best:
            best = score
            atomic_save({'model': model.state_dict(), 'classes': classes, 'config': cfg}, directory / 'best.pt')
        atomic_save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                     'epoch': epoch + 1, 'best': best, 'history': history, 'torch_rng': torch.get_rng_state()}, directory / 'last.pt')
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        print(f'{directory.name} epoch {epoch+1}/{epochs}: val macro-F1={score:.4f}', flush=True)
    model.load_state_dict(torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)['model'])
    dump_json(directory / 'training.json', {**parameter_counts(model), 'elapsed_seconds': time.monotonic() - begin,
              'best_validation_macro_f1': best, 'actual_train_images': len(train_frame), 'classes': classes})
    return model


def supervised(cfg, experiment, seed, method, fraction, resume=False):
    seed_all(seed)
    manifest = read_manifest(cfg['manifest'])
    classes = sorted(manifest.label.unique())
    train = subset(manifest[manifest.split == 'train'], fraction, seed)
    val, test = manifest[manifest.split == 'val'], manifest[manifest.split == 'test']
    scratch = experiment == 4
    if scratch and method != 'full':
        raise ValueError('Experiment 4 trains all randomly initialised backbone weights; use --method full')
    features = yolo_features(cfg, method, scratch)
    seed_all(seed)  # Identical linear-head initialisation across PEFT methods.
    model = Classifier(features, len(classes))
    directory = Path(cfg['output']) / f'exp{experiment}' / f'{method}_seed{seed}_p{int(fraction*100):03d}'
    if directory.exists() and not resume:
        raise FileExistsError(f'{directory} exists; use --resume or a fresh output root')
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / 'training_subset.csv', index=False)
    dump_json(directory / 'run.json', {'experiment': experiment, 'method': method, 'fraction': fraction, 'seed': seed})
    provenance(cfg, directory)
    model = fit_classifier(model, cfg, train, val, classes, directory, resume)
    return classification_eval(model, cfg, test, classes, directory, train)


class DINOProjection(nn.Module):
    def __init__(self, features, cfg):
        super().__init__()
        self.features = features
        h, out = cfg['dino']['hidden_dim'], cfg['dino']['projection_dim']
        self.projector = nn.Sequential(nn.Linear(features.dim, h), nn.GELU(), nn.Linear(h, 256))
        self.output = nn.utils.parametrizations.weight_norm(nn.Linear(256, out, bias=False))

    def forward(self, x):
        return self.output(F.normalize(self.projector(self.features(x)), dim=-1))


class DINOLoss(nn.Module):
    """Centred, sharpened cross-view teacher targets. Labels are never inputs."""
    def __init__(self, dim, student_temperature=.1, centre_momentum=.9):
        super().__init__()
        self.register_buffer('centre', torch.zeros(1, dim))
        self.student_temperature, self.centre_momentum = student_temperature, centre_momentum

    def forward(self, student_views, teacher_views, teacher_temperature):
        teacher_probs = [F.softmax((x.detach() - self.centre) / teacher_temperature, dim=-1) for x in teacher_views]
        losses = [-(q * F.log_softmax(p / self.student_temperature, dim=-1)).sum(-1).mean()
                  for i, q in enumerate(teacher_probs) for j, p in enumerate(student_views) if i != j]
        loss = torch.stack(losses).mean()
        with torch.no_grad():
            self.centre.mul_(self.centre_momentum).add_(torch.cat(teacher_views).mean(0, keepdim=True), alpha=1-self.centre_momentum)
        return loss


def dino_adapt(cfg, seed, method, resume=False):
    if method not in ['full', 'lora', 'block']:
        raise ValueError('DINO adaptation uses full, lora or block')
    seed_all(seed)
    frame = read_manifest(cfg['manifest'])
    train = frame[frame.split == 'train']
    directory = Path(cfg['output']) / 'exp3' / f'{method}_seed{seed}'
    if directory.exists() and not resume:
        raise FileExistsError(f'{directory} exists; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    provenance(cfg, directory)
    train.to_csv(directory / 'unlabelled_training_pool.csv', index=False)
    features = yolo_features(cfg, method)
    seed_all(seed)  # Identical projection-head initialisation across methods.
    student = DINOProjection(features, cfg).to(cfg['device'])
    teacher = deepcopy(student).requires_grad_(False).eval()
    criterion = DINOLoss(cfg['dino']['projection_dim'], cfg['dino']['student_temperature']).to(cfg['device'])
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    epochs = cfg['dino']['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    start, history = 0, []
    if resume and (directory / 'dino_last.pt').exists():
        state = torch.load(directory / 'dino_last.pt', map_location='cpu', weights_only=False)
        student.load_state_dict(state['student']); teacher.load_state_dict(state['teacher'])
        criterion.load_state_dict(state['criterion']); opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, history = state['epoch'], state['history']
        torch.set_rng_state(state['torch_rng'])
    # Images(label_free=True) does not return labels to the optimiser.
    data_loader = loader(train, cfg, [], train=True, transform=MultiCrop(cfg), label_free=True)
    for epoch in range(start, epochs):
        student.train()
        teacher.eval()
        total, count = 0., 0
        warmup = max(1, cfg['dino']['warmup_epochs'])
        progress = min(1., epoch / warmup)
        temperature = (1-progress)*cfg['dino']['warmup_teacher_temperature'] + progress*cfg['dino']['teacher_temperature']
        for views in data_loader:
            if len(views[0]) < 2:
                continue
            views = [x.to(cfg['device']) for x in views]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t = [teacher(x) for x in views[:2]]
            loss = criterion([student(x) for x in views], t, temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite DINO loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.)
            opt.step()
            ema_update(teacher, student, cfg['dino']['teacher_momentum'])
            total += loss.item(); count += 1
        if count == 0:
            raise ValueError('DINO needs at least two training images in a batch')
        scheduler.step()
        history.append({'epoch': epoch+1, 'loss': total/count})
        pd.DataFrame(history).to_csv(directory / 'dino_history.csv', index=False)
        atomic_save({'student': student.state_dict(), 'teacher': teacher.state_dict(), 'criterion': criterion.state_dict(),
                     'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch+1,
                     'history': history, 'torch_rng': torch.get_rng_state()}, directory / 'dino_last.pt')
        print(f'DINO {method} epoch {epoch+1}/{epochs}: loss={total/count:.4f}', flush=True)
    # Fixed-epoch final teacher; never select DINO weights using target test labels.
    atomic_save({'features': teacher.features.state_dict(), 'method': method, 'seed': seed, 'config': cfg}, directory / 'teacher_features.pt')
    return directory


def dino_probe(cfg, seed, method, fraction, resume=False):
    seed_all(seed)
    base = Path(cfg['output']) / 'exp3' / f'{method}_seed{seed}'
    state = torch.load(base / 'teacher_features.pt', map_location='cpu', weights_only=False)
    features = yolo_features(cfg, method)
    features.load_state_dict(state['features'])
    features.requires_grad_(False).eval()
    manifest = read_manifest(cfg['manifest'])
    classes = sorted(manifest.label.unique())
    train = subset(manifest[manifest.split == 'train'], fraction, seed)
    directory = base / f'probe_p{int(fraction*100):03d}'
    if directory.exists() and not resume:
        raise FileExistsError(f'{directory} exists; use --resume')
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / 'training_subset.csv', index=False)
    dump_json(directory / 'run.json', {'experiment': 3, 'method': method, 'fraction': fraction, 'seed': seed})
    provenance(cfg, directory)
    seed_all(seed)
    model = fit_classifier(Classifier(features, len(classes)), cfg, train, manifest[manifest.split == 'val'], classes, directory, resume, probe=True)
    return classification_eval(model, cfg, manifest[manifest.split == 'test'], classes, directory, train)
