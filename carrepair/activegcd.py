"""Kaggle port of ActiveGCD (CVPR 2024), not a YOLO or DBSCAN substitute.

Official heads/losses are vendored under their original MIT licence. Adaptations:
balanced guaranteed-labelled minibatches; safe small-pool query fallback; fixed
epoch checkpoints instead of choosing best test epochs; all oracle access audited.
"""
from copy import deepcopy
from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from torch import nn
from torch.nn import functional as F
from .common import seed_all, dump_json, atomic_save, provenance
from .data import read_manifest, subset, loader, transforms
from .models import ema_update, parameter_counts
from .vendor.activegcd.model import DINOHead_feature, DistillLoss, SupConLoss, info_nce_logits, get_params_groups


class TwoViews:
    def __init__(self):
        self.transform = transforms(224, train=True, imagenet=True, scale=(.5, 1.))

    def __call__(self, im):
        return [self.transform(im), self.transform(im)]


def adaptive_query(probabilities, n, n_old, round_number, adaptive_round=2):
    """Official Adaptive-Novel margin ordering and per-novel-slot quotas.

    Fallback fills an underfull query with remaining margin-ranked images when
    fewer than n are predicted novel. It never uses oracle labels.
    """
    probs = np.asarray(probabilities)
    n = min(n, len(probs))
    if n < 1:
        return np.array([], dtype=int), 0
    n_novel = probs.shape[1] - n_old
    if n_novel < 1:
        raise ValueError('At least one novel prediction slot is required')
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    order = np.argsort(margins if round_number >= adaptive_round else -margins, kind='stable')
    predictions = probs.argmax(1)
    selected = []
    for slot in range(n_old, probs.shape[1]):
        candidates = [int(i) for i in order if predictions[i] == slot]
        selected.extend(candidates[:n // n_novel])
    remaining_novel = [int(i) for i in order if predictions[i] >= n_old and int(i) not in selected]
    selected += remaining_novel[:n-len(selected)]
    fallback = n-len(selected)
    if fallback:
        selected += [int(i) for i in order if int(i) not in selected][:fallback]
    return np.asarray(selected, dtype=int), fallback


def label_mapping(predicted_slots, observed_labels, n_classes):
    """Hungarian matching from VERIFIED labels only (official mapping direction)."""
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (np.asarray(predicted_slots, dtype=int), np.asarray(observed_labels, dtype=int)), 1)
    slots, labels = linear_sum_assignment(counts.max() - counts)
    observed = set(map(int, observed_labels))
    return {int(label): int(slot) for slot, label in zip(slots, labels) if int(label) in observed}


def build_model(cfg, n_classes, backbone=None):
    if backbone is None:
        backbone = torch.hub.load(cfg['activegcd']['dino_repo'], 'dino_vitb16', trust_repo=True)
    backbone.requires_grad_(False)
    for name, p in backbone.named_parameters():
        if name.startswith('blocks.') and int(name.split('.')[1]) >= cfg['activegcd']['grad_from_block']:
            p.requires_grad = True
    dim = getattr(backbone, 'embed_dim', 768)
    return nn.Sequential(backbone, DINOHead_feature(dim, n_classes, nlayers=3))


def ag_loader(frame, cfg, classes, train=False, label_free=False):
    return loader(frame, {**cfg, 'image_size': 224}, classes, train,
                  TwoViews() if train else transforms(224, imagenet=True), label_free)


@torch.no_grad()
def predict(model, frame, cfg, classes):
    model.eval()
    logits, y = [], []
    for x, labels, _ in ag_loader(frame, cfg, classes):
        logits.append(model(x.to(cfg['device']))[-1].cpu())
        y.extend(labels.tolist())
    return torch.cat(logits), np.asarray(y)


def train_phase(model, teacher, initial, pool, queried, cfg, classes, epochs, output):
    """Default SimGCD objective, EMA, and separate mapped active supervision."""
    a = cfg['activegcd']
    model.to(cfg['device']); teacher.to(cfg['device'])
    opt = torch.optim.SGD(get_params_groups(model), lr=a['lr'], momentum=.9, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=a['lr']*.001)
    distill = DistillLoss(min(30, epochs), epochs, 2, .07, .04)
    supcon = SupConLoss()
    half_cfg = {**cfg, 'batch_size': max(2, a['batch_size']//2)}
    labelled_loader = ag_loader(initial, half_cfg, classes, train=True)
    unlabelled_loader = ag_loader(pool, half_cfg, classes, train=True, label_free=True)
    history = []
    for epoch in range(epochs):
        model.train()
        li, ui = iter(labelled_loader), iter(unlabelled_loader)
        steps = max(len(labelled_loader), len(unlabelled_loader))
        losses = []
        for _ in range(steps):
            try:
                lx, ly, _ = next(li)
            except StopIteration:
                li = iter(labelled_loader); lx, ly, _ = next(li)
            try:
                ux = next(ui)
            except StopIteration:
                ui = iter(unlabelled_loader); ux = next(ui)
            ly = ly.to(cfg['device'])
            views = [torch.cat([lx[v], ux[v]]).to(cfg['device']) for v in range(2)]
            _, proj, out = model(torch.cat(views))
            n_l = len(ly)
            log1, log2 = out.chunk(2)
            cls_loss = F.cross_entropy(torch.cat([log1[:n_l], log2[:n_l]])/.1, ly.repeat(2))
            cluster_loss = distill(out, out.detach(), epoch)
            avg = (out/.1).softmax(1).mean(0)
            cluster_loss += a['memax_weight'] * (math.log(len(classes)) + (avg * avg.clamp_min(1e-8).log()).sum())
            contrastive_logits, contrastive_targets = info_nce_logits(proj, device=cfg['device'])
            contrastive = F.cross_entropy(contrastive_logits, contrastive_targets)
            p1, p2 = proj.chunk(2)
            supervised = supcon(F.normalize(torch.stack([p1[:n_l], p2[:n_l]], 1), dim=-1), labels=ly)
            loss = (1-a['sup_weight'])*(cluster_loss+contrastive) + a['sup_weight']*(cls_loss+supervised)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite ActiveGCD loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ema_update(teacher, model, a['ema_decay'])
            losses.append(loss.item())
        mapping = None
        if len(queried):
            verified = pd.concat([initial, queried], ignore_index=True)
            slots, labels = predict(teacher, verified, cfg, classes)
            mapping = label_mapping(slots.argmax(1).numpy(), labels, len(classes))
            model.train()
            for views, labels, _ in ag_loader(queried, {**cfg, 'batch_size': a['al_batch_size']}, classes, train=True):
                mapped = torch.tensor([mapping[int(i)] for i in labels], device=cfg['device'])
                _, proj, out = model(torch.cat(views).to(cfg['device']))
                cls = F.cross_entropy(out/.1, mapped.repeat(2))
                p1, p2 = proj.chunk(2)
                con = supcon(F.normalize(torch.stack([p1, p2], 1), dim=-1), labels=labels.to(cfg['device']))
                loss = a['sup_weight'] * (cls+con)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                ema_update(teacher, model, a['ema_decay'])
        scheduler.step()
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses))})
        pd.DataFrame(history).to_csv(Path(output) / 'history.csv', index=False)
        atomic_save({'model': model.state_dict(), 'teacher': teacher.state_dict(), 'optimizer': opt.state_dict(),
                     'epoch': epoch+1, 'mapping': mapping}, Path(output) / 'last.pt')
        print(f'ActiveGCD {Path(output).name} epoch {epoch+1}/{epochs} loss={history[-1]["loss"]:.4f}', flush=True)


def evaluate(model, test, verified, cfg, classes, n_old, directory):
    # No operational label mapping is computed using test labels.
    known_logits, known_y = predict(model, verified, cfg, classes)
    mapping = label_mapping(known_logits.argmax(1).numpy(), known_y, len(classes))
    inverse = {v: k for k, v in mapping.items()}
    logits, y = predict(model, test, cfg, classes)
    slots = logits.argmax(1).numpy()
    semantic = np.array([inverse.get(int(i), -1) for i in slots])
    old = y < n_old
    # Permutation-invariant clustering metrics are OFFLINE evaluation only.
    oracle_mapping = label_mapping(slots, y, len(classes))
    oracle_inverse = {v: k for k, v in oracle_mapping.items()}
    aligned = np.array([oracle_inverse.get(int(i), -1) for i in slots])
    def acc(mask, pred):
        return float((pred[mask] == y[mask]).mean()) if mask.any() else None
    result = {'semantic_accuracy': float((semantic == y).mean()), 'old_accuracy': acc(old, semantic),
              'novel_semantic_accuracy': acc(~old, semantic), 'offline_cluster_accuracy': float((aligned == y).mean()),
              'offline_nmi': float(normalized_mutual_info_score(y, slots)),
              'offline_ari': float(adjusted_rand_score(y, slots)),
              'observed_novel_categories': len(set(known_y[known_y >= n_old])),
              'test_mapping_used_for_training': False}
    directory = Path(directory)
    out = test[['id', 'path', 'label']].copy().reset_index(drop=True)
    out['slot'] = slots
    out['prediction'] = [classes[i] if i >= 0 else f'novel_slot_{slot}' for i, slot in zip(semantic, slots)]
    out['confidence'] = logits.softmax(1).max(1).values.numpy()
    out['correct'] = semantic == y
    out.to_csv(directory / 'predictions.csv', index=False)
    dump_json(directory / 'metrics.json', {'activegcd': result})
    dump_json(directory / 'verified_label_mapping.json', {classes[k]: v for k, v in mapping.items()})
    return result


def run(cfg, seed, strategy='adaptive', resume=False, backbone=None):
    if resume:
        raise ValueError('ActiveGCD round restart is not implemented: use a fresh output root. Completed phases/checkpoints are retained.')
    seed_all(seed)
    a = cfg['activegcd']
    classes = a['known'] + a['novel']
    if len(set(classes)) != len(classes):
        raise ValueError('Known/novel class lists must be disjoint')
    manifest = read_manifest(cfg['manifest'])
    if set(classes) - set(manifest.label):
        raise ValueError('Configured ActiveGCD categories are missing from the manifest')
    train = manifest[(manifest.split == 'train') & manifest.label.isin(classes)].copy()
    known_train = train[train.label.isin(a['known'])]
    initial = subset(known_train, a['labelled_known_fraction'], seed)
    pool = train[~train.id.isin(initial.id)].reset_index(drop=True)
    test = manifest[(manifest.split == 'test') & manifest.label.isin(classes)]
    if not len(pool) or not len(initial):
        raise ValueError('ActiveGCD requires nonempty initial labelled and unlabelled discovery pools')
    base = Path(cfg['output']) / 'exp5b' / f'{strategy}_seed{seed}'
    if base.exists():
        raise FileExistsError(base)
    base.mkdir(parents=True)
    provenance(cfg, base)
    initial.to_csv(base / 'initial_labelled.csv', index=False)
    pool[['id', 'path', 'group', 'split']].to_csv(base / 'discovery_pool_without_labels.csv', index=False)
    model = build_model(cfg, len(classes), backbone).to(cfg['device'])
    # Legacy weight_norm heads cannot safely deepcopy on current PyTorch.
    teacher = build_model(cfg, len(classes), deepcopy(model[0])).to(cfg['device'])
    teacher.load_state_dict(model.state_dict()); teacher.requires_grad_(False).eval()
    queried = train.iloc[:0].copy()
    phase = base / 'round00'; phase.mkdir()
    train_phase(model, teacher, initial, pool, queried, cfg, classes, a['base_epochs'], phase)
    rows = [{'round': 0, 'annotations': 0, **evaluate(teacher, test, initial, cfg, classes, len(a['known']), phase)}]
    available = np.arange(len(pool))
    rng = np.random.default_rng(seed)
    for rd in range(1, a['rounds']+1):
        if not len(available):
            break
        current = pool.iloc[available].reset_index(drop=True)
        if strategy == 'random':
            local = rng.choice(len(current), size=min(a['queries'], len(current)), replace=False)
            fallback = 0
        elif strategy == 'adaptive':
            logits, _ = predict(teacher, current, cfg, classes)
            local, fallback = adaptive_query((logits/.1).softmax(1).numpy(), a['queries'], len(a['known']), rd, a['adaptive_round'])
        else:
            raise ValueError(strategy)
        # Oracle reveal occurs ONLY AFTER query selection. Selection never receives labels.
        selected = current.iloc[local].copy()
        queried = pd.concat([queried, selected], ignore_index=True)
        available = np.setdiff1d(available, available[local])
        phase = base / f'round{rd:02d}'; phase.mkdir()
        selected.to_csv(phase / 'queried_annotations.csv', index=False)
        dump_json(phase / 'selection.json', {'strategy': strategy, 'fallback_images': fallback,
            'note': 'Fallback uses remaining margin order when predicted-novel pool is smaller than query budget.'})
        train_phase(model, teacher, initial, pool, queried, cfg, classes, a['round_epochs'], phase)
        verified = pd.concat([initial, queried], ignore_index=True)
        rows.append({'round': rd, 'annotations': len(queried), **evaluate(teacher, test, verified, cfg, classes, len(a['known']), phase)})
        pd.DataFrame(rows).to_csv(base / 'round_summary.csv', index=False)
    pd.DataFrame(rows).to_csv(base / 'round_summary.csv', index=False)
    dump_json(base / 'protocol.json', {'total_class_count_supplied': len(classes), 'known': a['known'],
        'novel': a['novel'], 'excluded': sorted(set(manifest.label)-set(classes)),
        'upstream_commit': '66598129185020a8cb8842696f8ea660e59d5861', **parameter_counts(model)})
    return rows
