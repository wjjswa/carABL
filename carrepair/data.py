"""Immutable, grouped splits; nested training budgets; label-free SSL views."""
from pathlib import Path
import math
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .common import dump_json, file_hash


def open_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def dhash(path):
    a = np.asarray(open_image(path).convert('L').resize((9, 8)))
    return int(''.join('1' if x else '0' for x in (a[:, 1:] > a[:, :-1]).flat), 2)


def prepare(root, out, csv=None, path_column=None, label_column=None, group_column=None, seed=42, near_distance=2):
    root, out = Path(root).resolve(), Path(out)
    if (out / 'manifest.csv').exists():
        raise FileExistsError('Prepared split already exists. Use a NEW output directory to change the protocol.')
    if csv:
        frame = pd.read_csv(csv)
        path_column = path_column or next((x for x in ['image', 'path', 'filename', 'file', 'images'] if x in frame), None)
        label_column = label_column or next((x for x in ['classes', 'class', 'label', 'category'] if x in frame), None)
        if not path_column or not label_column:
            raise ValueError(f'CSV columns are {list(frame)}; specify --path-column and --label-column.')
        rows = [{'path': str(r[path_column]), 'label': str(r[label_column]),
                 'user_group': str(r[group_column]) if group_column and pd.notna(r[group_column]) else ''} for _, r in frame.iterrows()]
    else:
        rows = [{'path': str(p.relative_to(root)), 'label': p.parent.name, 'user_group': ''}
                for p in sorted(root.rglob('*')) if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    if not rows:
        raise ValueError('No images found')
    # File paths remain relative, allowing relocation to remote machines.
    for r in rows:
        p = (root / r['path']).resolve()
        r['path'] = str(p.relative_to(root))
        r['sha256'], r['dhash'] = file_hash(p), dhash(p)
    # Remove exact duplicates; contradictory labels need human resolution.
    frame = pd.DataFrame(rows)
    if frame.groupby('sha256').label.nunique().max() > 1:
        raise ValueError('Exact duplicate image has conflicting labels; resolve annotations before splitting.')
    original = len(frame)
    frame = frame.drop_duplicates('sha256').reset_index(drop=True)
    n = len(frame)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    hashes, user_groups = list(map(int, frame.dhash)), frame.user_group.tolist()
    for i in range(n):
        for j in range(i):
            same_user = bool(user_groups[i]) and user_groups[i] == user_groups[j]
            close = (hashes[i] ^ hashes[j]).bit_count() <= near_distance
            if same_user or close:
                parent[find(i)] = find(j)
    frame['group'] = [str(find(i)) for i in range(n)]
    # Group-stratified greedy assignment works also for mixed-label vehicle groups.
    counts = frame.label.value_counts().to_dict()
    totals = {s: {c: 0 for c in counts} for s in ['train', 'val', 'test']}
    ratios = {'train': .7, 'val': .15, 'test': .15}
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby('group', sort=True))
    rng.shuffle(groups)
    groups.sort(key=lambda x: len(x[1]), reverse=True)
    assignments = {}
    for group, part in groups:
        cls_counts = part.label.value_counts().to_dict()
        score = {s: sum((ratios[s] * counts[c] - totals[s][c]) / max(ratios[s] * counts[c], 1)
                        * num for c, num in cls_counts.items()) for s in ratios}
        s = max(score, key=score.get)
        assignments[group] = s
        for c, num in cls_counts.items():
            totals[s][c] += num
    frame['split'] = frame.group.map(assignments)
    for split in ratios:
        missing = set(counts) - set(frame[frame.split == split].label)
        if missing:
            raise ValueError(f'{split} lacks {missing}; insufficient independent groups. Inspect the dataset or supply better grouping.')
    frame['id'] = frame.sha256.str[:20]
    frame = frame[['id', 'path', 'label', 'group', 'split', 'sha256']]
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'manifest.csv', index=False)
    dump_json(out / 'audit.json', {'raw_count': original, 'deduplicated_count': len(frame), 'seed': seed,
        'near_duplicate_hamming_threshold': near_distance, 'counts': totals,
        'limitations': 'Perceptual hashing cannot identify every related vehicle photograph or CarDD overlap. Inspect groups and supply vehicle IDs where available.'})
    return frame


def read_manifest(path):
    f = pd.read_csv(path, dtype={'group': str})
    needed = {'id', 'path', 'label', 'group', 'split'}
    if not needed <= set(f):
        raise ValueError(f'Manifest must contain {needed}')
    if f.id.duplicated().any() or f.path.duplicated().any():
        raise ValueError('Manifest image IDs and paths must be unique')
    if f.groupby('group').split.nunique().max() > 1 or f.groupby('path').split.nunique().max() > 1:
        raise ValueError('Data leakage: group or image occurs in multiple splits')
    if not set(f.split) <= {'train', 'val', 'test'}:
        raise ValueError('Unknown split name')
    return f


def subset(frame, fraction, seed):
    if not 0 < fraction <= 1 or (frame.split != 'train').any():
        raise ValueError('Subsets must be drawn ONLY from training data with 0 < fraction <= 1')
    keep = []
    # Per-class deterministic permutation, shared between methods and nested across fractions.
    for label, part in frame.groupby('label', sort=True):
        groups = sorted(part.group.unique(), key=lambda g: hashlib.sha256(f'{seed}:{label}:{g}'.encode()).hexdigest())
        keep.extend(groups[:max(1, math.ceil(len(groups) * fraction))])
    return frame[frame.group.isin(set(keep))].copy().reset_index(drop=True)


def transforms(size, train=False, imagenet=False, scale=(.7, 1.0)):
    ops = [T.RandomResizedCrop(size, scale=scale), T.RandomHorizontalFlip()] if train else [T.Resize((size, size))]
    if train:
        ops += [T.RandomApply([T.ColorJitter(.2, .2, .2, .05)], p=.5)]
    ops += [T.ToTensor()]
    if imagenet:
        ops += [T.Normalize((.485, .456, .406), (.229, .224, .225))]
    return T.Compose(ops)


class Images(Dataset):
    def __init__(self, frame, root, classes, transform, label_free=False):
        self.frame, self.root = frame.reset_index(drop=True), Path(root)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform, self.label_free = transform, label_free

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        r = self.frame.iloc[i]
        x = self.transform(open_image(self.root / r.path))
        if self.label_free:
            return x  # The SSL training loop never receives labels.
        return x, self.class_to_idx[r.label], i


class MultiCrop:
    def __init__(self, cfg):
        d = cfg['dino']
        self.global_transform = transforms(cfg['image_size'], True, scale=tuple(d['global_scale']))
        self.local_transform = transforms(d['local_size'], True, scale=tuple(d['local_scale']))
        self.n_local = d['local_crops']

    def __call__(self, im):
        return [self.global_transform(im) for _ in range(2)] + [self.local_transform(im) for _ in range(self.n_local)]


def loader(frame, cfg, classes, train=False, transform=None, label_free=False):
    ds = Images(frame, cfg['data_root'], classes, transform or transforms(cfg['image_size'], train), label_free)
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=train,
                      num_workers=cfg['workers'], pin_memory=str(cfg['device']).startswith('cuda'), drop_last=False)
