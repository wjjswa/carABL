"""Experiments E6 (AGE-inspired) and E7 (DP-BOA paper-based) on carABL.

This is a documented adaptation, NOT a faithful reproduction of the authors'
end-to-end training. A shared frozen DINO ViT-B/16 encoder plus a known-only
linear probe permits controlled tests of the online discovery algorithms.

Strict prequential protocol: the stream contains held-out test images. Every
prediction is made BEFORE the corresponding feature is used for an unsupervised
online update. Ground-truth stream labels are accessed only by evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_t
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .common import dump_json, file_hash, provenance, read_config, seed_all
from .data import loader, read_manifest, subset


def partition_open_world(manifest, known, exclude, fraction, seed):
    """Return ONLY known-class training/validation data and a held-out stream.

    The discovery methods receive support/validation features and labels, and
    stream features alone. Novel test labels never enter a training procedure.
    """
    known, exclude = list(known), set(exclude)
    if len(known) < 2 or len(set(known)) != len(known):
        raise ValueError('At least two distinct known categories are required')
    if set(known) & exclude:
        raise ValueError('Known and excluded classes overlap')
    if not 0 < fraction <= 1:
        raise ValueError('support_fraction must be in (0, 1]')
    if set(known) - set(manifest.label):
        raise ValueError('Known categories missing from the manifest')
    tr = manifest[(manifest.split == 'train') & manifest.label.isin(known)].copy()
    support = subset(tr, fraction, seed)
    val = manifest[(manifest.split == 'val') & manifest.label.isin(known)].copy()
    stream = manifest[(manifest.split == 'test') & ~manifest.label.isin(exclude)].copy()
    if not len(support) or not len(val) or not len(stream):
        raise ValueError('Empty support, known validation or test stream')
    if set(support.label) != set(known) or set(val.label) != set(known):
        raise ValueError('Every known class needs support and validation examples')
    if not (~stream.label.isin(known)).any():
        raise ValueError('The held-out stream contains no new category')
    # Both image hashes and near-duplicate group splits are checked by read_manifest.
    if (set(support.id) & set(stream.id)) or (set(val.id) & set(stream.id)):
        raise ValueError('Train/validation and test image overlap')
    return support.reset_index(drop=True), val.reset_index(drop=True), stream.reset_index(drop=True)


def _regular_cov(features, floor=1e-5):
    x = np.asarray(features, dtype=np.float64)
    d = x.shape[1]
    if len(x) <= 2:
        return np.eye(d) * max(floor, 0.1)
    cov = LedoitWolf().fit(x).covariance_
    return (cov + cov.T) / 2 + floor * np.eye(d)


def _gaussian_logpdf(x, mean, cov):
    d = len(x)
    cov = np.asarray(cov, dtype=np.float64)
    for retry in range(4):
        try:
            chol = np.linalg.cholesky(cov + 10**retry * 1e-8 * np.eye(d))
            z = np.linalg.solve(chol, x - mean)
            return float(-.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + z @ z))
        except np.linalg.LinAlgError:
            pass
    raise np.linalg.LinAlgError('Could not regularize category covariance')


class AGEOnline:
    """Adapted AGE discovery: soft OSR threshold + Gaussian category expansion.

    Mirrors the author's overall decision rule: known predictions pass through
    the supervised classifier, whereas rejected examples are assigned to an
    existing novel Gaussian or cause a new Gaussian to be initialized if their
    closest density model belongs to a known category.
    """

    def __init__(self, support, support_y, val_prob, val_y, count_known,
                 beta=.5, std_scale=.8, covariance_prior=2., ridge=1e-5,
                 window=32):
        self.k = int(count_known)
        self.ridge = float(ridge)
        self.s = float(covariance_prior)
        self.window = int(window)
        if self.window < 2 or self.s <= 0:
            raise ValueError('AGE window must be >=2 and covariance_prior positive')
        conf = np.max(val_prob, axis=1)
        per_class = []
        for c in range(self.k):
            sample = conf[val_y == c]
            per_class.append(float(sample.mean() - std_scale * sample.std()) if len(sample) else float(conf.mean()))
        global_threshold = float(np.mean(per_class))
        self.thresholds = np.clip(beta * np.asarray(per_class) + (1-beta) * global_threshold, 0, 1)
        self.global_cov = _regular_cov(support, ridge)
        self.centers = []
        self.covariances = []
        self.counts = []
        self.recent = []
        for c in range(self.k):
            part = support[support_y == c]
            cov = _regular_cov(part, ridge) if len(part) > 2 else self.global_cov.copy()
            n = len(part)
            cov = (n * cov + self.s * self.global_cov) / (n + self.s)
            self.centers.append(part.mean(0).copy())
            self.covariances.append(cov + ridge*np.eye(support.shape[1]))
            self.counts.append(n)
            self.recent.append([])
        self.births = 0

    def predict_update(self, x, known_prob):
        c = int(np.argmax(known_prob))
        if known_prob[c] >= self.thresholds[c]:
            return c, 'known_classifier', float(known_prob[c])
        likelihood = np.array([_gaussian_logpdf(x, mu, cov)
                               for mu, cov in zip(self.centers, self.covariances)])
        winner = int(np.argmax(likelihood))
        score = float(likelihood[winner])
        if winner < self.k:
            winner = len(self.centers)
            self.centers.append(np.asarray(x, dtype=np.float64).copy())
            self.covariances.append(self.global_cov.copy())
            self.counts.append(1)
            self.recent.append([np.asarray(x).copy()])
            self.births += 1
            return winner, 'new_cluster', score
        # Novel groups are updated without using the unknown class label.
        self.counts[winner] += 1
        n = self.counts[winner]
        self.centers[winner] += (x - self.centers[winner]) / n
        memory = self.recent[winner]
        memory.append(np.asarray(x).copy())
        if len(memory) > self.window:
            memory.pop(0)
        local = _regular_cov(np.stack(memory), self.ridge) if len(memory) > 2 else self.global_cov
        cov = (n * local + self.s * self.global_cov) / (n + self.s)
        self.covariances[winner] = cov + self.ridge * np.eye(len(x))
        return winner, 'existing_novel_cluster', score

    @property
    def discovered(self):
        return len(self.centers) - self.k


@dataclass
class NIWCategory:
    """Online Welford moments, sufficient for NIW posterior updates."""
    count: int
    mean: np.ndarray
    scatter: np.ndarray

    @classmethod
    def empty(cls, d):
        return cls(0, np.zeros(d, dtype=np.float64), np.zeros((d, d), dtype=np.float64))

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        old_mean = self.mean.copy()
        self.count += 1
        self.mean += (x - old_mean) / self.count
        self.scatter += np.outer(x - old_mean, x - self.mean)


class DPBOAOnline:
    """NIW posterior-predictive birth-or-assign, adapted from ECCV 2026 DP-BOA.

    Official author source was not verified available when this code was written.
    Hyperparameters are explicit, not claimed to reproduce the paper's tuning.
    """

    def __init__(self, support, support_y, count_known, alpha=10.,
                 kappa0=.25, nu_extra=4., ridge=1e-4):
        self.k = int(count_known)
        self.d = support.shape[1]
        self.alpha = float(alpha)
        self.kappa0 = float(kappa0)
        self.nu0 = self.d + 1 + float(nu_extra)
        self.ridge = float(ridge)
        if min(self.alpha, self.kappa0, nu_extra, self.ridge) <= 0:
            raise ValueError('All DP-BOA prior parameters must be positive')
        self.mu0 = support.mean(0).copy()
        pooled_cov = _regular_cov(support, self.ridge)
        # NIW E[Sigma] = Psi0 / (nu0 - d - 1).
        self.psi0 = pooled_cov * (self.nu0 - self.d - 1)
        self.states = [NIWCategory.empty(self.d) for _ in range(self.k)]
        for feat, c in zip(support, support_y):
            self.states[int(c)].update(feat)
        self.births = 0

    def _predictive_logpdf(self, x, state):
        n = state.count
        kappa = self.kappa0 + n
        nu = self.nu0 + n
        if n:
            loc = (self.kappa0 * self.mu0 + n * state.mean) / kappa
            delta = state.mean - self.mu0
            psi = self.psi0 + state.scatter + (
                (self.kappa0 * n / kappa) * np.outer(delta, delta))
        else:
            loc = self.mu0
            psi = self.psi0
        df = nu - self.d + 1
        scale = ((kappa + 1) / (kappa * df)) * psi + self.ridge * np.eye(self.d)
        return float(multivariate_t.logpdf(x, loc=loc, shape=scale, df=df))

    def predict_update(self, x, known_prob=None):
        del known_prob  # DP-BOA relies on posterior-predictive evidence, not the OSR head.
        scores = [math.log(s.count) + self._predictive_logpdf(x, s)
                  if s.count > 0 else float('-inf') for s in self.states]
        birth_score = math.log(self.alpha) + self._predictive_logpdf(x, NIWCategory.empty(self.d))
        winner = int(np.argmax(scores))
        if birth_score > scores[winner]:
            winner = len(self.states)
            self.states.append(NIWCategory.empty(self.d))
            event = 'new_cluster'
            self.births += 1
        else:
            event = 'known_category' if winner < self.k else 'existing_novel_cluster'
        self.states[winner].update(x)
        return winner, event, float(birth_score - max(scores))

    @property
    def discovered(self):
        return len(self.states) - self.k


def offline_hungarian(true_labels, predicted):
    """Offline-only test-label matching; forbidden for operational prediction."""
    truth = np.asarray(true_labels)
    pred = np.asarray(predicted)
    labels = list(pd.unique(truth))
    groups = list(np.unique(pred))
    counts = np.zeros((len(labels), len(groups)), dtype=np.int64)
    for i, label in enumerate(labels):
        for j, cluster in enumerate(groups):
            counts[i, j] = np.count_nonzero((truth == label) & (pred == cluster))
    rows, columns = linear_sum_assignment(-counts)
    assignment = {int(groups[j]): labels[i] for i, j in zip(rows, columns)}
    aligned = np.array([assignment.get(int(p), '__unmapped__') for p in pred], dtype=object)
    return aligned, assignment


def evaluate_stream(stream, order, features, predictions, events, scores, known):
    """Only this function reads held-out stream labels."""
    stream = stream.iloc[order].reset_index(drop=True)
    truth = stream.label.astype(str).to_numpy()
    pred = np.asarray(predictions, dtype=int)
    k = len(known)
    old = np.isin(truth, known)
    novel = ~old
    pred_names = np.array([known[p] if p < k else 'novel_cluster_%03d' % (p-k+1)
                           for p in pred], dtype=object)
    aligned, mapping = offline_hungarian(truth, pred)
    true_new = int(len(set(truth[novel])))
    created = int(len(set(pred[pred >= k])))
    result = {
        'test_images': len(stream),
        'test_known': int(old.sum()),
        'test_novel': int(novel.sum()),
        'known_semantic_accuracy': float(np.mean(pred_names[old] == truth[old])),
        'novel_detection_recall': float(np.mean(pred[novel] >= k)),
        'known_false_novel_rate': float(np.mean(pred[old] >= k)),
        'offline_hungarian_all_accuracy': float(np.mean(aligned == truth)),
        'offline_hungarian_novel_accuracy': float(np.mean(aligned[novel] == truth[novel])),
        'nmi': float(normalized_mutual_info_score(truth, pred)),
        'ari': float(adjusted_rand_score(truth, pred)),
        'estimated_novel_class_count': created,
        'true_novel_class_count_evaluation_only': true_new,
        'novel_class_count_abs_error': abs(created - true_new),
        'test_labels_used_to_train_or_update': False,
        'test_labels_used_for_offline_alignment_only': True,
    }
    rows = []
    for label in sorted(set(truth)):
        mask = truth == label
        rows.append({
            'category_evaluation_only': label,
            'status': 'known' if label in known else 'novel',
            'support': int(mask.sum()),
            'semantic_accuracy_if_known': float(np.mean(pred_names[mask] == truth[mask])) if label in known else None,
            'offline_hungarian_recall': float(np.mean(aligned[mask] == truth[mask])),
            'predicted_as_novel_rate': float(np.mean(pred[mask] >= k)),
        })
    per_image = stream[['id', 'path', 'label']].copy()
    per_image.insert(0, 'stream_step', np.arange(1, len(stream)+1))
    per_image['predicted_slot'] = pred
    per_image['prediction_without_test_labels'] = pred_names
    per_image['event'] = events
    per_image['decision_score'] = scores
    per_image['offline_hungarian_prediction_evaluation_only'] = aligned
    return result, pd.DataFrame(rows), per_image, mapping


def _feature_batch(frame, cfg, backbone):
    import torch
    items = []
    with torch.no_grad():
        for batch in loader(frame, {**cfg, 'image_size': 224}, [], train=False,
                            transform=None, label_free=True):
            # loader's default transform has no ImageNet normalization; use an
            # explicit normalized loader below instead.
            raise AssertionError('Use normalized image features only')


def _extract_features(frame, cfg, backbone):
    import torch
    from .data import transforms
    all_features = []
    local_cfg = {**cfg, 'image_size': 224}
    dataloader = loader(frame, local_cfg, [], train=False,
                        transform=transforms(224, imagenet=True), label_free=True)
    with torch.inference_mode():
        for images in dataloader:
            output = backbone(images.to(cfg['device']))
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim != 2:
                raise RuntimeError('DINO encoder should produce [batch, embedding_dim]')
            all_features.append(output.detach().cpu().numpy())
    return np.concatenate(all_features).astype(np.float64)


def load_or_extract_features(cfg, seed, support, val, stream, dino_ref):
    import torch
    root = Path(cfg['output']) / 'open_world_feature_cache'
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ('seed%d.npz' % seed)
    meta = {
        'manifest_sha256': file_hash(cfg['manifest']),
        'dino_ref': dino_ref,
        'ids': [list(map(str, f.id)) for f in (support, val, stream)],
        'image_size': 224,
        'imagenet_normalize': True,
    }
    with (root / ('seed%d.lock' % seed)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                if json.loads(str(z['meta'])) == meta:
                    return z['support'], z['val'], z['stream']
            raise RuntimeError('Feature cache mismatch: remove or rename %s before rerun' % cache)
        print('Extracting DINO ViT-B/16 features; first run may download torch.hub weights.', flush=True)
        backbone = torch.hub.load(dino_ref, 'dino_vitb16', trust_repo=True).to(cfg['device']).eval()
        backbone.requires_grad_(False)
        arrays = [_extract_features(frame, cfg, backbone) for frame in (support, val, stream)]
        temp = cache.with_suffix('.npz.tmp')
        with temp.open('wb') as fd:
            np.savez_compressed(fd, meta=json.dumps(meta), support=arrays[0],
                                val=arrays[1], stream=arrays[2])
        temp.replace(cache)
        return arrays


def run(cfg, method, seed):
    method = str(method).lower()
    if method not in ('e6', 'e7'):
        raise ValueError('Choose e6 or e7')
    seed_all(seed)
    settings = cfg['open_world']
    known = list(settings['known'])
    support, val, stream = partition_open_world(
        read_manifest(cfg['manifest']), known, settings.get('exclude', ['unknown']),
        float(settings.get('support_fraction', .8)), seed)
    dino_ref = settings.get('dino_repo', cfg['activegcd']['dino_repo'])
    target = Path(cfg['output']) / ('exp6' if method == 'e6' else 'exp7') / ('seed%d' % seed)
    if target.exists():
        raise FileExistsError('Output already exists: %s. Do not overwrite an earlier run.' % target)
    target.mkdir(parents=True)
    provenance({**cfg, 'open_world_method': method, 'run_seed': seed}, target)
    support[['id', 'path', 'label', 'group', 'split']].to_csv(target / 'known_support.csv', index=False)
    val[['id', 'path', 'label', 'group', 'split']].to_csv(target / 'known_validation.csv', index=False)
    # Withheld test labels are intentionally absent from the streaming input file.
    stream[['id', 'path', 'group', 'split']].to_csv(target / 'stream_without_labels.csv', index=False)
    features, val_features, test_features = load_or_extract_features(
        cfg, seed, support, val, stream, dino_ref)
    scaler = PCA(n_components=min(int(settings.get('pca_dim', 24)),
                                  features.shape[1], len(features)-1),
                 random_state=seed)
    support_z = scaler.fit_transform(features)
    validation_z = scaler.transform(val_features)
    stream_z = scaler.transform(test_features)
    classes = {name: index for index, name in enumerate(known)}
    support_y = support.label.map(classes).to_numpy(dtype=int)
    val_y = val.label.map(classes).to_numpy(dtype=int)
    head = LogisticRegression(max_iter=2000, random_state=seed)
    head.fit(features, support_y)
    # Classifier calibration uses KNOWN validation photos only.
    val_prob = head.predict_proba(val_features)
    if not np.array_equal(head.classes_, np.arange(len(known))):
        raise RuntimeError('Logistic regression class order is inconsistent')
    if method == 'e6':
        p = settings.get('age', {})
        online = AGEOnline(
            support_z, support_y, val_prob, val_y, len(known),
            beta=float(p.get('beta', .5)), std_scale=float(p.get('std_scale', .8)),
            covariance_prior=float(p.get('covariance_prior', 2.)),
            ridge=float(p.get('ridge', 1e-5)),
            window=int(p.get('window', 32)))
    else:
        p = settings.get('dpboa', {})
        online = DPBOAOnline(
            support_z, support_y, len(known),
            alpha=float(p.get('alpha', 10.)),
            kappa0=float(p.get('kappa0', .25)),
            nu_extra=float(p.get('nu_extra', 4.)),
            ridge=float(p.get('ridge', 1e-4)))
    order = np.random.default_rng(seed).permutation(len(stream))
    predictions, events, scores = [], [], []
    for index in order:
        z = stream_z[index]
        prob = head.predict_proba(features.reshape(1, -1))[0] if False else head.predict_proba(test_features[index:index+1])[0]
        # No ground-truth field is passed to either online method.
        prediction, event, score = online.predict_update(z, prob)
        predictions.append(int(prediction))
        events.append(event)
        scores.append(float(score))
    result, categories, output, offline_map = evaluate_stream(
        stream, order, test_features, predictions, events, scores, known)
    result.update({'method': method, 'seed': seed, 'known_class_count_given': len(known),
                   'new_class_count_given_to_algorithm': None,
                   'discovered_cluster_objects': int(online.discovered),
                   'encoder': dino_ref,
                   'feature_backbone_frozen': True,
                   'classifier_training': 'known-only logistic regression probe',
                   'pca_components': int(scaler.n_components_)})
    output.to_csv(target / 'stream_predictions.csv', index=False)
    categories.to_csv(target / 'per_category.csv', index=False)
    dump_json(target / 'metrics.json', result)
    dump_json(target / 'offline_mapping_EVALUATION_ONLY.json',
              {str(k): str(v) for k, v in offline_map.items()})
    dump_json(target / 'protocol.json', {
        'method': method, 'seed': seed, 'known': known,
        'excluded_from_stream': settings.get('exclude', ['unknown']),
        'initial_support_images': len(support), 'validation_known_images': len(val),
        'stream_test_images': len(stream), 'stream_shuffle_seed': seed,
        'new_class_count_given_to_algorithm': None,
        'training_images_from_novel_categories': 0,
        'test_truth_used_during_online_decision': False,
        'adaptation_not_faithful_official_reproduction': True,
        'settings': settings,
    })
    print(json.dumps(result, indent=2))
    print('Results saved to', target, flush=True)
    return result


def aggregate(cfg):
    root = Path(cfg['output'])
    all_rows = []
    for method, folder in [('e6', 'exp6'), ('e7', 'exp7')]:
        for path in sorted((root / folder).glob('seed*/metrics.json')):
            row = json.loads(path.read_text())
            all_rows.append(row)
    if not all_rows:
        raise FileNotFoundError('No E6/E7 metrics.json found')
    frame = pd.DataFrame(all_rows)
    destination = root / 'open_world_comparison'
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / 'per_seed.csv', index=False)
    scores = ['known_semantic_accuracy', 'novel_detection_recall',
              'known_false_novel_rate', 'offline_hungarian_all_accuracy',
              'offline_hungarian_novel_accuracy', 'nmi', 'ari',
              'estimated_novel_class_count', 'novel_class_count_abs_error']
    summary = frame.groupby('method')[scores].agg(['mean','std'])
    summary.columns = [a + '_' + b for a,b in summary.columns]
    summary['n_seeds'] = frame.groupby('method').seed.nunique()
    summary.to_csv(destination / 'mean_std.csv')
    print(summary.to_string())
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['e6','e7'], default='e6')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--device')
    parser.add_argument('--output')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.device:
        cfg['device'] = args.device
    if args.output:
        cfg['output'] = args.output
    if args.aggregate:
        return aggregate(cfg)
    return run(cfg, args.method, args.seed)


if __name__ == '__main__':
    main()
