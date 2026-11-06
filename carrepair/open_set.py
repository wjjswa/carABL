"""Seven-class training and referral of unseen miscellaneous damage photos."""
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .common import dump_json, provenance, seed_all
from .data import loader, read_manifest
from .evaluation import extract, normalise
from .models import Classifier, yolo_features
from .training import fit_classifier

UNKNOWN = 'unknown'
COVERAGES = (0.90, 0.80)
NEIGHBOURS = 5
RATE_COLUMNS = [
    'known_accuracy_before_review', 'known_macro_f1_before_review',
    'known_correct_auto_rate', 'known_wrong_auto_rate', 'known_referral_rate',
    'unknown_referral_recall', 'unknown_false_accept_rate', 'workload_rate',
    'unfamiliarity_auroc',
]


def split_open_set(manifest):
    if manifest[['label', 'group', 'split']].isna().any().any():
        raise ValueError('Every image needs a label, split and duplicate group.')
    known = manifest.label != UNKNOWN
    train = manifest.loc[known & (manifest.split == 'train')].copy()
    val = manifest.loc[known & (manifest.split == 'val')].copy()
    test = manifest.loc[manifest.split == 'test'].copy()
    classes = sorted(train.label.unique())
    if len(classes) < 2 or len(train) < NEIGHBOURS:
        raise ValueError('Too few known training images or classes.')
    if set(val.label) != set(classes) or set(test.label) != set(classes) | {UNKNOWN}:
        raise ValueError('Expected known validation classes and known plus unknown test classes.')
    return train, val, test, classes


def similarity_score(bank, queries):
    """Cosine similarity to the fifth-nearest known training feature."""
    similarities = normalise(queries) @ normalise(bank).T
    return np.partition(similarities, -NEIGHBOURS, axis=1)[:, -NEIGHBOURS]


def referral_outcomes(frame, prediction, similarity, threshold):
    known = frame.label.to_numpy() != UNKNOWN
    correct = prediction == frame.label.to_numpy()
    review = similarity < threshold
    outcome = np.select(
        [review & known, review & ~known, ~known, correct],
        ['known_referred', 'unknown_referred', 'unknown_accepted', 'known_correct_auto'],
        default='known_wrong_auto',
    )
    n_known, n_unknown = int(known.sum()), int((~known).sum())
    counts = {name: int((outcome == name).sum()) for name in (
        'known_correct_auto', 'known_wrong_auto', 'known_referred', 'unknown_referred', 'unknown_accepted')}
    rates = {
        'known_correct_auto_rate': counts['known_correct_auto'] / n_known,
        'known_wrong_auto_rate': counts['known_wrong_auto'] / n_known,
        'known_referral_rate': counts['known_referred'] / n_known,
        'unknown_referral_recall': counts['unknown_referred'] / n_unknown,
        'unknown_false_accept_rate': counts['unknown_accepted'] / n_unknown,
        'workload_rate': float(review.mean()),
    }
    return outcome, {'test_known': n_known, 'test_unknown': n_unknown, **counts, **rates}


@torch.no_grad()
def evaluate(model, cfg, train, val, test, classes, directory):
    model.eval()
    device = cfg['device']
    bank, _ = extract(model.features, loader(train, cfg, classes), device)
    val_features, _ = extract(model.features, loader(val, cfg, classes), device)
    val_score = similarity_score(bank, val_features)
    # Lock both policies using known validation photos before scoring the mixed test set.
    cutoffs = [(coverage, float(np.quantile(val_score, 1 - coverage))) for coverage in COVERAGES]
    val.assign(similarity=val_score).to_csv(directory / 'calibration.csv', index=False)

    features, labels = extract(model.features, loader(test, cfg, classes + [UNKNOWN]), device)
    true = test.label.to_numpy()
    if not np.array_equal(np.asarray(classes + [UNKNOWN])[labels], true):
        raise RuntimeError('Test loader and manifest are in different orders.')
    probabilities = model.head(torch.as_tensor(features, device=device)).softmax(dim=1).cpu().numpy()
    prediction = np.asarray(classes)[probabilities.argmax(axis=1)]
    similarity = similarity_score(bank, features)
    known = true != UNKNOWN
    base = {
        'known_accuracy_before_review': float(np.mean(prediction[known] == true[known])),
        'known_macro_f1_before_review': float(f1_score(true[known], prediction[known], labels=classes,
                                                     average='macro', zero_division=0)),
        'unfamiliarity_auroc': float(roc_auc_score(~known, -similarity)),
    }
    photos = test[['id', 'path', 'label']].copy()
    photos['predicted_known_class'] = prediction
    photos['max_softmax'] = probabilities.max(axis=1)
    photos['similarity'] = similarity
    rows = []
    for coverage, threshold in cutoffs:
        outcome, result = referral_outcomes(test, prediction, similarity, threshold)
        photos[f'outcome_cov{round(100 * coverage)}'] = outcome
        rows.append({'initialisation': cfg['initialisation'], 'seed': cfg['run_seed'],
                     'target_known_coverage': coverage, 'similarity_threshold': threshold,
                     'validation_known_coverage': float(np.mean(val_score >= threshold)),
                     'train_known': len(train), 'val_known': len(val), **base, **result})
    photos.to_csv(directory / 'predictions.csv', index=False)
    return pd.DataFrame(rows)


def train_one(cfg, partitions, root, initialisation, seed, resume):
    train, val, test, classes = partitions
    run_cfg = {**cfg, 'initialisation': initialisation, 'run_seed': seed, 'open_set_protocol': 1,
               'target_known_coverage': list(COVERAGES), 'similarity_neighbours': NEIGHBOURS}
    directory = root / f'{initialisation}_seed{seed}'
    directory.mkdir(exist_ok=resume)
    provenance(run_cfg, directory)
    report = directory / 'review_metrics.csv'
    if resume and report.is_file():
        print(f'Reusing completed {directory.name}', flush=True)
        return pd.read_csv(report)

    train.to_csv(directory / 'training_subset.csv', index=False)
    dump_json(directory / 'classes.json', classes)
    seed_all(seed)
    features = yolo_features(cfg, 'full', scratch=initialisation == 'scratch')
    seed_all(seed)  # Same initial head and augmentation/shuffle RNG in the paired runs.
    model = Classifier(features, len(classes))
    print(f'{directory.name}: {len(train)} train, {len(val)} validation, {len(test)} test', flush=True)
    model = fit_classifier(model, run_cfg, train, val, classes, directory, resume=resume)
    rows = evaluate(model, run_cfg, train, val, test, classes, directory)
    # The final report is also the completion marker. Predictions and calibration precede it.
    tmp = report.with_suffix('.tmp')
    rows.to_csv(tmp, index=False)
    tmp.replace(report)
    return rows


def run(cfg, resume=False):
    cfg = dict(cfg)
    cfg['epochs'] = cfg.pop('open_set_epochs', 100)
    if not isinstance(cfg['epochs'], int) or cfg['epochs'] < 1:
        raise ValueError('open_set_epochs must be a positive integer.')
    partitions = split_open_set(read_manifest(cfg['manifest']))
    root = Path(cfg['output']) / 'open_set'
    root.mkdir(exist_ok=True, parents=True)
    with (root / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another job is writing {root}.') from exc
        fits = []
        for seed in cfg['seeds']:
            for init in ('finetune', 'scratch'):
                fits.append(train_one(cfg, partitions, root, init, seed, resume))
        results = pd.concat(fits, ignore_index=True)
        results.to_csv(root / 'per_seed.csv', index=False)
        summary = results.groupby(['initialisation', 'target_known_coverage'])[RATE_COLUMNS].agg(['mean', 'std'])
        summary.columns = ['_'.join(column) for column in summary.columns]
        summary['n_seeds'] = results.groupby(['initialisation', 'target_known_coverage']).seed.nunique()
        summary.to_csv(root / 'comparison.csv')
        print(summary[[f'{name}_mean' for name in ['known_accuracy_before_review', 'known_correct_auto_rate',
                                                  'unknown_referral_recall', 'workload_rate']]].to_string())
        print(f'Results: {root / "comparison.csv"}', flush=True)
