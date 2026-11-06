import numpy as np
import pandas as pd
import pytest

from carrepair.open_set import referral_outcomes, similarity_score, split_open_set


def test_unknown_never_enters_training_or_calibration():
    rows = [(split, label) for split in ['train', 'val', 'test'] for label in
            ['bumper_dent', 'glass_shatter', 'bumper_dent', 'glass_shatter', 'bumper_dent', 'unknown']]
    frame = pd.DataFrame(rows, columns=['split', 'label']).assign(group=range(len(rows)))
    train, val, test, classes = split_open_set(frame)
    assert classes == ['bumper_dent', 'glass_shatter']
    assert len(train) == len(val) == 5
    assert 'unknown' not in set(train.label) | set(val.label)
    assert set(test.label) == set(classes) | {'unknown'}
    # A mistyped or unexpected target label should never silently become a training class.
    with pytest.raises(ValueError):
        split_open_set(frame.replace({'glass_shatter': 'unknown'}))


def test_referrals_and_denominators_include_all_test_images():
    frame = pd.DataFrame({'label': ['a', 'b', 'a', 'unknown', 'unknown']})
    predicted = np.array(['a', 'a', 'a', 'b', 'b'])
    outcomes, report = referral_outcomes(frame, predicted, np.array([.6, .8, .2, .3, .9]), .6)
    assert outcomes.tolist() == ['known_correct_auto', 'known_wrong_auto', 'known_referred',
                                 'unknown_referred', 'unknown_accepted']
    assert report['known_correct_auto_rate'] == report['known_referral_rate'] == 1/3
    assert report['unknown_referral_recall'] == report['unknown_false_accept_rate'] == .5
    assert report['workload_rate'] == .4


def test_fifth_nearest_known_image_defines_gate_score():
    bank = np.array([[1, 0]] * 4 + [[0, 1], [-1, 0]])
    assert similarity_score(bank, np.array([[1, 0]])).item() == 0


def test_open_set_defaults_to_100_epochs_for_both_initialisations(tmp_path, monkeypatch):
    from carrepair import open_set
    from carrepair.common import read_config

    cfg = read_config('configs/default.yaml')
    assert cfg['open_set_epochs'] == 100
    original_epochs = cfg['epochs']
    cfg.update(output=str(tmp_path), seeds=[0])
    monkeypatch.setattr(open_set, 'read_manifest', lambda _: None)
    monkeypatch.setattr(open_set, 'split_open_set', lambda _: None)
    fits = []

    def capture(run_cfg, partitions, root, initialisation, seed, resume):
        fits.append((initialisation, run_cfg['epochs']))
        return pd.DataFrame([{'initialisation': initialisation, 'seed': seed, 'target_known_coverage': .9,
                              **dict.fromkeys(open_set.RATE_COLUMNS, 0.)}])

    monkeypatch.setattr(open_set, 'train_one', capture)
    open_set.run(cfg)
    assert fits == [('finetune', 100), ('scratch', 100)]
    assert cfg['epochs'] == original_epochs
