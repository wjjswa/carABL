"""CPU-only deterministic checks for E6/E7 no-oracle online discovery."""
import numpy as np
import pandas as pd
import pytest
from carrepair.open_world import (AGEOnline, DPBOAOnline, partition_open_world,
                                  offline_hungarian, evaluate_stream)

def _manifest():
    rows=[]
    for split in ("train","val","test"):
        for name in ("known_a","known_b","future_c"):
            for i in range(8):
                if split != "test" and name == "future_c":
                    # Novel training/validation examples exist but are forbidden.
                    pass
                rows.append({"id":f"{split}_{name}_{i}", "path":f"{split}/{name}/{i}.jpg",
                             "group":f"{split}_{name}_{i}", "split":split, "label":name})
    return pd.DataFrame(rows)

def test_partition_blocks_future_categories():
    support, val, stream = partition_open_world(_manifest(), ["known_a","known_b"], [], .8, 0)
    assert set(support.label) == {"known_a","known_b"}
    assert set(val.label) == {"known_a","known_b"}
    assert set(stream.label) == {"known_a","known_b","future_c"}
    assert not set(support.id) & set(stream.id)
    assert not set(val.id) & set(stream.id)
    assert len(stream) == 24

def test_partition_requires_novel_stream():
    with pytest.raises(ValueError, match="no new category"):
        partition_open_world(_manifest(), ["known_a","known_b"], ["future_c"], 1.0, 0)


def test_partition_requires_known_test_examples():
    frame = _manifest()
    frame = frame[(frame.split != 'test') | (frame.label == 'future_c')]
    with pytest.raises(ValueError, match='no known category'):
        partition_open_world(frame, ['known_a', 'known_b'], [], 1.0, 0)

def _gauss_inputs():
    rng = np.random.default_rng(41)
    x = np.r_[rng.normal([-3.,0.],.20,(25,2)),rng.normal([3.,0.],.20,(25,2))]
    y = np.array([0]*25 + [1]*25)
    probs = np.r_[np.tile([.94,.06],(12,1)),np.tile([.06,.94],(12,1))]
    labels = np.array([0]*12+[1]*12)
    return x,y,probs,labels

def test_age_creates_clusters_without_ground_truth():
    x,y,p,labels = _gauss_inputs()
    model=AGEOnline(x,y,p,labels,2,std_scale=0.1)
    # Force the sample into the rejected/novel branch via valid low confidence.
    idx,event,_=model.predict_update(np.array([12.,12.]),np.array([.51,.49]))
    assert idx==2 and event=="new_cluster"
    assert model.discovered==1
    idx2,event2,_=model.predict_update(np.array([12.01,12.02]),np.array([.51,.49]))
    assert idx2 >= 2
    assert event2 in ("existing_novel_cluster","new_cluster")

def test_dpboa_birth_and_existing_assignment():
    x,y,_,_= _gauss_inputs()
    model=DPBOAOnline(x,y,2,alpha=100.,kappa0=.25,nu_extra=4.,ridge=.001)
    slot,event,_=model.predict_update(np.array([25.,25.]))
    assert slot>=2 and event=="new_cluster"
    assert model.discovered>=1
    slot2,event2,_=model.predict_update(np.array([25.01,25.01]))
    assert slot2>=2
    assert event2 in ("new_cluster","existing_novel_cluster")

def test_evaluation_only_offline_hungarian():
    pred=np.array([4,4,8,8])
    truth=np.array(["known_a","known_a","future_c","future_c"])
    aligned, mapping=offline_hungarian(truth,pred)
    assert np.array_equal(aligned,truth)
    assert mapping[4]=="known_a" and mapping[8]=="future_c"

def test_prequential_metrics_do_not_set_novel_semantics():
    frame=pd.DataFrame([{"id":"a","path":"a.jpg","label":"known_a"},
                        {"id":"b","path":"b.jpg","label":"future_c"}])
    metrics, categories, records, mapping = evaluate_stream(frame,np.array([0,1]),
        [0,2],["known_classifier","new_cluster"],[.9,-12.],["known_a","known_b"])
    assert metrics["known_semantic_accuracy"]==1.0
    assert metrics["novel_detection_recall"]==1.0
    assert metrics["estimated_novel_class_count"] == 1
    assert metrics["true_novel_class_count_evaluation_only"] == 1
    assert metrics["test_labels_used_to_train_or_update"] is False
    assert records.loc[1,"prediction_without_test_labels"]=="novel_cluster_001"
    assert "future_c" not in records.loc[1,"prediction_without_test_labels"]


def test_niw_moments_and_predecision_score():
    import math
    from carrepair.open_world import NIWCategory
    x, y, _, _ = _gauss_inputs()
    state = NIWCategory.empty(2)
    for point in x:
        state.update(point)
    assert np.allclose(state.mean, x.mean(0))
    assert np.allclose(state.scatter, (x-x.mean(0)).T @ (x-x.mean(0)))
    model = DPBOAOnline(x, y, 2)
    point = x[0]
    counts = [s.count for s in model.states]
    scores = [math.log(s.count) + model._predictive_logpdf(point, s) for s in model.states]
    birth = math.log(model.alpha) + model._predictive_logpdf(point, NIWCategory.empty(2))
    slot, event, margin = model.predict_update(point)
    assert slot == np.argmax(scores) and event == 'known_category'
    assert margin == pytest.approx(birth - max(scores))
    assert model.states[slot].count == counts[slot] + 1


def test_synthetic_image_pipeline_cache_and_complete_comparison(tmp_path, monkeypatch):
    """Exercise real image loading/PCA/head/online scoring; no pretrained download."""
    import json
    import torch
    from PIL import Image
    from carrepair.common import read_config
    from carrepair import open_world
    cfg = read_config('configs/default.yaml')
    cfg.update(data_root=str(tmp_path/'images'), manifest=str(tmp_path/'manifest.csv'),
               output=str(tmp_path/'outputs'), device='cpu', workers=0, seeds=[0, 1])
    cfg['open_world'].update(known=['known_a', 'known_b'], exclude=[], pca_dim=3)
    frame = _manifest()
    frame.to_csv(cfg['manifest'], index=False)
    rng = np.random.default_rng(19)
    for row in frame.itertuples():
        path = tmp_path/'images'/row.path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
    class TinyFrozenEncoder(torch.nn.Module):
        def forward(self, images):
            assert not self.training and not torch.is_grad_enabled()
            return torch.cat([images.mean((2, 3)), images.std((2, 3))], dim=1)
    loads = []
    def fake_hub(*args, **kwargs):
        loads.append(args)
        return TinyFrozenEncoder()
    monkeypatch.setattr(torch.hub, 'load', fake_hub)
    open_world.run(cfg, 'e6', 0)
    with pytest.raises(FileNotFoundError, match='Incomplete'):
        open_world.aggregate(cfg)
    for method, seed in [('e7', 0), ('e6', 1), ('e7', 1)]:
        open_world.run(cfg, method, seed)
    assert len(loads) == 2  # Both methods share one extraction per seed.
    destination = open_world.aggregate(cfg)
    summary = pd.read_csv(destination/'mean_std.csv')
    assert summary.n_seeds.tolist() == [2, 2]
    for seed in cfg['seeds']:
        e6 = pd.read_csv(tmp_path/f'outputs/exp6/seed{seed}/stream_predictions.csv')
        e7 = pd.read_csv(tmp_path/f'outputs/exp7/seed{seed}/stream_predictions.csv')
        assert e6.id.tolist() == e7.id.tolist()
    assert 'label' not in pd.read_csv(tmp_path/'outputs/exp6/seed0/stream_without_labels.csv')
    # Completed E6/E7 provenance must be accepted by the shared submission CLI.
    import yaml
    from carrepair import submit
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(submit, '__file__', str(tmp_path/'carrepair/submit.py'))
    monkeypatch.chdir(tmp_path)
    submit.main(['--config', str(config_path), '--experiments', '6', '7',
                 '--registry', str(tmp_path/'registry.json')])
    plan = pd.read_csv(tmp_path/'logs/experiment_plan.csv')
    assert plan.state.tolist() == ['SKIP_COMPLETED'] * 4
    with pytest.raises(FileExistsError):
        open_world.run(cfg, 'e6', 0)
    protocol_path = tmp_path/'outputs/exp7/seed0/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['stream_order_sha256'] = 'changed'
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='input or runtime mismatch'):
        open_world.aggregate(cfg)
    support, val, stream = open_world.partition_open_world(frame, ['known_a', 'known_b'], [], .8, 0)
    with pytest.raises(RuntimeError, match='cache mismatch'):
        open_world.load_or_extract_features({**cfg, 'data_root': str(tmp_path/'different')},
            0, support, val, stream, cfg['open_world']['dino_repo'])
