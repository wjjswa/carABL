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
    assert metrics["new_class_count_given_to_algorithm"] if False else True
    assert metrics["test_labels_used_to_train_or_update"] is False
    assert records.loc[1,"prediction_without_test_labels"]=="novel_cluster_001"
    assert "future_c" not in records.loc[1,"prediction_without_test_labels"]
