"""Experiment 1: target inference with original boxes/masks, image-level scoring only."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from .common import dump_json, provenance
from .data import read_manifest

MAPPING = {'bumper_dent': 'dent', 'door_dent': 'dent', 'bumper_scratch': 'scratch',
           'door_scratch': 'scratch', 'glass_shatter': 'glass_shatter',
           'head_lamp': 'lamp_broken', 'tail_lamp': 'lamp_broken'}


def direct(cfg, confidence=.25, image_size=1024):
    from ultralytics import YOLO
    from sklearn.metrics import accuracy_score, recall_score
    frame = read_manifest(cfg['manifest'])
    test = frame[frame.split == 'test']
    directory = Path(cfg['output']) / 'exp1'
    directory.mkdir(parents=True, exist_ok=True)
    provenance(cfg, directory)
    model = YOLO(cfg['checkpoint'])
    if model.task != 'segment':
        raise ValueError('Experiment 1 requires the v2 segmentation checkpoint')
    rows = []
    shared = ['dent', 'scratch', 'glass_shatter', 'lamp_broken']
    for _, row in test.iterrows():
        case = directory / 'cases' / row.id
        case.mkdir(parents=True, exist_ok=True)
        source = Path(cfg['data_root']) / row.path
        shutil.copy2(source, case / ('original' + source.suffix.lower()))
        result = model.predict(source=str(source), conf=confidence, imgsz=image_size, device=cfg['device'], verbose=False)[0]
        result.save(filename=str(case / 'masks_overlay.jpg'))
        Image.fromarray(result.plot(masks=False)[..., ::-1]).save(case / 'boxes_overlay.jpg')
        instances = []
        scores = {c: 0. for c in shared}
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            np.savez_compressed(case / 'predicted_masks.npz', masks=masks)
        for i, box in enumerate(result.boxes):
            name, conf = result.names[int(box.cls.item())], float(box.conf.item())
            instances.append({'class': name, 'confidence': conf, 'xyxy': box.xyxy[0].tolist(), 'instance_index': i})
            if name in scores:
                scores[name] = max(scores[name], conf)
        pred = max(scores, key=scores.get) if max(scores.values()) > 0 else 'abstain'
        dump_json(case / 'predictions.json', {'instances': instances, 'mapped_image_prediction': pred,
                  'confidence_threshold': confidence, 'note': 'Predicted masks and boxes; not ground truth or calibrated probabilities.'})
        mapped = MAPPING.get(row.label)
        rows.append({'id': row.id, 'path': row.path, 'label': row.label, 'mapped_label': mapped,
                     'prediction': pred, 'confidence': max(scores.values()), 'correct': pred == mapped,
                     'scorable': mapped is not None, 'case': str(case)})
    table = pd.DataFrame(rows)
    table.to_csv(directory / 'predictions.csv', index=False)
    part = table[table.scorable]
    if len(part):
        dump_json(directory / 'metrics.json', {'direct': {
            'mapped_accuracy': float(accuracy_score(part.mapped_label, part.prediction)),
            'per_class_recall': dict(zip(shared, recall_score(part.mapped_label, part.prediction, labels=shared, average=None, zero_division=0).tolist())),
            'abstention_rate': float((part.prediction == 'abstain').mean()), 'scored_images': len(part),
            'excluded_images': len(table)-len(part),
            'note': 'Image-level proxy with merged classes. No detection/mask mAP without target location annotations.'}})
    return table
