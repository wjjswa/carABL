from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .common import dump_json


def metrics(y, pred, classes):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=list(range(len(classes))), average='macro', zero_division=0)),
            'report': classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, zero_division=0, output_dict=True),
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist()}


@torch.no_grad()
def extract(model, data_loader, device):
    model.eval()
    features, labels = [], []
    for x, y, _ in data_loader:
        features.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def normalise(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_predict(bank, bank_y, query, n_classes, k, temperature):
    if not len(bank) or k < 1 or k > len(bank) or temperature <= 0:
        raise ValueError('Invalid k-NN parameters')
    sim = normalise(query) @ normalise(bank).T
    indices = np.argsort(-sim, axis=1, kind='stable')[:, :k]
    values = np.take_along_axis(sim, indices, axis=1)
    weights = np.exp((values - values.max(axis=1, keepdims=True)) / temperature)
    scores = np.zeros((len(query), n_classes))
    for j in range(k):
        scores[np.arange(len(query)), np.asarray(bank_y)[indices[:, j]]] += weights[:, j]
    return scores.argmax(1), scores / scores.sum(axis=1, keepdims=True), indices


def select_knn(bank, bank_y, val, val_y, n_classes, ks, temperatures):
    best = None
    for k in ks:
        if k > len(bank):
            continue
        for temperature in temperatures:
            pred, _, _ = knn_predict(bank, bank_y, val, n_classes, k, temperature)
            score = f1_score(val_y, pred, labels=list(range(n_classes)), average='macro', zero_division=0)
            if best is None or score > best['validation_macro_f1']:
                best = {'k': int(k), 'temperature': float(temperature), 'validation_macro_f1': float(score)}
    if best is None:
        raise ValueError('No valid k values')
    return best


def save_predictions(directory, frame, y, pred, scores, classes, name):
    out = frame[['id', 'path', 'label']].copy().reset_index(drop=True)
    out['prediction'] = [classes[p] if 0 <= p < len(classes) else 'abstain' for p in pred]
    out['confidence'] = np.asarray(scores).max(axis=1)
    out['correct'] = np.asarray(y) == np.asarray(pred)
    Path(directory).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(directory) / f'{name}_predictions.csv', index=False)
    np.savez_compressed(Path(directory) / f'{name}_scores.npz', scores=scores, targets=y)


def classification_eval(model, cfg, frame, classes, directory, bank_frame=None):
    from .data import loader
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for x, y, _ in loader(frame, cfg, classes):
            logits.append(model(x.to(cfg['device'])).cpu())
            targets.append(y)
    scores = F.softmax(torch.cat(logits), 1).numpy()
    y = torch.cat(targets).numpy()
    result = {'linear': metrics(y, scores.argmax(1), classes)}
    save_predictions(directory, frame, y, scores.argmax(1), scores, classes, 'linear')
    if bank_frame is not None:
        bank, bank_y = extract(model.features, loader(bank_frame, cfg, classes), cfg['device'])
        val_frame = __import__('carrepair.data', fromlist=['read_manifest']).read_manifest(cfg['manifest'])
        val_frame = val_frame[val_frame.split == 'val']
        val, val_y = extract(model.features, loader(val_frame, cfg, classes), cfg['device'])
        features, _ = extract(model.features, loader(frame, cfg, classes), cfg['device'])
        chosen = select_knn(bank, bank_y, val, val_y, len(classes), cfg['knn_k'], cfg['knn_temperature'])
        pred, votes, neighbours = knn_predict(bank, bank_y, features, len(classes), chosen['k'], chosen['temperature'])
        result['knn'] = {**metrics(y, pred, classes), **chosen}
        save_predictions(directory, frame, y, pred, votes, classes, 'knn')
        bank_frame.reset_index(drop=True).to_csv(Path(directory) / 'knn_reference_bank.csv', index=False)
        np.savez_compressed(Path(directory) / 'knn_features.npz', bank=bank, bank_y=bank_y, test=features, neighbours=neighbours)
    dump_json(Path(directory) / 'metrics.json', result)
    return result


def export_results(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for p in Path(root).rglob('metrics.json'):
        content = json.loads(p.read_text())
        config_file = p.parent / 'run.json'
        meta = json.loads(config_file.read_text()) if config_file.exists() else {}
        for method, r in content.items():
            if isinstance(r, dict) and 'macro_f1' in r:
                rows.append({'run': str(p.parent), **meta, 'evaluation': method,
                             'macro_f1': r['macro_f1'], 'accuracy': r['accuracy']})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(root) / 'summary.csv', index=False)
    if len(frame) and 'fraction' in frame:
        for ev, part in frame.groupby('evaluation'):
            fig, ax = plt.subplots(figsize=(9, 5))
            for name, group in part.groupby(['experiment', 'method']):
                g = group.groupby('fraction').macro_f1.agg(['mean', 'std']).fillna(0)
                ax.errorbar(g.index * 100, g['mean'], yerr=g['std'], marker='o', label='/'.join(map(str, name)))
            ax.set(xlabel='Labelled training pool (%)', ylabel='Macro-F1', ylim=(0, 1), title=f'{ev} evaluation')
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(Path(root) / f'{ev}_label_efficiency.png', dpi=200)
            plt.close(fig)
    return frame
