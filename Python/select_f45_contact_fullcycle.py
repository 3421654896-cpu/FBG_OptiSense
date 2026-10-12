"""Development-only contact detector including post-release silicone response.

Previously scored v7 rounds are development holdouts, not a new blind test.
Every evaluation uses frames outside the baseline-estimation window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from audit_f45_contact_drift import trace_features, phase_masks
from select_f45_activation_threshold_v6 import load_manifest


def load(item):
    path = Path(item['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('capture hash mismatch')
    data, seq, features = trace_features(path)
    return item, features, phase_masks(data, seq)


def represent(values, variant):
    values = np.asarray(values, float)
    if variant == 'raw':
        return values
    pair = values[:, :90]
    norm = np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1.0)
    shape = pair / norm
    energy = np.log1p(np.sqrt(np.mean(pair.reshape(-1, 2, 9, 5)**2, axis=(1,3))))
    parts = [values, shape, energy]
    if variant == 'shape':
        return np.concatenate(parts, axis=1)
    if variant != 'history':
        raise ValueError(variant)
    # Strictly causal samples; no future-frame averaging or pressure smoothing.
    for lag in (4, 16):
        previous = pair[np.maximum(np.arange(len(pair)) - lag, 0)]
        delta = pair - previous
        parts.append(delta)
        parts.append(delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1.0))
    return np.concatenate(parts, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', required=True, type=Path)
    p.add_argument('--development-round', action='append', required=True, type=Path)
    p.add_argument('--idle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--shape-history', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        p.error('output must be new')
    lock = json.loads(a.lock.read_text('utf-8'))
    training = [load(i) for i in lock['training']['captures']]
    testing = [load(i) for m in a.development_round for i in load_manifest(m)]
    _, seq, idle = trace_features(a.idle)
    results = []
    candidates = [
        ('raw', 'lda_balanced', make_pipeline(StandardScaler(),
          LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[.5,.5]))),
        ('raw', 'extra_trees_leaf10', ExtraTreesClassifier(n_estimators=300,
          min_samples_leaf=10, max_features='sqrt', class_weight='balanced',
          random_state=20260908, n_jobs=-1)),
    ]
    if a.shape_history:
        candidates = [(variant, f'extra_trees_leaf{leaf}', ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=leaf, max_features='sqrt',
            class_weight='balanced', random_state=20260908, n_jobs=-1))
            for variant in ('shape', 'history') for leaf in (1, 3, 10)]
    for variant, name, model in candidates:
        x, y = [], []
        for _, f, masks in training:
            features = represent(f, variant)
            negative = features[masks['pre_descent_unseen'] | masks['after_safe_retraction']]
            if len(negative) > 64:
                negative = negative[np.linspace(0, len(negative)-1, 64).astype(int)]
            positive = features[masks['held']]
            x.extend([negative, positive])
            y.extend([np.zeros(len(negative), int), np.ones(len(positive), int)])
        x, y = np.concatenate(x), np.concatenate(y)
        model.fit(x, y)
        rows = []
        for item, f, masks in testing:
            pred = model.predict(represent(f, variant))
            phases = {n: {'frames': int(m.sum()), 'contact_predictions': int(pred[m].sum())}
                      for n,m in masks.items() if n != 'baseline_fit_window'}
            rows.append({'round': item['round'], 'point': item['point'], 'phases': phases})
        totals = {}
        for row in rows:
            for phase, counts in row['phases'].items():
                t = totals.setdefault(phase, {'frames':0,'contact_predictions':0})
                for k,v in counts.items(): t[k] += v
        idle_features = represent(idle, variant)[seq >= 80]
        result = {'model':name, 'representation':variant, 'totals':totals, 'captures':rows,
            'independent_idle_frames':len(idle_features),
            'independent_idle_false_positives':int(model.predict(idle_features).sum())}
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='captures'}),flush=True)
    report = {'schema':'f45_contact_fullcycle_development_v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'training_contact_frames':int(y.sum()), 'training_idle_frames':int((y==0).sum()),
        'training_lock_sha256':hashlib.sha256(a.lock.read_bytes()).hexdigest(),
        'previously_scored_rounds_reused_for_development':True,
        'random_frame_split_used':False, 'temporal_smoothing_used':False,
        'deployment_eligible':False, 'candidates':results}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')


if __name__=='__main__': main()
