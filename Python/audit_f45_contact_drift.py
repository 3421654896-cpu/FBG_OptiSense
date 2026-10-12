"""Audit frozen v7 contact detection on disjoint idle and post-release frames."""
from pathlib import Path
import argparse
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from select_f45_activation_detector_v6 import phase_features


def trace_features(path):
    data = json.loads(Path(path).read_text('utf-8'))
    frames = data['frames']
    seq = np.array([f['sequence'] for f in frames])
    first = np.array([[r['first_code'] for r in f['records']] for f in frames], float)
    second = np.array([[r['second_code'] for r in f['records']] for f in frames], float)
    baseline_start = data.get('commissioning', {}).get('baseline_start_sequence',16)
    mask = (seq >= baseline_start) & (seq < baseline_start+64)
    if mask.sum() != 64:
        raise ValueError('Missing baseline')
    a = first - np.median(first[mask], axis=0)
    b = second - np.median(second[mask], axis=0)
    return data, seq, np.concatenate([a, b, b-a], axis=1)


def phase_masks(data, seq):
    c = data['commissioning']
    receipts = {r['sequence']: r['host_monotonic_ns'] for r in c['frame_receipts']}
    times = np.array([receipts[int(s)] for s in seq], dtype=np.int64)
    events = {e['kind']: e['host_monotonic_ns'] for e in c['events']}
    start = c.get('baseline_start_sequence',16)
    return {
        'baseline_fit_window': (seq >= start) & (seq < start+64),
        'pre_descent_unseen': (seq >= start+64) & (times < events['contact_descent_started']),
        'held': (times >= events['contact_reached_commanded_depth'] + 200_000_000)
                & (times < events['release_started']),
        'after_safe_retraction': times >= events['safe_height_reached'] + 200_000_000,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', required=True, type=Path)
    p.add_argument('--idle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('output must be new')
    lock = json.loads(args.lock.read_text('utf-8'))
    records = lock['training']['captures']
    parts, labels = [], []
    for item in records:
        path = Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Training hash mismatch')
        base, held = phase_features(path)
        parts.extend([base, held])
        labels.extend([np.zeros(len(base), int), np.ones(len(held), int)])
    model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10,
        max_features='sqrt', class_weight='balanced', random_state=20260908,
        n_jobs=-1).fit(np.concatenate(parts), np.concatenate(labels))
    totals = {}
    per_capture = []
    for item in records:
        data, seq, features = trace_features(item['path'])
        prediction = model.predict(features)
        row = {'round': item['round'], 'point': item['point'], 'phases': {}}
        for name, mask in phase_masks(data, seq).items():
            n, positive = int(mask.sum()), int(prediction[mask].sum())
            row['phases'][name] = {'frames': n, 'contact_predictions': positive}
            total = totals.setdefault(name, {'frames': 0, 'contact_predictions': 0})
            total['frames'] += n
            total['contact_predictions'] += positive
        per_capture.append(row)
    data, seq, features = trace_features(args.idle)
    pred = model.predict(features[seq >= 80])
    result = {'schema': 'f45_contact_drift_audit_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_lock': str(args.lock.resolve()),
        'training_windows_are_in_sample': True,
        'later_no_contact_windows_excluded_from_fit': True,
        'totals': totals, 'captures': per_capture,
        'independent_idle': {'path': str(args.idle.resolve()),
            'frames': len(pred), 'false_positive_frames': int(pred.sum())},
        'deployment_eligible': False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({'totals': totals, 'idle': result['independent_idle']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
