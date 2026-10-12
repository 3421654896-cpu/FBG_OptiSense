"""Development-only cross-capture distance study; never an acceptance test."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_fresh_sentinel_position import audit
from evaluate_fresh_sentinel_position import compare, metrics, unit


def study(train, test):
    compare(train, test)  # enforce independent captures and identical acquisition
    a, b = train['records'], test['records']
    xy = np.asarray([r['model_xy_mm'] for r in a])
    # Scaling is learned from training data only. Test labels only score predictions.
    scale = np.maximum(np.std([r['delta9'] for r in a], axis=0), 1.)
    variants = {}
    for name in ('cosine_delta', 'euclidean_delta', 'train_scaled_delta', 'fractional_delta'):
        def transform(values, record):
            values = np.asarray(values, dtype=float)
            if name == 'cosine_delta':
                return unit(values)
            if name == 'train_scaled_delta':
                return values / scale
            if name == 'fractional_delta':
                return values / np.maximum(record['baseline9'], 5.)
            return values
        templates = np.asarray([transform(r['delta9'], r) for r in a])
        errors, frame_errors, predictions = [], [], []
        for record in b:
            vectors = transform(record['hold_delta9'], record)
            med = transform(record['delta9'], record)
            nearest = int(np.argmin(np.sum((templates-med)**2, axis=1)))
            frame_nearest = np.argmin(np.sum((vectors[:, None, :]-templates[None, :, :])**2, axis=2), axis=1)
            truth = np.asarray(record['model_xy_mm'])
            errors.append(float(np.linalg.norm(xy[nearest]-truth)))
            frame_errors.extend(np.linalg.norm(xy[frame_nearest]-truth, axis=1).tolist())
            predictions.append({'truth_xy': truth.tolist(), 'predicted_xy': xy[nearest].tolist()})
        variants[name] = {'contact': metrics(errors, np.asarray(errors)<1e-6),
                          'frame': metrics(frame_errors, np.asarray(frame_errors)<1e-6),
                          'predictions': predictions}
    paired = []
    for r in b:
        matches = [q for q in a if np.linalg.norm(np.asarray(q['model_xy_mm'])-r['model_xy_mm']) < 1e-6]
        if len(matches) != 1:
            continue
        q = matches[0]
        paired.append({'xy': r['model_xy_mm'], 'train_delta9': q['delta9'],
                       'test_delta9': r['delta9'],
                       'baseline_change9': (np.asarray(r['baseline9'])-q['baseline9']).tolist(),
                       'cosine': float(unit(q['delta9']) @ unit(r['delta9']))})
    return {'purpose': 'development_algorithm_comparison_not_locked_validation',
            'deployment_eligible': False, 'physical_15hz_verified': False,
            'area_verified': False, 'train_sha256': train['source_sha256'],
            'test_sha256': test['source_sha256'], 'variants': variants, 'paired': paired,
            'next_validation': 'Any selected algorithm needs a new untouched capture.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train', type=Path)
    p.add_argument('test', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = study(audit(args.train), audit(args.test))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: {m: v[m] for m in ('contact', 'frame')} for k,v in result['variants'].items()}, indent=2))
