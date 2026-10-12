"""Offline five-position pilot. CNC labels score results, never enter inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_current_three_positions import capture_parts, spectral_feature, nearest
from evaluate_fast_fullmap_position import _capture_feature

TRAINING = [
    'f45_visible_p06_v46_20260907_r1',
    'f45_visible_p08_v46_20260907_r1',
    'f45_visible_center_v46_20260907_r2',
    'f45_visible_p10_v46_20260907_r1',
    'f45_visible_p13_v46_20260907_r1',
]


def evaluate(root, queries, include_shallow=False, feature='shape_delta', include_all_positions=False):
    names = list(TRAINING)
    if include_shallow:
        names += [f'f45_visible_p{point:02d}_depth010_v46_20260907_r1'
                  for point in (8, 9, 10, 13)]
    if include_all_positions:
        names += [f'f45_visible_p{point:02d}_depth020_v46_20260907_r1'
                  for point in (1, 2, 3, 4, 7, 11, 12, 14, 15, 16, 17)]
        names += ['f45_visible_p05_depth020_v46_20260907_r2']
    paths = [root / name / 'capture.json' for name in names]
    if any(q.resolve() in {p.resolve() for p in paths} for q in queries):
        raise ValueError('Training captures cannot be validation queries')
    records = [_capture_feature(p) for p in paths]
    parts = [capture_parts(p) for p in paths]
    vectors = [spectral_feature(b, np.median(h, axis=0), feature) for b, h in parts]
    results = []
    for query in queries:
        truth = _capture_feature(query)
        b, held = capture_parts(query)
        predictions, errors = [], []
        for sample in held:
            index, _ = nearest(vectors, spectral_feature(b, sample, feature))
            predictions.append(records[index]['point_index'])
            errors.append(float(np.linalg.norm(records[index]['xy'] - truth['xy'])))
        results.append(dict(source=str(query.resolve()), sha256=hashlib.sha256(query.read_bytes()).hexdigest(),
            truth=truth['point_index'], predictions=predictions,
            correct=sum(p == truth['point_index'] for p in predictions), total=len(predictions),
            p95_position_error_mm=float(np.percentile(errors, 95))))
    return dict(schema='current_position_pilot_v2',
        training=[dict(source=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            point=r['point_index'], prototype=v.tolist()) for p, r, v in zip(paths, records, vectors)],
        results=results, include_shallow=include_shallow, include_all_positions=include_all_positions,
        feature=feature, deployed=False, physical_15hz_verified=False,
        scope='Fixed position prototypes at recorded depths; separate contacts, not independent adjacent frames. Not full silicone validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queries', nargs='+', type=Path)
    parser.add_argument('--root', type=Path, default=Path('../outputs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-shallow', action='store_true')
    parser.add_argument('--include-all-positions', action='store_true')
    parser.add_argument('--feature', choices=('shape_delta', 'raw_delta'), default='shape_delta')
    args = parser.parse_args()
    result = evaluate(args.root, args.queries, args.include_shallow, args.feature, args.include_all_positions)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['results'], indent=2))
