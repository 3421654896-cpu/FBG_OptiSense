"""Evaluate one fixed 0.20 mm prototype for each of 17 positions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from evaluate_current_three_positions import capture_parts, nearest, spectral_feature
from evaluate_fast_fullmap_position import _capture_feature

TRAINING = {
    1: 'f45_visible_p01_depth020_v46_20260907_r1',
    2: 'f45_visible_p02_depth020_v46_20260907_r1',
    3: 'f45_visible_p03_depth020_v46_20260907_r1',
    4: 'f45_visible_p04_depth020_v46_20260907_r1',
    5: 'f45_visible_p05_depth020_v46_20260907_r2',
    6: 'f45_visible_p06_depth020_v46_20260907_r2',
    7: 'f45_visible_p07_depth020_v46_20260907_r1',
    8: 'f45_visible_p08_depth020_v46_20260907_r1',
    9: 'f45_visible_p09_depth020_v46_20260907_r1',
    10: 'f45_visible_p10_depth020_v46_20260907_r1',
    11: 'f45_visible_p11_depth020_v46_20260907_r1',
    12: 'f45_visible_p12_depth020_v46_20260907_r1',
    13: 'f45_visible_p13_depth020_v46_20260907_r1',
    14: 'f45_visible_p14_depth020_v46_20260907_r1',
    15: 'f45_visible_p15_depth020_v46_20260907_r1',
    16: 'f45_visible_p16_depth020_v46_20260907_r1',
    17: 'f45_visible_p17_depth020_v46_20260907_r1',
}


def evaluate(root, queries, baseline_mode='early64', idle_noise=None):
    paths = [root / TRAINING[p] / 'capture.json' for p in sorted(TRAINING)]
    resolved = {p.resolve() for p in paths}
    if any(q.resolve() in resolved for q in queries):
        raise ValueError('Training captures cannot be validation queries')
    records = [_capture_feature(p) for p in paths]
    if [r['point_index'] for r in records] != list(range(1, 18)):
        raise ValueError('Training label order is not 1..17')
    if any(abs(r['depth_mm'] - 0.20) > 1e-6 for r in records):
        raise ValueError('Training depth is not uniformly 0.20 mm')
    parts = [capture_parts(p, baseline_mode) for p in paths]
    vectors = [spectral_feature(b, np.median(h, axis=0), 'shape_delta') for b, h in parts]
    matrix = np.eye(45)
    noise_source = None
    if idle_noise is not None:
        from idle_spectral_noise import load_idle_metric
        if idle_noise.resolve() in resolved or any(idle_noise.resolve() == q.resolve() for q in queries):
            raise ValueError('Noise record must be independent')
        matrix, noise_source = load_idle_metric(idle_noise)
        for path in paths + list(queries):
            capture = json.loads(path.read_text('utf-8'))
            if (capture['table_codes_crc32'] != noise_source['table_crc32'] or
                    capture['table_sha256'] != noise_source['firmware_table_sha256']):
                raise ValueError('Noise and contact DAC tables differ')
    vectors = np.asarray(vectors) @ matrix
    results = []
    for query in queries:
        truth = _capture_feature(query)
        b, held = capture_parts(query, baseline_mode)
        labels, errors = [], []
        for sample in held:
            index, _ = nearest(vectors, spectral_feature(b, sample, 'shape_delta') @ matrix)
            labels.append(records[index]['point_index'])
            errors.append(float(np.linalg.norm(records[index]['xy'] - truth['xy'])))
        median_index, distances = nearest(vectors, spectral_feature(b, np.median(held, axis=0), 'shape_delta') @ matrix)
        results.append(dict(source=str(query.resolve()), sha256=hashlib.sha256(query.read_bytes()).hexdigest(),
            truth=truth['point_index'], depth_mm=truth['depth_mm'], held_frames=len(held),
            median_prediction=records[median_index]['point_index'],
            frame_confusion=dict(sorted(Counter(labels).items())),
            correct_frames=sum(x == truth['point_index'] for x in labels),
            p95_position_error_mm=float(np.percentile(errors, 95)),
            nearest_rms=float(distances[median_index])))
    return dict(schema='current_uniform17_position_validation_v1',
        baseline_mode=baseline_mode,
        method=('shape_delta, unscaled Euclidean nearest single held-median prototype' if idle_noise is None
                else 'shape_delta, independent idle covariance with fixed 50% shrinkage; nearest held-median prototype'),
        noise_source=noise_source,
        training=[dict(point=r['point_index'], source=str(p.resolve()),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p, r in zip(paths, records)],
        validation=results, deployed=False, physical_15hz_verified=False,
        full_spatial_accuracy_verified=False,
        scope='Uniform 0.20 mm 17-position pilot; adjacent held frames are correlated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queries', nargs='+', type=Path)
    parser.add_argument('--root', type=Path, default=Path('../outputs'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', choices=('early64', 'recent64'), default='early64')
    parser.add_argument('--idle-noise', type=Path)
    args = parser.parse_args()
    result = evaluate(args.root, args.queries, args.baseline, args.idle_noise)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['validation'], indent=2))
