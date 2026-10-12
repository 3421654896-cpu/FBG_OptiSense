"""Offline linearized peak-shift features from an independent dense teacher.

Fixed 11-point triangular envelope smoothing before observing validation.
No deployment, hardware I/O, or label-dependent feature tuning.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_current_dense9_teacher import validated_groups
from evaluate_current_uniform17 import TRAINING
from evaluate_current_three_positions import capture_parts, nearest
from evaluate_fast_fullmap_position import _capture_feature


def triangular_envelope(y):
    y = np.asarray(y, dtype=float)
    kernel = np.r_[np.arange(1, 7), np.arange(5, 0, -1)].astype(float)
    kernel /= np.sum(kernel)
    return np.convolve(np.pad(y, (5,5), mode='edge'), kernel, mode='valid')


def load_templates(path):
    data = json.loads(path.read_text('utf-8'))
    groups = validated_groups(data)
    templates = []
    for group in groups:
        dense = group['dense']
        x = np.array([r['measured_wavelength_nm'] for r in dense])
        if np.any(np.diff(x) <= 0):
            raise ValueError('Dense wavelength axis is not monotonic')
        y = triangular_envelope([r['ch1_adc_code'] for r in dense])
        anchor_x = [p['samples'][0]['sample']['measured_wavelength_nm'] for p in group['paired_anchors']]
        ref = np.interp(anchor_x, x, y)
        slope = np.interp(anchor_x, x, np.gradient(y, x))
        templates.append((ref, slope))
    return templates


def shift_feature(templates, baseline, sample):
    b = np.asarray(baseline, dtype=float).reshape(9, 5)
    y = np.asarray(sample, dtype=float).reshape(9, 5)
    if not np.isfinite([b,y]).all():
        raise ValueError('Nonfinite ADC values')
    result = []
    for group, ((reference, slope), base, value) in enumerate(zip(templates, b, y), start=1):
        # Fit only baseline amplitude and offset. The query never updates the teacher.
        design = np.column_stack((reference, np.ones(5)))
        amplitude, _ = np.linalg.lstsq(design, base, rcond=None)[0]
        if amplitude <= 0:
            raise ValueError(f'Group {group} baseline has no positive teacher amplitude ({amplitude:.6g})')
        derivative = -amplitude * slope
        nuisance = np.column_stack((np.ones(5), base))
        derivative -= nuisance @ np.linalg.lstsq(nuisance, derivative, rcond=None)[0]
        energy = np.dot(derivative, derivative)
        if energy < 1e-6:
            raise ValueError('Insufficient shift sensitivity after nuisance projection')
        result.append(float(np.dot(value-base, derivative)/energy))
    return np.array(result)


def evaluate(root, teacher, queries):
    templates = load_templates(teacher)
    paths = [root/TRAINING[i]/'capture.json' for i in range(1,18)]
    if set(p.resolve() for p in paths) & set(p.resolve() for p in queries):
        raise ValueError('Training/validation overlap')
    records = [_capture_feature(p) for p in paths]
    if [r['point_index'] for r in records] != list(range(1,18)) or any(abs(r['depth_mm']-.2)>1e-6 for r in records):
        raise ValueError('Training labels/depth changed')
    parts = [capture_parts(p) for p in paths]
    vectors = []
    for path, (b,h) in zip(paths, parts):
        try:
            vectors.append(shift_feature(templates, b, np.median(h, axis=0)))
        except ValueError as exc:
            return dict(schema='dense_shift_feature_experiment_v1', status='rejected_before_validation',
                teacher=str(teacher.resolve()), teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
                rejected_training_source=str(path.resolve()),
                rejected_training_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reason=str(exc), validation=[], deployed=False, physical_15hz_verified=False,
                method='11-point triangular envelope with positive baseline amplitude prerequisite')
    validations = []
    for path in queries:
        record = _capture_feature(path)
        baseline, held = capture_parts(path)
        features = [shift_feature(templates, baseline, sample) for sample in held]
        indices = [nearest(vectors, feature)[0] for feature in features]
        labels = [records[i]['point_index'] for i in indices]
        errors = [float(np.linalg.norm(records[i]['xy']-record['xy'])) for i in indices]
        median_index = nearest(vectors, shift_feature(templates, baseline, np.median(held, axis=0)))[0]
        validations.append(dict(source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            truth=record['point_index'], held_frames=len(held), correct_frames=labels.count(record['point_index']),
            confusion=dict(Counter(labels)), median_prediction=records[median_index]['point_index'],
            p95_position_error_mm=float(np.percentile(errors,95)),
            median_linearized_shifts_pm=(np.median(features,axis=0)*1000).tolist()))
    return dict(schema='dense_shift_feature_experiment_v1', teacher=str(teacher.resolve()),
        teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
        method='11-point triangular dense envelope; baseline affine fit; offset/amplitude-projected linear shift, nine features, Euclidean prototype',
        training=[dict(source=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        validation=validations, deployed=False, physical_15hz_verified=False,
        scope='Exploratory replay on previously inspected physical repeats; independent teacher not independent final validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('../outputs'))
    parser.add_argument('queries', type=Path, nargs='+')
    args = parser.parse_args()
    report = evaluate(args.root, args.teacher, args.queries)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report if not report['validation'] else report['validation'], indent=2))
