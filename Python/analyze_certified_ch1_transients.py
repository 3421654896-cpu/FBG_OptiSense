"""Offline, causal transient-to-held-target analysis. Never controls hardware.

The source is held 100 ms, not verified stable. These unloaded pair transitions do not certify continuous scanning, physical
15 Hz pressure response, localization, or contact area. Models are evaluated
with entire gratings held out and are NOT exported for deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TIA_OHM = {0: 2000, 1: 40000, 2: 5000, 3: 20000}
ADC_V_PER_CODE = 2.5 / 4096
CUTOFFS_US = (100, 300, 650, 1000, 2000, 3000, 5000, 10000, 20000)


def causal_read(record, cutoff_us):
    """Use only conversions whose entire six-channel group has finished.

    The firmware reports the group START. The first dense samples show about
    55 us/group. Require a subsequent group's start <= cutoff as conservative
    completion evidence, so CH1 never silently receives a future sample.
    """
    times = np.asarray(record['sample_start_after_target_write_us'], dtype=float)
    values = np.asarray(record['direct_adc'], dtype=float)[:, 1]
    if len(times) != len(values) or np.any(np.diff(times) <= 0):
        raise ValueError('invalid sample clock')
    completed = np.flatnonzero(times[1:] <= cutoff_us)
    if not len(completed):
        raise ValueError('no provably completed ADC group before cutoff')
    index = int(completed[-1])
    return float(values[0]), float(values[index]), index, float(times[index + 1])


def held_group_gain(first, last, truth, groups, valid):
    """Predict first+gain*(last-first), never fit a held-out grating's label.

    One no-intercept gain is trained across the remaining gratings. Clipping
    is explicitly counted, not hidden in a metric. Degenerate folds are absent.
    """
    predicted = np.full(len(truth), np.nan)
    folds = []
    delta = last - first
    for group in sorted(set(groups[valid])):
        train = valid & (groups != group)
        test = valid & (groups == group)
        energy = float(delta[train] @ delta[train])
        if np.count_nonzero(train) < 10 or energy < 1:
            continue
        gain = float(delta[train] @ (truth[train] - first[train]) / energy)
        # The bound prevents fitting a numerically explosive predictor.
        gain = float(np.clip(gain, 0, 20))
        raw = first[test] + gain * delta[test]
        predicted[test] = np.clip(raw, 0, 4095)
        folds.append({'held_out_group': int(group), 'gain': gain,
                      'train_count': int(train.sum()), 'test_count': int(test.sum()),
                      'clipped_predictions': int(np.count_nonzero((raw < 0) | (raw > 4095)))})
    return predicted, folds


def causal_shape_features(record, cutoff_us):
    """Three past-to-latest differences, preserving unit response to DC.

    There is no row/grating identity, fitted target, or future ADC input.
    Differences encode early curvature as well as the total early step.
    """
    first, last, _, _ = causal_read(record, cutoff_us)
    times = np.asarray(record['sample_start_after_target_write_us'])
    values = np.asarray(record['direct_adc'])[:, 1]
    history = [first]
    for fraction in (.2, .5):
        completed = np.flatnonzero(times[1:] <= cutoff_us * fraction)
        history.append(float(values[completed[-1]]) if len(completed) else first)
    return last, np.asarray([last - value for value in history])


def _ridge_correction(features, residual, fraction):
    gram = features.T @ features
    strength = max(float(np.trace(gram)) / features.shape[1], 1) * fraction
    coefficients = np.linalg.solve(gram + strength*np.eye(features.shape[1]), features.T @ residual)
    norm = np.linalg.norm(coefficients)
    if norm > 20:
        coefficients *= 20/norm
    return coefficients


def nested_shape_prediction(features, last, truth, groups, valid):
    """Choose regularization ONLY within training gratings of each outer fold."""
    prediction = np.full(len(truth), np.nan)
    folds = []
    unique_groups = sorted(set(groups[valid]))
    if len(unique_groups) < 3:
        return prediction, folds
    residual = truth - last
    for group in unique_groups:
        train = valid & (groups != group)
        test = valid & (groups == group)
        candidates = []
        for fraction in (.001, .01, .1, 1.0):
            errors = []
            for inner in unique_groups:
                if inner == group:
                    continue
                inner_train = train & (groups != inner)
                inner_test = train & (groups == inner)
                coeff = _ridge_correction(features[inner_train], residual[inner_train], fraction)
                errors.extend((features[inner_test] @ coeff - residual[inner_test]).tolist())
            candidates.append((float(np.mean(np.square(errors))), fraction))
        _, fraction = min(candidates)
        coeff = _ridge_correction(features[train], residual[train], fraction)
        raw = last[test] + features[test] @ coeff
        prediction[test] = np.clip(raw, 0, 4095)
        folds.append({'held_out_grating': int(group), 'ridge_fraction': fraction,
                      'coefficients': coeff.tolist(), 'train_count': int(train.sum()),
                      'test_count': int(test.sum()),
                      'clipped_predictions': int(np.count_nonzero((raw < 0) | (raw > 4095)))})
    return prediction, folds


def metrics(truth, prediction, peak_span, tia):
    valid = np.isfinite(truth) & np.isfinite(prediction) & (peak_span > 0)
    if not np.any(valid):
        return {'count': 0}
    error = prediction[valid] - truth[valid]
    relative = error / peak_span[valid]
    return {
        'count': int(valid.sum()),
        'rmse_codes': float(np.sqrt(np.mean(error**2))),
        'rmse_v': float(np.sqrt(np.mean(error**2)) * ADC_V_PER_CODE),
        'rmse_input_equivalent_ua': float(np.sqrt(np.mean(error**2)) * ADC_V_PER_CODE / tia * 1e6),
        'peak_span_normalized_rmse_percent': float(100*np.sqrt(np.mean(relative**2))),
        'peak_span_normalized_p95_absolute_percent': float(100*np.percentile(abs(relative), 95)),
        'within_5_percent_peak_span_percent': float(100*np.mean(abs(relative) <= .05)),
    }


def analyze(payload):
    if not payload.get('complete') or not payload.get('paired_stable_teacher'):
        raise ValueError('requires a completed capture with per-curve held-target teacher')
    cleanup = payload.get('safety_cleanup', {})
    if not cleanup.get('exact_disarm_ack') or not cleanup.get('exact_soa_shutter_ack'):
        raise ValueError('capture safety cleanup not confirmed')
    records = payload['captures']
    tia = TIA_OHM[payload['selector']]
    target = np.asarray([r['target_index'] for r in records])
    groups = target // 5
    repeats = np.asarray([r['repeat'] for r in records])
    if any('stable_teacher' not in r for r in records):
        raise ValueError('missing stable teacher')
    truth = np.asarray([r['stable_teacher']['ch1_adc_code'] for r in records])
    valid = np.asarray([r['stable_teacher']['stable'] and
                        not r['stable_teacher']['ch1_saturated'] for r in records])
    valid &= np.isfinite(truth) & (truth >= 0) & (truth < 4080)
    # Evaluation normalization only. These scales never enter model training.
    spans = np.full(len(records), np.nan)
    for group in sorted(set(groups)):
        per_point = [np.median(truth[valid & (target == point)])
                     for point in range(int(group)*5, int(group)*5+5)
                     if np.any(valid & (target == point))]
        if len(per_point) == 5:
            span = float(np.ptp(per_point))
            if span >= 10:
                spans[groups == group] = span
    eval_valid = valid & np.isfinite(spans)
    result = {
        'selector': payload['selector'], 'transimpedance_ohm': tia,
        'total_curves': len(records), 'stable_teachers': int(valid.sum()),
        'evaluated_curves': int(eval_valid.sum()),
        'invalid_teacher_targets': sorted(set(map(int, target[~valid]))),
        'scope': 'unloaded transitions from 100 ms source hold (source stability unverified); held-target >=300 ms measured truth; NOT pressure validation',
        'deployment_allowed': False,
        'normalization': 'per-grating five-point teacher max-minus-min; at least 10 ADC codes',
        'cutoffs': [],
    }
    for cutoff in CUTOFFS_US:
        first, last, count, completed_at = zip(*(causal_read(r, cutoff) for r in records))
        first, last = np.asarray(first), np.asarray(last)
        predicted, folds = held_group_gain(first, last, truth, groups, eval_valid)
        repeat_prediction, repeat_folds = held_group_gain(first, last, truth, repeats, eval_valid)
        features = np.asarray([causal_shape_features(r, cutoff)[1] for r in records])
        shape_prediction, shape_folds = nested_shape_prediction(features, last, truth, groups, eval_valid)
        direct = np.where(eval_valid, last, np.nan)
        result['cutoffs'].append({
            'cutoff_us_after_target_write': cutoff,
            'minimum_completed_group_count': int(min(count) + 1),
            'latest_completion_evidence_us': float(max(completed_at)),
            'direct_adc': metrics(truth, direct, spans, tia),
            'held_out_grating_prediction': metrics(truth, predicted, spans, tia),
            'held_out_repeat_prediction': metrics(truth, repeat_prediction, spans, tia),
            'nested_held_out_grating_shape_prediction': metrics(truth, shape_prediction, spans, tia),
            'shape_folds': shape_folds,
            'grating_folds': folds, 'repeat_folds': repeat_folds,
        })
    # Same-point teacher repeatability establishes a floor without pretending
    # unloaded repeat tests represent a changed pressure distribution.
    dispersion = []
    for point in sorted(set(target)):
        values = truth[valid & (target == point)]
        if len(values) >= 2:
            dispersion.append({'target_index': int(point), 'n': len(values),
                               'range_codes': float(np.ptp(values)),
                               'std_codes': float(np.std(values))})
    result['teacher_repeatability'] = dispersion
    result['teacher_repeatability_median_std_codes'] = float(np.median([r['std_codes'] for r in dispersion])) if dispersion else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'schema': 'certified_transient_causal_analysis_v1', 'experiments': []}
    for path in args.inputs:
        content = path.read_bytes()
        item = analyze(json.loads(content))
        item['source_path'] = str(path.resolve())
        item['source_sha256'] = hashlib.sha256(content).hexdigest()
        result['experiments'].append(item)
        print(f"TIA={item['transimpedance_ohm']} stable={item['stable_teachers']}/{item['total_curves']}")
        for row in item['cutoffs']:
            d = row['direct_adc'].get('peak_span_normalized_rmse_percent')
            p = row['held_out_grating_prediction'].get('peak_span_normalized_rmse_percent')
            s = row['nested_held_out_grating_shape_prediction'].get('peak_span_normalized_rmse_percent')
            print(f"  cutoff_us={row['cutoff_us_after_target_write']} direct_NRMSE={d} held_grating_NRMSE={p} shape_NRMSE={s}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
