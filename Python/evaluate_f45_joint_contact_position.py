"""Offline development evaluation: class 0 is no contact, 1..17 position.

Uses whole acquisition rounds, never a random frame split. Already inspected
development rounds are not blind evidence and cannot authorize deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from select_f45_contact_fullcycle import load, represent
from select_f45_activation_threshold_v6 import load_manifest
from audit_f45_contact_drift import trace_features


def score(prediction, masks, point):
    result = {}
    for name, mask in masks.items():
        if name == 'baseline_fit_window':
            continue
        selected = prediction[mask]
        result[name] = dict(frames=len(selected),
            contact_predictions=int(np.count_nonzero(selected)),
            correct_predictions=int(np.count_nonzero(selected == (point if name == 'held' else 0))))
    return result


def decode_contact_probability(probabilities, classes, threshold):
    """Sum all contact classes; do not confuse a dispersed location vote with idle."""
    classes = np.asarray(classes)
    probabilities = np.asarray(probabilities, float)
    if not 0 <= threshold <= 1 or probabilities.ndim != 2:
        raise ValueError('Invalid threshold or probabilities')
    if len(classes) != probabilities.shape[1] or np.count_nonzero(classes == 0) != 1:
        raise ValueError('Expected exactly one idle class')
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Probabilities must sum to one')
    contact = classes != 0
    if not contact.any():
        raise ValueError('Missing contact classes')
    votes = probabilities[:, contact]
    location = classes[contact][votes.argmax(axis=1)]
    return np.where(votes.sum(axis=1) >= threshold, location, 0)


def checked(item):
    data = json.loads(Path(item['path']).read_text('utf-8'))
    if tuple(data.get('feedback_selectors', (0, 2))) != (0, 2):
        raise ValueError('Non-production gain')
    if any(tuple(f.get('feedback_selectors', (0, 2))) != (0, 2) for f in data['frames']):
        raise ValueError('Mixed frame gain')
    return load(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--development-round', type=Path, action='append', required=True)
    parser.add_argument('--idle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probability-sweep', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be new')
    lock = json.loads(args.lock.read_text('utf-8'))
    train = lock['training']['captures']
    test = [i for path in args.development_round for i in load_manifest(path)]
    if {i['round'] for i in train} & {i['round'] for i in test}:
        raise ValueError('Acquisition round overlap')
    if {i['sha256'] for i in train} & {i['sha256'] for i in test}:
        raise ValueError('Capture overlap')
    training, testing = [checked(i) for i in train], [checked(i) for i in test]
    idle_data, seq, idle = trace_features(args.idle)
    if tuple(idle_data.get('feedback_selectors', (0, 2))) != (0, 2):
        raise ValueError('Idle gain mismatch')
    baseline_end = idle_data.get('commissioning', {}).get('baseline_start_sequence', 16) + 64
    candidates = []
    for variant in ('raw', 'shape'):
        xs, ys = [], []
        for item, values, masks in training:
            features = represent(values, variant)
            negative = features[masks['pre_descent_unseen'] | masks['after_safe_retraction']]
            if len(negative) > 64:
                negative = negative[np.linspace(0, len(negative)-1, 64).astype(int)]
            positive = features[masks['held']]
            xs.extend((negative, positive))
            ys.extend((np.zeros(len(negative), int), np.full(len(positive), item['point'], int)))
        for leaf in (1, 3):
            model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=leaf,
                max_features='sqrt', class_weight='balanced', random_state=20260908,
                n_jobs=-1).fit(np.concatenate(xs), np.concatenate(ys))
            rows, totals, cached = [], {}, []
            for item, values, masks in testing:
                probability = model.predict_proba(represent(values, variant))
                cached.append((item, probability, masks))
                phases = score(model.classes_[probability.argmax(axis=1)], masks, item['point'])
                rows.append(dict(round=item['round'], point=item['point'], phases=phases))
                for phase, counts in phases.items():
                    total = totals.setdefault(phase, dict.fromkeys(counts, 0))
                    for key, value in counts.items():
                        total[key] += value
            idle_probability = model.predict_proba(represent(idle, variant)[seq >= baseline_end])
            idle_pred = model.classes_[idle_probability.argmax(axis=1)]
            result = dict(representation=variant, leaf=leaf, totals=totals,
                independent_idle_frames=len(idle_pred),
                independent_idle_false_contacts=int(np.count_nonzero(idle_pred)), captures=rows)
            if args.probability_sweep:
                sweep = []
                for threshold in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
                    by_round = {}
                    for item, probability, masks in cached:
                        prediction = decode_contact_probability(probability, model.classes_, threshold)
                        phases = score(prediction, masks, item['point'])
                        aggregate = by_round.setdefault(item['round'], {})
                        for phase, counts in phases.items():
                            total = aggregate.setdefault(phase, dict.fromkeys(counts, 0))
                            for key, value in counts.items():
                                total[key] += value
                    idle_prediction = decode_contact_probability(idle_probability, model.classes_, threshold)
                    sweep.append(dict(threshold=threshold, by_round=by_round,
                        independent_idle_false_contacts=int(np.count_nonzero(idle_prediction))))
                result['development_probability_sweep'] = sweep
            candidates.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != 'captures'}), flush=True)
    report = dict(schema='f45_joint_contact_position_development_v1',
        training_lock_sha256=hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        development_captures=test, idle_sha256=hashlib.sha256(args.idle.read_bytes()).hexdigest(),
        deployment_eligible=False, new_blind_test=False, random_frame_split=False,
        temporal_smoothing=False, physical_15hz_verified=False, candidates=candidates)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')


if __name__ == '__main__':
    main()
