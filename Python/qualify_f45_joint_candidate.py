"""Freeze development-selected classifier; evaluate new idle captures without refit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from evaluate_f45_joint_contact_position import checked, decode_contact_probability, score
from audit_f45_contact_drift import trace_features, phase_masks
from run_f45_depth_round_visible import verify_capture


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--lock', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    idle = sub.add_parser('idle')
    idle.add_argument('--model', type=Path, required=True)
    idle.add_argument('--capture', type=Path, required=True)
    idle.add_argument('--output', type=Path, required=True)
    contact = sub.add_parser('contact')
    contact.add_argument('--model', type=Path, required=True)
    contact.add_argument('--capture', type=Path, required=True)
    contact.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be new')
    if a.action == 'freeze':
        lock = json.loads(a.lock.read_text('utf-8'))
        xs, ys = [], []
        for item in lock['training']['captures']:
            item, values, masks = checked(item)
            neg = values[masks['pre_descent_unseen'] | masks['after_safe_retraction']]
            if len(neg) > 64:
                neg = neg[np.linspace(0, len(neg)-1, 64).astype(int)]
            pos = values[masks['held']]
            xs.extend((neg, pos))
            ys.extend((np.zeros(len(neg), int), np.full(len(pos), item['point'], int)))
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced', random_state=20260908,
            n_jobs=-1).fit(np.concatenate(xs), np.concatenate(ys))
        # Keep fitting parallel, but avoid per-frame thread scheduling overhead.
        # Real-frame benchmark preserves probabilities with a single worker.
        model.n_jobs = 1
        payload = dict(schema='f45_joint_candidate_v1', model=model, threshold=.4,
            representation='raw135', baseline_start=16, baseline_count=64,
            inference_n_jobs=1,
            feedback_selectors=(0, 2), created_utc=datetime.now(timezone.utc).isoformat(),
            training_lock_sha256=sha(a.lock), captures=lock['training']['captures'],
            deployment_eligible=False, physical_15hz_verified=False)
        joblib.dump(payload, a.output)
        print(json.dumps(dict(model=str(a.output), sha256=sha(a.output), threshold=.4)))
        return
    payload = joblib.load(a.model)
    if payload['schema'] != 'f45_joint_candidate_v1':
        raise ValueError('Unknown candidate')
    data, seq, features = trace_features(a.capture)
    if a.action == 'idle' and (not data.get('complete') or data.get('cnc_motion_commands') != 0):
        raise ValueError('Expected complete no-motion capture')
    if a.action == 'contact':
        point = int(data['commissioning']['target']['point_index'])
        depth = float(data['commissioning']['depth_mm'])
        verify_capture(a.capture, point, depth, ch1_selector=2)
    cleanup = data.get('safety_cleanup', {})
    if not cleanup.get('exact_disarm_ack') or not cleanup.get('exact_soa_shutter_ack'):
        raise ValueError('Optical shutdown not confirmed')
    if data.get('commissioning', {}).get('baseline_start_sequence', 16) != payload['baseline_start']:
        raise ValueError('Baseline protocol mismatch')
    if tuple(data.get('feedback_selectors', ())) != tuple(payload['feedback_selectors']):
        raise ValueError('Gain mismatch')
    if any(tuple(f.get('feedback_selectors', ())) != tuple(payload['feedback_selectors']) for f in data['frames']):
        raise ValueError('Frame gain mismatch')
    if datetime.fromisoformat(data['created_utc']) <= datetime.fromisoformat(payload['created_utc']):
        raise ValueError('Capture must follow model freeze')
    valid = seq >= payload['baseline_start']+payload['baseline_count']
    if a.action == 'contact':
        probability = payload['model'].predict_proba(features)
        prediction = decode_contact_probability(probability, payload['model'].classes_, payload['threshold'])
        masks = phase_masks(data, seq)
        held = prediction[masks['held']]
        labels, counts = np.unique(held, return_counts=True)
        report = dict(schema='f45_joint_frozen_contact_validation_v1',
            model_sha256=sha(a.model), capture_sha256=sha(a.capture), point=point,
            depth_mm=depth, phases=score(prediction, masks, point),
            held_prediction_counts={str(k):int(v) for k,v in zip(labels, counts)},
            hardware_analysis=data['analysis'], safety_cleanup=cleanup,
            physical_15hz_verified=False, deployment_eligible=False)
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    probability = payload['model'].predict_proba(features[valid])
    predictions = decode_contact_probability(probability, payload['model'].classes_, payload['threshold'])
    report = dict(schema='f45_joint_frozen_idle_validation_v1', model_sha256=sha(a.model),
        capture_sha256=sha(a.capture), evaluated_frames=len(predictions),
        false_contact_frames=int(np.count_nonzero(predictions)),
        false_contact_sequences=seq[valid][predictions != 0].tolist(),
        hardware_analysis=data['analysis'], safety_cleanup=data['safety_cleanup'],
        physical_15hz_verified=False, deployment_eligible=False)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
