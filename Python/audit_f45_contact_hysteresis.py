"""Development-only causal contact hysteresis; no smoothing of position or ADC."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from audit_f45_contact_drift import trace_features, phase_masks
from evaluate_f45_joint_contact_position import score
from select_f45_activation_threshold_v6 import load_manifest


def hysteresis(probability, on=.4, off=.3):
    if not 0 <= off <= on <= 1:
        raise ValueError('Invalid thresholds')
    active = False
    result = []
    for value in probability:
        if not np.isfinite(value) or not -1e-12 <= value <= 1+1e-12:
            raise ValueError('Invalid probability')
        value = np.clip(value, 0, 1)
        active = value >= (off if active else on)
        result.append(active)
    return np.asarray(result, bool)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--round', type=Path, action='append', required=True)
    p.add_argument('--idle', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('New output required')
    payload = joblib.load(a.model)
    model = payload['model']
    contact = model.classes_ != 0
    records = []
    for manifest in a.round:
        for item in load_manifest(manifest):
            data, seq, values = trace_features(item['path'])
            if tuple(data.get('feedback_selectors', (0,2))) != (0,2):
                raise ValueError('Gain mismatch')
            probabilities = model.predict_proba(values)[:, contact]
            records.append((item, data, seq, probabilities))
    results = []
    for off in (.2, .25, .3, .35, .4):
        rows = []
        for item, data, seq, probabilities in records:
            # Model state starts only after baseline is ready, never uses labels.
            valid = seq >= 80
            state = np.zeros(len(seq), bool)
            state[valid] = hysteresis(probabilities[valid].sum(axis=1), .4, off)
            pred = np.where(state, model.classes_[contact][probabilities.argmax(axis=1)], 0)
            rows.append(dict(round=item['round'], point=item['point'],
                phases=score(pred, phase_masks(data,seq),item['point'])))
        idle_results = []
        for path in a.idle:
            data, seq, values = trace_features(path)
            probabilities = model.predict_proba(values[seq >= 80])[:,contact].sum(axis=1)
            state = hysteresis(probabilities, .4, off)
            idle_results.append(dict(path=str(path), frames=len(state), false_contacts=int(state.sum())))
        totals = {}
        for row in rows:
            for phase, counts in row['phases'].items():
                target = totals.setdefault(phase,dict.fromkeys(counts,0))
                for k,v in counts.items(): target[k] += v
        result = dict(on=.4, off=off, totals=totals, captures=rows, idle=idle_results)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='captures'}),flush=True)
    a.output.write_text(json.dumps(dict(schema='f45_hysteresis_development_v1',
        results=results, deployment_eligible=False, new_blind_test=False,
        adc_or_position_smoothed=False, physical_15hz_verified=False),indent=2),'utf-8')


if __name__=='__main__': main()
