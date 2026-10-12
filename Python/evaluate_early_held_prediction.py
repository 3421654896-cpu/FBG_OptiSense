"""Chronological static feasibility test; not physical 15 Hz qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from analyze_sparse_held_trajectory import audit_record


def features(record, horizon=750):
    available = [r for r in record['curve'] if r['end_us'] <= horizon]
    if len(available) < 2:
        raise ValueError('insufficient causally available samples')
    first, last = available[0], available[-1]
    a, b = np.array(first['direct_adc'], float), np.array(last['direct_adc'], float)
    if a.shape != (6,) or b.shape != (6,) or not np.isfinite([a,b]).all():
        raise ValueError('invalid ADC channels')
    if not 0 < first['end_us'] < last['end_us'] <= horizon:
        raise ValueError('invalid timing')
    # Same target, early six-channel levels and observed changes only.
    return np.r_[b, b-a], float(b[1]), last['end_us']


def evaluate(path, train_repeats=7):
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc['schema'] != 'sparse_held_trajectory_v1' or not doc['complete']:
        raise ValueError('incomplete capture')
    safe = doc['safety_cleanup']
    if not safe['exact_disarm_ack'] or not safe['exact_soa_shutter_ack']:
        raise ValueError('cleanup unconfirmed')
    groups, excluded = {}, []
    for r in doc['captures']:
        try:
            audit_record(r)  # independently validate held teacher and intervals
            if r['prefix_profile'] != 2:
                raise ValueError('not the preselected profile 2 route')
        except ValueError as exc:
            excluded.append({'point':r['point_index'],'repeat':r['repeat'],'reason':str(exc)})
            continue
        groups.setdefault(r['point_index'], []).append(r)
    predictions, models, unavailable = [], [], []
    for point, records in sorted(groups.items()):
        repeats = [r['repeat'] for r in records]
        if repeats != sorted(set(repeats)):
            raise ValueError('requires chronological unique repeats')
        if len({r['prefix_profile'] for r in records}) != 1:
            raise ValueError('mixed prefix routes')
        x = np.array([features(r)[0] for r in records])
        y = np.array([r['teacher']['median'][3] for r in records],float)
        training = np.array([r['repeat'] < train_repeats for r in records])
        testing = np.flatnonzero(~training)
        if training.sum() < 4 or len(testing) < 1:
            unavailable.append({'point':point,'train_count':int(training.sum()),
                                'test_count':len(testing),'reason':'insufficient qualified chronological train/test records'})
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[training], y[training])
        baseline = float(np.median(y[training]))
        predicted = model.predict(x[testing])
        models.append({'point':point,'qualified_training_count':int(training.sum()),
                       'training_teacher_range':[float(y[training].min()),float(y[training].max())]})
        for i, pred in zip(testing, predicted):
            _, direct, end = features(records[i])
            predictions.append({'point':point,'repeat':records[i]['repeat'],'teacher':float(y[i]),
                                'predicted':float(pred),'direct':direct,
                                'training_median':baseline,'last_feature_end_us':end})
    scores = {}
    for name in ('predicted','direct','training_median'):
        if not predictions: break
        err = np.array([r[name]-r['teacher'] for r in predictions])
        scores[name] = {'mae_codes':float(np.abs(err).mean()),
                        'rmse_codes':float(np.sqrt(np.mean(err**2))),
                        'max_abs_codes':float(np.abs(err).max())}
    return {'schema':'early_held_feasibility_v1','source':str(path.resolve()),
            'source_sha256':hashlib.sha256(raw).hexdigest(),
            'policy':'per-point StandardScaler/Ridge alpha=10, fixed before test; first 7 repeats train, final 3 test; no tuning on test',
            'scope':'static short-horizon feasibility only; no contact variation, no physical 15 Hz claim, no deployment',
            'models':models,'unavailable_points':unavailable,'excluded':excluded,
            'all_requested_points_evaluated':not unavailable and len(groups)==3,
            'test_count':len(predictions),'scores':scores,'predictions':predictions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = evaluate(args.input)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['scores']))
