"""Experimental, grouped evaluation of causal CH1-to-stable predictors.

No hardware access or deployment. Inputs exclude teacher values, wavelength,
table row, grating identity, timestamps identifying a trial, and future samples.
The models are homogeneous in CH1 amplitude (zero input cannot become a stored
reflection peak). This property alone does not prove pressure generalisation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_production_pair_teacher import analyze, finite_code
from capture_production_pair_teacher import GAIN_KOHM

MODEL_NAMES = ('ridge_pair', 'ridge_optical', 'ratio_optical')
BASELINES = ('last_direct', 'firmware_rc')
CONTRACT = 'same_trial_causal_homogeneous_v1'


def causal_features(r, kind):
    """Read only causal fields; teacher/sample identity deliberately unused."""
    if kind not in MODEL_NAMES:
        raise ValueError('unknown model')
    reads = r['production_reads']
    if [v['kind'] for v in reads] not in (['first','second'], ['first','second','slow_recheck']):
        raise ValueError('wrong production read order')
    previous_end = -1
    for v in reads:
        finite_code(v['direct_ch1'])
        if not previous_end < v['start_us'] < v['end_us'] < 300000:
            raise ValueError('noncausal production read times')
        previous_end = v['end_us']
    gain = GAIN_KOHM[r['feedback_selectors'][1]] / 2.0
    first, last = [float(v['direct_ch1']) for v in reads[-2:]]
    if kind == 'ridge_pair':
        # Four independently learned gain conditions, each without an intercept.
        onehot = np.array([float(r['feedback_selectors'][1] == s) for s in range(4)])
        return np.outer([first/gain, last/gain], onehot).ravel(), previous_end
    state = r['optical_state']
    if not previous_end < state['start_us'] < state['end_us'] < 300000:
        raise ValueError('optical state is not a causal post-pair feature')
    if len(state['direct_adc']) != 6:
        raise ValueError('invalid optical state channels')
    for v in state['direct_adc']:
        finite_code(v)
    ch1, pdt, pdr = [float(state['direct_adc'][i]) for i in (1,4,5)]
    context = np.array([1.0, np.log2(gain), pdt/4095, pdr/4095,
                        pdt/max(pdt+pdr, 1.0)])
    if kind == 'ridge_optical':
        return np.outer(np.array([first,last,ch1])/gain, context).ravel(), state['end_us']
    if last == 0:
        return None, state['end_us']  # multiplicative model cannot recover a zero read
    x = np.r_[first/last, ch1/last, context[1:],
              (reads[-1]['end_us']-reads[-2]['end_us'])/1000,
              float(len(reads)==3)]
    return x, state['end_us']


def load_dataset(paths, schedule_version=5):
    rows, sources, skipped = [], [], Counter()
    source_hashes, trial_ids, table_hashes = set(), set(), set()
    for path in paths:
        path = Path(path)
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        if digest in source_hashes:
            raise ValueError('duplicate input file content')
        source_hashes.add(digest)
        doc = json.loads(data)
        audited = analyze(doc)
        if not audited['source_capture_complete']:
            raise ValueError('incomplete acquisition cannot be used for fitting')
        table_hashes.add(hashlib.sha256(json.dumps(
            [r['codes'] for r in doc['rows']], separators=(',',':')).encode()).hexdigest())
        sources.append({'path':str(path.resolve()), 'sha256':digest})
        for raw, checked in zip(doc['captures'], audited['records'], strict=True):
            if raw['schedule_version'] != schedule_version:
                skipped['other_schedule'] += 1; continue
            if not checked['teacher_stable']:
                skipped['unstable_teacher'] += 1; continue
            # Capture tags identify actual trials across copied/overlapping files.
            key = (raw['request_tag'], raw['frame_start_ms'], raw['point_index'])
            if key in trial_ids:
                raise ValueError('duplicate captured trial')
            trial_ids.add(key)
            features = {name: causal_features(raw,name) for name in MODEL_NAMES}
            rows.append({'id': f'{digest[:12]}:{raw["request_tag"]}', 'raw':raw,
                         'features':features, 'teacher':checked['teacher_code'],
                         'gain_scale':checked['gain_kohm']/2.0,
                         'grating':raw['grating'], 'repeat':raw['repeat'],
                         'source_sha256':digest})
    if len(table_hashes) != 1 or not rows:
        raise ValueError('one verified table and nonempty records required')
    return rows, {'sources':sources, 'table_sha256':next(iter(table_hashes)),
                  'schedule_version':schedule_version, 'qualified_rows':len(rows),
                  'excluded':dict(skipped)}


def grouped_folds(rows, group_field):
    groups = sorted({r[group_field] for r in rows})
    if len(groups) < 2:
        raise ValueError('at least two independent groups required')
    for group in groups:
        train = [i for i,r in enumerate(rows) if r[group_field] != group]
        test = [i for i,r in enumerate(rows) if r[group_field] == group]
        if len(train) < 8 or not test:
            raise ValueError('insufficient grouped training rows')
        yield group, train, test


def train_candidate(name, rows):
    selected = [r for r in rows if r['features'][name][0] is not None]
    if len(selected) < 8:
        raise ValueError('insufficient eligible causal samples')
    x = np.stack([r['features'][name][0] for r in selected])
    if name == 'ratio_optical':
        y = np.array([r['teacher']/r['raw']['production_reads'][-1]['direct_ch1'] for r in selected])
        estimator = ExtraTreesRegressor(n_estimators=96, max_depth=3,
                                       min_samples_leaf=4, random_state=20260905, n_jobs=1)
    else:
        y = np.array([r['teacher']/r['gain_scale'] for r in selected])
        # Fit scaling ONLY on this training fold; no centring/intercept keeps
        # exact amplitude homogeneity and prevents a point-independent offset.
        estimator = make_pipeline(StandardScaler(with_mean=False),
                                  Ridge(alpha=1.0, fit_intercept=False))
    estimator.fit(x,y)
    return estimator


def predict_candidate(name, estimator, rows):
    out = np.full(len(rows), np.nan)
    available = [i for i,r in enumerate(rows) if r['features'][name][0] is not None]
    if available:
        x = np.stack([rows[i]['features'][name][0] for i in available])
        factors = [rows[i]['raw']['production_reads'][-1]['direct_ch1'] if name == 'ratio_optical'
                   else rows[i]['gain_scale'] for i in available]
        out[available] = estimator.predict(x) * np.asarray(factors)
    return out  # never clip poor predictions into apparent agreement


def metrics(rows, predicted):
    y = np.array([r['teacher'] for r in rows]); scale = np.array([r['gain_scale'] for r in rows])
    finite = np.isfinite(predicted); error = predicted[finite]-y[finite]
    if not len(error):
        return {'total':len(rows), 'predicted':0, 'coverage':0.0}
    normalized = error/scale[finite]
    return {'total':len(rows), 'predicted':int(np.sum(finite)), 'coverage':float(np.mean(finite)),
            'mae_adc_codes':float(np.mean(np.abs(error))),
            'rmse_adc_codes':float(np.sqrt(np.mean(error**2))),
            'max_abs_adc_codes':float(np.max(np.abs(error))),
            'mae_2k_equivalent_codes':float(np.mean(np.abs(normalized))),
            'max_abs_2k_equivalent_codes':float(np.max(np.abs(normalized))),
            'within_diagnostic_tolerance_count':int(np.sum(np.abs(error)<=np.maximum(3,.1*y[finite]))),
            'out_of_adc_range_count':int(np.sum((predicted[finite]<0)|(predicted[finite]>4095)))}


def evaluate_grouping(rows, grouping):
    predictions = {name:np.full(len(rows),np.nan) for name in MODEL_NAMES}
    folds = []
    for group,train,test in grouped_folds(rows, grouping):
        training = [rows[i] for i in train]; testing = [rows[i] for i in test]
        assert set(r['id'] for r in training).isdisjoint(r['id'] for r in testing)
        for name in MODEL_NAMES:
            estimator = train_candidate(name, training)
            predictions[name][test] = predict_candidate(name,estimator,testing)
        folds.append({'held_out_group':group, 'training_ids':[r['id'] for r in training],
                      'testing_ids':[r['id'] for r in testing]})
    predictions['last_direct'] = np.array([r['raw']['production_reads'][-1]['direct_ch1'] for r in rows])
    predictions['firmware_rc'] = np.array([r['raw']['firmware_rc_estimate'] for r in rows])
    return {'grouping':grouping, 'folds':folds, 'random_frame_split':False,
            'metrics':{name:metrics(rows,p) for name,p in predictions.items()},
            'per_grating':{str(g):{name:metrics([rows[i] for i in ids],p[ids]) for name,p in predictions.items()}
                           for g in sorted({r['grating'] for r in rows})
                           for ids in [[i for i,r in enumerate(rows) if r['grating']==g]]},
            'predictions':[{'id':r['id'], 'grating':r['grating'], 'repeat':r['repeat'],
                            'teacher':r['teacher'], 'gain_scale':r['gain_scale'],
                            'predicted':{name:float(p[i]) if np.isfinite(p[i]) else None for name,p in predictions.items()}}
                           for i,r in enumerate(rows)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--save-candidates',type=Path)
    args = p.parse_args()
    rows, provenance = load_dataset(args.inputs)
    report = {'schema':CONTRACT, 'provenance':provenance, 'deployable':False,
              'physical_pressure_tested':False, 'real_contact_area_tested':False,
              'feature_policy':__doc__, 'preset_hyperparameters_not_test_tuned':True,
              'notes':['Only no-commanded-motion trials; not pressure/area ground truth.',
                       'Optical candidates require an additional six-channel acquisition, not available in normal frames.',
                       '2k-equivalent normalization is nominal transimpedance scaling, not calibrated pressure.',
                       'No wavelength, grating, row or future teacher is a predictor input.',
                       'Prospective data must be disjoint from all fitted records before claiming validation.'],
              'evaluations':{key:evaluate_grouping(rows,key) for key in ('repeat','grating')}}
    if args.save_candidates:
        if args.save_candidates.exists():
            raise FileExistsError('refusing to overwrite a frozen candidate artifact')
        payload = {'contract':CONTRACT, 'deployable':False, 'provenance':provenance,
                   'training_ids':[r['id'] for r in rows],
                   'models':{name:train_candidate(name,rows) for name in MODEL_NAMES}}
        joblib.dump(payload,args.save_candidates)
        report['frozen_candidate_file'] = str(args.save_candidates.resolve())
        report['frozen_candidate_sha256'] = hashlib.sha256(args.save_candidates.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:result['metrics'] for key,result in report['evaluations'].items()},indent=2))


if __name__ == '__main__': main()
