"""Fixed raw-delta nearest-prototype check on independent physical repeats.

Not a deployed model, full spatial validation, dynamic test, or area estimator.
The feature and metric are fixed before inspecting the new repeat captures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_fast_fullmap_position import _capture_feature


def nearest(train_vectors, query):
    train = np.asarray(train_vectors, dtype=float)
    query = np.asarray(query, dtype=float)
    if train.ndim != 2 or query.shape != train.shape[1:]:
        raise ValueError('Inconsistent feature dimensions')
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError('Nonfinite features')
    distances = np.sqrt(np.mean((train-query)**2,axis=1))
    return int(np.argmin(distances)), distances


def spectral_feature(baseline, sample, mode):
    baseline=np.asarray(baseline,dtype=float)
    sample=np.asarray(sample,dtype=float)
    if baseline.shape!=(45,) or sample.shape!=(45,) or not np.isfinite([baseline,sample]).all():
        raise ValueError('45 finite physical samples required')
    if mode=='raw_delta': return sample-baseline
    if mode!='shape_delta': raise ValueError('Unknown feature')
    def shape(values):
        segments=values.reshape(9,5)
        segments=segments-np.min(segments,axis=1,keepdims=True)
        return segments/np.maximum(np.linalg.norm(segments,axis=1,keepdims=True),1.0)
    return (shape(sample)-shape(baseline)).ravel()


def capture_parts(path, baseline_mode='early64'):
    _capture_feature(path)  # protocol, completeness, shutdown and motion gates
    data=json.loads(path.read_text('utf-8'))
    baseline=np.median([[r['first_code'] for r in f['records']]
                        for f in data['frames'] if 16 <= f['sequence'] < 80],axis=0)
    events={e['kind']:e['host_monotonic_ns'] for e in data['commissioning']['events']}
    receipts={r['sequence']:r['host_monotonic_ns'] for r in data['commissioning']['frame_receipts']}
    if baseline_mode == 'recent64':
        candidates = [f for f in data['frames'] if f['sequence'] >= 16
                      and receipts[f['sequence']] < events['contact_descent_started']]
        candidates.sort(key=lambda f: f['sequence'])
        if len(candidates) < 64:
            raise ValueError('Insufficient pre-descent baseline frames')
        baseline = np.median([[r['first_code'] for r in f['records']]
                              for f in candidates[-64:]], axis=0)
    elif baseline_mode != 'early64':
        raise ValueError('Unknown baseline mode')
    held=np.asarray([[r['first_code'] for r in f['records']] for f in data['frames']
        if events['contact_reached_commanded_depth']+200_000_000 <= receipts[f['sequence']] < events['release_started']],dtype=float)
    return baseline,held


def evaluate(root, mode='raw_delta'):
    training = [root/'f45_visible_p08_v46_20260907_r1/capture.json',
                root/'f45_visible_center_v46_20260907_r2/capture.json',
                root/'f45_visible_p10_v46_20260907_r1/capture.json']
    validation = [root/'f45_visible_p08_v46_20260907_r2/capture.json',
                  root/'f45_visible_center_v46_20260907_r3/capture.json',
                  root/'f45_visible_p10_v46_20260907_r2/capture.json']
    if set(p.resolve() for p in training)&set(p.resolve() for p in validation):
        raise ValueError('Training/validation file overlap')
    tr = [_capture_feature(p) for p in training]
    va = [_capture_feature(p) for p in validation]
    parts=[capture_parts(p) for p in training]
    vectors = [spectral_feature(b,np.median(h,axis=0),mode) for b,h in parts]
    rows = []
    for p, record in zip(validation,va):
        baseline,held=capture_parts(p)
        index, distances = nearest(vectors,spectral_feature(baseline,np.median(held,axis=0),mode))
        predictions=[]
        for sample in held:
            predicted,_=nearest(vectors,spectral_feature(baseline,sample,mode))
            predictions.append(tr[predicted]['point_index'])
        rows.append(dict(source=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            truth=record['point_index'],predicted=tr[index]['point_index'],
            position_error_mm=float(np.linalg.norm(record['xy']-tr[index]['xy'])),
            rms_distances_to_points_8_9_10=distances.tolist(),
            held_frame_predictions=predictions,
            held_frame_correct=sum(p==record['point_index'] for p in predictions),
            held_frame_total=len(predictions)))
    return dict(schema='current_three_position_repeat_check_v1',
        method=f'{mode}, unscaled Euclidean nearest prototype; training held medians, single-frame queries',
        training=[dict(source=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in training],
        validation=rows,correct=sum(r['truth']==r['predicted'] for r in rows),total=len(rows),
        scope='Three fixed positions at 0.30 mm only; exploratory centre repeat already inspected',
        deployed=False,full_spatial_accuracy_verified=False,physical_15hz_verified=False,area_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('../outputs'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--feature',choices=('raw_delta','shape_delta'),default='raw_delta')
    args=parser.parse_args()
    result=evaluate(args.root,args.feature)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
