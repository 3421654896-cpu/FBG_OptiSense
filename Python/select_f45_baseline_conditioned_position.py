"""Offline whole-round position audit using causal baseline-conditioned features.

Projection coefficients are descriptive features, not calibrated wavelength or
force measurements. Contact detection is assessed separately from this audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from audit_f45_contact_drift import trace_features, phase_masks
from select_f45_activation_threshold_v6 import load_manifest
from select_f45_spatial_temporal_decoder_v5 import Decoder, decode


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True),1e-12)


def features(path, wavelength):
    wavelength=np.asarray(wavelength,float)
    if wavelength.shape!=(45,) or not np.isfinite(wavelength).all():
        raise ValueError('Expected 45 finite calibrated wavelength values')
    if any(len(np.unique(wavelength[i:i+5]))!=5 for i in range(0,45,5)):
        raise ValueError('Duplicate wavelength within a grating')
    data, seq, delta = trace_features(path)
    if data.get('feedback_selectors',[0,2]) != [0,2]:
        raise ValueError('Different analog gains must have independent qualification')
    masks=phase_masks(data,seq)
    raw=np.array([[[r['first_code'],r['second_code']] for r in f['records']]
                  for f in data['frames']],float).transpose(0,2,1)
    base=np.median(raw[masks['baseline_fit_window']],axis=0)
    pair=delta[:,:90]
    normalized_base=unit(base.reshape(1,90))
    contextual=np.concatenate([unit(pair),np.repeat(normalized_base,len(pair),axis=0)],axis=1)
    coefficients=[]
    for sample in range(2):
        for sensor in range(9):
            section=slice(sensor*5,(sensor+1)*5)
            x=wavelength[section]
            b=base[sample,section]
            gradient=np.gradient(b,x)
            design=np.column_stack([gradient,b,np.ones(5)])
            scale=np.maximum(np.linalg.norm(design,axis=0),1e-9)
            # Ridge stabilizes nearly flat/collinear five-point shapes.
            design=design/scale
            inverse=np.linalg.solve(design.T@design+np.eye(3)*.01,design.T)
            d=pair[:,sample*45+sensor*5:sample*45+sensor*5+5]
            coefficients.append(d@inverse.T)
    projection=np.concatenate(coefficients,axis=1)
    return {'pair_unit90':unit(pair)[masks['held']],
            'baseline_context180':contextual[masks['held']],
            'projection_plus_pair':np.concatenate([unit(pair),unit(projection)],axis=1)[masks['held']]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--development-round',type=Path,action='append',default=[])
    p.add_argument('--external-capture',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('all','lda','extra_trees'),default='all')
    p.add_argument('--tree-hard',action='store_true')
    a=p.parse_args()
    if a.output.exists(): p.error('output must be new')
    lock=json.loads(a.lock.read_text('utf-8'))
    rows=json.loads(Path(__file__).with_name('mode_tables_from_fullband_2001.json').read_text('utf-8'))['modes']['stress']['rows']
    wavelengths=np.array([r['measured_wavelength_nm'] for r in rows])
    records=list(lock['training']['captures'])
    for manifest in a.development_round: records.extend(load_manifest(manifest))
    captures=[]
    for item in records:
        path=Path(item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('capture hash mismatch')
        captures.append({**item,'features':features(path,wavelengths)})
    external=features(a.external_capture,wavelengths) if a.external_capture else None
    external_truth=(json.loads(a.external_capture.read_text('utf-8'))['commissioning']['target']['point_index']
                    if a.external_capture else None)
    coords={int(k):np.array(v) for k,v in lock['position']['coordinates_by_point'].items()}
    decoder=Decoder('posterior_temperature_3.0_ema_0.25','posterior_ema',3.,.25)
    rounds=sorted({r['round'] for r in captures})
    results=[]
    for variant in ('pair_unit90','baseline_context180','projection_plus_pair'):
        for name,template in [('lda',make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))),
             ('extra_trees',ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,
                         max_features='sqrt',random_state=20260908,n_jobs=-1))]:
            if a.model != 'all' and a.model != name:
                continue
            active_decoder = Decoder('hard_xy','hard',0.) if name=='extra_trees' and a.tree_hard else decoder
            folds=[]
            for held_round in rounds:
                train=[r for r in captures if r['round']!=held_round]
                test=[r for r in captures if r['round']==held_round]
                model=clone(template).fit(np.concatenate([r['features'][variant] for r in train]),
                    np.concatenate([np.full(len(r['features'][variant]),r['point']) for r in train]))
                details=[]
                for row in test:
                    xy,labels=decode(model,row['features'][variant],active_decoder,coords)
                    centre=xy.mean(axis=0)
                    label=min(coords,key=lambda k:np.linalg.norm(coords[k]-centre))
                    error=np.linalg.norm(xy-coords[row['point']],axis=1)
                    details.append({'point':row['point'],'predicted':int(label),
                        'trial_error_mm':float(np.linalg.norm(centre-coords[row['point']])),
                        'frames':len(labels),'frames_within4':int((error<=4).sum())})
                fold={'round':held_round,'exact':sum(d['point']==d['predicted'] for d in details),
                      'total':len(details),'within4':sum(d['trial_error_mm']<=4 for d in details),'details':details}
                folds.append(fold)
                print(json.dumps({'variant':variant,'model':name,'round':held_round,'exact':fold['exact'],'total':len(details)}),flush=True)
            result={'variant':variant,'model':name,'decoder':active_decoder.name,'folds':folds,
                'total_exact':sum(f['exact'] for f in folds),'total_contacts':len(captures),
                'worst_round_exact_fraction':min(f['exact']/f['total'] for f in folds),
                'total_within4':sum(f['within4'] for f in folds)}
            if external is not None:
                model=clone(template).fit(np.concatenate([r['features'][variant] for r in captures]),
                    np.concatenate([np.full(len(r['features'][variant]),r['point']) for r in captures]))
                xy,labels=decode(model,external[variant],active_decoder,coords)
                label=min(coords,key=lambda k:np.linalg.norm(coords[k]-xy.mean(axis=0)))
                result['later_capture']={'truth':external_truth,'predicted':int(label),
                    'trial_error_mm':float(np.linalg.norm(xy.mean(axis=0)-coords[external_truth]))}
            results.append(result)
            report={'schema':'f45_baseline_conditioned_position_development_v1',
                'created_utc':datetime.now(timezone.utc).isoformat(),
                'source_lock_sha256':hashlib.sha256(a.lock.read_bytes()).hexdigest(),
                'wavelengths_nm':wavelengths.tolist(),
                'development_rounds':[str(m.resolve()) for m in a.development_round],
                'later_capture_path':str(a.external_capture.resolve()) if a.external_capture else None,
                'contact_detection_evaluated':False,'random_frame_split_used':False,
                'deployment_eligible':False,'complete':False,'candidates':results}
            a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    report['complete']=True
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')


if __name__=='__main__': main()
