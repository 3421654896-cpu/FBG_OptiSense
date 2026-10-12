"""Separate detector/localizer development test; no deployment or test refit."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from select_f45_contact_fullcycle import load
from select_f45_activation_threshold_v6 import load_manifest
from audit_f45_contact_drift import trace_features, phase_masks
from audit_f45_contact_transition import validated_gain_identity, transition
from audit_f45_contact_hysteresis import hysteresis


def shape(x):
    pair = np.asarray(x,float)[:,:90]
    return pair/np.maximum(np.linalg.norm(pair,axis=1,keepdims=True),1e-12)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--joint-model',type=Path,required=True)
    p.add_argument('--round',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    lock=json.loads(a.lock.read_bytes())
    train=lock['training']['captures']
    tests=load_manifest(a.round)
    if {r['sha256'] for r in train}&{r['sha256'] for r in tests}:
        raise ValueError('training/test overlap')
    px,py,bx,by=[],[],[],[]
    for i,item in enumerate(train):
        _,x,masks=load(item)
        pos=x[masks['held']]
        neg=x[masks['pre_descent_unseen']|masks['after_safe_retraction']]
        if len(neg)>64: neg=neg[np.linspace(0,len(neg)-1,64).astype(int)]
        px.append(shape(pos)); py.append(np.full(len(pos),item['point']))
        bx.extend((neg,pos)); by.extend((np.zeros(len(neg)),np.ones(len(pos))))
        if i%80==0: print(f'loaded training {i+1}/{len(train)}',flush=True)
    position=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))
    position.fit(np.concatenate(px),np.concatenate(py))
    detector=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5]))
    detector.fit(np.concatenate(bx),np.concatenate(by))
    joint=joblib.load(a.joint_model)['model']
    rows=[]
    for item in tests:
        data,seq,x=trace_features(item['path'])
        validated_gain_identity(data)
        c=data['commissioning']; valid=seq>=c.get('baseline_start_sequence',16)+64
        receipts={r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
        events={r['kind']:r['host_monotonic_ns'] for r in c['events']}
        times=[receipts[int(s)] for s in seq[valid]]
        location=position.predict(shape(x[valid]))
        probabilities=joint.predict_proba(x[valid])[:,joint.classes_!=0].sum(axis=1)
        states={'frozen_joint_contact':hysteresis(probabilities,.4,.3),
                'separate_lda_contact':detector.predict(x[valid]).astype(bool)}
        for name,state in states.items():
            pred=np.where(state,location,0)
            masks=phase_masks(data,seq)
            held=masks['held'][valid]
            labels,counts=np.unique(location[held],return_counts=True)
            rows.append({'point':item['point'],'detector':name,
                         'pre_descent_false_contacts':int(np.count_nonzero(pred[np.asarray(times)<events['contact_descent_started']])),
                         'position_votes_before_gate':dict(zip(map(str,labels),map(int,counts))),
                         'phases':transition(times,pred,events,item['point'])})
    report={'schema':'separate_contact_position_development_v1','training_captures':len(train),
            'training_lock_sha256':hashlib.sha256(a.lock.read_bytes()).hexdigest(),
            'test_round':str(a.round.resolve()),'rows':rows,'deployment_eligible':False,
            'physical_15hz_verified':False,'test_refit':False,'new_blind_test':False,
            'position_smoothing':False}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)
