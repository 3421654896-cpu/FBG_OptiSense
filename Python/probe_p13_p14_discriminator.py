"""Offline local shape discriminator experiment, never a deployed override."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from select_f45_contact_fullcycle import load
from evaluate_separate_contact_position import shape
from audit_f45_contact_drift import trace_features,phase_masks
from audit_f45_contact_transition import validated_gain_identity


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--model-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.model_output.exists(): p.error('new outputs required')
    lock=json.loads(a.lock.read_bytes())
    train=[r for r in lock['training']['captures'] if r['point'] in (13,14)]
    hashes={r['sha256'] for r in train}
    if any(hashlib.sha256(c.read_bytes()).hexdigest() in hashes for c in a.capture):
        raise ValueError('training/test overlap')
    xs,ys=[],[]
    for item in train:
        _,x,masks=load(item)
        values=shape(x[masks['held']])
        xs.append(values); ys.append(np.full(len(values),item['point']))
    model=make_pipeline(StandardScaler(),SVC(C=1.0,gamma='scale',class_weight='balanced'))
    model.fit(np.concatenate(xs),np.concatenate(ys))
    payload={'schema':'p13_p14_shape_svc_experiment_v1','model':model,
             'created_utc':datetime.now(timezone.utc).isoformat(),'training':train,
             'representation':'unit90','deployment_eligible':False}
    joblib.dump(payload,a.model_output)
    rows=[]
    for path in a.capture:
        data,seq,x=trace_features(path)
        validated_gain_identity(data)
        point=data['commissioning']['target']['point_index']
        if point not in (13,14): raise ValueError('local test requires point 13 or 14')
        masks=phase_masks(data,seq)
        selected=shape(x[masks['held']])
        pred=model.predict(selected)
        labels,counts=np.unique(pred,return_counts=True)
        rows.append({'source':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'point':point,'frames':len(pred),'correct':int((pred==point).sum()),
                     'predictions':dict(zip(map(str,labels),map(int,counts)))})
    report={'schema':'local_p13_p14_development_probe_v1','training_captures':len(train),
            'fixed_parameters':{'C':1,'gamma':'scale','class_weight':'balanced'},
            'model_sha256':hashlib.sha256(a.model_output.read_bytes()).hexdigest(),
            'rows':rows,'contact_detector_bypassed':True,'deployment_eligible':False,
            'physical_15hz_verified':False,'new_blind_test':False}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
