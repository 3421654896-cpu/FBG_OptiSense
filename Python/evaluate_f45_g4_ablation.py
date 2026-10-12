"""Development-only G4 ablation. No deployment or hardware changes."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from evaluate_f45_joint_contact_position import checked
from audit_f45_contact_drift import trace_features
from audit_f45_contact_transition import validated_gain_identity, transition
from audit_f45_contact_hysteresis import hysteresis
from select_f45_activation_threshold_v6 import load_manifest
from f45_shape_amplitude import features as shape_amplitude_features


def kept_columns():
    return np.array([i for i in range(135) if i%45 not in range(15,20)])


def candidate_features(data,x,variant):
    if variant=='without_g4': return x[:,kept_columns()]
    frames=data['frames']; start=data.get('commissioning',{}).get('baseline_start_sequence',16)
    base=[f for f in frames if start<=f['sequence']<start+64]
    if len(base)!=64: raise ValueError('Incomplete baseline')
    if variant=='shape_amplitude':
        def codes(items):
            return [[r['first_code'] for r in f['records']]+[r['second_code'] for r in f['records']] for f in items]
        return shape_amplitude_features(codes(frames),codes(base))
    baseline=np.median([[r['first_code'] for r in f['records']] +
                        [r['second_code'] for r in f['records']] for f in base],axis=0)
    # Only pre-contact information: no held, release, label or depth included.
    return np.concatenate([x,np.broadcast_to(baseline,(len(x),90))],axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--reference-model',type=Path,required=True)
    p.add_argument('--round',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=('without_g4','baseline_context','shape_amplitude'),default='without_g4')
    p.add_argument('--idle',type=Path,action='append',default=[])
    a=p.parse_args()
    if a.output.exists(): p.error('output exists')
    lock=json.loads(a.lock.read_bytes()); xs=[]; ys=[]; training_hashes=set()
    for item in lock['training']['captures']:
        item,x,masks=checked(item)
        x=candidate_features(json.loads(Path(item['path']).read_bytes()),x,a.variant)
        training_hashes.add(item['sha256'])
        neg=x[masks['pre_descent_unseen']|masks['after_safe_retraction']]
        if len(neg)>64: neg=neg[np.linspace(0,len(neg)-1,64).astype(int)]
        pos=x[masks['held']]
        xs.extend([neg,pos]); ys.extend([np.zeros(len(neg),int),np.full(len(pos),item['point'],int)])
    keep=kept_columns()
    candidate=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=1,max_features='sqrt',
        class_weight='balanced',random_state=20260908,n_jobs=-1)
    candidate.fit(np.concatenate(xs),np.concatenate(ys)); candidate.n_jobs=1
    reference=joblib.load(a.reference_model)['model']; reference.n_jobs=1
    rows=[]
    for manifest in a.round:
        for item in load_manifest(manifest):
            path=Path(item['path']); digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in training_hashes: raise ValueError('Training/test overlap')
            data,seq,x=trace_features(path); validated_gain_identity(data)
            c=data['commissioning']; valid=seq>=c.get('baseline_start_sequence',16)+64
            receipts={r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
            times=[receipts[int(s)] for s in seq[valid]]
            events={e['kind']:e['host_monotonic_ns'] for e in c['events']}
            row=dict(point=item['point'],round=item['round'],sha256=digest,models={})
            candidate_x=candidate_features(data,x,a.variant)
            for name,model,values in [('all9',reference,x[valid]),(a.variant,candidate,candidate_x[valid])]:
                mask=model.classes_!=0; probability=model.predict_proba(values)[:,mask]
                active=hysteresis(probability.sum(axis=1),.4,.3)
                pred=np.where(active,model.classes_[mask][probability.argmax(axis=1)],0)
                row['models'][name]=transition(times,pred,events,item['point'])
            rows.append(row)
    idle_results=[]
    for path in a.idle:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in training_hashes: raise ValueError('Idle overlaps training')
        data,seq,x=trace_features(path); validated_gain_identity(data)
        if not data['complete'] or data['cnc_motion_commands']!=0: raise ValueError('Invalid idle')
        values=candidate_features(data,x,a.variant)[seq>=80]
        mask=candidate.classes_!=0; prob=candidate.predict_proba(values)[:,mask].sum(axis=1)
        idle_results.append(dict(source=str(path),sha256=digest,frames=len(prob),
            false_contact_frames=int(hysteresis(prob,.4,.3).sum())))
    report=dict(rows=rows,idle_results=idle_results,training_lock_sha256=hashlib.sha256(a.lock.read_bytes()).hexdigest(),
        reference_model_sha256=hashlib.sha256(a.reference_model.read_bytes()).hexdigest(),
        candidate_variant=a.variant,
        excluded_feature_columns=sorted(set(range(135))-set(map(int,keep))) if a.variant=='without_g4' else [],
        development_only=True,blind_test=False,deployed=False,physical_15hz_verified=False)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(idle_results),flush=True)
    for row in rows:
        print(row['round'],row['point'],{k:v['whole_depth_hold'] for k,v in row['models'].items()},flush=True)


if __name__=='__main__': main()
