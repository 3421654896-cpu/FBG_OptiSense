"""Replay real frames to compare scheduling cost without changing model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from audit_f45_contact_drift import trace_features
from audit_f45_contact_transition import validated_gain_identity


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('new output required')
    data,seq,x=trace_features(a.capture)
    validated_gain_identity(data)
    eligible=np.flatnonzero(seq>=80)
    indices=eligible[np.linspace(0,len(eligible)-1,96).astype(int)]
    m=joblib.load(a.model)['model']
    outputs, timing={},{}
    for jobs in (-1,1):
        m.n_jobs=jobs
        m.predict_proba(x[indices[:1]])
        ms=[]; predictions=[]
        for i in indices:
            begin=time.perf_counter_ns()
            pred=m.predict_proba(x[i:i+1])
            ms.append((time.perf_counter_ns()-begin)/1e6)
            predictions.append(pred[0])
        outputs[jobs]=np.array(predictions)
        timing[str(jobs)]={'median_ms':float(np.median(ms)),
                          'p95_ms':float(np.percentile(ms,95)),'max_ms':max(ms),
                          'over_31p8ms':int(np.sum(np.array(ms)>31.8))}
    difference=float(np.max(np.abs(outputs[-1]-outputs[1])))
    labels_same=bool(np.array_equal(outputs[-1].argmax(axis=1),outputs[1].argmax(axis=1)))
    report={'schema':'joint_real_frame_inference_benchmark_v1','frames':len(indices),
            'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
            'capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),
            'timing_by_worker_count':timing,'max_probability_difference':difference,
            'same_argmax_classes':labels_same,'equivalent_within_1e_12':difference<=1e-12 and labels_same,
            'hardware_or_model_modified':False,'physical_15hz_verified':False,
            'scope':'model.predict_proba only; excludes acquisition, baseline, GUI and transport costs'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
