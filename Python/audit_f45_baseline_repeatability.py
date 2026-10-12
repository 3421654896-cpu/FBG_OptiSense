"""Offline baseline and contact-shape audit; no hardware commands or model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_f45_contact_drift import trace_features, phase_masks
from audit_f45_contact_transition import validated_gain_identity


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def inspect(path, expected_hash):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_hash:
        raise ValueError('Capture hash changed')
    data,seq,features=trace_features(path)
    validated_gain_identity(data)
    masks=phase_masks(data,seq)
    c=data['commissioning']
    receipts={r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
    events={e['kind']:e['host_monotonic_ns'] for e in c['events']}
    base=masks['baseline_fit_window']
    if max(receipts[int(s)] for s in seq[base])>=events['contact_descent_started']:
        raise ValueError('Baseline overlaps descent')
    first=np.array([[r['first_code'] for r in f['records']] for f in data['frames']],float)
    baseline=np.median(first[base],axis=0)
    held=np.median(features[masks['held'],:45],axis=0)
    unseen=np.median(features[masks['pre_descent_unseen'],:45],axis=0)
    return dict(source=str(path),sha256=expected_hash,depth_mm=c['depth_mm'],
        baseline_codes=baseline.tolist(),held_delta_codes=held.tolist(),
        baseline_noise_rms=rms(np.std(first[base],axis=0)),
        unseen_idle_offset_rms=rms(unseen),
        held_response_rms=rms(held),
        baseline_per_grating_peak_codes=baseline.reshape(9,5).max(axis=1).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('output exists')
    source=json.loads(a.comparison.read_bytes())
    rows=[inspect(Path(r['capture']),r['sha256']) for r in source['captures']]
    pairs=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            b=np.array(rows[j]['baseline_codes'])-rows[i]['baseline_codes']
            h=np.array(rows[j]['held_delta_codes'])-rows[i]['held_delta_codes']
            pairs.append(dict(indices=[i,j],baseline_difference_rms=rms(b),
                held_response_difference_rms=rms(h),
                baseline_difference_by_grating=np.sqrt(np.mean(b.reshape(9,5)**2,axis=1)).tolist()))
    report=dict(rows=rows,pairs=pairs,hardware_commands_sent=False,
        limitation='Idle spectral changes cannot distinguish optical drift from residual mechanical relaxation; no physical 15 Hz verification.')
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=[{k:v for k,v in r.items() if k not in ('baseline_codes','held_delta_codes','source','sha256')} for r in rows],pairs=pairs)))


if __name__=='__main__': main()
