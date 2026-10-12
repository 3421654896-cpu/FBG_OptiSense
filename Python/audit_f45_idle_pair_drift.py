"""Attribute idle variation to shared versus within-pair ADC components, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_f45_contact_transition import validated_gain_identity


def analyze(path):
    data=json.loads(path.read_bytes())
    validated_gain_identity(data)
    if not data['complete'] or data['cnc_motion_commands']!=0:
        raise ValueError('Requires complete optical-only capture')
    if not all(data['safety_cleanup'].get(k) for k in ('exact_disarm_ack','exact_soa_shutter_ack')):
        raise ValueError('Missing shutdown confirmation')
    frames=data['frames'][80:]
    a=np.array([[r['first_code'] for r in f['records']] for f in frames],float)
    b=np.array([[r['second_code'] for r in f['records']] for f in frames],float)
    rows=[]
    for i in range(45):
        corr=float(np.corrcoef(a[:,i],b[:,i])[0,1]) if np.std(a[:,i])*np.std(b[:,i])>0 else None
        rows.append(dict(row=i,grating=i//5+1,first_sd=float(np.std(a[:,i])),
            second_sd=float(np.std(b[:,i])),pair_difference_sd=float(np.std(b[:,i]-a[:,i])),
            first_second_correlation=corr))
    groups=[]
    for g in range(9):
        sl=slice(g*5,g*5+5)
        groups.append(dict(grating=g+1,
            first_variance=float(np.var(a[:,sl],axis=0).sum()),
            second_variance=float(np.var(b[:,sl],axis=0).sum()),
            difference_variance=float(np.var(b[:,sl]-a[:,sl],axis=0).sum())))
    total=sum(g['first_variance'] for g in groups)
    for g in groups: g['first_variance_share']=g['first_variance']/total if total else 0
    return dict(source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frames=len(frames),rows=rows,groups=groups,
        limitation='Correlation does not identify optical versus electronic cause; two reads are not independent physical measurements. No physical bandwidth claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('output exists')
    report=analyze(args.source)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(groups=report['groups'],largest_rows=sorted(report['rows'],key=lambda r:r['first_sd'],reverse=True)[:8])))
