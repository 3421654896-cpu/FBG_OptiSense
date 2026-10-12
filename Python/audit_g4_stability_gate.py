"""Replay explicit low-signal gates; thresholds are diagnostic, not qualified limits."""
import argparse
import json
from pathlib import Path
import numpy as np


def window_quality(samples, sigma_limit=1., drift_limit=2.):
    values=np.asarray(samples,float)
    if values.ndim!=2 or values.shape[1]!=6 or len(values)<15 or not np.isfinite(values).all():
        raise ValueError('Requires at least 15 finite six-channel samples')
    x=values[-25:,3]
    median=float(np.median(x)); sigma=float(1.4826*np.median(abs(x-median)))
    n=max(3,len(x)//3)
    drift=float(abs(np.median(x[-n:])-np.median(x[:n])))
    return dict(median=median,sigma=sigma,drift=drift,
                window_pass=sigma<=sigma_limit and drift<=drift_limit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('output exists')
    data=json.loads(a.source.read_bytes())
    if not data['complete']: raise ValueError('Incomplete scan')
    rows={}
    for r in data['rows']:
        key=(r['repeat'],r['index'])
        if key in rows: raise ValueError('Duplicate record')
        rows[key]=window_quality(r['adc']['monitor_samples'])
    pairs=[]
    for index in data['indices']:
        first,second=rows[0,index],rows[1,index]
        delta=abs(first['median']-second['median'])
        pairs.append(dict(index=index,both_windows_pass=first['window_pass'] and second['window_pass'],
            repeat_difference=delta,repeat_within_3_codes=delta<=3.))
    report=dict(diagnostic_only=True,deployed=False,sigma_limit_codes=1.,drift_limit_codes=2.,
        repeat_limit_codes=3.,window_passes=sum(r['window_pass'] for r in rows.values()),
        total_windows=len(rows),pairs=pairs,
        passing_window_pairs_with_failed_repeat=sum(r['both_windows_pass'] and not r['repeat_within_3_codes'] for r in pairs))
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'}))


if __name__=='__main__': main()
