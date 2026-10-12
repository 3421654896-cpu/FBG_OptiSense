"""Compare paired local static scans without fitting or changing table/model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(path):
    data=json.loads(path.read_bytes())
    if not data['complete'] or data['cnc_motion_commands']!=0: raise ValueError('Incomplete/nonstatic run')
    if not all(data['safety_cleanup'][k] for k in ('exact_disarm_ack','exact_soa_shutter_ack')):
        raise ValueError('Unconfirmed shutdown')
    rows={(r['repeat'],r['index']):r['adc'] for r in data['rows']}
    if len(rows)!=82 or len(data['rows'])!=82: raise ValueError('Missing/duplicate point')
    values=np.array([[rows[rep,i]['ch1_adc_code'] for i in data['indices']] for rep in (0,1)])
    differences=values[1]-values[0]
    normalization={}
    for name,column in [('PDT',0),('PDR',1)]:
        pd=np.array([[np.median(np.asarray(rows[rep,i]['monitor_samples'])[:,column])
                      for i in data['indices']] for rep in (0,1)],float)
        if not np.isfinite(pd).all() or (pd<=4).any():
            normalization[name]={'valid':False,'reason':'PD at or below diagnostic floor'}
            continue
        # Common scaling retains ADC-equivalent units for comparison. This is
        # an offline ratio-of-medians diagnostic, never applied to live data.
        normalized=values/pd*np.median(pd,axis=0)
        normalization[name]=dict(valid=True,
            repeat_rmse_adc_equivalent=float(np.sqrt(np.mean((normalized[1]-normalized[0])**2))),
            maximum_pd_relative_change=float(np.max(abs(pd[1]-pd[0])/np.mean(pd,axis=0))))
    return dict(source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        repeat_rmse_codes=float(np.sqrt(np.mean(differences**2))),max_difference_codes=float(abs(differences).max()),
        power_normalization_diagnostic=normalization,
        peak_table_indices=[data['indices'][int(np.argmax(v))] for v in values],
        stable_windows=sum(bool(r['stable']) for r in rows.values()),
        paired=[dict(index=i,first=float(values[0,j]),second=float(values[1,j]),difference=float(differences[j])) for j,i in enumerate(data['indices'])],
        current_wavelength_accuracy_verified=False,fast_path_qualified=False,
        limitation='Host-paced 0.3s hold; existing fullband codes, not installed F45 codes; table wavelengths historical. Static maximum sample is not a fitted Bragg center.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists')
    result=analyze(a.source)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='paired'}))
