"""Compare reference drift to meter and internal PD monitors; no causal claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from capture_ch1_teacher_spectrum import sha256_file


def audit(before,after):
    data=[json.loads(p.read_text('utf-8')) for p in (before,after)]
    if any(not d.get('complete') or len(d.get('samples',[]))!=54 for d in data):
        raise ValueError('Complete references required')
    if data[0]['route_rows']!=data[1]['route_rows']:
        raise ValueError('Route changed')
    monitor=np.array([[np.median(s['point']['monitor_samples'],axis=0)
        for s in d['samples']] for d in data]).reshape(2,2,27,6).mean(axis=1)
    power=np.array([[s['meter']['power_mw'] for s in d['samples']]
        for d in data]).reshape(2,2,27).mean(axis=1)
    wave=np.array([[s['meter']['wavelength_nm'] for s in d['samples']]
        for d in data]).reshape(2,2,27).mean(axis=1)
    rows=[]
    for i in range(27):
        rows.append(dict(local=i,active_order=data[0]['route_rows'][i]['order'],
            ch1_codes=monitor[:,i,3].tolist(),pdt_codes=monitor[:,i,0].tolist(),
            pdr_codes=monitor[:,i,1].tolist(),meter_power_mw=power[:,i].tolist(),
            wavelength_drift_pm=float((wave[1,i]-wave[0,i])*1000),
            power_change_percent=float((power[1,i]/power[0,i]-1)*100),
            ch1_change_codes=float(monitor[1,i,3]-monitor[0,i,3])))
    return dict(schema='active_reference_monitor_audit_v1',
        before_sha256=sha256_file(before),after_sha256=sha256_file(after),rows=rows,
        adc_meter_samples_simultaneous=False,causal_origin_identified=False,
        note='Small source-meter drift does not establish reflection stability or ADC correctness.',
        deployment_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('New output required')
    report=audit(a.before,a.after)
    a.output.write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(sorted(report['rows'],key=lambda r:abs(r['ch1_change_codes']),reverse=True)[:3],indent=2))
