"""Compare exact-route raw ADC with settled teacher; not simultaneous reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from capture_ch1_teacher_spectrum import sha256_file


def compare(probe_path,captures,after_path=None):
    teacher=json.loads(probe_path.read_text('utf-8'))
    if not teacher.get('complete') or len(teacher['samples'])!=54:
        raise ValueError('Complete teacher required')
    reference=np.array([s['point']['ch1_adc_code'] for s in teacher['samples']]).reshape(2,27)
    target=reference.mean(axis=0)
    after_target=None
    if after_path is not None:
        after=json.loads(after_path.read_text('utf-8'))
        if (not after.get('complete') or not after.get('disarm_ack') or not after.get('shutter_ack')
                or after.get('route_rows')!=teacher['route_rows']
                or after.get('feedback_selectors')!=teacher['feedback_selectors']
                or after.get('hidden_predecessor_codes')!=teacher['hidden_predecessor_codes']
                or len(after.get('samples',[]))!=54):
            raise ValueError('After-reference identity or completion mismatch')
        after_target=np.array([s['point']['ch1_adc_code'] for s in after['samples']]).reshape(2,27).mean(axis=0)
    results=[]
    for path in captures:
        d=json.loads(path.read_text('utf-8'))
        if (not d.get('complete') or not d.get('disarm_ack') or not d.get('shutter_ack')
                or d.get('qualification_sha256')!=sha256_file(probe_path)):
            raise ValueError('Incomplete or mismatched capture')
        for actual,expected in zip(d['route_rows'],teacher['route_rows']):
            if actual['codes']!=expected['codes'] or actual['active_mode_order']!=expected['order']:
                raise ValueError('Route changed')
        matrix=np.array([[r['second_code'] for r in f['records']] for f in d['frames'] if f['sequence']>=80])
        if matrix.shape[1:]!=(27,) or len(matrix)<32:
            raise ValueError('Insufficient data')
        median=np.median(matrix,axis=0); errors=median-target
        result=dict(path=str(path),sha256=sha256_file(path),
            first_delay_us=d['first_delay_us'],spacing_us=d['spacing_us'],
            board_rate_hz=d['analysis']['board_rate_hz'],
            median_absolute_error_codes=float(np.median(abs(errors))),
            maximum_absolute_error_codes=float(np.max(abs(errors))),
            rms_error_codes=float(np.sqrt(np.mean(errors**2))),
            per_point=[dict(local=i,reference_code=float(target[i]),median_code=float(median[i]),
                error_codes=float(errors[i]),std_codes=float(matrix[:,i].std())) for i in range(27)])
        if after_target is not None:
            low=np.minimum(target,after_target); high=np.maximum(target,after_target)
            distance=np.maximum(np.maximum(low-median,median-high),0)
            result['after_reference_rms_error_codes']=float(np.sqrt(np.mean((median-after_target)**2)))
            result['outside_reference_envelope_max_codes']=float(distance.max())
            result['outside_reference_envelope_rms_codes']=float(np.sqrt(np.mean(distance**2)))
        results.append(result)
    report=dict(schema='active_route_timing_comparison_v1',results=results,
        teacher_repeat_max_difference_codes=float(np.max(abs(reference[1]-reference[0]))),
        reference_is_simultaneous=False, drift_confounded=True,
        physical_15hz_verified=False,deployment_eligible=False)
    if after_target is not None:
        report['after_reference_sha256']=sha256_file(after_path)
        report['reference_drift_max_codes']=float(np.max(abs(after_target-target)))
        report['reference_drift_rms_codes']=float(np.sqrt(np.mean((after_target-target)**2)))
        report['reference_drift_is_only_endpoint_evidence']=True
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--after-probe',type=Path)
    a=p.parse_args()
    if a.output.exists(): p.error('New output required')
    result=compare(a.probe,a.capture,a.after_probe)
    a.output.write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='per_point'} for r in result['results']],indent=2))
