"""Measure within-frame acquisition skew, not physical pressure bandwidth."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(data, frequency=15.):
    frames = [f for f in data['frames'] if f['sequence'] >= 80]
    if not data.get('complete') or len(frames) < 2:
        raise ValueError('Complete multi-frame capture required')
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError('Invalid frequency')
    if any(not f.get('point_timestamps_are_physical') or
           not f.get('second_code_is_physical_sample') for f in frames):
        raise ValueError('Requires physical ADC timestamps')
    offsets = np.array([[r['second_end_us'] for r in f['records']] for f in frames],float)
    if offsets.shape[1:] != (45,) or not np.isfinite(offsets).all():
        raise ValueError('Expected 45 timed points')
    if any([r['row'] for r in f['records']] != list(range(45)) for f in frames):
        raise ValueError('Unexpected scan order')
    group = offsets.reshape(-1,9,5)
    center = group.mean(axis=2)
    span = offsets.max(axis=1)-offsets.min(axis=1)
    centerspan = center.max(axis=1)-center.min(axis=1)
    phase = lambda us: np.asarray(us)*frequency*360/1e6
    def stats(values):
        return dict(mean=float(np.mean(values)), p95=float(np.quantile(values,.95)),
                    maximum=float(np.max(values)))
    return dict(schema='f45_measured_sampling_skew_v1', frequency_hz=frequency,
        frames=len(frames), second_adc_acquisition_span_us=stats(span),
        nine_grating_center_span_us=stats(centerspan),
        second_adc_phase_span_degrees=stats(phase(span)),
        nine_grating_center_phase_span_degrees=stats(phase(centerspan)),
        within_grating_phase_span_degrees=stats(phase(np.ptp(group,axis=2))),
        mean_second_adc_offsets_us=offsets.mean(axis=0).tolist(),
        measured_physical_pressure_response=False,
        timestamp_meaning='ADC read completion, not exact optical exposure midpoint',
        implication='A static 45-point vector is not simultaneous; dynamic localization requires explicit time handling or faster scanning.',
        deployment_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('New output required')
    result=audit(json.loads(a.capture.read_text('utf-8')))
    result['capture_sha256']=hashlib.sha256(a.capture.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='mean_second_adc_offsets_us'},indent=2))
