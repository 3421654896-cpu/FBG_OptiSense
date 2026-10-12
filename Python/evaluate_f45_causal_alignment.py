"""Synthetic signal replay on measured timestamps; never a hardware bandwidth claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def align_previous_current(times, values):
    """Align to the earliest current sample, using only previous/current frames."""
    times, values = np.asarray(times,float), np.asarray(values,float)
    if times.shape != values.shape or times.ndim != 2 or len(times) < 2:
        raise ValueError('Expected matching frame-by-point matrices')
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError('Non-finite data')
    if (np.diff(times,axis=0) <= 0).any():
        raise ValueError('Sample times must increase')
    target = times[1:].min(axis=1)
    weight = (target[:,None]-times[:-1])/(times[1:]-times[:-1])
    if ((weight < 0) | (weight > 1)).any():
        raise ValueError('Cannot bracket shared time without extrapolation')
    return target, values[:-1]+weight*(values[1:]-values[:-1])


def evaluate(times, frequency):
    omega = 2*np.pi*frequency
    values = np.sin(omega*times)
    target, aligned = align_previous_current(times,values)
    design = np.column_stack((np.sin(omega*target),np.cos(omega*target),np.ones(len(target))))
    def metrics(matrix):
        coefficients = np.linalg.lstsq(design,matrix,rcond=None)[0]
        amplitude = np.hypot(coefficients[0],coefficients[1])
        phase = np.degrees(np.arctan2(coefficients[1],coefficients[0]))
        error = matrix-np.sin(omega*target)[:,None]
        return dict(amplitude_ratio_min=float(amplitude.min()),
            amplitude_ratio_max=float(amplitude.max()),
            absolute_phase_error_degrees_max=float(np.abs(phase).max()),
            normalized_rms_error=float(np.sqrt(np.mean(error**2))/.7071067811865476),
            rms_false_spatial_spread=float(np.sqrt(np.mean(np.var(matrix,axis=1)))))
    return dict(frequency_hz=frequency, unaligned=metrics(values[1:]),
        linear_causal_aligned=metrics(aligned),
        alignment_output_age_ms_max=float(np.max(times[1:].max(axis=1)-target)*1000))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Output must be new')
    data=json.loads(a.capture.read_text('utf-8'))
    if not data.get('complete'): raise ValueError('Incomplete capture')
    frames=[f for f in data['frames'] if f['sequence']>=80]
    if any(not f.get('point_timestamps_are_physical') for f in frames):
        raise ValueError('Requires measured timestamps')
    # Unwrap the board microsecond counter before adding within-frame offsets.
    starts=np.array([f['cycle_start_us'] for f in frames],dtype=np.int64)
    gaps=(np.diff(starts) % (2**32))
    if np.any(gaps>1e6): raise ValueError('Long gap or reordered frames')
    starts=np.concatenate(([0],np.cumsum(gaps)))
    times=(starts[:,None]+np.array([[r['second_end_us'] for r in f['records']] for f in frames]))/1e6
    report=dict(schema='f45_causal_alignment_synthetic_replay_v1',
        capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),
        waveform_source='synthetic identical sinusoid at all 45 points',
        timestamp_source='measured second ADC completion times',
        physical_pressure_test=False, deployment_eligible=False,
        results=[evaluate(times,hz) for hz in (1.,5.,10.,15.)])
    a.output.write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
