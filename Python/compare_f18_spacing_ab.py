"""No-motion raw ADC A/B/A comparison; not dynamic localization validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_scan_alignment_bandwidth import verified_times


def load(path):
    raw=path.read_bytes(); data=json.loads(raw)
    verified_times(data)
    if not data['complete'] or data['cnc_motion_commands']!=0: raise ValueError('Not a complete idle capture')
    if not all(data['safety_cleanup'][k] for k in ('exact_disarm_ack','exact_soa_shutter_ack')):
        raise ValueError('Shutdown not confirmed')
    frames=[f for f in data['frames'] if 2_000_000<=f['cycle_start_us']<=2_600_000]
    if len(frames)<20: raise ValueError('Incomplete shared post-start time window')
    codes=np.array([[r[k] for k in ('first_code','second_code') for r in f['records']] for f in frames])
    return data,codes,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
        spacing_us=data['requested_spacing_us'],frames=len(frames),
        median_codes=np.median(codes,axis=0).tolist(),
        standard_deviation_codes=np.std(codes,axis=0).tolist(),
        board_rate_hz=data['analysis']['board_rate_hz'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',type=Path,required=True);p.add_argument('--b',type=Path,required=True)
    p.add_argument('--a-return',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    items=[load(path) for path in (a.a,a.b,a.a_return)]
    if len({i[0]['table_codes_crc32'] for i in items})!=1:raise ValueError('Table differs')
    if [i[2]['spacing_us'] for i in items]!=[600,50,600]:raise ValueError('Expected 600/50/600')
    med=np.array([i[2]['median_codes'] for i in items])
    rows=[]
    for name,section in [('first',slice(0,18)),('second',slice(18,36))]:
        delta=med[1,section]-(med[0,section]+med[2,section])/2
        repeat=med[2,section]-med[0,section]
        rows.append(dict(read=name,short_vs_bracket_rms_codes=float(np.sqrt(np.mean(delta**2))),
            long_repeat_rms_codes=float(np.sqrt(np.mean(repeat**2))),short_vs_bracket_per_point=delta.tolist()))
    result=dict(captures=[i[2] for i in items],comparison=rows,window_seconds=[2,2.6],
        physical_15hz_verified=False,model_transfer_qualified=False,
        limitation='Same post-start time window, fixed table/gain, no movement. Different spacing changes sampling age and laser residence; this is not a pure ADC-noise experiment or a stable-spectrum truth.')
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
