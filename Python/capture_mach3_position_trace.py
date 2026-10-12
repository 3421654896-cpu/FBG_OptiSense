"""Bounded read-only CNC DRO trace. Never commands motion or resets E-stop.

Host receive timestamps are not encoder measurement timestamps. Raw read
intervals are retained; clock synchronization and true DRO refresh are unproven.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from mach3_controller import Mach3Controller


def summarize(samples):
    if len(samples)<2:
        raise ValueError('At least two samples required')
    starts=np.asarray([r['host_request_perf_ns'] for r in samples],dtype=np.int64)
    ends=np.asarray([r['host_received_perf_ns'] for r in samples],dtype=np.int64)
    if np.any(ends<starts) or np.any(np.diff(ends)<=0) or np.any(starts[1:]<ends[:-1]):
        raise ValueError('Invalid or overlapping host timing')
    xyz=np.asarray([[r['position'][f'machine_{a}'] for a in 'xyz'] for r in samples],dtype=float)
    if not np.all(np.isfinite(xyz)):
        raise ValueError('Nonfinite coordinates')
    intervals=np.diff(ends)/1e6
    return dict(samples=len(samples), receive_rate_hz=1000/float(np.mean(intervals)),
        maximum_receive_gap_ms=float(np.max(intervals)),
        request_roundtrip_p95_ms=float(np.percentile((ends-starts)/1e6,95)),
        observed_coordinate_span_mm=np.ptp(xyz,axis=0).tolist(),
        encoder_refresh_rate_verified=False, synchronized_to_adc=False,
        dynamic_15hz_reference_verified=False)


def capture(seconds, output):
    if not math.isfinite(seconds) or not .5<=seconds<=10:
        raise ValueError('Read-only duration must be .5..10 seconds')
    if output.exists():
        raise FileExistsError('Do not overwrite a reference trace')
    samples=[]
    with Mach3Controller() as controller:
        before=controller.status()
        deadline=time.perf_counter()+seconds
        while time.perf_counter()<deadline and len(samples)<4096:
            started=time.perf_counter_ns()
            position=controller.position_sample()
            ended=time.perf_counter_ns()
            samples.append(dict(host_request_perf_ns=started,host_received_perf_ns=ended,
                                host_received_monotonic_ns=time.monotonic_ns(),position=position))
            # Modest polling rate. Observed intervals, not this target, are used.
            time.sleep(.005)
        after=controller.status()
    report=dict(schema='mach3_readonly_position_trace_v1',before=before,after=after,
        motion_commands=0,estop_release_commands=0,pressure_units=None,
        summary=summarize(samples),samples=samples)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(capture(args.seconds,args.output)['summary'],indent=2))
