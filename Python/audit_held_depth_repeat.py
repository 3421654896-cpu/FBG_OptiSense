"""Audit identical-depth held captures against actual CNC telemetry, not commands."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from causal_transient_features import extract
from held_capture_position import reference


def inspect(source):
    summary_path = source/'summary.json'
    summary = json.loads(summary_path.read_text('utf-8'))
    if not summary['complete'] or summary['error']:
        raise ValueError('Successful closed run required')
    rows = []
    for contact in summary['contacts']:
        path = Path(contact['capture'])
        raw = path.read_bytes()
        doc = json.loads(raw)
        if not doc['complete'] or not contact['motion_success'] or not all(
                doc['safety_cleanup'][key] for key in ('exact_disarm_ack','exact_soa_shutter_ack')):
            raise ValueError('Successful capture and retraction required')
        for r in doc['captures']:
            if not r['teacher']['stable'] or r['teacher']['dac_rewritten']:
                raise ValueError('Same-DAC stable teacher required')
            brackets = r['machine_position_bracket']
            start, end = (brackets[k] for k in ('before_received_monotonic_ns','after_received_monotonic_ns'))
            trace = [t for t in summary['machine_trace'] if start <= t['received_ns'] <= end]
            if not trace:
                raise ValueError('No telemetry during capture')
            positions = [reference(contact['target'], t['status']) for t in trace]
            xyz = np.asarray([p['machine_xyz_mm'] for p in positions])
            times = [start]+[t['received_ns'] for t in trace]+[end]
            max_gap_ms = max(np.diff(times))/1e6
            span = np.ptp(xyz, axis=0)
            rows.append(dict(requested_depth_mm=contact['requested_depth_mm'],
                actual_indentation_mm=positions[0]['pressure_reference']['indentation_mm'],
                machine_xyz_mm=xyz[0].tolist(), recorded_axis_span_mm=span.tolist(),
                machine_samples=len(trace), max_observation_gap_ms=float(max_gap_ms),
                stationary_observations_pass=bool(np.all(span<=.003) and max_gap_ms<=150),
                continuous_motion_excluded=False,
                teacher=r['teacher'], profile=r['prefix_profile'],
                schedule_version=r['schedule_version'], feedback_selectors=r['feedback_selectors'],
                early_features={str(h):extract(r['curve'],h) for h in (750,1500,3000)},
                source=str(path.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                table_sha256=hashlib.sha256(json.dumps([doc['rows'],doc['hidden_codes']],sort_keys=True).encode()).hexdigest()))
    return rows


def compare(left, right):
    rows=[]
    for a in left:
        matches=[b for b in right if b['requested_depth_mm']==a['requested_depth_mm']]
        if len(matches)!=1:
            raise ValueError('Unique paired depth required')
        b=matches[0]
        if a['source_sha256']==b['source_sha256'] or a['table_sha256']!=b['table_sha256']:
            raise ValueError('Independent captures with identical tables required')
        for key in ('schedule_version','feedback_selectors','profile'):
            if a[key] != b[key]:
                raise ValueError(f'Acquisition identity mismatch: {key}')
        rows.append(dict(depth_mm=a['requested_depth_mm'],
            actual_xyz_difference_mm=(np.asarray(b['machine_xyz_mm'])-a['machine_xyz_mm']).tolist(),
            teacher_ch1_pair=[a['teacher']['median'][3],b['teacher']['median'][3]],
            left=a,right=b))
    return dict(pairs=rows, purpose='repeatability diagnostic, no fitted model',
                pressure_source='actual_indentation_proxy',
                physical_15hz_verified=False, area_verified=False, deployment_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=compare(inspect(args.left),inspect(args.right))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    for row in report['pairs']:
        print(json.dumps({k:row[k] for k in ('depth_mm','actual_xyz_difference_mm','teacher_ch1_pair')}))
        print([(row[k]['machine_samples'],row[k]['max_observation_gap_ms'],row[k]['recorded_axis_span_mm']) for k in ('left','right')])
