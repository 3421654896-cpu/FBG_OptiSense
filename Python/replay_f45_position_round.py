"""Replay raw F45 frames causally through the existing model, without fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from evaluate_fast_fullmap_position import _capture_feature
from fast_fullmap_realtime import FastFullMapRealtimeLocalizer, DEFAULT_MODEL_PATH


def replay(path, model_path):
    _capture_feature(path)  # existing provenance, completeness and safe-release gates
    raw = path.read_bytes()
    data = json.loads(raw)
    c = data['commissioning']
    events = {r['kind']:r['host_monotonic_ns'] for r in c['events']}
    receipts = {r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
    model = FastFullMapRealtimeLocalizer(model_path)
    truth = int(c['target']['point_index'])
    truth_xy = c['target'].get('calibration_work_xy_mm',
                              [c['target']['work_x'],c['target']['work_y']])
    held, baseline = [], []
    for frame in data['frames']:
        result = model.update(frame)  # no phase, CNC coordinate or label supplied
        received = receipts[frame['sequence']]
        if events['contact_reached_commanded_depth']+200_000_000 <= received < events['release_started']:
            held.append(result)
        elif frame['sequence'] < c['trigger_sequence'] and result.baseline_ready:
            baseline.append(result)
    if not held:
        raise ValueError('No held frames')
    errors = [float(np.linalg.norm(np.asarray([r.provisional_x_mm,r.provisional_y_mm])-
              truth_xy)) for r in held if r.contact_detected]
    return dict(source=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),truth_point=truth,
        depth_mm=float(c['depth_mm']),
        held_frames=len(held), exact_frames=sum(r.predicted_point==truth for r in held),
        abstained_frames=sum(not r.contact_detected for r in held),
        wrong_frames=sum(r.contact_detected and r.predicted_point!=truth for r in held),
        baseline_frames=len(baseline),baseline_false_contact_frames=sum(r.contact_detected for r in baseline),
        predicted_points=sorted({r.predicted_point for r in held if r.predicted_point is not None}),
        detected_frame_p95_error_mm=float(np.percentile(errors,95)) if errors else None,
        held_sampling_gate_failures=sum(not r.sampling_gate_for_15hz for r in held))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--model',type=Path,default=DEFAULT_MODEL_PATH)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text('utf-8'))
    if manifest['schema']!='f45_position_validation_round_v1': raise ValueError('Wrong manifest')
    rows=[replay((args.manifest.parent/manifest['round'][str(i)]).resolve(),args.model) for i in range(1,18)]
    count=sum(r['held_frames'] for r in rows)
    depths={round(float(r['depth_mm']),6) for r in rows}
    if len(depths)!=1:
        raise ValueError('Validation round mixes indentation depths')
    trial_exact=sum(
        r['exact_frames'] > r['wrong_frames']
        and r['exact_frames'] > r['abstained_frames']
        for r in rows
    )
    result=dict(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        purpose=manifest.get('purpose','qualified F45 fixed-grid validation round'),
        model_refitted=False,
        scope=('Causal replay of independently captured, provenance-checked F45 frames; '
               'the CNC labels are used only for scoring and never enter inference.'),
        indentation_depth_mm=next(iter(depths)),
        trial_count=len(rows),trial_exact_count=trial_exact,
        trial_exact_percent=100*trial_exact/len(rows),
        held_frames=count,exact_frames=sum(r['exact_frames'] for r in rows),
        exact_frame_percent=100*sum(r['exact_frames'] for r in rows)/count,
        abstained_frames=sum(r['abstained_frames'] for r in rows),
        baseline_false_contact_frames=sum(r['baseline_false_contact_frames'] for r in rows),
        held_sampling_gate_failures=sum(r['held_sampling_gate_failures'] for r in rows),
        captures=rows,physical_15hz_verified=False,
        nominal_tool_face_diameter_mm=8.0,
        nominal_tool_face_area_mm2=math.pi*4.0**2,
        actual_contact_area_verified=False)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='captures'},indent=2))
