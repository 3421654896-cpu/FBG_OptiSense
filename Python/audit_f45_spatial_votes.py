"""Inspect spatial votes of a frozen classifier, without temporal smoothing."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from audit_f45_contact_drift import trace_features
from audit_f45_contact_transition import validated_gain_identity,require_fresh_capture
from commission_fast_flank_center_contact import coverage_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): p.error('output exists')
    payload=joblib.load(a.model); data,seq,x=trace_features(a.capture)
    digest=hashlib.sha256(a.capture.read_bytes()).hexdigest()
    require_fresh_capture(payload,data,digest); validated_gain_identity(data)
    model=payload['model']; model.n_jobs=1
    c=data['commissioning']; point=c['target']['point_index']
    events={e['kind']:e['host_monotonic_ns'] for e in c['events']}
    receipts={r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
    held=np.array([events['contact_reached_commanded_depth']<=receipts[int(s)]<events['release_started'] for s in seq])
    contact=model.classes_!=0; labels=model.classes_[contact]
    targets=[coverage_target(int(i)) for i in labels]
    # Calibration XY in mm; a common work-coordinate offset does not change distances.
    coords=np.array([[t['work_x'],t['work_y']] for t in targets])
    truth=coords[list(labels).index(point)]
    probability=model.predict_proba(x[held])[:,contact]
    rows=[]
    for sequence,prob in zip(seq[held],probability):
        mass=float(prob.sum())
        # Same tie rule as np.argmax in the frozen classifier: first class wins.
        order=np.argsort(-prob,kind='stable')
        winner=int(labels[order[0]])
        weights=prob/mass if mass>0 else np.zeros(len(prob))
        centroid=weights@coords if mass>0 else None
        rows.append(dict(sequence=int(sequence),winner=winner,contact_mass=mass,
            winner_error_mm=float(np.linalg.norm(coords[order[0]]-truth)),
            top_two_margin=float(prob[order[0]]-prob[order[1]]),
            top_three=[dict(point=int(labels[i]),vote=float(prob[i])) for i in order[:3]],
            centroid_error_mm=float(np.linalg.norm(centroid-truth)) if centroid is not None else None))
    report=dict(point=point,source_sha256=digest,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        frames=rows,wrong_frames=[r for r in rows if r['winner']!=point],
        mean_winner_error_mm=float(np.mean([r['winner_error_mm'] for r in rows])),
        mean_centroid_error_mm=float(np.mean([r['centroid_error_mm'] for r in rows])),
        limitation='Centroid is an unvalidated diagnostic, not a deployed continuous-position estimate. All contact votes retained; no time smoothing. Grid evaluation does not prove off-grid accuracy.')
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}))


if __name__=='__main__': main()
