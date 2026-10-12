"""Static repeatability audit of contemporaneous held ADC groups, not bandwidth."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_sparse_held_trajectory import audit_record


def spread(values):
    a = np.asarray(values,float)
    mean = float(a.mean())
    return {'values':a.tolist(),'min':float(a.min()),'max':float(a.max()),
            'relative_std_percent':float(100*a.std()/abs(mean)) if mean else None}


def audit(path):
    raw = path.read_bytes()
    doc = json.loads(raw)
    if not doc['complete'] or not all(doc['safety_cleanup'][k] for k in
                                    ('exact_disarm_ack','exact_soa_shutter_ack')):
        raise ValueError('incomplete or unsafe capture')
    groups, excluded = {}, []
    for record in doc['captures']:
        try: audit_record(record)
        except ValueError as exc:
            excluded.append({'point':record['point_index'],'tag':record['request_tag'],'reason':str(exc)})
            continue
        # Profile number alone is insufficient: selected segments alter path.
        key = (record['point_index'],record['schedule_version'],record['prefix_profile'],
               tuple(record['planned_indices']),tuple(record['feedback_selectors']))
        samples = np.array([r['direct_adc'] for r in record['teacher_samples']],float)
        if samples.shape != (25,6) or not np.isfinite(samples).all():
            raise ValueError('invalid direct held data')
        groups.setdefault(key,[]).append(samples)
    rows = []
    for key, captures in sorted(groups.items()):
        medians = np.array([np.median(c,axis=0) for c in captures])
        row = {'point':key[0],'schedule':key[1],'profile':key[2],
               'planned_indices':list(key[3]),'feedback_selectors':list(key[4]),
               'repeat_count':len(captures),'ch1':spread(medians[:,1]),
               'pdt':spread(medians[:,4]),'pdr':spread(medians[:,5])}
        for channel,name in ((4,'pdt'),(5,'pdr')):
            if any((c[:,channel] <= 4).any() for c in captures):
                row['ch1_over_'+name] = {'unavailable':'denominator near ADC floor'}
            else:
                ratios = [np.median(c[:,1]/c[:,channel]) for c in captures]
                row['ch1_over_'+name] = spread(ratios)
        rows.append(row)
    return {'schema':'held_pd_normalization_audit_v1','source_sha256':hashlib.sha256(raw).hexdigest(),
            'scope':'static repeatability only; same conversion group, channels sequential not simultaneous; no motion or deployment',
            'causal_origin_identified':False,'rows':rows,'excluded':excluded}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = audit(args.input)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['rows'],indent=2))
