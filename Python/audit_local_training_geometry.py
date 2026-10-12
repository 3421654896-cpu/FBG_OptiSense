"""Check physical-coordinate label consistency without commanding hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def physical_target(target):
    offset=np.asarray(target['calibration_machine_minus_work'],float)
    xyz=np.asarray([target['work_x'],target['work_y'],target['contact_z']],float)+offset
    if not np.isfinite(xyz).all(): raise ValueError('nonfinite coordinates')
    return xyz


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--reference-round',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    lock=json.loads(a.lock.read_bytes())
    manifest=json.loads(a.reference_round.read_bytes())
    reference={}
    for point in (13,14):
        data=json.loads(Path(manifest['captures'][str(point)]['capture']).read_bytes())
        reference[point]=physical_target(data['commissioning']['target'])
    rows=[]
    for item in lock['training']['captures']:
        if item['point'] not in reference: continue
        path=Path(item['path']); raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']: raise ValueError('hash mismatch')
        data=json.loads(raw); c=data['commissioning']; target=c['target']
        xyz=physical_target(target)
        event=next(e for e in c['events'] if e['kind']=='contact_reached_commanded_depth')
        actual=np.array([event['status']['machine_'+axis] for axis in 'xyz'])
        endpoint=xyz.copy(); endpoint[2]-=float(c['depth_mm'])
        rows.append({'point':item['point'],'depth_mm':c['depth_mm'],'source':str(path),
            'label_matches_target':item['point']==target['point_index'],
            'target_machine_contact_xyz_mm':xyz.tolist(),
            'target_delta_from_current_mm':(xyz-reference[item['point']]).tolist(),
            'reported_endpoint_error_mm':(actual-endpoint).tolist(),
            'table_crc32':data['table_codes_crc32']})
    report={'schema':'local_training_geometry_audit_v1','rows':rows,
            'reference_machine_contact_xyz':{str(k):v.tolist() for k,v in reference.items()},
            'scope':'Mach3-reported position consistency only; cannot verify real fixture drift or actual contact depth'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'records':len(rows),
        'max_target_difference_mm':max(float(np.max(np.abs(r['target_delta_from_current_mm']))) for r in rows),
        'max_endpoint_error_mm':max(float(np.max(np.abs(r['reported_endpoint_error_mm']))) for r in rows),
        'all_labels_match':all(r['label_matches_target'] for r in rows),
        'table_crcs':sorted({r['table_crc32'] for r in rows})}))
