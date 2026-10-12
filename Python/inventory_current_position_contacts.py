"""Audit current guarded captures without hardware access or model training."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_current_three_positions import capture_parts


def inventory(root):
    valid, rejected = [], []
    for path in sorted(root.glob('f45_visible_*v46_20260907*/capture.json')):
        try:
            _, held = capture_parts(path)
            data = json.loads(path.read_text('utf-8'))
            target = data['commissioning']['target']
            valid.append(dict(source=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                point_index=target['point_index'], depth_mm=target['depth_mm'],
                held_frames=len(held), role='unassigned; not automatically training or validation'))
        except (ValueError, KeyError) as error:
            rejected.append(dict(source=str(path.resolve()), reason=str(error)))
    covered = sorted({r['point_index'] for r in valid})
    return dict(schema='current_contact_inventory_v1', valid=valid, rejected=rejected,
        covered_points=covered, missing_points=sorted(set(range(1,18))-set(covered)),
        scope='Current visible v46 captures only; file completeness is not model accuracy.',
        physical_15hz_verified=False, full_spatial_accuracy_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('../outputs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print('valid contacts:', len(result['valid']))
    print('covered:', result['covered_points'])
    print('missing:', result['missing_points'])
    print('rejected:', result['rejected'])
