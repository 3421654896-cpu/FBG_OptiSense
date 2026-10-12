"""Offline repeat-versus-neighbour comparison; no hardware commands."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent / 'outputs'
names = ['f18_visible_p06_depth020_v46_20260907_r1',
         'f18_visible_p06_depth020_v46_20260907_r2',
         'f18_visible_p05_depth020_v46_20260907_r1']
vectors = []
for name in names:
    data = json.loads((root / name / 'capture.json').read_text('utf-8'))
    c = data['commissioning']
    if not data['complete'] or c['capture_error'] or c['motion_failures']:
        raise ValueError('Incomplete capture')
    if not all(data['safety_cleanup'][key] for key in ('exact_disarm_ack', 'exact_soa_shutter_ack')):
        raise ValueError('Missing shutdown acknowledgment')
    events = {e['kind']: e['host_monotonic_ns'] for e in c['events']}
    receipts = {e['sequence']: e['host_monotonic_ns'] for e in c['frame_receipts']}
    baseline = np.median([[r['first_code'] for r in f['records']] for f in data['frames']
                          if 16 <= f['sequence'] < 80], axis=0)
    held = np.array([[r['first_code'] for r in f['records']] for f in data['frames']
                     if events['contact_reached_commanded_depth'] + 200_000_000
                     <= receipts[f['sequence']] < events['release_started']], dtype=float)
    if len(held) < 25 or held.shape[1:] != (18,):
        raise ValueError('Insufficient held samples')
    vectors.append(np.median(held, axis=0) - baseline)
values = np.asarray(vectors)
unit = values / np.linalg.norm(values, axis=1, keepdims=True)
result = dict(sources=names,
    delta_rms_distances=np.sqrt(np.mean((values[:, None] - values[None, :]) ** 2, axis=2)).tolist(),
    cosine_similarity=(unit @ unit.T).tolist(),
    scope='Two repeats of point 6 and one point 5 contact at 0.20 mm; exploratory only',
    physical_15hz_verified=False, position_accuracy_verified=False)
with (root / 'current_f18_repeat_neighbor_audit_20260907.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
