"""Offline continuous XY/depth comparison; never commands or deploys hardware."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from evaluate_f45_joint_contact_position import checked
from audit_f45_contact_transition import validated_gain_identity
from audit_f45_contact_drift import trace_features
from audit_local_training_geometry import physical_target
from select_f45_activation_threshold_v6 import load_manifest


def target_label(data, item):
    c = data['commissioning']
    if c['target']['point_index'] != item['point']:
        raise ValueError('Position label mismatch')
    depth = float(c['depth_mm'])
    if not 0 < depth <= .8 or not np.isclose(depth, item['depth_mm'], atol=1e-9):
        raise ValueError('Depth label mismatch')
    # Use machine coordinates, not work coordinates changed by operator zeroing.
    return np.r_[physical_target(c['target'])[:2], depth]


def metrics(predicted, truth):
    predicted = np.asarray(predicted, float)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or not len(predicted):
        raise ValueError('Expected nonempty XYZ/depth predictions')
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite prediction')
    xy = np.linalg.norm(predicted[:, :2] - truth[:2], axis=1)
    depth = np.abs(predicted[:, 2] - truth[2])
    return dict(frames=len(predicted), xy_mean_mm=float(xy.mean()),
                xy_p95_mm=float(np.percentile(xy, 95)), xy_max_mm=float(xy.max()),
                depth_mae_mm=float(depth.mean()), depth_bias_mm=float((predicted[:, 2]-truth[2]).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--round', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weighting', choices=('capture', 'position_depth'), default='capture')
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists')
    train = json.loads(a.lock.read_bytes())['training']['captures']
    test = [i for m in a.round for i in load_manifest(m)]
    if {i['sha256'] for i in train} & {i['sha256'] for i in test}:
        raise ValueError('Capture overlap')
    if {i['round'] for i in train} & {i['round'] for i in test}:
        raise ValueError('Round overlap')
    xs, ys, weights, excluded, included = [], [], [], [], []
    for index, item in enumerate(train):
        _, values, masks = checked(item)
        data = json.loads(Path(item['path']).read_bytes())
        try:
            validated_gain_identity(data)
        except ValueError as exc:
            if str(exc) != 'historical firmware protocol not reviewed': raise
            excluded.append(dict(path=item['path'], sha256=item['sha256'], reason=str(exc)))
            continue
        label = target_label(data, item)
        held = values[masks['held']]
        if not len(held): raise ValueError('Empty training hold')
        xs.append(held); ys.append(np.broadcast_to(label, (len(held), 3)))
        weights.append(np.full(len(held), 1/len(held)))
        included.append(item)
        if index % 50 == 0: print('validated training captures', index+1, flush=True)
    if {i['point'] for i in included} != {i['point'] for i in train}:
        raise ValueError('Reviewed firmware subset does not cover all training positions')
    print('included', len(included), 'excluded legacy protocol', len(excluded), flush=True)
    cells = Counter((i['point'], float(i['depth_mm'])) for i in included)
    if a.weighting == 'position_depth':
        weights = [w/cells[(i['point'], float(i['depth_mm']))] for i, w in zip(included, weights)]
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    # Separate objectives: millimetre XY span must not swamp sub-mm depth loss.
    xy = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
        max_features='sqrt', random_state=20260908, n_jobs=-1).fit(x, y[:, :2], sample_weight=weight)
    depth = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
        max_features='sqrt', random_state=20260908, n_jobs=-1).fit(x, y[:, 2], sample_weight=weight)
    xy.n_jobs = depth.n_jobs = 1
    rows = []
    for item in test:
        checked(item)
        data, seq, values = trace_features(Path(item['path']))
        validated_gain_identity(data)
        label = target_label(data, item)
        c = data['commissioning']
        events = {e['kind']: e['host_monotonic_ns'] for e in c['events']}
        receipt = {r['sequence']: r['host_monotonic_ns'] for r in c['frame_receipts']}
        mask = np.array([events['contact_reached_commanded_depth'] <= receipt[int(s)] < events['release_started'] for s in seq])
        predicted = np.c_[xy.predict(values[mask]), depth.predict(values[mask])]
        row = dict(round=item['round'], point=item['point'], sha256=item['sha256'],
            truth_machine_xy_and_commanded_depth_mm=label.tolist(), metrics=metrics(predicted, label),
            predictions=[dict(sequence=int(s), machine_x_mm=float(v[0]), machine_y_mm=float(v[1]),
                              commanded_depth_proxy_mm=float(v[2])) for s, v in zip(seq[mask], predicted)])
        rows.append(row)
        print(row['round'], row['point'], row['metrics'], flush=True)
    report = dict(rows=rows, training_captures=len(included), training_frames=len(x),
        included_captures=included, excluded_captures=excluded,
        training_lock_sha256=hashlib.sha256(a.lock.read_bytes()).hexdigest(),
        weighting=a.weighting, depth_capture_counts=dict(Counter(str(i['depth_mm']) for i in included)),
        position_depth_cell_counts=[dict(point=k[0], depth_mm=k[1], captures=v) for k, v in sorted(cells.items())],
        includes_first_200ms=True,
        temporal_smoothing=False, development_only=True, blind_test=False, deployed=False,
        physical_15hz_verified=False, actual_contact_area_verified=False,
        limitation='Contact-held evaluation only; not a contact detector. Depth is commanded displacement, not force. Grid-labelled training and tests do not verify off-grid localization. No hardware commands.')
    a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
