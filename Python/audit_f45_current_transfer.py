"""Read-only comparison of current centre captures and historical prototypes."""
import json
from pathlib import Path
import numpy as np
from evaluate_fast_fullmap_position import _capture_feature
from fast_fullmap_realtime import FastFullMapRealtimeLocalizer


def audit(root):
    manifest = json.loads((root/'fast_fullmap_position_rounds_0p30_20260906.json').read_text('utf-8'))
    paths = {f'{r}_p{p}': root / entries[str(p)]
             for r, entries in manifest['rounds'].items() for p in (8, 9)}
    paths.update({f'current_r{r}': root/f'f45_visible_center_v46_20260907_r{r}'/'capture.json' for r in (2,3)})
    model = FastFullMapRealtimeLocalizer()
    vectors, records, baselines = {}, {}, {}
    for name, path in paths.items():
        feature = _capture_feature(path)
        data = json.loads(path.read_text('utf-8'))
        selected = [f for f in data['frames'] if 16 <= f['sequence'] < 80]
        baselines[name] = np.median([[r['first_code'] for r in f['records']] for f in selected],axis=0)
        packet = bytes.fromhex(data['frames'][0]['wire_hex'])
        write_times = np.median([[r['write_end_us'] for r in f['records']] for f in selected],axis=0)
        second_times = np.median([[r['second_end_us'] for r in f['records']] for f in selected],axis=0)
        vectors[name] = feature['features']['first_delta45']
        transformed = (feature['features'][model.feature]-model.centre)/model.scale
        distances = np.linalg.norm(model.prototype_vectors-transformed, axis=1)
        order = np.argsort(distances)[:4]
        events = {e['kind']:e for e in data['commissioning']['events']}
        status = events['contact_reached_commanded_depth']['status']
        records[name] = dict(source=str(path.resolve()),
            table_codes_crc32=data.get('table_codes_crc32'), table_sha256=data.get('table_sha256'),
            firmware_version=int.from_bytes(packet[8:12],'big'), feedback_selectors=list(packet[32:34]),
            first_frame_metadata={k:v for k,v in data['frames'][0].items() if k not in ('records','wire_hex')},
            baseline_first_by_grating=baselines[name].reshape(9,5).tolist(),
            median_write_end_us=write_times.tolist(),
            median_write_to_second_us=(second_times-write_times).tolist(),
            machine_contact_xyz=[status.get('machine_'+a) for a in 'xyz'],
            response_rms_codes=feature['response_rms_codes'],
            first_delta_by_grating=vectors[name].reshape(9,5).tolist(),
            nearest_prototypes=[dict(point=int(model.prototype_points[i]),distance=float(distances[i])) for i in order])
    pairs = []
    for current in ('current_r2','current_r3'):
        for other in vectors:
            if other == current: continue
            a,b = vectors[current],vectors[other]
            pairs.append(dict(current=current,other=other,
                delta_rmse_codes=float(np.sqrt(np.mean((a-b)**2))),
                baseline_rmse_codes=float(np.sqrt(np.mean((baselines[current]-baselines[other])**2))),
                cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))))
    return dict(records=records,comparisons=pairs,model_refitted=False,
                interpretation='Descriptive transfer audit; no causal fault attribution or new accuracy claim.')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('../outputs'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
