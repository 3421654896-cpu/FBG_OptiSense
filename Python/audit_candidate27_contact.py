"""Audit independent C27 physical contacts; CNC labels never enter features."""
import argparse
import json
from pathlib import Path
import numpy as np
from capture_candidate27_stream import decode as decode_v1, FIRMWARE
from capture_candidate27_guarded_stream import decode as decode_v2
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file


def rms(values): return float(np.sqrt(np.mean(np.asarray(values,dtype=float)**2)))


def shape(values):
    groups=np.asarray(values,dtype=float).reshape(9,3)
    # No subtraction of the minimum: only three measured points per FBG, and
    # forcing one to zero would discard a third of the already sparse shape.
    return (groups/np.maximum(np.linalg.norm(groups,axis=1,keepdims=True),1)).ravel()


def load_contact(path, expected_firmware=FIRMWARE):
    data=json.loads(path.read_text('utf-8'))
    c=data['commissioning']
    if (not data['complete'] or not data['disarm_ack'] or not data['shutter_ack']
            or c['capture_error'] or c['motion_failures']
            or c['stream_protocol'] not in ('candidate27', 'candidate27_guarded')
            or not 0 < c['depth_mm'] <= .8
            or data['expected_firmware']!=expected_firmware):
        raise ValueError('Incomplete/unsafe/incompatible capture')
    terminal=data['terminal']
    if terminal['stop_reason'] or terminal['status'] or terminal['completed_cycles']!=len(data['frames']):
        raise ValueError('Stream did not finish')
    command=bytes.fromhex(data['command_hex'])
    tag=int.from_bytes(command[8:12],'big')
    for i,f in enumerate(data['frames']):
        if c['stream_protocol'] == 'candidate27_guarded':
            decoded=decode_v2(bytes.fromhex(f['wire_hex']),tag,data['route_crc32'],
                              data['first_delay_us'],data['spacing_us'],data['guard'],
                              expected_firmware=expected_firmware)
        else:
            decoded=decode_v1(bytes.fromhex(f['wire_hex']),tag,data['route_crc32'],
                              data['first_delay_us'],data['spacing_us'],
                              expected_firmware=expected_firmware)
        if decoded!=f or f['sequence']!=i or f['i2c_error_delta'] or f['spi_error_delta']:
            raise ValueError('Invalid raw frame/provenance')
    events={e['kind']:e for e in c['events']}
    required=('validated_safe_start','contact_descent_started','contact_reached_commanded_depth',
              'release_started','safe_height_reached','final_safe_state_confirmed')
    if any(name not in events for name in required): raise ValueError('Missing physical event')
    if any(events[b]['host_monotonic_ns']<=events[a]['host_monotonic_ns'] for a,b in zip(required,required[1:])):
        raise ValueError('Noncausal physical event order')
    target=c['target']
    for name,z in (('contact_reached_commanded_depth',target['press_z']),('final_safe_state_confirmed',target['safe_z'])):
        status=events[name]['status']
        if not status['stopped'] or status['moving'] or status['estop'] or abs(status['work_z']-z)>.003:
            raise ValueError('Wrong physical endpoint')
        if max(abs(status['work_'+a]-target['work_'+a]) for a in 'xy')>.01:
            raise ValueError('Wrong physical XY label')
    receipts={r['sequence']:r['host_monotonic_ns'] for r in c['frame_receipts']}
    if len(receipts)!=len(data['frames']): raise ValueError('Missing host timestamps')
    frames=data['frames']
    times=np.array([receipts[f['sequence']] for f in frames],dtype=np.int64)
    baseline_mask=np.array([16<=f['sequence']<80 for f in frames])
    if np.max(times[baseline_mask])>=events['contact_descent_started']['host_monotonic_ns']:
        raise ValueError('Baseline overlaps commanded contact')
    held_mask=(times>=events['contact_reached_commanded_depth']['host_monotonic_ns']+200_000_000)&(times<events['release_started']['host_monotonic_ns'])
    if np.sum(held_mask)<20: raise ValueError('Insufficient independent held-window samples')
    features={}
    sample_stats=[]
    for field in ('first_code','second_code'):
        values=np.array([[r[field] for r in f['records']] for f in frames],dtype=float)
        baseline=np.median(values[baseline_mask],axis=0)
        held=values[held_mask]
        delta=np.median(held,axis=0)-baseline
        vector_shape=shape(np.median(held,axis=0))-shape(baseline)
        features[field]=(delta,vector_shape)
        sample_stats.append(dict(field=field,baseline=baseline.tolist(),held_median=np.median(held,axis=0).tolist(),
            delta=delta.tolist(),shape_delta=vector_shape.tolist(),held_frames=len(held),
            signal_rms_codes=rms(delta),baseline_noise_rms_codes=rms(np.std(values[baseline_mask],axis=0)),
            held_noise_rms_codes=rms(np.std(held,axis=0)),
            group_signal_rms_codes=[rms(g) for g in delta.reshape(9,3)],
            held_shape_features=[(shape(v)-shape(baseline)).tolist() for v in held]))
    item=dict(source=str(path.resolve()),sha256=sha256_file(path),point_index=target['point_index'],
              depth_mm=c['depth_mm'],route_crc32=data['route_crc32'],first_delay_us=data['first_delay_us'],
              spacing_us=data['spacing_us'],stream_protocol=c['stream_protocol'],
              rate_hz=data['analysis']['board_rate_hz'],samples=sample_stats,
              nominal_tool_area_mm2=c['nominal_tool_area_mm2'])
    return item,features


def audit(paths):
    items=[]; features=[]
    for p in paths:
        item,values=load_contact(p); items.append(item); features.append(values)
    if len({(x['route_crc32'],x['first_delay_us'],x['spacing_us'],x['depth_mm']) for x in items})!=1:
        raise ValueError('Mixed routes/timings/depths cannot be pooled')
    distances={}
    for field in ('first_code','second_code'):
        distances[field]={}
        for index,kind in enumerate(('raw_delta','shape_delta')):
            values=np.array([f[field][index] for f in features])
            distances[field][kind]=np.sqrt(np.mean((values[:,None]-values[None,:])**2,axis=2)).tolist()
    return dict(schema='candidate27_contact_audit_v1',captures=items,distances=distances,deployed=False,
                physical_15hz_verified=False,full_spatial_accuracy_verified=False,
                scope='Exploratory independent contacts at recorded positions; correlated held frames are not independent trials.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures',type=Path,nargs='+'); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result=audit(args.captures); save_checkpoint(args.output,result)
    for row in result['captures']:
        print(row['point_index'],[(r['field'],r['signal_rms_codes'],r['baseline_noise_rms_codes']) for r in row['samples']])
    print(json.dumps(result['distances'],indent=2))
