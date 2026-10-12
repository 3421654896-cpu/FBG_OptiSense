"""Capture normal CH1 reads then a same-DAC held teacher (firmware >=1.0.30).

No CNC motion, arbitrary currents, table replacement or trained-model deployment.
This diagnostic stops/shutters after each target and cannot prove bandwidth.
Pressure is not independently measured; records are NOT pressure/area labels.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter, _wait_multirate_ack, wait_single
from capture_preserved_ch1_raw import normal_frames
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint
from capture_sparse_held_trajectory import command as held_command
from frame_telemetry import sentinel_offsets_for_version
from stress_multirate_protocol import MultirateConfiguration, REFERENCE_MAP_PERIOD, build_safe_arm_command_sequence

MAGIC = b'\xd8\x8d\x01'
LENGTH = 604
CHANNELS = ['CH0','CH1','CH2','CH3','PDT','PDR']
GAIN_KOHM = {0:2,1:40,2:5,3:20}


def command(point, tag):
    b=bytearray(held_command(point,tag)); b[3]=4
    return bytes(b)


def decode(packet, point, tag, selector, schedule_version):
    if len(packet) != LENGTH or packet[:3] != MAGIC or packet[-2:] != b'\x8d\xd8':
        raise ValueError('invalid production-pair teacher packet')
    u16=lambda at:int.from_bytes(packet[at:at+2],'big')
    u32=lambda at:int.from_bytes(packet[at:at+4],'big')
    if packet[3] != 0:
        raise ValueError('capture aborted, target not naturally fresh, or shutter not confirmed')
    if u32(4) != tag or u16(12) != point:
        raise ValueError('stale tag or wrong target')
    if schedule_version not in (4,5) or u32(8) < 0x1001e or packet[14] != schedule_version:
        raise ValueError('firmware or negotiated schedule mismatch')
    if packet[32:34] != bytes((selector,selector)) or packet[34] not in (3,7):
        raise ValueError('feedback IO or production-read mask mismatch')
    if packet[35:38] != bytes((25,6,1)) or packet[50:52] != bytes((2,1)):
        raise ValueError('invalid teacher shape, channel mask, or provenance flags')
    profile,primary,secondary=packet[15:18]
    if profile not in (0,1,2,3): raise ValueError('unknown profile')
    roi=set()
    if profile == 2:
        if not (primary < 9 and secondary < 9 and primary != secondary):
            raise ValueError('invalid wide ROI')
        roi={primary,secondary}
    elif profile == 3:
        if not (primary < 9 and secondary == 255): raise ValueError('invalid single ROI')
        roi={primary}
    elif (primary,secondary) != (255,255): raise ValueError('MAP/SURVEY ROI invalid')
    offsets=sentinel_offsets_for_version(schedule_version)
    expected=set(range(45)) if profile == 0 else {
        g*5+o for g in range(9) for o in ((1,2,3) if g in roi else (offsets[g],))}
    planned={p for p in range(45) if packet[38+p//8] & (1<<(p%8))}
    if packet[43]&0xe0 or len(planned)!=packet[18] or planned!=expected or point not in planned:
        raise ValueError('target/prefix freshness contradicts negotiated contract')
    if packet[19]&128 and profile!=0: raise ValueError('invalid prefix gap flag')
    if u32(44)>1_000_000 or u16(48)>4095: raise ValueError('invalid delay or RC estimate')
    reads=[]; previous_end=-1
    for i in range(3):
        at=52+i*10; start,end,value=u32(at),u32(at+4),u16(at+8)
        if not packet[34] & (1<<i):
            if any(packet[at:at+10]): raise ValueError('absent recheck contains stale data')
            continue
        if not previous_end < start < end < 300000 or value>4095:
            raise ValueError('invalid production CH1 value or time order')
        reads.append({'kind':('first','second','slow_recheck')[i],
                      'start_us':start,'end_us':end,'direct_ch1':value})
        previous_end=end
    groups=[]
    for i in range(26):
        at=82+i*20; start,end=u32(at),u32(at+4)
        adc=[u16(at+8+c*2) for c in range(6)]
        requested=0 if i==0 else 300000+(i-1)*20000
        if not previous_end < start < end <= 850000 or start<requested or any(x>4095 for x in adc):
            raise ValueError('invalid post-pair/teacher ADC or timestamps')
        # Labels must come from their scheduled windows, not a delayed burst.
        if i and start>requested+10000: raise ValueError('teacher missed scheduled window')
        if i==0 and end>=300000: raise ValueError('optical state is not an early feature')
        groups.append({'start_us':start,'end_us':end,'direct_adc':adc})
        previous_end=end
    monitor=[[r['direct_adc'][c] for c in (4,5,0,1,2,3)] for r in groups[1:]]
    median,sigma,drift,stable=app.fullband_accuracy_statistics(monitor)
    return {'request_tag':tag,'point_index':point,'firmware_version':u32(8),
            'wire_packet_hex':packet.hex(),
            'bandwidth_discontinuity':bool(packet[19]&128), 'map_age_frames':packet[19]&127,
            'schedule_version':schedule_version,'profile':profile,'primary_segment':primary,
            'secondary_segment':secondary,'plan_sequence':u32(20),'frame_start_ms':u32(24),
            'header_build_frame_offset_us':u32(28),'configured_wave_delay_us':u32(44),
            'feedback_selectors':list(packet[32:34]),'planned_indices':sorted(planned),
            'production_reads':reads,'firmware_rc_estimate':u16(48),
            'pair_available_us':reads[-1]['end_us'],'optical_state':groups[0],
            'extended_features_available_us':groups[0]['end_us'],
            'teacher_samples':groups[1:],
            'teacher':{'channels':['PDT','PDR','CH0','CH1','CH2','CH3'],
                       'median':median.tolist(),'sigma':sigma.tolist(),'drift':drift.tolist(),
                       'stable':bool(stable),'adc_value_kind':'direct_adc','dac_rewritten':False},
            'same_trial':True,'diagnostic_interrupts_scan':True,'firmware_shutter_confirmed':True,
            'pressure_ground_truth':None,
            'timing_note':'origin is post-DAC marker; read intervals include API instrumentation; extra six-channel state follows all production reads'}


def capture(device, point, tag, selector, schedule_version):
    device.reset_input_buffer(); device.write(command(point,tag))
    buffer=bytearray(); deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        buffer.extend(device.read(4096)); at=buffer.find(MAGIC)
        if at>=0 and len(buffer)>=at+LENGTH:
            return decode(bytes(buffer[at:at+LENGTH]),point,tag,selector,schedule_version)
    raise TimeoutError('same-trial production teacher missing; requires firmware >=1.0.30')


def selected_points(grating, version, *, full_map_reference=False):
    if grating not in range(1,10) or version not in (4,5):
        raise ValueError('invalid grating or schedule version')
    base=(grating-1)*5
    if full_map_reference:
        return list(range(base,base+5))
    return [base+sentinel_offsets_for_version(version)[grating-1]]


def validate_capture_profile(record, *, full_map_reference=False):
    # A packet from a different acquisition path is not interchangeable merely
    # because it targets the same DAC row. Preserve this constraint in the file.
    if full_map_reference:
        if (record['firmware_version'] < 0x1001f or record['profile'] != 0 or
                record['planned_indices'] != list(range(45)) or
                record['bandwidth_discontinuity'] is not True or record['map_age_frames'] != 0):
            raise ValueError('same-trial response is not the requested full-MAP reference')
    elif record['bandwidth_discontinuity']:
        raise ValueError('dynamic-path teacher unexpectedly came from a reference gap')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='COM6')
    p.add_argument('--versions',type=int,nargs='+',choices=(4,5),default=[4,5])
    p.add_argument('--selectors',type=int,nargs='+',choices=range(4),default=[2])
    p.add_argument('--gratings',type=int,nargs='+',choices=range(1,10),default=list(range(1,10)))
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--full-map-reference',action='store_true',
                   help='All five rows per selected grating; explicit gapped MAP path, NOT sparse dynamics')
    p.add_argument('--warmup-frames',type=int,default=100,
                   help='Complete actual frames before each independent held target (20..200)')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.repeats<=10 or any(len(a)!=len(set(a)) for a in (args.versions,args.selectors,args.gratings)):
        raise ValueError('bounded repetitions and unique settings required')
    if not 20<=args.warmup_frames<=200:
        raise ValueError('warmup-frames must be 20..200')
    if args.output.exists():
        raise FileExistsError('A previous capture must not be overwritten or silently resumed')
    rows,hidden=load_installed_rows()
    report={'schema':'production_pair_same_hold_teacher_v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),'scope':__doc__,
            'input_condition':'no commanded motion; external pressure not independently instrumented',
            'rows':rows,'hidden_codes':hidden,'channels':CHANNELS,'captures':[],
            'complete':False,'trained_model':False,'schedule_versions':args.versions,
            'full_map_reference':args.full_map_reference,'warmup_frames':args.warmup_frames,
            'requested_gratings':args.gratings,'requested_selectors':args.selectors,'requested_repeats':args.repeats,
            'training_eligible':False,'physical_15hz_verified':False,
            'contact_area_ground_truth_mm2':None,
            'assembly_note':'Each target is a separate same-DAC hold; the assembled 45 rows are not a simultaneous spectrum or one pressure event.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(args.output,report)
    with serial.Serial(args.port,2000000,timeout=.02,write_timeout=3) as device:
        try:
            device.dtr=True; tag=int(time.time_ns()&0x7fffffff) or 1
            for repeat in range(args.repeats):
                for grating in args.gratings:
                    # Interleave path/gain conditions; do not gather all v4 then v5.
                    settings=[(v,s) for v in args.versions for s in args.selectors]
                    if (repeat+grating)%2: settings.reverse()
                    trials=[(v,s,point) for v,s in settings for point in selected_points(
                        grating,v,full_map_reference=args.full_map_reference)]
                    # Reverse target order on alternate repetitions to expose
                    # drift; the optical prefix itself always remains forward.
                    if repeat%2: trials.reverse()
                    for version,selector,point in trials:
                        gain=GAIN_KOHM[selector]
                        config=MultirateConfiguration(enabled=True,map_period_frames=(
                            REFERENCE_MAP_PERIOD if args.full_map_reference else 0),
                            activate_threshold_codes=max(1,round(gain)),
                            release_threshold_codes=max(1,round(3*gain/5))).validate()
                        _safe_disarm(device)
                        device.write(app.build_stress_feedback_command(selector,selector))
                        if wait_single(device,app.decode_stress_feedback_status)!=((selector,selector),app.ACK_VALUE):
                            raise RuntimeError('feedback IO ACK missing')
                        _,arm,stress=build_safe_arm_command_sequence(config,schedule_version=version)
                        device.reset_input_buffer(); device.write(arm)
                        _wait_multirate_ack(device,config,requested_version=version)
                        device.reset_input_buffer(); device.write(stress)
                        warmup=normal_frames(device,args.warmup_frames,initial=True,schedule_version=version,
                                             reference_only=args.full_map_reference)
                        tag=(tag+1)&0xffffffff or 1
                        record=capture(device,point,tag,selector,version)
                        validate_capture_profile(record,full_map_reference=args.full_map_reference)
                        record.update(repeat=repeat,grating=grating,warmup=warmup,
                                      captured_utc=datetime.now(timezone.utc).isoformat(),
                                      host_received_monotonic_ns=time.monotonic_ns(),
                                      activation=config.activate_threshold_codes,release=config.release_threshold_codes)
                        report['captures'].append(record)
                        save_checkpoint(args.output,report)
                        print(f"v{version} G{grating} p{point} {gain}k rep={repeat+1} profile={record['profile']} raw={[r['direct_ch1'] for r in record['production_reads']]} RC={record['firmware_rc_estimate']} teacher={record['teacher']['median'][3]} stable={record['teacher']['stable']} pair_us={record['pair_available_us']}",flush=True)
            report['complete']=True
        except BaseException as exc:
            report['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            errors=[]; disarmed=shuttered=False
            try: disarmed=_safe_disarm(device)
            except Exception as exc: errors.append(f'disarm: {exc}')
            try: shuttered=_close_shutter(device)
            except Exception as exc: errors.append(f'shutter: {exc}')
            report['safety_cleanup']={'exact_disarm_ack':disarmed,'exact_soa_shutter_ack':shuttered,'errors':errors}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            report['finished_utc']=datetime.now(timezone.utc).isoformat()
            save_checkpoint(args.output,report)
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}',flush=True)
            if not disarmed or not shuttered: raise RuntimeError('safe shutdown not confirmed')


if __name__ == '__main__': main()
