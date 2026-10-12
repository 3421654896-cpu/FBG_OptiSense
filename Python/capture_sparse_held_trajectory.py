"""Actual sparse-prefix -> held installed target -> stable ADC teacher.

Opt-in USB diagnostic, not a real-time frame or pressure accuracy test. Firmware
stops and shutters after each capture. Each repeat starts a fresh warmed session.
No CNC control, arbitrary currents, model deployment, or table replacement.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter, _wait_multirate_ack, wait_single
from capture_preserved_ch1_raw import normal_frames
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_certified_ch1_transients import OFFSETS_US, CHANNELS
from frame_telemetry import sentinel_offsets_for_version
from stress_multirate_protocol import MultirateConfiguration, REFERENCE_MAP_PERIOD, build_safe_arm_command_sequence

LENGTH = 1206
MAGIC = b'\xd7\x7d\x01'


def command(point, tag):
    if not isinstance(point, int) or not 0 <= point < 45 or not isinstance(tag, int) or not 1 <= tag <= 0xffffffff:
        raise ValueError('installed point 0..44 and nonzero uint32 tag required')
    packet = bytearray(808)
    packet[:4] = b'\xff\xff\x03\x03'
    packet[4:6] = point.to_bytes(2, 'big')
    packet[6:10] = tag.to_bytes(4, 'big')
    return bytes(packet)


def decode(packet, point, tag, selector, schedule_version=4):
    if len(packet) != LENGTH or packet[:3] != MAGIC or packet[-2:] != b'\x7d\xd7':
        raise ValueError('invalid held-trajectory frame')
    u32 = lambda at: int.from_bytes(packet[at:at+4], 'big')
    if packet[3] != 0:
        raise ValueError('firmware capture aborted or shutter not confirmed')
    if u32(4) != tag or int.from_bytes(packet[12:14], 'big') != point:
        raise ValueError('wrong held target or stale request tag')
    minimum_firmware = 0x1001d if schedule_version == 5 else 0x1001c
    if schedule_version not in (4,5) or u32(8) < minimum_firmware or packet[14] != schedule_version or packet[33] != selector:
        raise ValueError('firmware, negotiated schedule or CH1 feedback mismatch')
    if packet[34:38] != bytes((33,25,6,1)):
        raise ValueError('unsupported ADC shape or missing diagnostic-gap flag')
    profile, primary, secondary = packet[15:18]
    if profile not in (0,1,2,3):
        raise ValueError('unknown prefix profile')
    roi = set()
    if profile == 2:
        if not (primary < 9 and secondary < 9 and primary != secondary):
            raise ValueError('invalid wide ROI')
        roi = {primary, secondary}
    elif profile == 3:
        if not (primary < 9 and secondary == 255):
            raise ValueError('invalid single ROI')
        roi = {primary}
    elif (primary, secondary) != (255,255):
        raise ValueError('MAP/SURVEY must not claim ROI')
    planned = {p for p in range(45) if packet[38+p//8] & (1 << (p%8))}
    offsets = sentinel_offsets_for_version(schedule_version)
    expected = set(range(45)) if profile == 0 else {
        g*5+offset for g in range(9)
        for offset in ((1,2,3) if g in roi else (offsets[g],))}
    if packet[43] & 0xe0 or planned != expected or len(planned) != packet[18]:
        raise ValueError('prefix sampling plan contradicts negotiated contract')
    records = []
    requested = list(OFFSETS_US) + [300000+i*20000 for i in range(25)]
    for index, offset in enumerate(requested):
        at = 44+index*20
        start, end = u32(at), u32(at+4)
        adc = [int.from_bytes(packet[c:c+2], 'big') for c in range(at+8,at+20,2)]
        if not offset <= start < end <= 850000 or any(x > 4095 for x in adc):
            raise ValueError('invalid ADC value or conversion interval')
        if records and start < records[-1]['end_us']:
            raise ValueError('overlapping/nonmonotonic conversion groups')
        records.append({'requested_us':offset,'start_us':start,'end_us':end,'direct_adc':adc})
    # Reorder only to match the existing single-mode stability gate API.
    monitor = [[r['direct_adc'][c] for c in (4,5,0,1,2,3)] for r in records[33:]]
    median, sigma, drift, stable = app.fullband_accuracy_statistics(monitor)
    return {'point_index':point,'request_tag':tag,'firmware_version':u32(8),
            'schedule_version':schedule_version,
            'prefix_profile':profile,'primary_segment':primary,'secondary_segment':secondary,
            'plan_sequence':u32(20),'frame_start_ms':u32(24),
            # Firmware reads this while constructing the header, a few us
            # after the probe's origin. Do not pass it off as exact DAC time.
            'header_build_frame_offset_us':u32(28),
            'planned_indices':sorted(planned),
            'target_would_normally_be_sampled':point in planned,
            'feedback_selectors':list(packet[32:34]),
            'prefix_bandwidth_discontinuity':bool(packet[19] & 128),
            'diagnostic_interrupts_scan':True,'firmware_shutter_confirmed':True,
            'curve':records[:33],'teacher_samples':records[33:],
            'teacher':{'channels':['PDT','PDR','CH0','CH1','CH2','CH3'],
                       'median':median.tolist(),'sigma':sigma.tolist(),
                       'drift':drift.tolist(),'stable':bool(stable),
                       'adc_value_kind':'direct_adc','dac_rewritten':False}}


def capture(device, point, tag, selector, schedule_version=4):
    device.reset_input_buffer()
    device.write(command(point,tag))
    rx = bytearray()
    deadline = time.monotonic()+4
    while time.monotonic() < deadline:
        rx.extend(device.read(4096))
        start = rx.find(MAGIC)
        if start >= 0 and len(rx) >= start+LENGTH:
            return decode(bytes(rx[start:start+LENGTH]),point,tag,selector,schedule_version)
    raise TimeoutError('held diagnostic missing; requires firmware >=1.0.28')


def validate_full_map_reference(record):
    if (record['firmware_version'] < 0x1001f or record['prefix_profile'] != 0
            or record['planned_indices'] != list(range(45))
            or not record['prefix_bandwidth_discontinuity']):
        raise ValueError('requested fixed full-MAP reference path was not returned')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default='COM6')
    parser.add_argument('--selector',type=int,choices=range(4),default=2)
    parser.add_argument('--schedule-version',type=int,choices=(4,5),default=4)
    parser.add_argument('--indices',type=int,nargs='+',default=[36])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--full-map-reference',action='store_true',
                        help='Fixed full-MAP diagnostic route; NOT fast-scan bandwidth evidence')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record-machine-position', action='store_true',
                        help='Read-only CNC coordinate brackets using the installed contact plane; no motion')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing capture must not be overwritten')
    if not 1 <= args.repeats <= 10 or not 1 <= len(args.indices) <= 45:
        raise ValueError('bounded captures required')
    for point in args.indices: command(point,1)
    rows, hidden = load_installed_rows()
    controller = target = None
    if args.record_machine_position:
        from mach3_controller import Mach3Controller
        from commission_fast_flank_center_contact import coverage_target
        from held_capture_position import reference, bracket
        target = coverage_target(9, 0.0)
        controller = Mach3Controller()
        try:
            reference(target, controller.status())
        except BaseException:
            controller.close()
            raise
    config = MultirateConfiguration(enabled=True,map_period_frames=(REFERENCE_MAP_PERIOD if args.full_map_reference else 0),
                                    activate_threshold_codes=5,release_threshold_codes=3).validate()
    report = {'schema':'sparse_held_trajectory_v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':__doc__,'channels':CHANNELS,'rows':rows,'hidden_codes':hidden,
              'configuration':{'selector':args.selector,'activate':5,'release':3,
                               'schedule_version':args.schedule_version,
                               'full_map_reference':args.full_map_reference},
              'captures':[],'complete':False}
    with ExitStack() as resources:
        if controller:
            resources.callback(controller.close)
        device = resources.enter_context(serial.Serial(args.port,2000000,timeout=.02,write_timeout=3))
        try:
            device.dtr = True
            tag = int(time.time_ns() & 0x7fffffff) or 1
            for repeat in range(args.repeats):
                for point in args.indices:
                    before_status = controller.status() if controller else None
                    before_status_ns = time.monotonic_ns()
                    if controller:
                        reference(target, before_status)
                    _safe_disarm(device)
                    device.write(app.build_stress_feedback_command(args.selector,args.selector))
                    if wait_single(device,app.decode_stress_feedback_status) != ((args.selector,args.selector),app.ACK_VALUE):
                        raise RuntimeError('feedback IO ACK missing')
                    _,arm,stress = build_safe_arm_command_sequence(config,schedule_version=args.schedule_version)
                    device.reset_input_buffer()
                    device.write(arm)
                    _wait_multirate_ack(device,config,requested_version=args.schedule_version)
                    device.reset_input_buffer()
                    device.write(stress)
                    warmup = normal_frames(device,100,initial=True,schedule_version=args.schedule_version)
                    tag = (tag+1) & 0xffffffff or 1
                    record = capture(device,point,tag,args.selector,args.schedule_version)
                    if args.full_map_reference:
                        validate_full_map_reference(record)
                    if controller:
                        after_status = controller.status()
                        record['machine_position_bracket'] = bracket(target, before_status, after_status)
                        record['machine_position_bracket'].update(
                            before_received_monotonic_ns=before_status_ns,
                            after_received_monotonic_ns=time.monotonic_ns())
                    record.update(repeat=repeat,warmup=warmup)
                    report['captures'].append(record)
                    print(f"point={point} repeat={repeat+1} profile={record['prefix_profile']} teacher_CH1={record['teacher']['median'][3]} stable={record['teacher']['stable']} curve_CH1={[r['direct_adc'][1] for r in record['curve']]}",flush=True)
            report['complete'] = True
        finally:
            errors, disarmed, shuttered = [],False,False
            try: disarmed = _safe_disarm(device)
            except Exception as exc: errors.append(f'disarm: {exc}')
            try: shuttered = _close_shutter(device)
            except Exception as exc: errors.append(f'shutter: {exc}')
            report['safety_cleanup'] = {'exact_disarm_ack':disarmed,'exact_soa_shutter_ack':shuttered,'errors':errors}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}',flush=True)
            if not disarmed or not shuttered: raise RuntimeError('safe shutdown not confirmed')


if __name__ == '__main__':
    main()
