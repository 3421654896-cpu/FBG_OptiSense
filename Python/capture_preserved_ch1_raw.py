"""Capture real sparse-profile ADC pairs without forcing MAP or adding reads.

No CNC control, arbitrary DAC values or model deployment. A diagnostic replaces
one USB frame, so this diagnoses signal history, NOT transport bandwidth.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import (
    _safe_disarm, _close_shutter, _wait_multirate_ack,
    _extract_scan_frames, _require_initial_current_map, wait_single,
)
from capture_ch1_rc_diagnostic import load_installed_rows
from frame_telemetry import decode_board_frame_schedule, decode_board_frame_timing
from stress_multirate_protocol import MultirateConfiguration, build_safe_arm_command_sequence
from verify_runtime_firmware import board_extension

POINTS = 45
LENGTH = 40 + POINTS * 13 + 2
MAGIC = b'\xd6\x6d\x02'


def command(tag):
    if not isinstance(tag, int) or not 1 <= tag <= 0xffffffff:
        raise ValueError('tag must be a nonzero uint32')
    packet = bytearray(808)
    packet[:5] = b'\xff\xff\x03\x02\x01'
    packet[5:9] = tag.to_bytes(4, 'big')
    return bytes(packet)


def decode(packet, tag, selector, schedule_version=4):
    if len(packet) != LENGTH or packet[:3] != MAGIC or packet[-2:] != b'\x6d\xd6':
        raise ValueError('invalid preserved-profile diagnostic frame')
    if int.from_bytes(packet[3:5], 'big') != POINTS or not packet[5] & 2:
        raise ValueError('45 points and active CH1 required')
    if int.from_bytes(packet[6:10], 'big') != tag:
        raise ValueError('mismatched diagnostic tag')
    if schedule_version not in (4,5) or packet[22] != schedule_version or packet[35] != selector:
        raise ValueError('schedule capability or CH1 feedback mismatch')
    version = int.from_bytes(packet[36:40], 'big')
    if version < 0x1001b:
        raise ValueError('requires installed firmware >=1.0.27')
    if schedule_version == 5 and version < 0x1001d:
        raise ValueError('v5 requires firmware >=1.0.29')
    duration = int.from_bytes(packet[18:22], 'big')
    if not 0 < duration <= 131068:
        raise ValueError('invalid frame duration')
    # Reuse production validation of profile, ROI, bitmap, gap and freshness.
    extension = bytearray(142)
    extension[32:36] = packet[18:22]
    extension[36:48] = packet[22:34]
    extension[48:52] = packet[14:18]
    records = []
    for point in range(POINTS):
        at = 40 + point*13
        values = [int.from_bytes(packet[p:p+2], 'big') for p in range(at, at+8, 2)]
        recheck = packet[at+8]
        offset = int.from_bytes(packet[at+9:at+13], 'big')
        fresh = bool(packet[28+point//8] & (1 << (point % 8)))
        if any(value > 4095 for value in values) or recheck not in (0, 1):
            raise ValueError('invalid ADC or recheck flag')
        if fresh:
            if not 0 <= offset <= duration:
                raise ValueError('fresh row has invalid timestamp')
            ticks = offset // 2
        else:
            if offset != 0xffffffff or any(values) or recheck:
                raise ValueError('unmeasured row contains stale data')
            ticks = 0xffff
        extension[52+point*2:54+point*2] = ticks.to_bytes(2, 'big')
        records.append({'point_index': point, 'fresh': fresh,
                        'first_direct_adc': values[0] if fresh else None,
                        'second_direct_adc': values[1] if fresh else None,
                        'slow_direct_adc': values[2] if fresh and recheck else None,
                        'rc_estimate': values[3] if fresh else None,
                        'recheck': bool(recheck),
                        'sample_end_offset_us': offset if fresh else None})
    schedule = decode_board_frame_schedule(bytes(extension))
    return {'request_tag': tag, 'schedule_version': schedule.version,
            'plan_sequence': int.from_bytes(packet[10:14], 'big'),
            'frame_start_ms': int.from_bytes(packet[14:18], 'big'),
            'acquisition_us': duration, 'profile': schedule.profile,
            'primary_segment': schedule.primary_segment,
            'secondary_segment': schedule.secondary_segment,
            'fresh_indices': list(schedule.fresh_indices),
            'bandwidth_discontinuity': schedule.bandwidth_discontinuity,
            'feedback_selectors': list(packet[34:36]), 'firmware_version': version,
            'records': records}


def capture(device, tag, selector, schedule_version=4):
    device.reset_input_buffer()
    device.write(command(tag))
    buffer = bytearray()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        buffer.extend(device.read(4096))
        at = buffer.find(MAGIC)
        if at >= 0 and len(buffer) >= at + LENGTH:
            return decode(bytes(buffer[at:at+LENGTH]), tag, selector, schedule_version)
    raise TimeoutError('preserved-profile diagnostic response missing')


def normal_frames(device, count, initial=False, schedule_version=4, *, reference_only=False):
    buffer = bytearray()
    observed = []
    deadline = time.monotonic() + max(5, count*.2)
    while len(observed) < count and time.monotonic() < deadline:
        buffer.extend(device.read(4096))
        for frame in _extract_scan_frames(buffer, POINTS):
            extension = board_extension(frame)
            schedule = decode_board_frame_schedule(extension)
            timing = decode_board_frame_timing(extension)
            if reference_only:
                if (schedule is None or timing is None or
                        not schedule.is_complete_map or not schedule.has_point_timing or
                        not schedule.bandwidth_discontinuity or
                        int.from_bytes(extension[24:28], 'big') < 0x1001f):
                    raise ValueError('explicit full-MAP reference warmup not confirmed')
            elif initial and not observed:
                _require_initial_current_map(schedule, timing, requested_version=schedule_version)
            if schedule is None or schedule.version != schedule_version:
                raise ValueError('not the explicitly negotiated normal scan')
            observed.append({'sequence': timing.sequence, 'profile': schedule.profile,
                             'schedule_version': schedule.version,
                             'fresh_indices': list(schedule.fresh_indices),
                             'bandwidth_discontinuity': schedule.bandwidth_discontinuity,
                             'frame_start_ms': schedule.frame_start_ms})
    if len(observed) < count:
        raise TimeoutError('normal scan warmup missing')
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM6')
    parser.add_argument('--selector', type=int, choices=range(4), default=2)
    parser.add_argument('--schedule-version',type=int,choices=(4,5),default=4)
    parser.add_argument('--activate', type=int, default=5)
    parser.add_argument('--release', type=int, default=3)
    parser.add_argument('--captures', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.captures <= 100:
        raise ValueError('capture count must be 1..100')
    rows, hidden = load_installed_rows()
    config = MultirateConfiguration(enabled=True, map_period_frames=0,
                                    activate_threshold_codes=args.activate,
                                    release_threshold_codes=args.release).validate()
    report = {'schema': 'preserved_ch1_raw_v2', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'selector': args.selector, 'activate': args.activate, 'release': args.release,
              'schedule_version': args.schedule_version,
              'scope': 'unchanged actual sparse ADC path; replaces diagnostic USB frame, NOT bandwidth proof',
              'dac_rows': rows, 'hidden_codes': hidden, 'captures': [], 'complete': False}
    with serial.Serial(args.port, 2000000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            _safe_disarm(device)
            device.write(app.build_stress_feedback_command(args.selector, args.selector))
            if wait_single(device, app.decode_stress_feedback_status) != ((args.selector,args.selector),app.ACK_VALUE):
                raise RuntimeError('feedback IO ACK missing')
            _, arm, stress = build_safe_arm_command_sequence(config, schedule_version=args.schedule_version)
            device.reset_input_buffer()
            device.write(arm)
            _wait_multirate_ack(device, config, requested_version=args.schedule_version)
            device.reset_input_buffer()
            device.write(stress)
            report['warmup'] = normal_frames(device, 100, initial=True, schedule_version=args.schedule_version)
            tag = int(time.time_ns() & 0x7fffffff) or 1
            for index in range(args.captures):
                tag = (tag+1) & 0xffffffff or 1
                record = capture(device, tag, args.selector, args.schedule_version)
                report['captures'].append(record)
                row = record['records'][38 if args.schedule_version == 5 else 36]
                print(f"capture={index+1}/{args.captures} profile={record['profile']} G8 first={row['first_direct_adc']} second={row['second_direct_adc']} rc={row['rc_estimate']}", flush=True)
                if index+1 < args.captures:
                    record['following_normal_frames'] = normal_frames(device, 8, schedule_version=args.schedule_version)
            report['complete'] = True
        finally:
            errors = []
            disarmed = shuttered = False
            try: disarmed = _safe_disarm(device)
            except Exception as exc: errors.append(f'disarm: {exc}')
            try: shuttered = _close_shutter(device)
            except Exception as exc: errors.append(f'shutter: {exc}')
            report['safety_cleanup'] = {'exact_disarm_ack': disarmed,
                                        'exact_soa_shutter_ack': shuttered, 'errors': errors}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}', flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError('safe shutdown not confirmed')
    print('profiles=', dict(Counter(r['profile'] for r in report['captures'])))
    print(f'report={args.output.resolve()}')


if __name__ == '__main__':
    main()
