"""Bounded continuous 18-row USB diagnostic; never CNC motion or table install.

Eight actual consecutive cycles and one target prefix use CH1-only direct pairs.
The last target is then held without a DAC rewrite for 25 later six-ADC samples.
Firmware shutters before returning, and the host independently disarms/shutters.
No optical wavelength, physical pressure, location or contact-area truth is measured.
"""
import argparse
import binascii
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from screen_ch1_reduced_pairs import candidate_indices

HERE = Path(__file__).resolve().parent
ROUTES = {'two_flanks': 0, 'g8_right_left': 1}
DELAYS = (0, 1500, 4350)
FIRMWARE = 0x10024
SUPPORTED_FIRMWARE = (0x10020, 0x10021, 0x10022, 0x10023, 0x10024)
CYCLES = 8


def command(route, target, extra_us, tag):
    if (not isinstance(route, str) or route not in ROUTES
            or type(target) is not int or target not in candidate_indices(route)
            or type(extra_us) is not int or extra_us not in DELAYS
            or type(tag) is not int or not 1 <= tag <= 0xffffffff):
        raise ValueError('unlisted route/target/delay or invalid tag')
    data = bytearray(808)
    data[:4] = b'\xff\xff\x03\x05'
    data[4] = ROUTES[route]
    data[5] = candidate_indices(route).index(target)
    data[6:8] = extra_us.to_bytes(2, 'big')
    data[8:12] = tag.to_bytes(4, 'big')
    data[12:16] = b'R18!'
    return bytes(data)


def packet_length(route, target):
    return 64 + (CYCLES*18 + candidate_indices(route).index(target)+1)*32 + 25*20 + 6


def codes_crc(rows):
    if len(rows) != 45:
        raise ValueError('exact installed 45-row table required')
    return binascii.crc32(b''.join(int(code).to_bytes(2, 'big') for row in rows for code in row['codes']))


def decode(wire, route, target, extra_us, tag, rows):
    request = command(route, target, extra_us, tag)
    u16 = lambda offset: int.from_bytes(wire[offset:offset+2], 'big')
    u32 = lambda offset: int.from_bytes(wire[offset:offset+4], 'big')
    expected = CYCLES*18 + request[5]+1
    if (len(wire) != packet_length(route, target) or wire[:4] != b'\xd9\x9d\x01\x00'
            or wire[-2:] != b'\x9d\xd9' or u32(len(wire)-6) != binascii.crc32(wire[:-6])):
        raise ValueError('short-route status/length/CRC/trailer mismatch')
    if (u32(4) != tag or u32(8) not in SUPPORTED_FIRMWARE or wire[12:16] != bytes((request[4], CYCLES, 18, request[5]))
            or u16(16) != extra_us or wire[18:20] != b'\x00\x02' or u16(20) != expected
            or wire[22:24] != b'\x19\x01' or u16(32) != expected or u16(34) != 0
            or u32(36) != codes_crc(rows) or u32(40) != 0
            or wire[44:62] != bytes(candidate_indices(route)) or wire[62:64] != b'\x01\x00'
            or not 0 < u32(24) <= 2_250_000):
        raise ValueError('short-route provenance, gain, count, errors or shutter mismatch')
    points = []
    previous_end = -1
    for index in range(expected):
        at = 64 + index*32
        row, cycle, local = u16(at), wire[at+2], wire[at+3]
        times = [u32(at+p) for p in (4, 8, 12, 16, 22, 26)]
        first, second = u16(at+20), u16(at+30)
        if (row != candidate_indices(route)[index%18] or (cycle, local) != divmod(index,18)
                or not previous_end < times[0] < times[1] <= times[2] < times[3] < times[4] < times[5]
                or times[2]-times[1] < 50 + (extra_us if row == 38 else 0)
                or times[4]-times[3] < 600 or max(first, second) > 4095):
            raise ValueError('noncausal timestamps, wrong row or invalid direct ADC')
        points.append(dict(target=row, cycle=cycle, local=local, write_start_us=times[0],
            write_end_us=times[1], first_start_us=times[2], first_end_us=times[3], first_code=first,
            second_start_us=times[4], second_end_us=times[5], second_code=second))
        previous_end = times[-1]
    if u32(28) != points[-1]['write_end_us']:
        raise ValueError('held target is not the last written fast point')
    teachers = []
    last_end = points[-1]['second_end_us']-u32(28)
    for index in range(25):
        at = 64 + expected*32 + index*20
        start, end = u32(at), u32(at+4)
        values = [u16(at+8+channel*2) for channel in range(6)]
        requested = 300000 + index*20000
        if (not last_end < start < end or not requested <= start < requested+10000
                or end+u32(28) > u32(24) or max(values) > 4095):
            raise ValueError('invalid or bunched held teacher samples')
        teachers.append(dict(start_us=start, end_us=end, adc=values))
        last_end = end
    # Reorder six-channel burst to the existing monitor-statistics convention.
    monitor = [[t['adc'][4],t['adc'][5],*t['adc'][:4]] for t in teachers]
    median, sigma, drift, stable = app.fullband_accuracy_statistics(monitor)
    return dict(route=route, target=target, extra_g8_right_delay_us=extra_us, tag=tag,
        firmware=u32(8), table_codes_crc32=codes_crc(rows), points=points,
        teacher_samples=teachers, teacher_ch1=float(median[3]), teacher_short_window_stable=bool(stable),
        teacher_ch1_sigma=float(sigma[3]), teacher_ch1_drift=float(drift[3]),
        firmware_shutter_asserted=True, elapsed_us=u32(24), wire_hex=wire.hex(),
        optical_wavelength_verified=False, physical_15hz_verified=False, training_eligible=False)


class RouteCaptureTimeout(TimeoutError):
    def __init__(self, received):
        super().__init__('continuous-route diagnostic not returned; recorded received bytes for review')
        self.received_wire_hex = bytes(received).hex()


class RouteRejected(RouteCaptureTimeout):
    def __init__(self, received, frame):
        super().__init__(received)
        self.reason_mask = int.from_bytes(frame[8:10], 'big')
        self.args = (f'route request rejected before output: reason mask 0x{self.reason_mask:04x}',)


def capture_packet(device, route, target, extra_us, tag):
    device.reset_input_buffer()
    device.write(command(route, target, extra_us, tag))
    data = bytearray()
    deadline = time.monotonic()+4
    length = packet_length(route, target)
    while time.monotonic() < deadline:
        data.extend(device.read(8192))
        reject = data.find(b'\xd9\x9d\x00\x7f')
        if reject >= 0 and len(data) >= reject+20:
            frame = data[reject:reject+20]
            if int.from_bytes(frame[4:8], 'big') == tag and frame[-2:] == b'\x9d\xd9':
                raise RouteRejected(data, frame)
        at = data.find(b'\xd9\x9d\x01')
        if at >= 0 and len(data) >= at+length:
            return bytes(data[at:at+length])
    raise RouteCaptureTimeout(data)


def execute(output, *, route='g8_right_left', targets=(36,38), extra_us=0, repeats=2, port='COM6'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (type(repeats) is not int or not 1 <= repeats <= 2 or not targets
            or len(set(targets)) != len(targets)):
        raise ValueError('one/two repeats and distinct targets required')
    for target in targets:
        command(route, target, extra_us, 1)
    rows, _ = load_installed_rows()
    report = dict(schema='ch1_continuous_short_route_teacher_v1', scope=__doc__, route=route,
        targets=list(targets), extra_us=extra_us, repeats=repeats, captures=[], complete=False,
        installed_rows=rows, table_sha256=sha256_file(HERE/'mode_tables_from_fullband_2001.json'),
        script_sha256=sha256_file(Path(__file__)), created_utc=datetime.now(timezone.utc).isoformat(),
        training_eligible=False, physical_15hz_verified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    with serial.Serial(port, 2_000_000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            tag = int(time.time_ns() & 0x7fffffff) or 1
            for repeat in range(repeats):
                for target in targets if repeat == 0 else targets[::-1]:
                    if not _safe_disarm(device) or not _close_shutter(device):
                        raise RuntimeError('initial disarm/shutter ACK missing')
                    worker = app.EqualIntervalWorker(device, (), {}, settle_s=0., feedback_selectors=(0,2))
                    if not worker._feedback_command_and_ack() or not worker._interruptible_wait(.01):
                        raise RuntimeError('feedback IO ACK/cancellation')
                    tag = (tag+1) & 0xffffffff or 1
                    record = dict(repeat=repeat, target=target, tag=tag)
                    report['captures'].append(record)
                    try:
                        wire = capture_packet(device, route, target, extra_us, tag)
                    except RouteCaptureTimeout as exc:
                        record['received_on_timeout_hex'] = exc.received_wire_hex
                        raise
                    record['wire_hex'] = wire.hex()
                    save_checkpoint(output, report)  # preserve rejected/aborted raw packets too
                    record['decoded'] = decode(wire, route, target, extra_us, tag, rows)
                    save_checkpoint(output, report)
                    last = record['decoded']['points'][-1]
                    print(f"trial={len(report['captures'])}/{len(targets)*repeats} p{target} "
                          f"early={last['first_code']}/{last['second_code']} "
                          f"held={record['decoded']['teacher_ch1']} "
                          f"shutter={record['decoded']['firmware_shutter_asserted']}", flush=True)
            report['complete'] = True
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            errors = []
            disarm = shutter = False
            try: disarm = _safe_disarm(device)
            except Exception as exc: errors.append(f'disarm: {exc}')
            try: shutter = _close_shutter(device)
            except Exception as exc: errors.append(f'shutter: {exc}')
            report['safety_cleanup'] = dict(exact_disarm_ack=disarm, exact_soa_shutter_ack=shutter, errors=errors)
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_checkpoint(output, report)
            print(f'disarm_confirmed={int(disarm)} soa_shutter_confirmed={int(shutter)}', flush=True)
            if not disarm or not shutter: raise RuntimeError('safe shutdown not confirmed')
    return report


def summarize(report):
    if report.get('schema') != 'ch1_continuous_short_route_teacher_v1' or not report.get('complete'):
        raise ValueError('complete continuous-route report required')
    if (type(report['repeats']) is not int or not 1 <= report['repeats'] <= 2
            or not report['targets'] or len(set(report['targets'])) != len(report['targets'])):
        raise ValueError('invalid capture set')
    for target in report['targets']:
        command(report['route'], target, report['extra_us'], 1)
    expected = {(target, repeat) for target in report['targets'] for repeat in range(report['repeats'])}
    seen, tags, trials = set(), set(), []
    for record in report['captures']:
        key = record['target'], record['repeat']
        if key in seen or key not in expected or record['tag'] in tags:
            raise ValueError('duplicate or unrequested capture')
        seen.add(key); tags.add(record['tag'])
        restored = decode(bytes.fromhex(record['wire_hex']), report['route'], record['target'],
                          report['extra_us'], record['tag'], report['installed_rows'])
        if restored != record['decoded']: raise ValueError('derived data differs from raw bytes')
        complete_cycles = [[p for p in restored['points'] if p['cycle']==c] for c in range(CYCLES)]
        # Direct same-point completion gaps across REAL cycles; never USB arrival intervals.
        gaps = [complete_cycles[c][i]['second_end_us']-complete_cycles[c-1][i]['second_end_us']
                for c in range(1,CYCLES) for i in range(18)]
        late_cycles = complete_cycles[4:]
        point_values = {str(row): [next(p['second_code'] for p in cycle if p['target']==row)
                                   for cycle in late_cycles] for row in candidate_indices(report['route'])}
        last = restored['points'][-1]
        trials.append(dict(target=record['target'], repeat=record['repeat'],
            same_trial_first=last['first_code'], same_trial_second=last['second_code'],
            same_trial_teacher=restored['teacher_ch1'],
            same_trial_error_codes=last['second_code']-restored['teacher_ch1'],
            short_teacher_stable=restored['teacher_short_window_stable'],
            same_point_gap_mean_us=statistics.fmean(gaps), same_point_gap_max_us=max(gaps),
            last_four_cycle_second_codes=point_values,
            per_cycle_scan_span_us=[cycle[-1]['second_end_us']-cycle[0]['write_start_us'] for cycle in complete_cycles]))
    if seen != expected: raise ValueError('missing capture')
    return dict(schema='ch1_continuous_short_route_analysis_v1', route=report['route'],
        extra_us=report['extra_us'], trials=trials, actual_consecutive_cycles_per_trial=CYCLES,
        pressure_bandwidth_verified=False, optical_wavelength_verified=False, training_eligible=False,
        scope=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', choices=tuple(ROUTES), default='g8_right_left')
    parser.add_argument('--targets', type=int, nargs='+', default=[36,38])
    parser.add_argument('--extra-us', type=int, choices=DELAYS, default=0)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--analyze', type=Path)
    args = parser.parse_args()
    if args.analyze:
        if args.output.exists(): raise FileExistsError(args.output)
        result = summarize(json.loads(args.analyze.read_text('utf-8')))
        result['source_sha256'] = sha256_file(args.analyze)
        result['analyzer_sha256'] = sha256_file(Path(__file__))
        save_checkpoint(args.output, result)
        print(f"redecoded={len(result['trials'])}")
    else:
        execute(args.output, route=args.route, targets=tuple(args.targets), extra_us=args.extra_us, repeats=args.repeats)
