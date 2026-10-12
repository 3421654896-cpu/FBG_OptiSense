"""USB-only guarded 27-point CH1 stream for the audited stream firmware.

The route and its single hidden transition guard live only in board RAM.  Every
frame carries direct ADC conversions plus guard provenance and board timing.
Nothing in this diagnostic moves CNC hardware or installs a production table.
"""
import argparse
import binascii
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file, now_text
from capture_fast_flank_stream import analyze as analyze_rate
from probe_current_candidate_route import load_route
from capture_fast_flank_stream import FIRMWARE

LIMITS = (58981, 58981, 32767, 24575, 24575)
PROTOCOL = 'last selected -> first; one hidden guard immediately before G8; 0.3 s target hold'


def _validate_route_and_guard(rows, guard, guard_before, guard_hold_us):
    if len(rows) != 27 or type(guard_before) is not int or not 1 <= guard_before < 27:
        raise ValueError('27 rows and an interior guard position are required')
    if (type(guard_hold_us) is not int or not 500 <= guard_hold_us <= 5000
            or guard_hold_us % 25):
        raise ValueError('Guard hold must be 500..5000 us in 25 us units')
    previous = -1
    for row in rows:
        index, codes = row['index'], tuple(row['codes'])
        if type(index) is not int or not previous < index <= 2000 or len(codes) != 5:
            raise ValueError('Forward unique 2001-table indices required')
        previous = index
        if any(type(code) is not int or not 0 <= code <= limit
               for code, limit in zip(codes, LIMITS)):
            raise ValueError('Candidate exceeds conservative laser current limits')
    index, codes = guard['index'], tuple(guard['codes'])
    if (type(index) is not int or len(codes) != 5
            or not rows[guard_before - 1]['index'] < index < rows[guard_before]['index']):
        raise ValueError('Guard must be a separate physical row between adjacent targets')
    if any(type(code) is not int or not 0 <= code <= limit
           for code, limit in zip(codes, LIMITS)):
        raise ValueError('Guard exceeds conservative laser current limits')


def command(rows, guard, cycles, tag, *, guard_before=21, guard_hold_us=1000,
            first_delay_us=400, spacing_us=50):
    if type(cycles) is not int or not 32 <= cycles <= 1024:
        raise ValueError('Cycles must be 32..1024')
    if type(tag) is not int or not 1 <= tag <= 0xffffffff:
        raise ValueError('Nonzero uint32 tag required')
    if any(type(value) is not int or not 50 <= value <= 600 or value % 25
           for value in (first_delay_us, spacing_us)) or first_delay_us + spacing_us > 650:
        raise ValueError('Invalid bounded sampling timing')
    _validate_route_and_guard(rows, guard, guard_before, guard_hold_us)
    packet = bytearray(808)
    packet[:6] = b'\xff\xff\x03\x0a\x02\x1b'
    packet[6:8] = first_delay_us.to_bytes(2, 'big')
    packet[8:12] = tag.to_bytes(4, 'big')
    packet[12:16] = b'C27!'
    packet[16:18] = cycles.to_bytes(2, 'big')
    packet[18] = 16
    packet[20:22] = spacing_us.to_bytes(2, 'big')
    packet[22] = guard_before
    packet[23] = guard_hold_us // 25
    for local, row in enumerate(rows):
        values = (row['index'], *row['codes'])
        packet[24 + local * 12:36 + local * 12] = b''.join(
            value.to_bytes(2, 'big') for value in values)
    packet[352:364] = b''.join(
        value.to_bytes(2, 'big') for value in (guard['index'], *guard['codes']))
    packet[364:368] = binascii.crc32(packet[:364]).to_bytes(4, 'big')
    return bytes(packet)


def extract_frames(buffer):
    while True:
        start = buffer.find(b'\xd9\x9d')
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return
        if start:
            del buffer[:start]
        if len(buffer) < 4:
            return
        kind, status = buffer[2:4]
        length = 20 if kind == 12 and status == 127 else 297 if kind == 12 else 48 if kind == 13 else 0
        if not length:
            del buffer[0]
            continue
        if len(buffer) < length:
            return
        wire = bytes(buffer[:length])
        del buffer[:length]
        yield wire


def decode(wire, tag, route_crc, first_delay_us, spacing_us, guard,
           expected_firmware=FIRMWARE):
    u16 = lambda at: int.from_bytes(wire[at:at + 2], 'big')
    u32 = lambda at: int.from_bytes(wire[at:at + 4], 'big')
    if wire[:4] == b'\xd9\x9d\x0c\x7f':
        if len(wire) != 20 or wire[-2:] != b'\x9d\xd9' or u32(4) != tag:
            raise ValueError('Guarded candidate rejection identity/trailer mismatch')
        raise RuntimeError(f'Guarded candidate rejected by board: flags={u16(8):04x}')
    if (len(wire) not in (297, 48) or wire[-2:] != b'\x9d\xd9'
            or u32(len(wire) - 6) != binascii.crc32(wire[:-6])):
        raise ValueError('Guarded frame length/CRC/trailer mismatch')
    if wire[:2] != b'\xd9\x9d' or u32(4) != tag or u32(8) != expected_firmware:
        raise ValueError('Guarded frame tag/firmware mismatch')
    guard_before = guard['before_order']
    guard_hold_us = guard['hold_us']
    guard_index = guard['index']
    if wire[2] == 13 and len(wire) == 48:
        if (u32(32) != route_crc or wire[18:22] != bytes((1, 1, 0, 2))
                or wire[36:38] != bytes((2, guard_before))
                or u16(38) != guard_hold_us or u16(40) != guard_index):
            raise ValueError('Guarded terminal route/shutter/feedback/provenance mismatch')
        return dict(kind='terminal', status=wire[3], completed_cycles=u16(12),
                    requested_cycles=u16(14), stop_reason=u16(16),
                    firmware_shutter_asserted=bool(wire[19]), i2c_error_delta=u16(22),
                    spi_error_delta=u32(24), elapsed_us=u32(28), wire_hex=wire.hex())
    if wire[2] != 12 or len(wire) != 297 or wire[3] & ~1 or u32(24) != route_crc:
        raise ValueError('Guarded data type/flags/route CRC mismatch')
    if (wire[28:34] != bytes((first_delay_us // 25, 27, 16,
                              spacing_us // 25, 0, 2))
            or wire[40:67] != bytes(range(27))):
        raise ValueError('Guarded timing/count/feedback/local order mismatch')
    if (wire[283:285] != bytes((2, guard_before)) or u16(285) != guard_hold_us
            or u16(287) != guard_index):
        raise ValueError('Guarded data provenance mismatch')
    sequence = u32(12)
    if bool(wire[3] & 1) != (sequence >= 16):
        raise ValueError('Noncausal warm-up flag')
    records = []
    previous = -1
    for local in range(27):
        at = 67 + local * 8
        first, second, write, end = (u16(at + offset) for offset in (0, 2, 4, 6))
        if (max(first, second) > 4095 or not previous < write < end <= u32(20) + 1
                or end - write < first_delay_us + spacing_us):
            raise ValueError('Invalid ADC code or conversion timing')
        previous = end
        records.append(dict(row=local, first_code=first, second_code=second,
                            write_end_us=write, second_end_us=end))
    guard_write_end_us = u16(289)
    if (not records[guard_before - 1]['second_end_us'] < guard_write_end_us
            or records[guard_before]['write_end_us'] - guard_write_end_us < guard_hold_us):
        raise ValueError('Hidden guard was not executed at the declared time')
    return dict(kind='data', sequence=sequence, cycle_start_us=u32(16),
                elapsed_us=u32(20), minimum_warmup_complete=sequence >= 16,
                guard_write_end_us=guard_write_end_us, records=records,
                i2c_error_delta=u16(34), spi_error_delta=u32(36), wire_hex=wire.hex())


def prepare_capture(candidate, probe_path, cycles=256, first_delay_us=400, spacing_us=50):
    rows, _ = load_route(candidate, 3)
    probe = json.loads(probe_path.read_text('utf-8'))
    source_guard = probe.get('guard', {})
    hold_us = round(float(source_guard.get('hold_s', float('nan'))) * 1_000_000)
    guard = dict(index=source_guard.get('index'), codes=source_guard.get('codes'),
                 before_order=source_guard.get('before_order'), hold_us=hold_us)
    if (probe.get('schema') != 'candidate_settled_guarded_route_probe_v1'
            or probe.get('protocol') != PROTOCOL or probe.get('cnc_commands') != []
            or not probe.get('complete') or not probe.get('disarm_ack')
            or not probe.get('shutter_ack') or probe.get('candidate_sha256') != sha256_file(candidate)
            or probe.get('route_rows') != rows or probe.get('feedback_selectors') != [0, 2]
            or len(probe.get('samples', [])) != 54 or len(probe.get('guard_events', [])) != 2):
        raise ValueError('Complete matching guarded settled-route optical probe is required')
    _validate_route_and_guard(rows, guard, guard['before_order'], guard['hold_us'])
    for index, event in enumerate(probe['guard_events']):
        if (event.get('repeat') != index + 1 or event.get('before_order') != guard['before_order']
                or event.get('guard_index') != guard['index']
                or event.get('completed_ns', 0) - event.get('started_ns', 0) < guard['hold_us'] * 1000):
            raise ValueError('Guard qualification event is invalid')
    for index, sample in enumerate(probe['samples']):
        point, meter = sample['point'], sample['meter']
        if (sample['order'] != index % 27 or sample['repeat'] != index // 27 + 1
                or point['index'] != rows[index % 27]['index'] or not point['stable']
                or point['dac_codes'] != rows[index % 27]['codes']
                or not math.isfinite(point['ch1_adc_code']) or not 0 <= point['ch1_adc_code'] < 4000
                or not math.isfinite(meter['wavelength_nm'])
                or abs(meter['wavelength_nm'] - rows[index % 27]['target_nm']) > .05
                or not math.isfinite(meter['power_mw']) or meter['power_mw'] <= 0
                or meter['side_mode_suppression_db'] is None
                or not math.isfinite(meter['side_mode_suppression_db'])
                or meter['side_mode_suppression_db'] < 20):
            raise ValueError('Settled guarded optical qualification record is invalid')
    tag = int(time.time_ns() & 0x7fffffff) or 1
    packet = command(rows, guard, cycles, tag, guard_before=guard['before_order'],
                     guard_hold_us=guard['hold_us'], first_delay_us=first_delay_us,
                     spacing_us=spacing_us)
    route_crc = binascii.crc32(packet[24:364])
    return dict(schema='candidate27_guarded_stream_v2', complete=False, created=now_text(),
                route_rows=rows, guard=guard, candidate_sha256=sha256_file(candidate),
                probe_sha256=sha256_file(probe_path), cycles_requested=cycles,
                first_delay_us=first_delay_us, spacing_us=spacing_us,
                route_crc32=route_crc, expected_firmware=FIRMWARE,
                command_hex=packet.hex(), frames=[], cnc_commands=[], deployed=False,
                physical_15hz_verified=False, disarm_ack=False, shutter_ack=False)


def execute_device(device, output, *, prepared, on_frame=None, should_stop=None):
    if output.exists():
        raise ValueError('Output must be new')
    report = deepcopy(prepared)
    packet = bytes.fromhex(report['command_hex'])
    tag = int.from_bytes(packet[8:12], 'big')
    guard = report['guard']
    rebuilt = command(report['route_rows'], guard, report['cycles_requested'], tag,
                      guard_before=guard['before_order'], guard_hold_us=guard['hold_us'],
                      first_delay_us=report['first_delay_us'], spacing_us=report['spacing_us'])
    route_crc = binascii.crc32(packet[24:364])
    if (packet != rebuilt or report['frames'] or report['complete']
            or report['route_crc32'] != route_crc or report['expected_firmware'] != FIRMWARE):
        raise ValueError('Prepared guarded capture has invalid identity/state')
    try:
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError('Initial shutdown not confirmed')
        worker = app.EqualIntervalWorker(device, (), {}, settle_s=0, feedback_selectors=(0, 2))
        if not worker._feedback_command_and_ack():
            raise RuntimeError('Feedback ACK missing')
        device.reset_input_buffer()
        device.write(packet)
        device.flush()
        deadline = time.monotonic() + min(55, 5 + report['cycles_requested'] * .05)
        buffer = bytearray()
        terminal = None
        while time.monotonic() < deadline and terminal is None:
            if should_stop is not None and should_stop():
                raise RuntimeError('External capture stop requested')
            buffer.extend(device.read(8192))
            for wire in extract_frames(buffer):
                row = decode(wire, tag, route_crc, report['first_delay_us'],
                             report['spacing_us'], guard)
                if row['i2c_error_delta'] or row['spi_error_delta']:
                    raise RuntimeError('Board I2C/SPI error')
                if row['kind'] == 'terminal':
                    terminal = row
                else:
                    if row['sequence'] != len(report['frames']):
                        raise ValueError('Missing/reordered cycle')
                    if (report['frames']
                            and row['cycle_start_us'] <= report['frames'][-1]['cycle_start_us']):
                        raise ValueError('Nonmonotonic board time')
                    report['frames'].append(row)
                    if on_frame is not None:
                        on_frame(row)
                    if len(report['frames']) % 64 == 0:
                        save_checkpoint(output, report)
        if terminal is None:
            raise TimeoutError('Guarded candidate terminal missing')
        report['terminal'] = terminal
        cycles = report['cycles_requested']
        if (terminal['status'] or terminal['stop_reason'] or terminal['completed_cycles'] != cycles
                or terminal['requested_cycles'] != cycles or len(report['frames']) != cycles):
            raise RuntimeError('Guarded candidate did not finish cleanly')
        report['analysis'] = analyze_rate(report['frames'])
        report['complete'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        errors = []
        try:
            report['disarm_ack'] = _safe_disarm(device)
        except Exception as exc:
            errors.append('disarm: ' + repr(exc))
        try:
            report['shutter_ack'] = _close_shutter(device)
        except Exception as exc:
            errors.append('shutter: ' + repr(exc))
        report['cleanup_errors'] = errors
        save_checkpoint(output, report)
    if not report['disarm_ack'] or not report['shutter_ack']:
        raise RuntimeError('Final shutdown not confirmed')
    print(json.dumps(report['analysis'], indent=2))
    return report


def execute(candidate, probe_path, output, cycles=256, first_delay_us=400, spacing_us=50):
    prepared = prepare_capture(candidate, probe_path, cycles, first_delay_us, spacing_us)
    with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=3) as device:
        return execute_device(device, output, prepared=prepared)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cycles', type=int, default=256)
    parser.add_argument('--first-delay-us', type=int, default=400)
    parser.add_argument('--spacing-us', type=int, default=50)
    args = parser.parse_args()
    execute(args.candidate, args.probe, args.output, args.cycles,
            args.first_delay_us, args.spacing_us)
