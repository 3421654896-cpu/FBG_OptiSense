"""Verify DAC command/I2C ACK, actual MCU route RAM, and CRC USB transport.

No optical-fit, waveform, centre-pm or reference-stability acceptance gates.
SWD never halts, resets, unlocks, or writes application RAM/flash. Reading MCU
route RAM is NOT reading the PI11210 DAC registers or measuring its outputs.
"""
import argparse
import binascii
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

from elftools.elf.elffile import ELFFile
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from beta_wire_trace import TraceSerial
from fbg_usb_ota import UsbOtaClient
from laser_dac_safety import FULLBAND_2001_CODE_LIMITS, validate_pi11210_codes
from machine_profile import get_runtime_machine_id, set_runtime_machine_id
from temporary_test_mode import command, execute_device, extract_frames, validate_plan_payload
from validate_beta_single_adc_live import validate_capture
from verify_lan_full_function_live import wait_single


def build_selected_dac_packet(codes):
    # This exact saved 63-row route uses the explicitly authorized 145mA
    # fullband profile, not the older 135mA/rounded calibration editor.
    expected = validate_pi11210_codes(codes, FULLBAND_2001_CODE_LIMITS)
    packet = app.build_single_value_dac_command(expected, soa_mode=0, fullband_2001=True)
    if tuple(int.from_bytes(packet[4+2*i:6+2*i], 'big') for i in range(5)) != expected:
        raise ValueError('Selected DAC packet changed exact source codes')
    return packet


def decode_dac_ack(frame, expected):
    if frame[:4] != b'\xff\xff\x02\x00': return None
    if len(frame) != 20: raise ValueError('Invalid DAC ACK size')
    actual = tuple(int.from_bytes(frame[4+2*i:6+2*i], 'big') for i in range(5))
    flags, errors = frame[15], int.from_bytes(frame[18:20], 'big')
    if actual != tuple(expected) or frame[14] != 0 or flags & 0x87 != 0x87 or flags & 0x38 or errors:
        raise RuntimeError('DAC echo/source/I2C-write acknowledgement mismatch or fault')
    return dict(echoed_codes=list(actual), all_i2c_writes_acknowledged=True,
                i2c_error_count=errors, flags=flags, raw_ack_hex=frame.hex(),
                actual_dac_register_readback=False)


def load_route_layout(path):
    with path.open('rb') as stream:
        elf = ELFFile(stream)
        candidates = [s for s in elf.get_section_by_name('.symtab').iter_symbols()
                      if s.name == 'candidateRoute' or s.name.endswith('.candidateRoute')]
        if len(candidates) != 1: raise ValueError('Ambiguous/missing route RAM symbol')
        symbol = candidates[0]
        address, size = symbol['st_value'], symbol['st_size']
        if not 0x20000000 <= address < address + size <= 0x20020000:
            raise ValueError('Route symbol outside SRAM')
        layouts = []
        for cu in elf.get_dwarf_info().iter_CUs():
            for die in cu.iter_DIEs():
                name = die.attributes.get('DW_AT_name')
                if die.tag == 'DW_TAG_typedef' and name and name.value == b'CandidateRoute':
                    obj = die.get_DIE_from_attribute('DW_AT_type')
                    if obj.attributes['DW_AT_byte_size'].value != size:
                        raise ValueError('DWARF/symbol route size differs')
                    layouts.append({m.attributes['DW_AT_name'].value.decode():
                                    m.attributes['DW_AT_data_member_location'].value
                                    for m in obj.iter_children() if m.tag == 'DW_TAG_member'})
        if not layouts or any(l != layouts[0] for l in layouts):
            raise ValueError('Missing or ambiguous route member layout')
        flash = []
        for segment in elf.iter_segments():
            if segment['p_type'] == 'PT_LOAD' and segment['p_filesz']:
                base, data = segment['p_paddr'], segment.data()
                if not 0x08008000 <= base < base + len(data) <= 0x08040000:
                    raise ValueError('ELF segment outside application flash')
                flash.append((base, data))
        if not flash: raise ValueError('Missing application flash segments')
    return address, size, layouts[0], flash


def validate_route_ram(raw, offsets, rows, waits, expected_crc):
    n = len(rows)
    unpack = lambda member, count: list(struct.unpack_from('<' + 'H'*count, raw, offsets[member]))
    if unpack('indices', n) != [r['index'] for r in rows]:
        raise RuntimeError('Actual MCU indices differ from host')
    if unpack('codes', n*5) != [v for r in rows for v in r['codes']]:
        raise RuntimeError('Actual MCU DAC codes differ from host')
    if unpack('pointDelayUs', n) != waits:
        raise RuntimeError('Actual MCU waits differ from host')
    for member, expected in dict(pointCount=n, protocolVersion=7, adcChannel=0,
                                  feedbackSelector=2, pointsPerPeak=7, fastDacWrites=0).items():
        if raw[offsets[member]] != expected: raise RuntimeError('MCU route metadata differs: '+member)
    crc = struct.unpack_from('<I', raw, offsets['tableCrc'])[0]
    if crc != expected_crc: raise RuntimeError('Actual MCU route CRC differs')
    return dict(rows_verified=n, individual_dac_codes_verified=n*5,
                physical_indices_verified=True, point_waits_verified=True,
                route_crc32=crc, dac_codes_exactly_match_host=True,
                actual_dac_register_readback=False)


def inspect_route_readonly(probe, elf_path, serial, rows, waits, crc, *, frequency=1000000):
    address, size, offsets, flash = load_route_layout(elf_path)
    options = dict(target_override='cortex_m', connect_mode='attach', auto_unlock=False,
                   resume_on_disconnect=False, no_config=True, frequency=frequency)
    options.update({'cache.enable_memory': False, 'cache.enable_register': False,
                    'cache.read_code_from_elf': False})
    session = ConnectHelper.session_with_chosen_probe(blocking=False, unique_id=probe, options=options)
    if session is None or session.probe.unique_id != probe:
        raise RuntimeError('Exact requested SWD probe unavailable')
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError('CPU not running; no resume/reset attempted')
        uid = [target.read32(0x1FFF7A10+4*i) for i in range(3)]
        actual_serial = f'{(uid[0]+uid[2]) & 0xFFFFFFFF:08X}{uid[1] >> 16:04X}'
        if actual_serial != serial: raise RuntimeError('SWD UID differs from USB; refusing RAM reads')
        for base, expected in flash:
            for offset in range(0, len(expected), 1024):
                wanted = expected[offset:offset+1024]
                if bytes(target.read_memory_block8(base+offset, len(wanted))) != wanted:
                    raise RuntimeError('Actual application flash differs from ELF')
        raw = bytes(target.read_memory_block8(address, size))
        if bytes(target.read_memory_block8(address, size)) != raw:
            raise RuntimeError('Route RAM changed during read; no atomic snapshot claimed')
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError('CPU stopped; no control action attempted')
        return dict(validate_route_ram(raw, offsets, rows, waits, crc),
                    swd_usb_identity_matched=True, actual_flash_matches_elf=True,
                    cpu_running_after=True, halt_reset_flash_ram_writes=False,
                    route_ram_sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--elf', type=Path, required=True)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--expected-version', required=True, type=lambda text: int(text, 0))
    parser.add_argument('--allow-light', action='store_true')
    args = parser.parse_args()
    if not args.allow_light: parser.error('Explicit optical source authorization required')
    for suffix in (args.output.suffix, '.capture.json', '.wire.bin', '.trace.json'):
        if args.output.with_suffix(suffix).exists(): parser.error('Output exists; historical evidence preserved')
    source = json.loads(args.session.read_text(encoding='utf-8'))
    if source['machine_id'] != 'machine_3' or source['signal_channel'] != 0 or source['feedback_selector'] != 2:
        raise ValueError('Requires actual machine3 CH0/5k source')
    plan = validate_plan_payload(source['plan'])
    if len(plan['rows']) != 63 or plan['points_per_peak'] != 7:
        raise ValueError('Requires the complete selected 63-row route')
    if source['use_point_waits'] or source['first_delay_us'] != 850 or source['boundary_wait_us'] != 850:
        raise ValueError('Saved settings differ from this 850us verification; no implicit wait change')
    load_route_layout(args.elf)
    for row in plan['rows']: build_selected_dac_packet(row['codes'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='selected_dac_usb_transport_verification_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), complete=False, checks={},
        command_and_usb_transport_passed=False,
        optical_accuracy_assessed=False, center_pm_gate_applied=False,
        actual_dac_register_readback=False, firmware_or_saved_parameters_changed=False)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    previous = get_runtime_machine_id()
    try:
        with UsbOtaClient(args.port, args.device_serial) as client: before = client.query_info()
        report['runtime_before'] = asdict(before)
        if before.firmware_version != args.expected_version or args.expected_version != 0x10076:
            raise RuntimeError('Requires explicitly selected actual v1.0.118')
        set_runtime_machine_id('machine_3')
        with TraceSerial(args.port, 2000000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            thermal = []
            device.thermal_observer.callback = lambda value: thermal.append(dict(value))
            try:
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial shutdown not confirmed')
                report['checks']['individual_dac_commands'] = []
                for row in plan['rows']:
                    expected = row['codes']
                    device.reset_input_buffer()
                    packet = build_selected_dac_packet(expected)
                    if device.write(packet) != len(packet): raise RuntimeError('Incomplete DAC command transfer')
                    device.flush()
                    ack = wait_single(device, lambda frame: decode_dac_ack(frame, expected))
                    report['checks']['individual_dac_commands'].append(dict(index=row['index'], **ack))
                if not _close_shutter(device): raise RuntimeError('Direct-command shutdown missing')
                device.raw.clear(); device.trace.clear()
                capture = execute_device(device, plan, args.output.with_suffix('.capture.json'), cycles=32,
                    first_delay_us=850, spacing_us=0, boundary_extra_us=0, point_delays_us=[850]*63,
                    adc_channel=0, feedback_selector=2, single_sample=True, fast_dac_writes=False)
                report['checks']['capture'] = validate_capture(capture, args.expected_version, 32, 63)
                wires = list(extract_frames(bytearray(device.raw)))
                first = next(w for w in wires if w[2] == 16 and w[3] != 127)
                tag = int.from_bytes(first[4:8], 'big')
                packet = command(plan['rows'],32,tag,first_delay_us=850,spacing_us=0,
                    point_delays_us=[850]*63,adc_channel=0,feedback_selector=2,single_sample=True)
                crc = binascii.crc32(packet[:796])
                tx = [v for v in device.trace if v['action'] == 'write' and v['header_hex'] == packet[:24].hex()]
                if (len(tx) != 1 or tx[0]['sha256'] != hashlib.sha256(packet).hexdigest()
                        or tx[0]['returned'] != 808): raise RuntimeError('Actual transmitted route differs')
                report['checks']['wire'] = dict(route_crc32=crc, actual_tx_packet_sha256=tx[0]['sha256'],
                    complete_808_byte_route_write=True, all_frame_crcs_tags_counts_orders_validated=True)
                args.output.with_suffix('.wire.bin').write_bytes(device.raw)
                args.output.with_suffix('.trace.json').write_text(json.dumps(device.trace), encoding='utf-8')
            finally:
                try: report['final_disarm_ack'] = bool(_safe_disarm(device))
                finally: report['final_shutter_ack'] = bool(_close_shutter(device))
            report['thermal_snapshots'] = thermal
            if (not thermal or any(not t['thermal_flags'] & 1 or t['thermal_flags'] & (4|8|32)
                                   or not 0<t['temperature_c']<85 for t in thermal)):
                raise RuntimeError('Missing or unsafe thermal telemetry')
        with UsbOtaClient(args.port, args.device_serial) as client: after = client.query_info()
        report['runtime_after'] = asdict(after)
        if before != after or not report['final_disarm_ack'] or not report['final_shutter_ack']:
            raise RuntimeError('Runtime changed or final shutdown unconfirmed')
        report['command_and_usb_transport_passed'] = True
        save()
        report['checks']['mcu_ram'] = inspect_route_readonly(args.probe,args.elf,args.device_serial,
                                                           plan['rows'],[850]*63,crc)
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        set_runtime_machine_id(previous)
        save()
        print(json.dumps(dict(complete=report['complete'], error=report.get('error'),
            command_and_usb_transport_passed=report['command_and_usb_transport_passed'],
            dac_command_ack_count=len(report['checks'].get('individual_dac_commands', [])),
            capture=report['checks'].get('capture'), mcu_ram=report['checks'].get('mcu_ram'),
            final_disarm_ack=report.get('final_disarm_ack'), final_shutter_ack=report.get('final_shutter_ack'),
            optical_accuracy_assessed=False, actual_dac_register_readback=False)), flush=True)


if __name__ == '__main__': main()
