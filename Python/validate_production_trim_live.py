"""Identity-locked production smoke test with zero laser DAC codes only.

No flash, installed table, Wi-Fi credentials, or user profile is modified.
SOURCE may be selected to exercise acquisition, but GAIN and SOA codes are
always zero. Every exit independently attempts the exact hardware shutter.
This is a protocol/safety test, not optical or mechanical metrology.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
from thermal_serial import ThermalSerial
from serial.tools import list_ports
import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
import dense_scan_transport as dense
from fbg_usb_ota import UsbOtaClient
from temporary_test_mode import execute_device
from stress_multirate_protocol import build_multirate_arm_command, decode_multirate_ack
from verify_lan_full_function_live import wait_single
from wifi_status_query import (query as query_wifi, build_status_query,
                               find_status_reply)

ZERO_CODES = (0, 0, 0, 0, 0)


def zero_source(device):
    device.reset_input_buffer()
    device.write(app.build_single_value_dac_command(ZERO_CODES))

    def decode(frame):
        if frame[:4] != b'\xff\xff\x02\x00':
            return None
        codes = tuple(int.from_bytes(frame[4+2*i:6+2*i], 'big') for i in range(5))
        if codes != ZERO_CODES or frame[14] != app.PI11210_SOA_SOURCE_MODE:
            raise RuntimeError('Zero-source acknowledgement does not match')
        app.UnlimitedAccuracyWorker._validate_ack_status(frame, 0)
        return dict(codes=codes, soa_mode=frame[14], pi_flags=frame[15],
                    raw_status=int.from_bytes(frame[16:18], 'big'),
                    i2c_error_count=int.from_bytes(frame[18:20], 'big'))

    return wait_single(device, decode)


def collect_monitors(device, duration_s=2):
    device.reset_input_buffer()
    deadline = time.monotonic() + duration_s
    buffer, adc, temperatures = bytearray(), [], []
    while time.monotonic() < deadline:
        buffer.extend(device.read(4096))
        for frame in app.extract_single_value_frames(buffer):
            values = app.decode_single_value_monitor_frame(frame)
            if values is not None:
                adc.append(values)
            if frame[:4] == b'\xff\xff\x02\x01':
                temperature = (int.from_bytes(frame[4:6], 'big') +
                               int.from_bytes(frame[6:8], 'big') / 10000)
                if not 0 < temperature < 85:
                    raise RuntimeError('Temperature reading invalid or unsafe')
                temperatures.append(temperature)
    if len(adc) < 30 or not temperatures or any(v > 4095 for row in adc for v in row):
        raise RuntimeError('Fresh ADC/temperature monitor coverage insufficient')
    return dict(adc_frames=len(adc), temperature_c=temperatures,
                rpm_verified=False, pid_target_runtime_verified=False)


def manual_zero(device):
    device.write(app.build_work_mode_command(1))
    time.sleep(.12)
    device.reset_input_buffer()
    device.write(app.build_single_value_dac_command(ZERO_CODES))
    try:
        # MANUAL replies use 01/00, not the EXTRA-only 02/* extractor.
        buffer, frame = bytearray(), None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and frame is None:
            buffer.extend(device.read(4096))
            for candidate in app.extract_manual_scan_frames(buffer):
                if app.manual_scan_ack_status(candidate) is not None:
                    frame = candidate
                    break
        if frame is None:
            raise RuntimeError('Manual calibration acknowledgement missing')
        if app.manual_scan_ack_status(frame) != app.ACK_VALUE:
            raise RuntimeError('Manual calibration zero command rejected')
        return dict(accepted=True, pdt_code=int.from_bytes(frame[5:7], 'big'),
                    pdr_code=int.from_bytes(frame[8:10], 'big'))
    finally:
        if not _close_shutter(device):
            raise RuntimeError('Manual calibration cleanup shutter missing')


def reject_legacy(device, mode):
    device.write(app.build_work_mode_command(mode))
    time.sleep(.12)
    device.reset_input_buffer()
    # This query reports the real work state, without requesting any scan.
    device.write(build_status_query())
    received = bytearray()
    deadline = time.monotonic() + 2
    result = None
    while time.monotonic() < deadline:
        received.extend(device.read(4096))
        result = find_status_reply(received)
        if result is not None:
            break
    if result is None:
        raise RuntimeError('Actual work-state status query missing')
    if result['work_state'] != 2:
        raise RuntimeError(f'Unsupported mode {mode} did not fall back to EXTRA')
    # Observe the actual gate before any explicit cleanup can hide a failure.
    capability = dense.probe(device)
    if capability is None or capability['soa_mode'] != 1:
        raise RuntimeError('Unsupported legacy mode did not leave the source shuttered')
    return dict(requested_mode=mode, actual_work_state=result['work_state'],
                actual_soa_mode=capability['soa_mode'], shutter_status_verified=True)


def reject_legacy_arm(device):
    results = []
    for version in (3, 4, 5):
        device.reset_input_buffer()
        device.write(build_multirate_arm_command(schedule_version=version))
        ack = wait_single(device, decode_multirate_ack)
        if ack.accepted or ack.configuration.enabled or ack.schedule_version != version:
            raise RuntimeError('Trimmed firmware accepted legacy ARM')
        results.append(asdict(ack))
    if not _safe_disarm(device):
        raise RuntimeError('Safe DISARM missing after rejected legacy ARM')
    return results


def dense_zero(device, count):
    worker = app.EqualIntervalWorker(device, (), {}, settle_s=.002,
                                    feedback_selectors=(0, 2))
    if not worker._feedback_command_and_ack():
        raise RuntimeError('Physical feedback acknowledgement missing')
    capability = dense.probe(device)
    if capability is None:
        raise RuntimeError('Dense capability removed unexpectedly')
    points = [app.FullbandAccuracyPoint(index=i, target_nm=1525+i*.02,
              measured_nm=1525+i*.02, codes=ZERO_CODES) for i in range(count)]
    started = time.monotonic()
    rows = list(dense.acquire(device, points, {}, settle_s=.002, selectors=(0, 2)))
    elapsed = time.monotonic() - started
    if [r['index'] for r in rows] != list(range(count)):
        raise RuntimeError('Dense point loss or reordering')
    return dict(points=len(rows), requested_wait_s=count*.002,
                wall_s=round(elapsed, 4), capability=capability,
                zero_codes_only=True, optical_accuracy_verified=False)


def temporary_feedback_zero_matrix(device, output):
    """Exercise frontend gains with absolute 850 us at every selected point."""
    rows = [dict(index=i, target_nm=1525+i*.02, measured_nm=1525+i*.02,
                 codes=list(ZERO_CODES)) for i in range(45)]
    results = []
    for channel in range(4):
        for selector in (range(4) if channel < 2 else (0,)):
            target = output.with_suffix(f'.ch{channel}.gain{selector}.json')
            if target.exists():
                raise FileExistsError(target)
            result = execute_device(device, dict(rows=rows), target, cycles=32,
                first_delay_us=850, spacing_us=50, boundary_extra_us=0,
                point_delays_us=[850]*45,
                feedback_selector=selector, adc_channel=channel)
            if (not result.get('complete') or result.get('received_frames') != 32
                    or not result.get('disarm_ack') or not result.get('shutter_ack')):
                raise RuntimeError(f'CH{channel} gain selector {selector} dark test incomplete')
            summary = dict(channel=channel, feedback_selector=selector,
                feedback_kohm=(2, 40, 5, 20)[selector], received_frames=32,
                complete=True, first_delay_us=850, boundary_extra_us=0,
                point_delays_us=[850]*45,
                disarm_ack=True, shutter_ack=True, zero_codes_only=True,
                mean_hz=result['mean_hz'], terminal=result['terminal'])
            results.append(summary)
            print(f'CH{channel} {summary["feedback_kohm"]} kOhm: 32 zero-current frames PASS',
                  flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--expected-version', type=lambda v: int(v, 0), required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.finger.json').exists():
        parser.error('Output exists; preserving previous measurements')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  device_serial=args.device_serial, port=args.port,
                  zero_codes_only=True, optical_accuracy_verified=False,
                  complete=False, final_shutter_ack=False, checks={})
    try:
        with UsbOtaClient(args.port, args.device_serial) as client:
            info = client.query_info()
            report['runtime_before'] = asdict(info)
        if info.firmware_version != args.expected_version:
            raise RuntimeError('Unexpected running firmware version')
        matching = [p for p in list_ports.comports() if p.device == args.port
                    and p.serial_number == args.device_serial
                    and p.vid == 0x483 and p.pid == 0x5740]
        if len(matching) != 1:
            raise RuntimeError('Exact STM32 identity no longer present')
        with ThermalSerial(args.port, 2_000_000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            thermal = []
            phase = ['single']
            device.thermal_observer.callback = lambda value: thermal.append(dict(value, phase=phase[0]))
            try:
                report['checks']['initial_shutter'] = _close_shutter(device)
                report['checks']['client_safe_disarm'] = _safe_disarm(device)
                report['checks']['single_zero_ack'] = zero_source(device)
                if not _close_shutter(device):
                    raise RuntimeError('Zero-source shutter missing')
                report['checks']['monitors'] = collect_monitors(device)
                phase[0] = 'manual'
                report['checks']['manual_calibration_zero'] = manual_zero(device)
                report['checks']['legacy_modes'] = [reject_legacy(device, mode) for mode in (0, 3)]
                report['checks']['legacy_arm_rejected'] = reject_legacy_arm(device)
                report['checks']['wifi_status'] = query_wifi(device)
                phase[0] = 'dense'
                report['checks']['dense_2001_zero'] = dense_zero(device, 2001)
                if not _close_shutter(device):
                    raise RuntimeError('Dense completion shutter missing')
                rows = [dict(index=i, target_nm=1525+i*.02, measured_nm=1525+i*.02,
                             codes=list(ZERO_CODES)) for i in range(45)]
                phase[0] = 'finger'
                finger = execute_device(device, dict(rows=rows),
                    args.output.with_suffix('.finger.json'), cycles=32,
                    first_delay_us=350, spacing_us=50, feedback_selector=2)
                report['checks']['finger_45_zero'] = dict(
                    complete=finger['complete'], received_frames=finger['received_frames'],
                    mean_hz=finger['mean_hz'], terminal=finger['terminal'],
                    disarm_ack=finger['disarm_ack'], shutter_ack=finger['shutter_ack'])
                if args.expected_version >= 0x0001006E:
                    report['checks']['temporary_feedback_zero_matrix'] = (
                        temporary_feedback_zero_matrix(device, args.output))
                print('Single, calibration, dense 2001 and finger 45 protocol checks passed', flush=True)
                report['checks']['thermal_snapshots'] = thermal
                if args.expected_version >= 0x0001006D:
                    for required in ('single', 'dense', 'finger'):
                        valid = [v for v in thermal if v['phase'] == required and v['thermal_flags'] & 1]
                        if not valid:
                            raise RuntimeError(f'No valid thermal telemetry during {required}')
            finally:
                report['final_shutter_ack'] = _close_shutter(device)
        with UsbOtaClient(args.port, args.device_serial) as client:
            after = client.query_info()
            report['runtime_after'] = asdict(after)
        if after != info or not report['final_shutter_ack']:
            raise RuntimeError('Runtime identity changed or final shutter unconfirmed')
        report['complete'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
