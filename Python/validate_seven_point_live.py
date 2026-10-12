"""Identity-locked USB seven-point acceptance with all five DAC codes zero.

Exercises the user's selected physical indices, not their optical currents.
No flash, Wi-Fi, SWD, installed table or persistent machine profile writes.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from serial.tools import list_ports

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_usb_ota import UsbOtaClient
from guarded_application_upgrade import zero_all_dacs_and_verify
from temporary_reference import acquire_selected_reference
from temporary_test_mode import execute_device, validate_plan_payload
from thermal_serial import ThermalSerial


def check_capture(capture, version, expected_frames, continuous=False):
    if (not capture['complete'] or capture['received_frames'] != expected_frames
            or not capture['disarm_ack'] or not capture['shutter_ack']
            or capture['cleanup_errors']
            or capture['point_count'] != 63 or capture['points_per_peak'] != 7):
        raise RuntimeError('Incomplete 63-point capture or shutdown')
    if continuous and not capture['stopped_by_user']:
        raise RuntimeError('Continuous stop not exercised')
    for frame in capture['frames']:
        if (frame['firmware'] != version or len(frame['records']) != 63
                or frame['i2c_errors'] or frame['spi_errors']):
            raise RuntimeError('63-point version/count/bus validation failed')
    return dict(complete=True, received_frames=expected_frames,
                point_count=63, points_per_peak=7,
                channel=capture['signal_channel'],
                feedback_selector=capture['feedback_selector'],
                mean_hz=capture['mean_hz'], p99_period_ms=capture['p99_period_ms'],
                disarm_ack=True, shutter_ack=True,
                stopped_by_user=capture['stopped_by_user'],
                terminal=capture.get('terminal'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--expected-version', type=lambda v: int(v, 0), required=True)
    parser.add_argument('--source-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserving previous acceptance records')
    source = json.loads(args.source_capture.read_text(encoding='utf-8'))
    selected = validate_plan_payload(source['plan'])
    if selected['point_count'] != 63 or selected['points_per_peak'] != 7:
        parser.error('Requires a validated nine-peak seven-point plan')
    # Retain grouping and real selected coordinates, but never source currents.
    rows = [dict(index=r['index'], group=r['group'], target_nm=r['target_nm'],
                 measured_nm=r['measured_nm'], codes=[0, 0, 0, 0, 0])
            for r in selected['rows']]
    plan = dict(rows=rows, points_per_peak=7, zero_codes_only=True,
                optical_accuracy_verified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  device_serial=args.device_serial, port=args.port,
                  source_capture=str(args.source_capture.resolve()),
                  selected_indices=[r['index'] for r in rows],
                  zero_codes_only=True, optical_accuracy_verified=False,
                  complete=False, final_disarm_ack=False, final_shutter_ack=False,
                  checks={})
    thermal, phase = [], ['initial']

    def capture(device, name, channel, selector, cycles=32, wait=850, continuous=False):
        destination = args.output.with_suffix(f'.{name}.json')
        if destination.exists() or destination.with_suffix('.frames.jsonl').exists():
            raise FileExistsError(destination)
        received = [0]

        def on_frame(_frame):
            received[0] += 1

        result = execute_device(device, plan, destination,
            cycles=0 if continuous else cycles,
            first_delay_us=850, spacing_us=50, boundary_extra_us=0,
            point_delays_us=[wait]*63, adc_channel=channel, feedback_selector=selector,
            on_frame=on_frame, should_stop=lambda: continuous and received[0] >= cycles)
        summary = check_capture(result, args.expected_version, cycles, continuous)
        report['checks'][name] = summary
        print(json.dumps(dict(check=name, **summary)), flush=True)

    try:
        with UsbOtaClient(args.port, args.device_serial) as client:
            before = client.query_info()
        report['runtime_before'] = asdict(before)
        if before.firmware_version != args.expected_version:
            raise RuntimeError('Unexpected running firmware version')
        matches = [p for p in list_ports.comports() if p.device == args.port
                   and p.vid == 0x0483 and p.pid == 0x5740
                   and p.serial_number == args.device_serial]
        if len(matches) != 1:
            raise RuntimeError('Exact STM32 USB identity no longer present')
        # This affects only this Python process, not the saved customer profile.
        app.set_runtime_machine_id('machine_3')
        with ThermalSerial(args.port, 2_000_000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            device.thermal_observer.callback = lambda value: thermal.append(
                dict(value, phase=phase[0]))
            try:
                if (not _close_shutter(device) or not _safe_disarm(device)
                        or not zero_all_dacs_and_verify(device) or not _close_shutter(device)):
                    raise RuntimeError('Initial zero-current/shutter confirmation missing')
                phase[0] = 'finite_63'
                channel, selector = source['signal_channel'], source['feedback_selector']
                capture(device, 'original_channel_63', channel, selector, cycles=64)
                for test_channel in range(4):
                    for gain in (range(4) if test_channel < 2 else (0,)):
                        if (test_channel, gain) == (channel, selector):
                            continue
                        capture(device, f'ch{test_channel}_gain{gain}_63', test_channel, gain)
                phase[0] = 'continuous_63'
                capture(device, 'continuous_stop_63', channel, selector, cycles=64, continuous=True)
                phase[0] = 'long_wait_63'
                capture(device, 'absolute_15ms_63', channel, selector, wait=15000)
                phase[0] = 'reference_63'
                reference = acquire_selected_reference(device,
                    args.output.with_suffix('.reference_63.json'), selected_rows=rows,
                    signal_channel=channel, feedback_selector=selector)
                if (not reference['complete'] or len(reference['rows']) != 63
                        or not reference['disarm_ack'] or not reference['shutter_ack']
                        or reference['stability_filter_applied']
                        or reference['requested_point_dwell_s'] != .63
                        or [r['index'] for r in reference['rows']] != report['selected_indices']
                        or any(tuple(r['dac_codes']) != (0, 0, 0, 0, 0)
                               for r in reference['rows'])):
                    raise RuntimeError('63-point raw 10 ms reference validation failed')
                report['checks']['raw_reference_63'] = {key: reference.get(key) for key in (
                    'complete', 'requested_point_count', 'settle_s', 'requested_point_dwell_s',
                    'acquisition_backend', 'acquisition_elapsed_s', 'board_acquisition_elapsed_s',
                    'total_elapsed_s', 'disarm_ack', 'shutter_ack', 'stability_filter_applied')}
                print(json.dumps(dict(check='raw_reference_63',
                                      **report['checks']['raw_reference_63'])), flush=True)
                valid = [v for v in thermal if v['thermal_flags'] & 1]
                faults = [v for v in thermal if not v['thermal_flags'] & 1
                          or v['thermal_flags'] & (4 | 8 | 32)]
                if not valid or any(not 0 < v['temperature_c'] < 85 for v in valid):
                    raise RuntimeError('Valid thermal telemetry missing or unsafe')
                report['thermal_snapshots'] = thermal
                report['thermal_summary'] = dict(valid_snapshots=len(valid),
                    fault_snapshots=len(faults),
                    sensor_age_max_ms=max(v['sensor_age_ms'] for v in valid),
                    temperature_min_c=min(v['temperature_c'] for v in valid),
                    temperature_max_c=max(v['temperature_c'] for v in valid),
                    rpm_min=min(v['fan_rpm'] for v in valid),
                    rpm_max=max(v['fan_rpm'] for v in valid),
                    duty_permille=sorted(set(v['fan_duty_permille'] for v in valid)))
                if faults:
                    raise RuntimeError('Thermal sensor/tach/PWM fault during seven-point acceptance')
            finally:
                try:
                    report['final_disarm_ack'] = _safe_disarm(device)
                finally:
                    report['final_shutter_ack'] = _close_shutter(device)
        with UsbOtaClient(args.port, args.device_serial) as client:
            after = client.query_info()
        report['runtime_after'] = asdict(after)
        if after != before or not report['final_disarm_ack'] or not report['final_shutter_ack']:
            raise RuntimeError('Runtime changed or final shutdown unconfirmed')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({key: report.get(key) for key in (
            'complete', 'error', 'final_disarm_ack', 'final_shutter_ack', 'thermal_summary')},
            ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
