"""Identity-locked independent five-DAC benchmark; no flash, no light.

All physical codes are zero. The explicit wait is identical before/after a
firmware comparison; no customer defaults, calibration, or wait tables change.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import app_JDSU as app
from beta_wire_trace import TraceSerial
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_usb_ota import UsbOtaClient
from guarded_application_upgrade import zero_all_dacs_and_verify
from temporary_test_mode import execute_device, validate_plan_payload
from validate_beta_single_adc_live import validate_capture


def zero_plan(payload):
    selected = validate_plan_payload(payload['plan'])
    if selected['point_count'] != 63 or selected['points_per_peak'] != 7:
        raise ValueError('Requires nine peaks, seven real points each')
    return dict(rows=[dict(index=r['index'], group=r['group'], target_nm=r['target_nm'],
                           measured_nm=r['measured_nm'], codes=[0] * 5)
                      for r in selected['rows']], points_per_peak=7, zero_codes_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--expected-version', required=True, type=lambda t: int(t, 0))
    parser.add_argument('--source-capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wait-us', type=int, default=850)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; previous evidence preserved')
    if args.expected_version not in (0x10073, 0x10075, 0x10076, 0x10077):
        parser.error('Requires explicitly selected independent-DAC Beta firmware')
    if not 50 <= args.wait_us <= 15000: parser.error('wait-us outside protocol limits')
    plan = zero_plan(json.loads(args.source_capture.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='independent_dac_polling_benchmark_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  device_serial=args.device_serial, port=args.port,
                  expected_version=args.expected_version, wait_us=args.wait_us,
                  zero_codes_only=True, optical_accuracy_verified=False,
                  complete=False, results={}, thermal_snapshots=[])
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()  # Fail output preflight before any device command.
    previous_machine = app.get_runtime_machine_id()
    try:
        with UsbOtaClient(args.port, args.device_serial) as client: before = client.query_info()
        report['runtime_before'] = asdict(before)
        if before.firmware_version != args.expected_version:
            raise RuntimeError('Actual runtime firmware does not match requested version')
        app.set_runtime_machine_id('machine_3')
        with TraceSerial(args.port, 2000000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            device.thermal_observer.callback = lambda value: report['thermal_snapshots'].append(dict(value))
            try:
                if (not _safe_disarm(device) or not zero_all_dacs_and_verify(device)
                        or not _close_shutter(device)):
                    raise RuntimeError('Initial all-zero / disarm / shutter not confirmed')
                for repeat in range(3):
                    key = f'single_63_{args.wait_us}_{repeat + 1:02d}'
                    path = args.output.with_suffix(f'.{key}.json')
                    device.trace.clear()
                    device.raw.clear()
                    try:
                        capture = execute_device(device, plan, path, cycles=64,
                            first_delay_us=args.wait_us, spacing_us=0, boundary_extra_us=0,
                            point_delays_us=[args.wait_us] * 63, adc_channel=0,
                            feedback_selector=2, single_sample=True, fast_dac_writes=False)
                    finally:
                        path.with_suffix('.wire.bin').write_bytes(device.raw)
                        path.with_suffix('.trace.json').write_text(json.dumps(device.trace), encoding='utf-8')
                    report['results'][key] = validate_capture(capture, args.expected_version, 64, 63)
                    save()
                    print(json.dumps(dict(check=key, **report['results'][key])), flush=True)
            finally:
                try: report['final_disarm_ack'] = bool(_safe_disarm(device))
                finally: report['final_shutter_ack'] = bool(_close_shutter(device))
        with UsbOtaClient(args.port, args.device_serial) as client: after = client.query_info()
        report['runtime_after'] = asdict(after)
        valid = [t for t in report['thermal_snapshots'] if t['thermal_flags'] & 1]
        faults = [t for t in report['thermal_snapshots'] if not t['thermal_flags'] & 1
                  or t['thermal_flags'] & (4 | 8 | 32)]
        report['thermal_summary'] = dict(valid_snapshots=len(valid), fault_snapshots=len(faults),
            sensor_age_max_ms=max((t['sensor_age_ms'] for t in valid), default=None),
            temperature_min_c=min((t['temperature_c'] for t in valid), default=None),
            temperature_max_c=max((t['temperature_c'] for t in valid), default=None))
        if (before != after or not report['final_disarm_ack'] or not report['final_shutter_ack']
                or not valid or faults or any(not 0 < t['temperature_c'] < 85 for t in valid)):
            raise RuntimeError('Identity changed or shutdown/thermal acceptance failed')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        app.set_runtime_machine_id(previous_machine)
        save()
        print(json.dumps({k: report.get(k) for k in ('complete', 'error', 'thermal_summary',
                         'final_disarm_ack', 'final_shutter_ack')}), flush=True)


if __name__ == '__main__': main()
