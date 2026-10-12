"""Identity-locked zero-current USB compatibility checks after OTA removal.

No flash, installed tables, persistent settings, Wi-Fi credentials or positive
DAC codes are changed. This is not optical accuracy or LAN throughput acceptance.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import app_JDSU as app
import dense_scan_transport as dense
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from fbg_usb_ota import UsbOtaClient
from thermal_serial import ThermalSerial
from temporary_test_mode import execute_device
from validate_production_trim_live import (zero_source, collect_monitors,
    manual_zero, reject_legacy, reject_legacy_arm)
from wifi_status_query import query as query_wifi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--expected-version', required=True, type=lambda t: int(t, 0))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; evidence preserved')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='usb_only_firmware_smoke_v1', zero_codes_only=True,
                  optical_accuracy_verified=False, complete=False, checks={})
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        with UsbOtaClient(args.port, args.device_serial) as client: before = client.query_info()
        report['runtime_before'] = asdict(before)
        if before.firmware_version != args.expected_version:
            raise RuntimeError('Actual runtime version differs from requested version')
        with ThermalSerial(args.port, 2000000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            thermal = []
            device.thermal_observer.callback = lambda value: thermal.append(dict(value))
            try:
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial shutdown not confirmed')
                report['checks']['single_zero'] = zero_source(device)
                report['checks']['monitors'] = collect_monitors(device)
                if not _close_shutter(device): raise RuntimeError('Monitor shutdown not confirmed')
                report['checks']['manual_zero'] = manual_zero(device)
                report['checks']['legacy_modes_rejected'] = [reject_legacy(device, m) for m in (0, 3)]
                report['checks']['legacy_arm_rejected'] = reject_legacy_arm(device)
                report['checks']['wifi_status'] = query_wifi(device)
                # Explicitly matches current startup dense wait, no saved setting writes.
                worker = app.EqualIntervalWorker(device, (), {}, settle_s=.003,
                                                feedback_selectors=(2, 0))
                if not worker._feedback_command_and_ack():
                    raise RuntimeError('Feedback selector acknowledgement missing')
                points = [app.FullbandAccuracyPoint(index=i, target_nm=1525 + i*.02,
                          measured_nm=1525 + i*.02, codes=(0, 0, 0, 0, 0)) for i in range(2001)]
                rows = list(dense.acquire(device, points, {}, settle_s=.003, selectors=(2, 0)))
                if [r['index'] for r in rows] != list(range(2001)):
                    raise RuntimeError('Dense acquisition lost or reordered points')
                report['checks']['dense_2001'] = dict(points=len(rows), wait_us=3000, zero_codes_only=True)
                if not _close_shutter(device): raise RuntimeError('Dense shutdown missing')
                plan = dict(rows=[dict(index=i, target_nm=1525+i*.02, measured_nm=1525+i*.02,
                                     codes=[0]*5) for i in range(45)])
                capture = execute_device(device, plan, args.output.with_suffix('.legacy_45.json'),
                    cycles=32, first_delay_us=850, spacing_us=50, boundary_extra_us=0,
                    point_delays_us=[850]*45, adc_channel=1, feedback_selector=2,
                    single_sample=False, fast_dac_writes=False)
                if (not capture.get('complete') or capture['received_frames'] != 32
                        or not capture['disarm_ack'] or not capture['shutter_ack']
                        or any(f['firmware'] != args.expected_version or f['i2c_errors']
                               or f['spi_errors'] for f in capture['frames'])):
                    raise RuntimeError('Legacy 45 compatibility capture failed')
                report['checks']['legacy_45'] = dict(received_frames=32, mean_hz=capture['mean_hz'],
                                                   disarm_ack=True, shutter_ack=True)
            finally:
                try: report['final_disarm_ack'] = bool(_safe_disarm(device))
                finally: report['final_shutter_ack'] = bool(_close_shutter(device))
            valid = [t for t in thermal if t['thermal_flags'] & 1]
            faults = [t for t in thermal if not t['thermal_flags'] & 1 or t['thermal_flags'] & (4 | 8 | 32)]
            report['thermal_snapshots'] = thermal
            if not valid or faults: raise RuntimeError('Thermal telemetry absent or faulty')
        with UsbOtaClient(args.port, args.device_serial) as client: after = client.query_info()
        report['runtime_after'] = asdict(after)
        if before != after or not report['final_disarm_ack'] or not report['final_shutter_ack']:
            raise RuntimeError('Runtime changed or final shutdown missing')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
        print(json.dumps({k: report.get(k) for k in ('complete', 'error', 'checks',
                           'final_disarm_ack', 'final_shutter_ack')}), flush=True)


if __name__ == '__main__': main()
