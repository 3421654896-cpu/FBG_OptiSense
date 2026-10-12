"""Identity-locked zero-current timing comparison; no flash or optical output."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_usb_ota import UsbOtaClient
from guarded_application_upgrade import zero_all_dacs_and_verify
from temporary_test_mode import execute_device
from thermal_serial import ThermalSerial


def summarize(report):
    frames = [f for f in report['frames'] if f.get('kind') == 'data'
              and f['sequence'] >= 16]
    if len(frames) < 16:
        raise RuntimeError('Insufficient steady-state frames')
    budget = (sum(report['point_delays_us']) if report['point_delays_us'] is not None
              else report['point_count'] * report['first_delay_us']
              + report['peak_count'] * report['boundary_extra_us'])
    budget += report['point_count'] * report['spacing_us']
    measurements = []
    for current, following in zip(frames, frames[1:]):
        records = current['records']
        point_start = records[0]['write_end_us'] + sum(
            record['write_end_us'] - previous['second_end_us']
            for previous, record in zip(records, records[1:]))
        held = sum(r['second_end_us'] - r['write_end_us'] for r in records)
        period = following['cycle_start_us'] - current['cycle_start_us']
        measurements.append(dict(period_ms=period / 1000,
            dac_and_point_start_ms=point_start / 1000,
            prescribed_wait_ms=budget / 1000,
            adc_and_hold_overhead_ms=(held - budget) / 1000,
            frame_gap_ms=(period - current['elapsed_us']) / 1000))
    result = {key: statistics.mean(row[key] for row in measurements)
              for key in measurements[0]}
    result.update(mean_hz=1000 / result['period_ms'],
                  p99_period_ms=report['p99_period_ms'],
                  received_frames=report['received_frames'],
                  disarm_ack=report['disarm_ack'], shutter_ack=report['shutter_ack'],
                  i2c_errors=max(f['i2c_errors'] for f in frames),
                  spi_errors=max(f['spi_errors'] for f in frames))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--expected-version', type=lambda s: int(s, 0), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserving prior measurements')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  zero_codes_only=True, optical_accuracy_verified=False,
                  complete=False, final_disarm_ack=False, final_shutter_ack=False,
                  results={})
    try:
        with UsbOtaClient(args.port, args.device_serial) as client:
            before = client.query_info()
        report['runtime_before'] = asdict(before)
        if before.firmware_version != args.expected_version:
            raise RuntimeError('Running firmware differs from requested test version')
        with ThermalSerial(args.port, 2_000_000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            try:
                if (not _safe_disarm(device) or not zero_all_dacs_and_verify(device)
                        or not _close_shutter(device)):
                    raise RuntimeError('Initial zero-current/shutter confirmation missing')
                rows = [dict(index=i, target_nm=1525+i*.02, measured_nm=1525+i*.02,
                             codes=[0]*5) for i in range(45)]
                for name, waits in (
                    ('legacy_additive_850_850', dict(boundary_extra_us=850)),
                    ('absolute_850_all_points', dict(boundary_extra_us=0,
                                                     point_delays_us=[850]*45)),
                ):
                    capture_path = args.output.with_suffix(f'.{name}.json')
                    if capture_path.exists():
                        raise FileExistsError(capture_path)
                    capture = execute_device(device, dict(rows=rows), capture_path,
                        cycles=128, first_delay_us=850, spacing_us=50,
                        feedback_selector=2, adc_channel=1, **waits)
                    if (not capture['complete'] or capture['received_frames'] != 128
                            or not capture['disarm_ack'] or not capture['shutter_ack']
                            or any(f.get('firmware') != args.expected_version
                                   for f in capture['frames'] if f.get('kind') == 'data')):
                        raise RuntimeError('Capture completion/version/cleanup validation failed')
                    summary = summarize(capture)
                    if summary['i2c_errors'] or summary['spi_errors']:
                        raise RuntimeError('Bus error during zero-current timing capture')
                    report['results'][name] = summary
                    print(json.dumps(dict(profile=name, **summary)), flush=True)
            finally:
                report['final_disarm_ack'] = _safe_disarm(device)
                report['final_shutter_ack'] = _close_shutter(device)
        with UsbOtaClient(args.port, args.device_serial) as client:
            after = client.query_info()
        report['runtime_after'] = asdict(after)
        if after != before or not report['final_disarm_ack'] or not report['final_shutter_ack']:
            raise RuntimeError('Runtime changed or final stop/shutter unconfirmed')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(complete=report['complete'],
                              final_disarm_ack=report['final_disarm_ack'],
                              final_shutter_ack=report['final_shutter_ack'])), flush=True)


if __name__ == '__main__':
    main()
