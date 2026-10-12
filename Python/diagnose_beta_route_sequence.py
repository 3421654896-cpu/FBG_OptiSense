"""Trace identity-locked zero-current long/legacy route transitions.

No flash, SWD, positive current, retry, or relaxed capture validation.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from serial.tools import list_ports

from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_usb_ota import UsbOtaClient
from guarded_application_upgrade import zero_all_dacs_and_verify
from temporary_test_mode import execute_device, validate_plan_payload
from beta_wire_trace import TraceSerial
from validate_beta_single_adc_live import validate_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.repetitions <= 12:
        raise ValueError('Repetitions must be 1..12')
    selected = validate_plan_payload(json.loads(args.source.read_text(encoding='utf-8'))['plan'])
    if selected['point_count'] != 63 or selected['points_per_peak'] != 7:
        raise ValueError('Needs a validated 63-point source plan')
    rows = [dict(row, codes=[0]*5) for row in selected['rows']]
    legacy = [rows[start+i] for start in range(0, 63, 7) for i in (0, 1, 3, 5, 6)]
    report = dict(zero_codes_only=True, complete=False, phases=[],
                  final_disarm_ack=False, final_shutter_ack=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    def run(device, name, route, group, single, wait):
        item = dict(name=name, complete=False)
        report['phases'].append(item)
        device.trace.clear()
        device.raw.clear()
        capture_path = args.output.with_suffix(f'.{name}.json')
        try:
            capture = execute_device(device, dict(rows=route, points_per_peak=group),
                capture_path, cycles=32, first_delay_us=850, spacing_us=50,
                boundary_extra_us=0, point_delays_us=[wait]*len(route),
                adc_channel=0, feedback_selector=2, single_sample=single,
                fast_dac_writes=single)
            item['budget'] = validate_capture(capture, 0x10073, 32, len(route))
            item['complete'] = True
        except BaseException as exc:
            item['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            item['trace'] = list(device.trace)
            item['raw_bytes'] = len(device.raw)
            item['raw_file'] = str(capture_path.with_suffix('.wire.bin'))
            capture_path.with_suffix('.wire.bin').write_bytes(device.raw)
            save()
            print(json.dumps(item | {'trace': f'{len(device.trace)} events'}), flush=True)

    try:
        ports = [p for p in list_ports.comports() if p.device == 'COM18'
                 and p.vid == 0x0483 and p.pid == 0x5740 and p.serial_number == '207933893232']
        if len(ports) != 1:
            raise RuntimeError('Machine 3 native USB identity is not confirmed')
        with UsbOtaClient('COM18', '207933893232') as client:
            before = client.query_info()
        report['runtime_before'] = asdict(before)
        if before.firmware_version != 0x10073:
            raise RuntimeError('Requires actual v1.0.115 Beta')
        with TraceSerial('COM18', 2_000_000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            try:
                if (not _safe_disarm(device) or not zero_all_dacs_and_verify(device)
                        or not _close_shutter(device)):
                    raise RuntimeError('Initial all-zero shutdown not confirmed')
                run(device, 'long63', rows, 7, True, 15000)
                for repetition in range(args.repetitions):
                    run(device, f'legacy45_{repetition}', legacy, 5, False, 850)
            finally:
                try:
                    report['final_disarm_ack'] = bool(_safe_disarm(device))
                finally:
                    report['final_shutter_ack'] = bool(_close_shutter(device))
        with UsbOtaClient('COM18', '207933893232') as client:
            after = client.query_info()
        report['runtime_after'] = asdict(after)
        if before != after:
            raise RuntimeError('Board runtime changed during test')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
        print(json.dumps({k: report.get(k) for k in
            ('complete', 'error', 'final_disarm_ack', 'final_shutter_ack')}), flush=True)


if __name__ == '__main__':
    main()
