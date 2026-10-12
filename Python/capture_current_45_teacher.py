"""Capture settled CH1 at the installed 45 rows; no CNC dependency."""
import argparse
import json
from pathlib import Path
import serial
from capture_ch1_rc_diagnostic import load_installed_rows, capture_teacher
from benchmark_stress_realtime import _safe_disarm, _close_shutter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    rows, hidden = load_installed_rows()
    result = dict(schema='current_settled45_teacher_v1', complete=False,
                  selector_ch1=2, dac_rows=rows, cnc_commands=[])
    try:
        with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=3) as device:
            try:
                device.dtr = True
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial optical shutdown acknowledgment missing')
                result['teacher'] = capture_teacher(device, rows, hidden, 2)
                result['complete'] = True
            finally:
                result['disarm_ack'] = _safe_disarm(device)
                result['shutter_ack'] = _close_shutter(device)
    finally:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    if not result['disarm_ack'] or not result['shutter_ack']:
        raise RuntimeError('Final shutdown acknowledgment missing')
    print('Settled teacher complete; optical shutdown acknowledged')
