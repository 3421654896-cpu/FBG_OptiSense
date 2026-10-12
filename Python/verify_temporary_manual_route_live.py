"""Verify one nonuniform temporary 45-point route on live USB hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import serial

import app_JDSU as app
from fullband_auto_calibration import best_effort_final_shutter
from temporary_test_mode import build_manual_dense_plan, execute_device, select_points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM6')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    dense = json.loads(args.reference.read_text(encoding='utf-8'))
    rows = dense.get('rows', [])
    if not dense.get('complete') or len(rows) != 2001:
        raise ValueError('需要完整的2001点密集参考')
    x = [float(row['measured_wavelength_nm']) for row in rows]
    y = [float(row['ch1_adc_code']) for row in rows]
    table = tuple(app.load_fullband_accuracy_table())
    automatic = select_points(
        table, x, y, baseline_fraction=.1, soft_baseline=True
    )
    groups = [
        [int(row['index']) for row in automatic['rows'][start:start + 5]]
        for start in range(0, 45, 5)
    ]
    group, point = 3, 1
    original = groups[group][point]
    candidate = original + 1
    if candidate >= groups[group][point + 1]:
        candidate = original - 1
    if candidate <= groups[group][point - 1] or candidate == original:
        raise ValueError('没有安全的单点非等间隔测试位置')
    groups[group][point] = candidate
    plan = build_manual_dense_plan(table, groups, x, y, source=str(args.reference))

    try:
        with serial.Serial(args.port, 2_000_000, timeout=.03, write_timeout=3) as device:
            device.dtr = True
            time.sleep(.15)
            report = execute_device(
                device, plan, args.output, cycles=32,
                first_delay_us=350, boundary_extra_us=1800,
                feedback_selector=2,
            )
    finally:
        best_effort_final_shutter(args.port)

    firmware = sorted({
        int(frame['firmware']) for frame in report.get('frames', [])
        if frame.get('kind') == 'data'
    })
    verified = bool(
        report.get('complete') and report.get('disarm_ack')
        and report.get('shutter_ack') and report.get('received_frames') == 32
        and firmware == [0x00010048]
    )
    print(json.dumps({
        'verified': verified,
        'firmware': firmware,
        'edited_group': group + 1,
        'edited_point': point + 1,
        'old_index': original,
        'new_index': candidate,
        'indices': groups[group],
        'received_frames': report.get('received_frames'),
        'disarm_ack': report.get('disarm_ack'),
        'shutter_ack': report.get('shutter_ack'),
        'output': str(args.output.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    if not verified:
        raise RuntimeError('非等间隔45点实机验证未通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
