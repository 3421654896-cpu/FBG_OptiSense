"""Bounded optical-only phase dither, measured wavelength and held CH1.

No CNC, firmware, or deployed table changes. All commands pass shared limits.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import acquire_settled_ch1
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import AQ6150B
from probe_current_segment4_meter import read_held_monitor


def phase_point(row, index, delta):
    if delta not in (-32, -16, 0, 16, 32):
        raise ValueError('Only bounded phase offsets allowed')
    codes = list(row['codes'])
    codes[2] += delta
    return app.FullbandAccuracyPoint(index, row['target_wavelength_nm'],
                                    row['measured_wavelength_nm'], validate_pi11210_codes(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output file')
    rows, _ = load_installed_rows()
    predecessor = phase_point(rows[15], 15, 0)
    deltas = (-32, -16, 0, 16, 32, 0)
    points = [phase_point(rows[16], 16, delta) for delta in deltas]
    result = dict(schema='local_phase_slope_v1', complete=False,
                  created=datetime.now().astimezone().isoformat(), samples=[],
                  cnc_commands=[], phase_offsets=list(deltas),
                  feedback_selectors=[0, 2], predecessor_codes=list(predecessor.codes),
                  disarm_ack=False, shutter_ack=False)
    meter = None
    try:
        meter = AQ6150B('GPIB0::7::INSTR')
        result['meter_identity'] = meter.identity
        with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=3) as device:
            try:
                device.dtr = True
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial shutdown unconfirmed')
                worker = app.EqualIntervalWorker(device, points, {}, settle_s=.3,
                                                feedback_selectors=(0, 2))
                if not worker._feedback_command_and_ack():
                    raise RuntimeError('Feedback not confirmed')
                for repeat in (1, 2):
                    sequence = list(zip(deltas, points))
                    if repeat == 2:
                        sequence.reverse()
                    for delta, point in sequence:
                        if worker._command_and_ack(predecessor.codes) is None:
                            raise RuntimeError('Predecessor DAC not confirmed')
                        worker._interruptible_wait(.3)
                        teacher = acquire_settled_ch1(worker, point, .3, retain_monitor_samples=True)
                        before = read_held_monitor(worker)
                        reading = meter.measure(point.measured_nm, select_main_peak=True)
                        after = read_held_monitor(worker)
                        result['samples'].append(dict(repeat=repeat, phase_delta=delta,
                            teacher=teacher, meter=asdict(reading),
                            held_before_meter=before, held_after_meter=after))
                        print(repeat, delta, round(reading.wavelength_nm, 6),
                              teacher['ch1_adc_code'], flush=True)
                        if abs(reading.wavelength_nm-point.measured_nm) > .025:
                            raise RuntimeError('Wavelength left bounded local region')
                        if (reading.side_mode_suppression_db is not None
                                and reading.side_mode_suppression_db < 20):
                            raise RuntimeError('Strong secondary optical peak; abort dither')
                result['complete'] = True
            finally:
                result['disarm_ack'] = _safe_disarm(device)
                result['shutter_ack'] = _close_shutter(device)
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        if meter is not None:
            meter.close()
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    if not result['disarm_ack'] or not result['shutter_ack']:
        raise RuntimeError('Final optical shutdown unconfirmed')


if __name__ == '__main__':
    main()
