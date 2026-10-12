"""Optical-only dense CH1 audit with paired fixed 5/40 kOhm checks.

Does not deploy a table/model or control CNC. Dense x coordinates are historical
table calibration; only explicit meter fields are new AQ6150 observations.
"""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import (acquire_settled_ch1, establish_forward_state,
                                         save_checkpoint, sha256_file, now_text)
from laser_power_calibration import AQ6150B

HERE = Path(__file__).resolve().parent


def make_plan(points, rows, margin_nm=.06):
    if not math.isfinite(margin_nm) or not .06 <= margin_nm <= .16:
        raise ValueError('Dense boundary extension must be within 0.06..0.16 nm')
    segments = []
    for group in range(9):
        anchors = rows[group*5:group*5+5]
        lower = min(row['target_wavelength_nm'] for row in anchors)-margin_nm
        upper = max(row['target_wavelength_nm'] for row in anchors)+margin_nm
        selected = [p.index for p in points if lower-1e-8 <= p.target_nm <= upper+1e-8]
        if not 20 <= len(selected) <= 90:
            raise ValueError('Dense group must contain 20..90 bounded table points')
        anchor_indices = []
        for row in anchors:
            matches = [p.index for p in points if abs(p.target_nm-row['target_wavelength_nm']) < 1e-7]
            if len(matches) != 1:
                raise ValueError('Installed anchor target absent from 2001-point grid')
            anchor_indices.append(matches[0])
        segments.append(dict(group=group+1, indices=selected, anchor_indices=anchor_indices,
            installed_anchor_rows=anchors,
            anchors_differing_from_dense_codes=[i for i, row in zip(anchor_indices, anchors)
                if tuple(points[i].codes) != tuple(row['codes'])]))
    return segments


def checked_sample(worker, point, guards):
    guard = guards.get(point.index)
    if guard is not None and not worker._apply_guard(guard):
        raise RuntimeError('Optical transition guard cancelled')
    sample = acquire_settled_ch1(worker, point, .3, retain_monitor_samples=True)
    if sample['ch1_adc_code'] >= 4000:
        raise RuntimeError('CH1 approaches saturation; fixed-gain scan aborted')
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--margin-nm', type=float, default=.06,
                        help='Bounded measured boundary extension, 0.06..0.16 nm')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path')
    points = app.load_fullband_accuracy_table(HERE)
    guards = app.load_fullband_transition_guards(points, HERE)
    rows, hidden = load_installed_rows()
    plan = make_plan(points, rows, args.margin_nm)
    print('plan_dense_points', sum(len(s['indices']) for s in plan), flush=True)
    if args.plan_only:
        for s in plan:
            print(s['group'], len(s['indices']), s['indices'][0], s['indices'][-1])
        return
    result = dict(schema='current_dense9_teacher_v1', complete=False,
        created=now_text(), cnc_commands=[], plan=plan, groups=[], margin_nm=args.margin_nm,
        source_table_sha256=sha256_file(HERE/'fullband_equal_power_operational_2001.csv'),
        installed_anchor_sha256=sha256_file(HERE/'mode_tables_from_fullband_2001.json'),
        guard_sha256=sha256_file(HERE/'fullband_equal_power_operational_2001_transition_guards.json'),
        table_axis_kind='historical wavelength-meter calibration, not new live readings',
        settle_s=.3, dense_ch1_selector=1, dense_ch1_kohm=40,
        paired_selectors=[2,1], paired_kohm=[5,40], disarm_ack=False, shutter_ack=False)
    result['installed_third_point_hidden_predecessor_codes'] = list(hidden)
    meter = None
    start = time.monotonic()
    try:
        meter = AQ6150B('GPIB0::7::INSTR')
        result['meter_identity'] = meter.identity
        with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=3) as device:
            try:
                device.dtr = True
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial shutdown not confirmed')
                workers = {s: app.EqualIntervalWorker(device, points, guards,
                            settle_s=.3, feedback_selectors=(0,s)) for s in (2,1)}
                for segment in plan:
                    group = dict(group=segment['group'], paired_anchors=[], dense=[], meter_checks=[])
                    result['groups'].append(group)
                    for index, row in zip(segment['anchor_indices'], segment['installed_anchor_rows']):
                        anchor = app.FullbandAccuracyPoint(index, row['target_wavelength_nm'],
                            row['measured_wavelength_nm'], tuple(row['codes']))
                        pair = dict(index=index, samples=[], dac_source='installed separately calibrated 45-point table')
                        group['paired_anchors'].append(pair)
                        for selector in (2,1):
                            worker = workers[selector]
                            if not worker._feedback_command_and_ack():
                                raise RuntimeError('Fixed gain ACK missing')
                            worker._rx_buffer.clear()
                            if segment['group'] == 1 and index == segment['anchor_indices'][2]:
                                if worker._command_and_ack(hidden) is None:
                                    raise RuntimeError('Installed hidden predecessor not confirmed')
                                worker._interruptible_wait(.005)
                            # This is an installed recalibrated point, not the dense-table row.
                            sample = checked_sample(worker, anchor, guards)
                            pair['samples'].append(dict(selector=selector, sample=sample))
                    worker = workers[1]
                    establish_forward_state(worker, points, segment['indices'][0])
                    for count, index in enumerate(segment['indices']):
                        if time.monotonic()-start > 900:
                            raise TimeoutError('15-minute diagnostic duration cap exceeded')
                        if count % 8 == 0 and not worker._feedback_command_and_ack():
                            raise RuntimeError('Periodic fixed gain ACK missing')
                        sample = checked_sample(worker, points[index], guards)
                        group['dense'].append(sample)
                        if index == segment['anchor_indices'][2]:
                            reading = meter.measure(points[index].measured_nm, select_main_peak=True)
                            group['meter_checks'].append(dict(index=index, meter=asdict(reading)))
                            if abs(reading.wavelength_nm-points[index].measured_nm) > .1:
                                raise RuntimeError('Meter detected unexpected wavelength branch')
                            if reading.side_mode_suppression_db is not None and reading.side_mode_suppression_db < 20:
                                raise RuntimeError('Strong secondary peak at checked anchor')
                        if count % 16 == 0:
                            save_checkpoint(args.output, result)
                            print('group', segment['group'], 'dense', count+1, '/', len(segment['indices']),
                                  'CH1', sample['ch1_adc_code'], flush=True)
                    save_checkpoint(args.output, result)
                    print('group_complete', segment['group'], flush=True)
                result['complete'] = True
            finally:
                errors = []
                try:
                    result['disarm_ack'] = _safe_disarm(device)
                except Exception as exc:
                    errors.append('disarm: '+repr(exc))
                try:
                    result['shutter_ack'] = _close_shutter(device)
                except Exception as exc:
                    errors.append('shutter: '+repr(exc))
                result['cleanup_errors'] = errors
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['finished'] = now_text()
        result['duration_s'] = time.monotonic()-start
        save_checkpoint(args.output, result)
        if meter is not None:
            meter.close()
    if not result['disarm_ack'] or not result['shutter_ack']:
        raise RuntimeError('Final shutdown not confirmed')
    print('complete; shutdown confirmed', flush=True)


if __name__ == '__main__':
    main()
