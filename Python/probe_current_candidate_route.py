"""Probe a new forward-only 27/45-point route at settled holds with AQ6150.

No CNC, table installation, firmware change or fast-timing qualification.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_teacher_spectrum import acquire_settled_ch1, save_checkpoint, sha256_file, now_text
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import AQ6150B

HERE = Path(__file__).resolve().parent


def load_route(source, per_group):
    if per_group not in (3,5):
        raise ValueError('Only 3 or 5 points per group supported')
    proposal = json.loads(source.read_text('utf-8'))
    if proposal['candidate_dac_source_sha256'] != sha256_file(HERE/'fullband_equal_power_operational_2001.csv'):
        raise ValueError('Source DAC table changed')
    rows = proposal['candidate_rows']
    if len(rows) != 45 or [r['group'] for r in rows] != [g for g in range(1,10) for _ in range(5)]:
        raise ValueError('Candidate must contain five ordered points for each of nine groups')
    full = app.load_fullband_accuracy_table(HERE)
    for r in rows:
        if not r['short_window_stable']:
            raise ValueError('Unstable candidate cannot be probed as selected stable route')
        codes = validate_pi11210_codes(r['codes'])
        p = full[r['index']]
        if codes != p.codes or abs(r['target_nm']-p.target_nm) > 1e-8:
            raise ValueError('Candidate identity disagrees with bounded source table')
    selected = rows if per_group == 5 else [r for i,r in enumerate(rows) if i%5 in (1,2,3)]
    if any(b['index'] <= a['index'] for a,b in zip(selected, selected[1:])):
        raise ValueError('Only strictly forward wavelength order is supported')
    return selected, [full[r['index']] for r in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--points-per-group', type=int, choices=(3,5), default=3)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    rows, points = load_route(args.candidate, args.points_per_group)
    print('route',len(points),[p.index for p in points],flush=True)
    if args.plan_only:
        return
    report = dict(schema='candidate_settled_route_probe_v1', complete=False,
        created=now_text(), candidate=str(args.candidate.resolve()),
        candidate_sha256=sha256_file(args.candidate), route_rows=rows,
        protocol='previous selected -> target; includes last-to-first wrap; 0.3 s minimum hold',
        fast_timing_verified=False, cnc_commands=[], samples=[],
        feedback_selectors=[0,2], disarm_ack=False, shutter_ack=False)
    meter = None
    start = time.monotonic()
    try:
        meter = AQ6150B('GPIB0::7::INSTR')
        report['meter_identity'] = meter.identity
        with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=3) as device:
            try:
                device.dtr = True
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial shutdown not confirmed')
                worker = app.EqualIntervalWorker(device, points, {}, settle_s=.3, feedback_selectors=(0,2))
                if not worker._feedback_command_and_ack():
                    raise RuntimeError('Feedback ACK missing')
                for repeat in (1,2):
                    if worker._command_and_ack(points[-1].codes) is None:
                        raise RuntimeError('Wrap predecessor not confirmed')
                    worker._interruptible_wait(.3)
                    for order, point in enumerate(points):
                        if time.monotonic()-start > 240:
                            raise TimeoutError('Four-minute probe cap exceeded')
                        sample = acquire_settled_ch1(worker, point, .3, retain_monitor_samples=True)
                        reading = meter.measure(point.target_nm, select_main_peak=True)
                        report['samples'].append(dict(repeat=repeat, order=order,
                            group=rows[order]['group'], point=sample, meter=asdict(reading)))
                        if abs(reading.wavelength_nm-point.target_nm) > .05:
                            raise RuntimeError('Unexpected wavelength branch in sparse route')
                        if reading.side_mode_suppression_db is not None and reading.side_mode_suppression_db < 20:
                            raise RuntimeError('Strong secondary peak in sparse route')
                        if sample['ch1_adc_code'] >= 4000:
                            raise RuntimeError('CH1 approaches saturation')
                        if order % 3 == 2:
                            save_checkpoint(args.output, report)
                            print('repeat',repeat,'group',rows[order]['group'],
                                  'nm',round(reading.wavelength_nm,6),'mW',round(reading.power_mw,5),flush=True)
                report['complete'] = True
            finally:
                errors = []
                try:
                    report['disarm_ack'] = _safe_disarm(device)
                except Exception as exc:
                    errors.append('disarm: '+repr(exc))
                try:
                    report['shutter_ack'] = _close_shutter(device)
                except Exception as exc:
                    errors.append('shutter: '+repr(exc))
                report['cleanup_errors'] = errors
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['finished'] = now_text()
        save_checkpoint(args.output, report)
        if meter is not None:
            meter.close()
    if not report['disarm_ack'] or not report['shutter_ack']:
        raise RuntimeError('Final shutdown unconfirmed')
    print('probe complete; shutdown confirmed',flush=True)


if __name__ == '__main__':
    main()
