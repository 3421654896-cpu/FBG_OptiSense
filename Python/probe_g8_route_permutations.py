"""Optical-only G8 order screen; no CNC, firmware or table installation.

Each trial rebuilds the same measured G7 state, then visits the same three G8
physical rows in one of six orders. AQ6150 measures the strongest line after a
settled CH1 window. This can reject route orders, not qualify fast timing.
"""
import argparse
from dataclasses import asdict
from itertools import permutations
import json
import math
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_teacher_spectrum import acquire_settled_ch1, save_checkpoint, sha256_file
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import AQ6150B

HERE = Path(__file__).resolve().parent
SEED = (1319, 1338, 1357)
TARGETS = (1523, 1541, 1559)


def make_cases(points):
    rows = {i: points[i] for i in (*SEED, *TARGETS)}
    for i, point in rows.items():
        if point.index != i:
            raise ValueError('Fullband index identity mismatch')
        validate_pi11210_codes(point.codes)
    return rows, [tuple(order) for order in permutations(TARGETS)]


def summarize(report):
    trials = {}
    for row in report['captures']:
        key = tuple(row['target_order'])
        item = trials.setdefault(key, dict(order=list(key), readings=0, wavelength_passes=0,
                                               stable_adc=0, smsr_passes=0, all_pass=True))
        item['readings'] += 1
        passed_wave = abs(row['wavelength_error_pm']) <= 50
        passed_smsr = row['meter']['side_mode_suppression_db'] is not None and row['meter']['side_mode_suppression_db'] >= 20
        item['wavelength_passes'] += int(passed_wave)
        item['stable_adc'] += int(row['adc']['stable'])
        item['smsr_passes'] += int(passed_smsr)
        item['all_pass'] &= bool(passed_wave and passed_smsr and row['adc']['stable'])
    return list(trials.values())


def execute(output, repeats=2, port='COM6', gpib='GPIB0::7::INSTR'):
    if output.exists(): raise FileExistsError(output)
    if type(repeats) is not int or not 1 <= repeats <= 2: raise ValueError('Repeats must be 1 or 2')
    points = app.load_fullband_accuracy_table(HERE)
    rows, cases = make_cases(points)
    report = dict(schema='g8_route_permutation_probe_v1', complete=False, cnc_commands=[],
        firmware_or_table_modified=False, fast_timing_qualified=False, physical_15hz_verified=False,
        repeats=repeats, seed_indices=list(SEED), target_indices=list(TARGETS), captures=[],
        source_table_sha256=sha256_file(HERE/'fullband_equal_power_operational_2001.csv'),
        settings=dict(ch0_selector=0, ch1_selector=2, digital_gain=1,
                      seed_hold_s=.05, target_settle_s=.3, wavelength_gate_pm=50, smsr_gate_db=20),
        disarm_ack=False, shutter_ack=False)
    save_checkpoint(output, report)
    meter = None
    with serial.Serial(port, 2_000_000, timeout=.03, write_timeout=3) as device:
        try:
            device.dtr = True
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError('Initial exact shutdown ACK missing')
            worker = app.EqualIntervalWorker(device, (), {}, settle_s=.3, feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack(): raise RuntimeError('Feedback ACK missing')
            meter = AQ6150B(gpib); report['meter_identity'] = meter.identity
            for repeat in range(repeats):
                trial_cases = cases if repeat == 0 else list(reversed(cases))
                for case_index, order in enumerate(trial_cases):
                    for index in SEED:
                        if worker._command_and_ack(rows[index].codes) is None:
                            raise RuntimeError('Seed DAC ACK missing')
                        if not worker._interruptible_wait(.05): raise RuntimeError('Interrupted')
                    for position, index in enumerate(order):
                        adc = acquire_settled_ch1(worker, rows[index], .3, retain_monitor_samples=True)
                        reading = meter.measure(rows[index].target_nm, select_main_peak=True)
                        error_pm = (reading.wavelength_nm-rows[index].target_nm)*1000
                        if (not math.isfinite(error_pm) or not math.isfinite(reading.power_mw)
                                or reading.power_mw <= 0): raise RuntimeError('Invalid meter result')
                        report['captures'].append(dict(repeat=repeat+1, case_index=case_index,
                            target_order=list(order), position=position, target_index=index,
                            target_nm=rows[index].target_nm, adc=adc, meter=asdict(reading),
                            wavelength_error_pm=error_pm))
                        save_checkpoint(output, report)
                    print('repeat',repeat+1,'order',order,
                          'errors_pm',[round(r['wavelength_error_pm'],1) for r in report['captures'][-3:]],flush=True)
            report['summary'] = summarize(report)
            report['complete'] = len(report['captures']) == repeats*len(cases)*len(TARGETS)
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'; raise
        finally:
            errors=[]
            try: report['disarm_ack'] = _safe_disarm(device)
            except Exception as exc: errors.append('disarm: '+repr(exc))
            try: report['shutter_ack'] = _close_shutter(device)
            except Exception as exc: errors.append('shutter: '+repr(exc))
            report['cleanup_errors']=errors
            if meter is not None: meter.close()
            save_checkpoint(output,report)
    if not report['disarm_ack'] or not report['shutter_ack']:
        raise RuntimeError('Final exact shutdown ACK missing')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args(); result=execute(args.output,args.repeats)
    print(json.dumps(result['summary'],indent=2))
