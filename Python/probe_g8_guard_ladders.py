"""Optical-only screen of bounded G7->G8 guard ladders; no CNC/deployment."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_teacher_spectrum import acquire_settled_ch1, save_checkpoint, sha256_file
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import AQ6150B

HERE=Path(__file__).resolve().parent
SEED=(1319,1338,1357)
TARGET=1523
CASES=(
    ('direct',()),
    ('one_1400',(1400,)),('one_1450',(1450,)),('one_1500',(1500,)),
    ('one_1510',(1510,)),('one_1520',(1520,)),
    ('four_stage',(1400,1450,1500,1520)),
    ('nine_stage',(1370,1390,1410,1430,1450,1470,1490,1510,1520)),
)


def validate_cases(points):
    indices=set((*SEED,TARGET,*(i for _,case in CASES for i in case)))
    for i in indices:
        if points[i].index!=i: raise ValueError('Fullband index mismatch')
        validate_pi11210_codes(points[i].codes)
    if len({name for name,_ in CASES})!=len(CASES): raise ValueError('Duplicate case')
    if any(any(b<=a for a,b in zip(case,case[1:])) or (case and case[-1]>=TARGET)
           for _,case in CASES): raise ValueError('Guard ladders must approach target in order')


def execute(output,repeats=2,stage_hold_s=.02,case_names=()):
    if output.exists(): raise FileExistsError(output)
    if type(repeats) is not int or not 1<=repeats<=10: raise ValueError('Repeats must be 1..10')
    if not math.isfinite(stage_hold_s) or not .0005<=stage_hold_s<=.05: raise ValueError('Stage hold must be 0.5..50 ms')
    points=app.load_fullband_accuracy_table(HERE);validate_cases(points)
    selected_cases=CASES if not case_names else tuple(case for case in CASES if case[0] in set(case_names))
    if len(selected_cases)!=len(case_names) or len(case_names)!=len(set(case_names)) or any(name not in {n for n,_ in CASES} for name in case_names):
        raise ValueError('Every requested case name must be unique and known')
    report=dict(schema='g8_guard_ladder_probe_v1',complete=False,cnc_commands=[],
        firmware_or_table_modified=False,training_eligible=False,physical_15hz_verified=False,
        seed_indices=list(SEED),target_index=TARGET,cases=[dict(name=n,guards=list(g)) for n,g in selected_cases],
        repeats=repeats,stage_hold_s=stage_hold_s,captures=[],
        source_table_sha256=sha256_file(HERE/'fullband_equal_power_operational_2001.csv'),
        disarm_ack=False,shutter_ack=False)
    save_checkpoint(output,report);meter=None
    with serial.Serial('COM6',2_000_000,timeout=.03,write_timeout=3) as device:
        try:
            device.dtr=True
            if not _safe_disarm(device) or not _close_shutter(device): raise RuntimeError('Initial shutdown ACK missing')
            worker=app.EqualIntervalWorker(device,(),{},settle_s=.3,feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack(): raise RuntimeError('Feedback ACK missing')
            meter=AQ6150B('GPIB0::7::INSTR');report['meter_identity']=meter.identity
            for repeat in range(repeats):
                cases=selected_cases if repeat==0 else tuple(reversed(selected_cases))
                for name,guards in cases:
                    for index in SEED:
                        if worker._command_and_ack(points[index].codes) is None: raise RuntimeError('Seed ACK missing')
                        if not worker._interruptible_wait(.05): raise RuntimeError('Interrupted')
                    for index in guards:
                        if worker._command_and_ack(points[index].codes) is None: raise RuntimeError('Guard ACK missing')
                        if not worker._interruptible_wait(stage_hold_s): raise RuntimeError('Interrupted')
                    adc=acquire_settled_ch1(worker,points[TARGET],.3,retain_monitor_samples=True)
                    reading=meter.measure(points[TARGET].target_nm,select_main_peak=True)
                    error=(reading.wavelength_nm-points[TARGET].target_nm)*1000
                    if not math.isfinite(error) or not math.isfinite(reading.power_mw): raise RuntimeError('Invalid meter result')
                    row=dict(repeat=repeat+1,name=name,guard_indices=list(guards),adc=adc,
                             meter=asdict(reading),wavelength_error_pm=error,
                             passes=bool(abs(error)<=50 and adc['stable'] and
                                reading.side_mode_suppression_db is not None and reading.side_mode_suppression_db>=20))
                    report['captures'].append(row);save_checkpoint(output,report)
                    print(repeat+1,name,'guards',len(guards),'error_pm',round(error,2),'ADC',adc['ch1_adc_code'],flush=True)
            report['summary']=[dict(name=name,guard_indices=list(guards),
                passed=sum(r['passes'] for r in report['captures'] if r['name']==name),trials=repeats,
                errors_pm=[r['wavelength_error_pm'] for r in report['captures'] if r['name']==name])
                for name,guards in selected_cases]
            report['complete']=len(report['captures'])==repeats*len(selected_cases)
        except BaseException as exc: report['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            errors=[]
            try: report['disarm_ack']=_safe_disarm(device)
            except Exception as exc: errors.append('disarm: '+repr(exc))
            try: report['shutter_ack']=_close_shutter(device)
            except Exception as exc: errors.append('shutter: '+repr(exc))
            report['cleanup_errors']=errors
            if meter is not None: meter.close()
            save_checkpoint(output,report)
    if not report['disarm_ack'] or not report['shutter_ack']: raise RuntimeError('Final shutdown ACK missing')
    print(json.dumps(report['summary'],indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--stage-hold-ms',type=float,default=20)
    p.add_argument('--case',action='append',default=[])
    a=p.parse_args();execute(a.output,a.repeats,a.stage_hold_ms/1000,a.case)
