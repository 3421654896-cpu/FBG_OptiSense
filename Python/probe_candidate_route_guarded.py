"""Qualify a 27-row candidate with one explicit hidden G7->G8 guard.

This is a host-paced, settled optical qualification only. It never controls CNC,
installs a table, or claims compatibility with an existing fast firmware route.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm,_close_shutter
from capture_ch1_teacher_spectrum import acquire_settled_ch1,save_checkpoint,sha256_file,now_text
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import AQ6150B
from probe_current_candidate_route import load_route

HERE=Path(__file__).resolve().parent


def execute(candidate,output,guard_index=1400,guard_hold_s=.005):
    if output.exists(): raise FileExistsError(output)
    if type(guard_index) is not int or not 0<=guard_index<=2000: raise ValueError('Physical guard index required')
    if not math.isfinite(guard_hold_s) or not .0005<=guard_hold_s<=.02: raise ValueError('Guard hold must be 0.5..20 ms')
    rows,points=load_route(candidate,3);full=app.load_fullband_accuracy_table(HERE)
    guard=full[guard_index]
    if guard.index!=guard_index or guard_index in {r['index'] for r in rows}: raise ValueError('Guard must be a separate table row')
    validate_pi11210_codes(guard.codes)
    if [r['group'] for r in rows] != [g for g in range(1,10) for _ in range(3)]: raise ValueError('Nine ordered triples required')
    report=dict(schema='candidate_settled_guarded_route_probe_v1',complete=False,created=now_text(),
        candidate=str(candidate.resolve()),candidate_sha256=sha256_file(candidate),route_rows=rows,
        protocol='last selected -> first; one hidden guard immediately before G8; 0.3 s target hold',
        guard=dict(index=guard_index,target_nm=guard.target_nm,measured_nm=guard.measured_nm,
                   codes=list(guard.codes),hold_s=guard_hold_s,before_order=21),
        feedback_selectors=[0,2],samples=[],guard_events=[],cnc_commands=[],
        firmware_or_table_modified=False,fast_timing_verified=False,physical_15hz_verified=False,
        disarm_ack=False,shutter_ack=False)
    save_checkpoint(output,report);meter=None;start=time.monotonic()
    with serial.Serial('COM6',2_000_000,timeout=.03,write_timeout=3) as device:
        try:
            device.dtr=True
            if not _safe_disarm(device) or not _close_shutter(device): raise RuntimeError('Initial exact shutdown missing')
            worker=app.EqualIntervalWorker(device,points,{},settle_s=.3,feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack(): raise RuntimeError('Feedback ACK missing')
            meter=AQ6150B('GPIB0::7::INSTR');report['meter_identity']=meter.identity
            for repeat in (1,2):
                if worker._command_and_ack(points[-1].codes) is None: raise RuntimeError('Wrap predecessor ACK missing')
                if not worker._interruptible_wait(.3): raise RuntimeError('Interrupted')
                for order,point in enumerate(points):
                    if time.monotonic()-start>240: raise TimeoutError('Four-minute cap exceeded')
                    if order==21:
                        before=time.monotonic_ns()
                        if worker._command_and_ack(guard.codes) is None: raise RuntimeError('Hidden guard ACK missing')
                        if not worker._interruptible_wait(guard_hold_s): raise RuntimeError('Interrupted')
                        report['guard_events'].append(dict(repeat=repeat,before_order=order,
                            guard_index=guard_index,started_ns=before,completed_ns=time.monotonic_ns()))
                    sample=acquire_settled_ch1(worker,point,.3,retain_monitor_samples=True)
                    reading=meter.measure(point.target_nm,select_main_peak=True)
                    error=(reading.wavelength_nm-point.target_nm)*1000
                    if (not sample['stable'] or sample['ch1_adc_code']>=4000 or not math.isfinite(error)
                            or abs(error)>50 or not math.isfinite(reading.power_mw) or reading.power_mw<=0
                            or reading.side_mode_suppression_db is None or reading.side_mode_suppression_db<20):
                        raise RuntimeError(f'Optical gate failed repeat={repeat} order={order} index={point.index} error_pm={error}')
                    report['samples'].append(dict(repeat=repeat,order=order,group=rows[order]['group'],
                                                  point=sample,meter=asdict(reading)))
                    if order%3==2:
                        save_checkpoint(output,report)
                        print('repeat',repeat,'group',rows[order]['group'],'nm',round(reading.wavelength_nm,6),
                              'ADC',sample['ch1_adc_code'],flush=True)
            report['complete']=len(report['samples'])==54 and len(report['guard_events'])==2
        except BaseException as exc: report['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            errors=[]
            try: report['disarm_ack']=_safe_disarm(device)
            except Exception as exc: errors.append('disarm: '+repr(exc))
            try: report['shutter_ack']=_close_shutter(device)
            except Exception as exc: errors.append('shutter: '+repr(exc))
            report['cleanup_errors']=errors
            if meter is not None:meter.close()
            save_checkpoint(output,report)
    if not report['disarm_ack'] or not report['shutter_ack']:raise RuntimeError('Final exact shutdown missing')
    print('complete; hidden guard route optically qualified');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--guard-index',type=int,default=1400)
    p.add_argument('--guard-hold-ms',type=float,default=5);a=p.parse_args()
    execute(a.candidate,a.output,a.guard_index,a.guard_hold_ms/1000)
