"""Qualify a reduced route using exact active DAC codes, never 2001-table substitutions."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import serial
import app_JDSU as app
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import acquire_settled_ch1, save_checkpoint, sha256_file, now_text
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from laser_power_calibration import AQ6150B


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Output must be new')
    rows,hidden=load_installed_rows()
    selected=[r for i,r in enumerate(rows) if i%5 in (1,2,3)]
    points=[app.FullbandAccuracyPoint(r['order'],r['target_wavelength_nm'],
        r['measured_wavelength_nm'],tuple(r['codes'])) for r in selected]
    root=Path(__file__).resolve().parent
    report=dict(schema='active_reduced_route_settled_probe_v1',created=now_text(),
        route_rows=selected,active_table_sha256=sha256_file(root/'mode_tables_from_fullband_2001.json'),
        hidden_predecessor_codes=list(hidden),guard_before_local=1,guard_hold_s=.005,
        feedback_selectors=[0,2],samples=[],guard_events=[],complete=False,
        cnc_commands=[],production_table_modified=False,fast_timing_verified=False,
        physical_15hz_verified=False,disarm_ack=False,shutter_ack=False)
    meter=None
    save_checkpoint(a.output,report)
    with serial.Serial('COM6',2_000_000,timeout=.03,write_timeout=3) as device:
        try:
            device.dtr=True
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError('Initial exact shutdown missing')
            worker=app.EqualIntervalWorker(device,points,{},settle_s=.3,feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack(): raise RuntimeError('Gain ACK missing')
            meter=AQ6150B('GPIB0::7::INSTR'); report['meter_identity']=meter.identity
            started=time.monotonic()
            for repeat in (1,2):
                if worker._command_and_ack(points[-1].codes) is None:
                    raise RuntimeError('Wrap predecessor ACK missing')
                if not worker._interruptible_wait(.3): raise RuntimeError('Interrupted')
                for local,point in enumerate(points):
                    if time.monotonic()-started>240: raise TimeoutError('Four-minute cap')
                    if point.index==2:
                        begin=time.monotonic_ns()
                        if worker._command_and_ack(hidden) is None: raise RuntimeError('Hidden ACK missing')
                        if not worker._interruptible_wait(.005): raise RuntimeError('Interrupted')
                        report['guard_events'].append(dict(repeat=repeat,before_local=local,
                            started_ns=begin,completed_ns=time.monotonic_ns()))
                    sample=acquire_settled_ch1(worker,point,.3,retain_monitor_samples=True)
                    reading=meter.measure(point.target_nm,select_main_peak=True)
                    error=(reading.wavelength_nm-point.target_nm)*1000
                    passed=bool(sample['stable'] and sample['ch1_adc_code']<4000
                        and math.isfinite(error) and abs(error)<=50
                        and math.isfinite(reading.power_mw) and reading.power_mw>0
                        and reading.side_mode_suppression_db is not None
                        and math.isfinite(reading.side_mode_suppression_db)
                        and reading.side_mode_suppression_db>=20)
                    report['samples'].append(dict(repeat=repeat,local=local,point=sample,
                        meter=asdict(reading),error_pm=error,passed=passed))
                    save_checkpoint(a.output,report)
                    print(json.dumps(dict(repeat=repeat,local=local,error_pm=error,
                        power_mw=reading.power_mw,passed=passed)),flush=True)
                    if not passed: raise RuntimeError('Settled optical gate failed; no fast route authorized')
            report['complete']=len(report['samples'])==54
        except BaseException as exc:
            report['error']=repr(exc)
            raise
        finally:
            errors=[]
            try: report['disarm_ack']=_safe_disarm(device)
            except Exception as exc: errors.append(repr(exc))
            try: report['shutter_ack']=_close_shutter(device)
            except Exception as exc: errors.append(repr(exc))
            report['cleanup_errors']=errors
            save_checkpoint(a.output,report)
            if meter is not None: meter.close()
            print('shutdown',report['disarm_ack'],report['shutter_ack'],flush=True)
    if not report['disarm_ack'] or not report['shutter_ack']:
        raise RuntimeError('Shutdown not confirmed')


if __name__=='__main__': main()
