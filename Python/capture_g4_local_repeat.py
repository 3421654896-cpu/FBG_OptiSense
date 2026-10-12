"""Bounded USB local static spectrum repeat, not a fast-path or wavelength certificate."""
import argparse
import json
from pathlib import Path
import time
import serial
import app_JDSU as app
from capture_ch1_teacher_spectrum import acquire_settled_ch1, save_checkpoint, sha256_file
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from laser_dac_safety import validate_pi11210_codes
from probe_ch1_g4_candidates import checked_command


def execute(output):
    if output.exists(): raise FileExistsError(output)
    points=app.load_fullband_accuracy_table()
    # 1539.70..1540.50 nm table targets; actual wavelengths not measured here.
    indices=list(range(735,776))
    for i in [734]+indices: validate_pi11210_codes(points[i].codes)
    report=dict(schema='g4_local_static_repeat_v1',repeats=2,indices=indices,
        table_sha256=sha256_file(Path('fullband_equal_power_operational_2001.csv')),
        ch1_selector=2,settle_s=.3,cnc_motion_commands=0,rows=[],complete=False,
        measured_wavelengths_are_historical=True,fast_path_qualified=False)
    with serial.Serial('COM6',2_000_000,timeout=.02,write_timeout=3) as device:
        try:
            device.dtr=True
            if not _safe_disarm(device) or not _close_shutter(device): raise RuntimeError('Initial shutdown ACK missing')
            worker=app.EqualIntervalWorker(device,(),{},settle_s=.3,feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack(): raise RuntimeError('Gain ACK missing')
            deadline=time.monotonic()+180
            for repeat in range(2):
                checked_command(worker,points[734].codes)
                if not worker._interruptible_wait(.3): raise RuntimeError('Interrupted')
                for index in indices:
                    if time.monotonic()>deadline: raise TimeoutError('Bounded scan expired')
                    row=acquire_settled_ch1(worker,points[index],.3,retain_monitor_samples=True)
                    report['rows'].append(dict(repeat=repeat,index=index,adc=row))
                    save_checkpoint(output,report)
                    print(repeat,index,row['ch1_adc_code'],flush=True)
            report['complete']=True
        except BaseException as exc:
            report['error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            errors=[]; disarm=shutter=False
            try: disarm=_safe_disarm(device)
            except Exception as exc: errors.append(str(exc))
            try: shutter=_close_shutter(device)
            except Exception as exc: errors.append(str(exc))
            report['safety_cleanup']=dict(exact_disarm_ack=disarm,exact_soa_shutter_ack=shutter,errors=errors)
            save_checkpoint(output,report)
            print(json.dumps(report['safety_cleanup']),flush=True)
            if not disarm or not shutter: raise RuntimeError('Final optical shutdown unconfirmed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    execute(p.parse_args().output)
