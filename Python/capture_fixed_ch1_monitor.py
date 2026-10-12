"""Bounded fixed-DAC monitor: no CNC commands, no repeated target writes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import serial
import app_JDSU as app
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint,now_text
from benchmark_stress_realtime import _safe_disarm,_close_shutter
from laser_power_calibration import AQ6150B


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dark',action='store_true',help='Keep SOA shuttered; no light-source command')
    a=p.parse_args()
    if a.output.exists(): p.error('New output required')
    rows,_=load_installed_rows(); target=rows[17]; predecessor=rows[16]
    report=dict(schema='fixed_ch1_monitor_v1',created=now_text(),target=target,
        predecessor=predecessor,feedback_selectors=[0,2],target_write_count=0,
        monitor=[],meter=[],cnc_commands=[],complete=False,dark=a.dark,
        host_receive_timestamps_not_adc_timestamps=True,disarm_ack=False,shutter_ack=False)
    report['adc_acquisition_method']='checkRT/ADC_Write_Read_Stable'
    report['adc_samples_are_uniform_raw_stream']=False
    meter=None
    with serial.Serial('COM6',2_000_000,timeout=.03,write_timeout=3) as device:
        try:
            device.dtr=True
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError('Initial shutdown missing')
            worker=app.EqualIntervalWorker(device,(),{},settle_s=.3,feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack(): raise RuntimeError('Gain ACK missing')
            if not a.dark:
                meter=AQ6150B('GPIB0::7::INSTR');report['meter_identity']=meter.identity
            for row in (() if a.dark else (predecessor,target)):
                if worker._command_and_ack(tuple(row['codes'])) is None:
                    raise RuntimeError('DAC ACK missing')
                if row is target: report['target_write_count']+=1
                if not worker._interruptible_wait(.3): raise RuntimeError('Interrupted')
            device.reset_input_buffer();worker._rx_buffer.clear()
            started=time.monotonic()
            for block in range(12):
                deadline=time.monotonic()+.8
                while time.monotonic()<deadline:
                    for frame in worker._read_available_frames():
                        values=app.decode_single_value_monitor_frame(frame)
                        if values is not None:
                            if any(not 0<=v<=4095 for v in values): raise ValueError('ADC range')
                            report['monitor'].append(dict(host_elapsed_s=time.monotonic()-started,
                                block=block,codes=list(values)))
                    time.sleep(.001)
                begin=time.monotonic()-started
                if meter is not None:
                    reading=meter.measure(target['target_wavelength_nm'],select_main_peak=True)
                    report['meter'].append(dict(start_s=begin,end_s=time.monotonic()-started,reading=asdict(reading)))
                save_checkpoint(a.output,report)
                print('block',block,'frames',len(report['monitor']),'dark',a.dark,flush=True)
                if time.monotonic()-started>45: raise TimeoutError('Fixed-output cap')
            report['complete']=True
        except BaseException as exc:
            report['error']=repr(exc);raise
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
    if not report['disarm_ack'] or not report['shutter_ack']: raise RuntimeError('Shutdown unconfirmed')
    x=np.array([r['codes'] for r in report['monitor']])
    print('CH1 min/median/max/std',x[:,3].min(),np.median(x[:,3]),x[:,3].max(),x[:,3].std())


if __name__=='__main__': main()
