"""Two exact-DAC passes through segment 4 with settled CH1 and AQ6150."""
import argparse
from dataclasses import asdict
import json
import time
import serial
from pathlib import Path
import app_JDSU as app
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import acquire_settled_ch1
from laser_power_calibration import AQ6150B
from benchmark_stress_realtime import _safe_disarm, _close_shutter

def read_held_monitor(worker, duration_s=.4):
    """Fresh monitor frames without rewriting DAC; timestamps are host receipt."""
    worker.port.reset_input_buffer()
    worker._rx_buffer.clear()
    start = time.monotonic()
    frames = []
    while time.monotonic() - start < duration_s:
        for frame in worker._read_available_frames():
            values = app.decode_single_value_monitor_frame(frame)
            if values is not None:
                frames.append(dict(host_monotonic_s=time.monotonic(), codes=list(values)))
    if not frames:
        raise RuntimeError('No fresh held-DAC monitor frames')
    return frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--hold-trace', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new')
    rows, _ = load_installed_rows()
    points = [app.FullbandAccuracyPoint(i, r['target_wavelength_nm'],
              r['measured_wavelength_nm'], tuple(r['codes'])) for i, r in enumerate(rows)]
    result = dict(schema='segment4_meter_repeat_v1', complete=False, samples=[], cnc_commands=[])
    meter = AQ6150B('GPIB0::7::INSTR')
    result['meter_identity'] = meter.identity
    try:
        with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=3) as device:
            try:
                device.dtr = True
                if not _safe_disarm(device) or not _close_shutter(device):
                    raise RuntimeError('Initial optical shutdown unconfirmed')
                worker = app.EqualIntervalWorker(device, points, {}, settle_s=.3, feedback_selectors=(0,2))
                if not worker._feedback_command_and_ack():
                    raise RuntimeError('Feedback ACK missing')
                for repeat in (1,2):
                    worker._command_and_ack(points[14].codes)
                    worker._interruptible_wait(.3)
                    for index in range(15,20):
                        teacher = acquire_settled_ch1(worker, points[index], .3, retain_monitor_samples=args.hold_trace)
                        before = read_held_monitor(worker) if args.hold_trace else []
                        reading = meter.measure(rows[index]['measured_wavelength_nm'], select_main_peak=True)
                        after = read_held_monitor(worker) if args.hold_trace else []
                        result['samples'].append(dict(repeat=repeat, row=index, teacher=teacher, meter=asdict(reading), held_before_meter=before, held_after_meter=after))
                        print(repeat,index,round(reading.wavelength_nm,6),round(reading.power_mw,5),teacher['ch1_adc_code'],flush=True)
                result['complete'] = True
            finally:
                result['disarm_ack'] = _safe_disarm(device)
                result['shutter_ack'] = _close_shutter(device)
    finally:
        meter.close()
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result,stream,indent=2)
    if not result['disarm_ack'] or not result['shutter_ack']:
        raise RuntimeError('Final optical shutdown unconfirmed')
