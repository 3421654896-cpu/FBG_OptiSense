"""Identity-locked Beta ADC/DAC timing acceptance, all physical DAC codes zero.

No flash, SWD, positive laser current, table/default/profile writes. This does
not validate optical settling or absolute wavelength/power accuracy.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from serial.tools import list_ports

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from capture_temporary_half_height_transients import capture_custom
from fbg_usb_ota import UsbOtaClient
from guarded_application_upgrade import zero_all_dacs_and_verify
from temporary_test_mode import execute_device, validate_plan_payload, points_per_peak_for_rows
from thermal_serial import ThermalSerial


def point_budget(capture):
    frames = [f for f in capture['frames'] if f['sequence'] >= 16]
    if len(frames) < 16:
        raise ValueError('Need at least 16 measured frames')
    prescribed = sum(capture['point_delays_us']) + capture['point_count'] * capture['spacing_us']
    samples = []
    for frame, following in zip(frames, frames[1:]):
        records = frame['records']
        end_key = 'sample_end_us' if capture['single_sample'] else 'second_end_us'
        point_work = records[0]['write_end_us'] + sum(
            cur['write_end_us'] - prev[end_key] for prev, cur in zip(records, records[1:]))
        held = sum(r[end_key] - r['write_end_us'] for r in records)
        period = following['cycle_start_us'] - frame['cycle_start_us']
        tail = frame['elapsed_us'] - records[-1][end_key]
        if min(point_work, held, period, tail, period-frame['elapsed_us']) < 0 or held < prescribed:
            raise ValueError('Inconsistent board point/frame timing')
        samples.append(dict(period_ms=period/1000,
                            dac_and_interpoint_ms=point_work/1000,
                            prescribed_wait_ms=prescribed/1000,
                            adc_and_hold_overhead_ms=(held-prescribed)/1000,
                            frame_tail_ms=tail/1000,
                            frame_gap_ms=(period-frame['elapsed_us'])/1000))
    result = {key: statistics.mean(row[key] for row in samples) for key in samples[0]}
    result.update(mean_hz=capture['mean_hz'],p99_period_ms=capture['p99_period_ms'],
                  point_count=capture['point_count'],samples_per_point=capture['samples_per_point'],
                  fast_dac_writes=capture['fast_dac_writes'],received_frames=capture['received_frames'],
                  disarm_ack=capture['disarm_ack'],shutter_ack=capture['shutter_ack'])
    return result


def validate_capture(capture, version, expected_frames, count):
    if (not capture.get('complete') or capture.get('received_frames') != expected_frames
            or len(capture['frames']) != expected_frames or capture['point_count'] != count
            or not capture.get('disarm_ack') or not capture.get('shutter_ack')
            or capture.get('cleanup_errors') or capture.get('error')):
        raise RuntimeError('Incomplete capture or unconfirmed shutdown')
    for sequence, frame in enumerate(capture['frames']):
        if (frame['firmware'] != version or frame['sequence'] != sequence
                or frame['i2c_errors'] or frame['spi_errors'] or len(frame['records']) != count
                or frame['adc_channel'] != capture['signal_channel']
                or frame['feedback_selector'] != capture['feedback_selector']):
            raise RuntimeError('Version/sequence/count/channel/bus mismatch')
        for record in frame['records']:
            if capture['single_sample']:
                if set(record) != {'adc_code','write_end_us','sample_end_us'}:
                    raise RuntimeError('Single ADC frame fabricated a second reading')
            elif 'first_code' not in record or 'second_code' not in record:
                raise RuntimeError('Legacy dual ADC frame is incomplete')
    return point_budget(capture)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--device-serial',required=True)
    parser.add_argument('--source-capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-version', type=lambda text: int(text, 0), default=0x10073,
                        help='Exact runtime version lock (v115=0x10073, v116=0x10074)')
    parser.add_argument('--wire-trace', action='store_true',
                        help='Save diagnostic raw wire and event trace for each route')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=json.loads(args.source_capture.read_text(encoding='utf-8'))
    selected=validate_plan_payload(source['plan'])
    if selected['point_count']!=63 or selected['points_per_peak']!=7:
        raise ValueError('Needs a valid nine-peak symmetric seven-point source plan')
    rows=[dict(index=r['index'],group=r['group'],target_nm=r['target_nm'],
               measured_nm=r['measured_nm'],codes=[0]*5) for r in selected['rows']]
    plan=dict(rows=rows,points_per_peak=7,zero_codes_only=True,optical_accuracy_verified=False)
    report=dict(schema='beta_single_adc_zero_current_acceptance_v1',
                created_utc=datetime.now(timezone.utc).isoformat(),device_serial=args.device_serial,
                port=args.port,zero_codes_only=True,optical_accuracy_verified=False,
                complete=False,results={},thermal_snapshots=[],final_disarm_ack=False,
                final_shutter_ack=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    phase=['initial']
    def save():
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def run(device,name,current_plan,*,single=True,fast=True,wait=850,
            channel=0,selector=2,cycles=64,continuous=False):
        phase[0]=name
        count=len(current_plan['rows'])
        output=args.output.with_suffix(f'.{name}.json')
        seen=[0]
        def arrived(_frame):seen[0]+=1
        if args.wire_trace:
            device.trace.clear()
            device.raw.clear()
        try:
            result=execute_device(device,current_plan,output,cycles=0 if continuous else cycles,
                first_delay_us=850,spacing_us=50,boundary_extra_us=0,point_delays_us=[wait]*count,
                adc_channel=channel,feedback_selector=selector,single_sample=single,fast_dac_writes=fast,
                on_frame=arrived,should_stop=lambda: continuous and seen[0]>=cycles)
        finally:
            if args.wire_trace:
                output.with_suffix('.wire.bin').write_bytes(device.raw)
                output.with_suffix('.trace.json').write_text(
                    json.dumps(device.trace, indent=2), encoding='utf-8')
        if continuous and not result['stopped_by_user']:
            raise RuntimeError('Continuous stop not exercised')
        report['results'][name]=validate_capture(result,args.expected_version,cycles,count)
        save()
        print(json.dumps(dict(check=name,**report['results'][name])),flush=True)
    previous_machine=app.get_runtime_machine_id()
    try:
        with UsbOtaClient(args.port,args.device_serial) as client:before=client.query_info()
        report['runtime_before']=asdict(before)
        if (args.expected_version not in (0x10073, 0x10074)
                or before.firmware_version != args.expected_version):
            raise RuntimeError('Requires the explicitly selected, actual running v1.0.115/116 Beta')
        ports=[p for p in list_ports.comports() if p.device==args.port and p.vid==0x0483
               and p.pid==0x5740 and p.serial_number==args.device_serial]
        if len(ports)!=1:raise RuntimeError('Exact native USB identity no longer present')
        app.set_runtime_machine_id('machine_3')
        if args.wire_trace:
            from beta_wire_trace import TraceSerial
            serial_class = TraceSerial
        else:
            serial_class = ThermalSerial
        with serial_class(args.port,2_000_000,timeout=.03,write_timeout=3) as device:
            device.dtr=True
            device.thermal_observer.callback=lambda value: report['thermal_snapshots'].append(
                dict(value,phase=phase[0]))
            try:
                if (not _safe_disarm(device) or not zero_all_dacs_and_verify(device)
                        or not _close_shutter(device)):
                    raise RuntimeError('Initial all-zero/disarm/shutter confirmation missing')
                for name,single,fast,wait in (
                        ('dual_63_850',False,False,850),
                        ('single_63_850',True,False,850),
                        ('merged_63_850',True,True,850),
                        ('single_63_50',True,False,50),
                        ('merged_63_50',True,True,50)):
                    run(device,name,plan,single=single,fast=fast,wait=wait)
                for channel in range(4):
                    for gain in (range(4) if channel<2 else (0,)):
                        run(device,f'merged_ch{channel}_gain{gain}',plan,
                            channel=channel,selector=gain,cycles=32)
                run(device,'continuous_63',plan,cycles=64,continuous=True)
                run(device,'merged_63_15ms',plan,wait=15000,cycles=32)
                legacy_rows=[]
                for start in range(0,63,7):
                    legacy_rows.extend(rows[start+i] for i in (0,1,3,5,6))
                legacy=dict(rows=legacy_rows,points_per_peak=5,zero_codes_only=True)
                run(device,'legacy_dual_45',legacy,single=False,fast=False,cycles=32)
                run(device,'single_45',legacy,cycles=32)
                phase[0]='custom_zero_transient'
                report['custom_zero_transients']=[]
                for fast in (False,True):
                    for repeat in range(3):
                        item=capture_custom(device,rows[0],rows[1],123000+repeat+int(fast)*10,
                                            fast_dac_writes=fast)
                        report['custom_zero_transients'].append(item)
                valid=[v for v in report['thermal_snapshots'] if v['thermal_flags']&1]
                faults=[v for v in report['thermal_snapshots'] if not v['thermal_flags']&1
                        or v['thermal_flags']&(4|8|32)]
                if not valid or faults or any(not 0<v['temperature_c']<85 for v in valid):
                    raise RuntimeError('Missing or faulty/unsafe thermal telemetry')
                report['thermal_summary']=dict(valid_snapshots=len(valid),fault_snapshots=len(faults),
                    sensor_age_max_ms=max(v['sensor_age_ms'] for v in valid),
                    temperature_min_c=min(v['temperature_c'] for v in valid),
                    temperature_max_c=max(v['temperature_c'] for v in valid),
                    rpm_min=min(v['fan_rpm'] for v in valid),rpm_max=max(v['fan_rpm'] for v in valid),
                    duty_permille=sorted(set(v['fan_duty_permille'] for v in valid)))
            finally:
                try:report['final_disarm_ack']=bool(_safe_disarm(device))
                finally:report['final_shutter_ack']=bool(_close_shutter(device))
        with UsbOtaClient(args.port,args.device_serial) as client:after=client.query_info()
        report['runtime_after']=asdict(after)
        if before!=after or not report['final_disarm_ack'] or not report['final_shutter_ack']:
            raise RuntimeError('Runtime changed or final shutdown not confirmed')
        report['complete']=True
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        app.set_runtime_machine_id(previous_machine)
        save()
        print(json.dumps({k:report.get(k) for k in ('complete','error','final_disarm_ack',
                                                  'final_shutter_ack','thermal_summary')}),flush=True)


if __name__=='__main__':main()
