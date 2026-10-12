"""Short, identity-locked raw USB trace for a zero-current temporary route."""
import argparse
import binascii
from dataclasses import asdict
import json
from pathlib import Path
import time

import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from fbg_usb_ota import UsbOtaClient
from guarded_application_upgrade import zero_all_dacs_and_verify
from temporary_test_mode import command, _feedback_pair, extract_frames, decode, validate_plan_payload
from thermal_serial import ThermalSerial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--channel',type=int,choices=range(4),default=0)
    p.add_argument('--group',type=int,choices=(5,7),default=5)
    p.add_argument('--pause-before-command',type=float,default=0)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not 0<=args.pause_before_command<=.2:raise ValueError('Diagnostic pause must be 0..0.2 seconds')
    selected=validate_plan_payload(json.loads(args.source.read_text(encoding='utf-8'))['plan'])
    source=selected['rows']
    if len(source)!=63:raise ValueError('Source must be current 63-point plan')
    route=[]
    for start in range(0,63,7):
        for offset in ((0,1,3,5,6) if args.group==5 else range(7)):
            row=dict(source[start+offset]);row['codes']=[0]*5;route.append(row)
    selector=2 if args.channel<2 else 0
    count=len(route);tag=876123
    packet=command(route,32,tag,first_delay_us=850,point_delays_us=[850]*count,
                   adc_channel=args.channel,feedback_selector=selector)
    crc=binascii.crc32(packet[:796] if packet[4]==6 else packet[:662])
    report=dict(zero_codes_only=True,pause_before_command_s=args.pause_before_command,
                group=args.group,channel=args.channel,protocol_version=packet[4],frames=[],
                command_hex=packet.hex(),complete=False)
    raw=bytearray()
    with UsbOtaClient('COM18','207933893232') as client:info=client.query_info()
    report['runtime']=asdict(info)
    if info.firmware_version!=0x10073:raise RuntimeError('Runtime is not v1.0.115 Beta')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with ThermalSerial('COM18',2000000,timeout=.03,write_timeout=3) as device:
            device.dtr=True
            try:
                if (not _safe_disarm(device) or not zero_all_dacs_and_verify(device)
                        or not _close_shutter(device)):
                    raise RuntimeError('Zero current/shutter not confirmed')
                worker=app.EqualIntervalWorker(device,(),{},settle_s=0,
                                               feedback_selectors=_feedback_pair(args.channel,selector))
                if not worker._feedback_command_and_ack():raise RuntimeError('Feedback not confirmed')
                time.sleep(args.pause_before_command)
                device.reset_input_buffer()
                report['bytes_written']=device.write(packet)
                device.flush()
                buffer=bytearray();deadline=time.monotonic()+8
                terminal=False
                while time.monotonic()<deadline and not terminal:
                    data=device.read(8192);raw.extend(data);buffer.extend(data)
                    for wire in extract_frames(buffer):
                        item=decode(wire,tag,crc,first_delay_us=850,point_delays_us=[850]*count,
                                    point_count=count,adc_channel=args.channel,feedback_selector=selector,
                                    points_per_peak=args.group)
                        if item['kind']=='terminal':
                            report['terminal']=item;terminal=True
                        else:report['frames'].append(item)
                report['complete']=terminal
            finally:
                try:report['disarm_ack']=bool(_safe_disarm(device))
                finally:report['shutter_ack']=bool(_close_shutter(device))
    except BaseException as exc:
        report['error']=str(exc)
        raise
    finally:
        report['raw_bytes']=len(raw)
        report['data_header_occurrences']=raw.count(b'\xd9\x9d')
        args.output.with_suffix('.wire.bin').write_bytes(raw)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:report.get(k) for k in ('complete','error','protocol_version','channel',
            'raw_bytes','data_header_occurrences','terminal','disarm_ack','shutter_ack')}),flush=True)


if __name__=='__main__':main()
