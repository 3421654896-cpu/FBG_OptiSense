"""Capture 32 uninterrupted 18-point direct-ADC cycles; never CNC motion.

This bounded diagnostic separates laser start-up from continuous-route drift.
It uses only installed current rows, shutters in firmware and again on the host,
and never claims wavelength, pressure, location or contact-area ground truth.
"""
import argparse
import binascii
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import serial

import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from capture_short_route_teacher import RouteCaptureTimeout, RouteRejected, codes_crc
from screen_ch1_reduced_pairs import candidate_indices

HERE = Path(__file__).resolve().parent
ROUTES = {'two_flanks': 0, 'g8_right_left': 1}
DELAYS = (0, 1500, 4350)
CYCLES = 32
POINTS = 18
FIRMWARE = 0x10024


def command(route, extra_us, tag):
    if (type(route) is not str or route not in ROUTES or type(extra_us) is not int
            or extra_us not in DELAYS or type(tag) is not int or not 1 <= tag <= 0xffffffff):
        raise ValueError('unlisted route/delay or invalid tag')
    data = bytearray(808)
    data[:4] = b'\xff\xff\x03\x06'
    data[4] = ROUTES[route]
    data[6:8] = extra_us.to_bytes(2, 'big')
    data[8:12] = tag.to_bytes(4, 'big')
    data[12:16] = b'R32!'
    return bytes(data)


def packet_length():
    return 64 + CYCLES*POINTS*32 + 6


def capture_packet(device, route, extra_us, tag):
    device.reset_input_buffer()
    device.write(command(route, extra_us, tag))
    data = bytearray()
    deadline = time.monotonic()+5.0
    length = packet_length()
    while time.monotonic() < deadline:
        data.extend(device.read(8192))
        rejected = data.find(b'\xd9\x9d\x00\x7f')
        if rejected >= 0 and len(data) >= rejected+20:
            frame = data[rejected:rejected+20]
            if int.from_bytes(frame[4:8],'big') == tag and frame[-2:] == b'\x9d\xd9':
                raise RouteRejected(data,frame)
        at = data.find(b'\xd9\x9d\x02')
        if at >= 0 and len(data) >= at+length:
            return bytes(data[at:at+length])
    raise RouteCaptureTimeout(data)


def decode(wire, route, extra_us, tag, rows):
    u16=lambda at:int.from_bytes(wire[at:at+2],'big')
    u32=lambda at:int.from_bytes(wire[at:at+4],'big')
    indices=candidate_indices(route)
    expected=CYCLES*POINTS
    if (len(wire)!=packet_length() or wire[:4]!=b'\xd9\x9d\x02\x00'
            or wire[-2:]!=b'\x9d\xd9' or u32(len(wire)-6)!=binascii.crc32(wire[:-6])):
        raise ValueError('long-route status/length/CRC/trailer mismatch')
    if (u32(4)!=tag or u32(8)!=FIRMWARE or wire[12:16]!=bytes((ROUTES[route],CYCLES,POINTS,0))
            or u16(16)!=extra_us or wire[18:20]!=b'\x00\x02' or u16(20)!=expected
            or wire[22:24]!=b'\x00\x02' or u16(32)!=expected or u16(34)!=0
            or u32(36)!=codes_crc(rows) or u32(40)!=0 or wire[44:62]!=bytes(indices)
            or wire[62:64]!=b'\x01\x00' or not 0<u32(24)<=2_250_000):
        raise ValueError('long-route provenance/count/errors/shutter mismatch')
    points=[];previous=-1
    for index in range(expected):
        at=64+index*32
        row,cycle,local=u16(at),wire[at+2],wire[at+3]
        times=[u32(at+p) for p in (4,8,12,16,22,26)]
        first,second=u16(at+20),u16(at+30)
        if (row!=indices[index%POINTS] or (cycle,local)!=divmod(index,POINTS)
                or not previous<times[0]<times[1]<=times[2]<times[3]<times[4]<times[5]
                or times[2]-times[1]<50+(extra_us if row==38 else 0)
                or times[4]-times[3]<600 or max(first,second)>4095):
            raise ValueError('long-route noncausal timestamp, row or ADC')
        points.append(dict(target=row,cycle=cycle,local=local,write_start_us=times[0],
            write_end_us=times[1],first_start_us=times[2],first_end_us=times[3],first_code=first,
            second_start_us=times[4],second_end_us=times[5],second_code=second))
        previous=times[-1]
    if u32(28)!=points[-1]['write_end_us']:
        raise ValueError('long-route final write timestamp mismatch')
    return dict(route=route,extra_g8_right_delay_us=extra_us,tag=tag,firmware=u32(8),
        table_codes_crc32=codes_crc(rows),elapsed_us=u32(24),points=points,
        firmware_shutter_asserted=True,wire_hex=wire.hex(),training_eligible=False,
        optical_wavelength_verified=False,physical_15hz_verified=False)


def _percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p/100;lo=math.floor(position);hi=math.ceil(position)
    return ordered[lo] if lo==hi else ordered[lo]*(hi-position)+ordered[hi]*(position-lo)


def analyze(decoded, rows):
    windows={'startup_0_3':range(0,4),'early_4_7':range(4,8),
             'middle_8_15':range(8,16),'steady_candidate_16_31':range(16,32)}
    series={};summary={}
    for grating in range(9):
        left,right=grating*5+1,grating*5+3
        sl,ml=rows[left]['template_signal_v'],rows[left]['template_slope_v_per_nm']
        sr,mr=rows[right]['template_signal_v'],rows[right]['template_slope_v_per_nm']
        values=[]
        for cycle in range(CYCLES):
            pair=[next(p['second_code'] for p in decoded['points']
                       if p['cycle']==cycle and p['target']==target) for target in (left,right)]
            if not pair[0]: raise ValueError('zero left flank prevents ratio estimate')
            ratio=pair[1]/pair[0];den=ratio*ml-mr
            if not den: raise ValueError('singular two-flank template estimate')
            values.append(1000*(sr-ratio*sl)/den)
        reference=statistics.median(values[16:])
        relative=[value-reference for value in values]
        series[f'G{grating+1}']=dict(left_row=left,right_row=right,
            apparent_shift_pm=values,relative_to_last16_median_pm=relative)
        summary[f'G{grating+1}']={}
        for name,cycles in windows.items():
            subset=[relative[c] for c in cycles];median=statistics.median(subset)
            mad=statistics.median(abs(v-median) for v in subset)
            summary[f'G{grating+1}'][name]=dict(median_pm=median,robust_sigma_pm=1.4826*mad,
                p95_absolute_from_own_median_pm=_percentile([abs(v-median) for v in subset],95),
                minimum_pm=min(subset),maximum_pm=max(subset))
    gaps=[]
    byrow={row:[] for row in candidate_indices(decoded['route'])}
    for point in decoded['points']: byrow[point['target']].append(point['second_end_us'])
    for times in byrow.values(): gaps.extend(b-a for a,b in zip(times,times[1:]))
    return dict(schema='ch1_long_route_stability_analysis_v1',cycles=CYCLES,
        duration_us=decoded['elapsed_us'],same_point_gap_mean_us=statistics.fmean(gaps),
        same_point_gap_max_us=max(gaps),series=series,windows=summary,
        deployment_eligible=False,training_eligible=False,physical_15hz_verified=False,
        note='Template-derived apparent shifts are relative repeatability diagnostics, not synchronized wavelength truth.')


def plot_analysis(analysis,path):
    figure,axis=plt.subplots(figsize=(12.5,6.5),layout='constrained')
    for name,item in analysis['series'].items(): axis.plot(range(CYCLES),item['relative_to_last16_median_pm'],marker='.',label=name)
    axis.axvspan(-.5,3.5,color='#f3a34a',alpha=.12,label='startup cycles 0-3')
    axis.axvline(15.5,color='#666',ls='--',lw=1)
    axis.set(xlabel='Continuous route cycle',ylabel='Template-derived apparent shift relative to last-16 median (pm)',
             title=f'Uninterrupted 18-point route, 32 cycles | worst same-point gap {analysis["same_point_gap_max_us"]/1000:.3f} ms')
    axis.grid(alpha=.2);axis.legend(ncol=5,fontsize=8)
    figure.savefig(path,dpi=145);plt.close(figure)


def execute(output,route='g8_right_left',extra_us=1500,port='COM6'):
    output=Path(output);png=output.with_suffix('.png')
    if output.exists() or png.exists(): raise FileExistsError(output)
    command(route,extra_us,1);rows,_=load_installed_rows()
    report=dict(schema='ch1_long_route_stability_capture_v1',scope=__doc__,complete=False,
        route=route,extra_us=extra_us,installed_rows=rows,
        table_sha256=sha256_file(HERE/'mode_tables_from_fullband_2001.json'),
        script_sha256=sha256_file(Path(__file__)),created_utc=datetime.now(timezone.utc).isoformat(),
        training_eligible=False,physical_15hz_verified=False)
    output.parent.mkdir(parents=True,exist_ok=True);save_checkpoint(output,report)
    with serial.Serial(port,2_000_000,timeout=.02,write_timeout=3) as device:
        try:
            device.dtr=True
            if not _safe_disarm(device) or not _close_shutter(device): raise RuntimeError('initial disarm/shutter ACK missing')
            worker=app.EqualIntervalWorker(device,(),{},settle_s=0.,feedback_selectors=(0,2))
            if not worker._feedback_command_and_ack() or not worker._interruptible_wait(.01): raise RuntimeError('feedback ACK/cancellation')
            tag=int(time.time_ns()&0x7fffffff) or 1
            try:wire=capture_packet(device,route,extra_us,tag)
            except RouteCaptureTimeout as exc:
                report['received_on_timeout_hex']=exc.received_wire_hex;raise
            report['wire_hex']=wire.hex();save_checkpoint(output,report)
            report['decoded']=decode(wire,route,extra_us,tag,rows)
            report['analysis']=analyze(report['decoded'],rows);report['complete']=True
        except BaseException as exc:
            report['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            errors=[]
            try:disarmed=_safe_disarm(device)
            except Exception as exc:disarmed=False;errors.append(f'disarm:{exc}')
            try:shuttered=_close_shutter(device)
            except Exception as exc:shuttered=False;errors.append(f'shutter:{exc}')
            report['safety_cleanup']=dict(exact_disarm_ack=disarmed,exact_soa_shutter_ack=shuttered,errors=errors)
            save_checkpoint(output,report)
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}',flush=True)
            if not disarmed or not shuttered: raise RuntimeError('exact final optical shutdown not confirmed')
    plot_analysis(report['analysis'],png)
    print(json.dumps({k:report['analysis'][k] for k in ('duration_us','same_point_gap_mean_us','same_point_gap_max_us')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--route',choices=tuple(ROUTES),default='g8_right_left')
    parser.add_argument('--extra-us',type=int,choices=DELAYS,default=1500);parser.add_argument('--port',default='COM6')
    args=parser.parse_args();execute(args.output,args.route,args.extra_us,args.port)
