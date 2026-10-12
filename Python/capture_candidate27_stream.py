"""USB-only RAM candidate diagnostic; requires the audited stream firmware.

The candidate is not installed. Every cycle carries fresh physical CH1 pairs,
requested timing, route CRC, sequence and board timestamps. No CNC interface.
"""
import argparse
import binascii
import json
import math
from copy import deepcopy
from pathlib import Path
import time
import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file, now_text
from capture_fast_flank_stream import analyze as analyze_rate
from probe_current_candidate_route import load_route
from capture_fast_flank_stream import FIRMWARE

LIMITS = (58981,58981,32767,24575,24575)


def command(rows, cycles, tag, first_delay_us=600, spacing_us=50):
    if type(cycles) is not int or not 32 <= cycles <= 1024:
        raise ValueError('Cycles must be 32..1024')
    if type(tag) is not int or not 1 <= tag <= 0xffffffff or len(rows) != 27:
        raise ValueError('Nonzero uint32 tag and 27 rows required')
    if any(type(v) is not int or not 50 <= v <= 600 or v%25
           for v in (first_delay_us,spacing_us)) or first_delay_us+spacing_us > 650:
        raise ValueError('Invalid bounded sampling timing')
    packet = bytearray(808)
    packet[:6] = b'\xff\xff\x03\x0a\x01\x1b'
    packet[6:8] = first_delay_us.to_bytes(2,'big')
    packet[8:12] = tag.to_bytes(4,'big')
    packet[12:16] = b'C27!'
    packet[16:18] = cycles.to_bytes(2,'big')
    packet[18] = 16
    packet[20:22] = spacing_us.to_bytes(2,'big')
    previous = -1
    for i,row in enumerate(rows):
        index, codes = row['index'], tuple(row['codes'])
        if type(index) is not int or not previous < index <= 2000 or len(codes) != 5:
            raise ValueError('Forward unique 2001-table indices required')
        previous = index
        if any(type(c) is not int or not 0 <= c <= limit for c,limit in zip(codes,LIMITS)):
            raise ValueError('Candidate exceeds conservative laser current limits')
        packet[24+i*12:36+i*12] = b''.join(v.to_bytes(2,'big') for v in (index,*codes))
    packet[348:352] = binascii.crc32(packet[:348]).to_bytes(4,'big')
    return bytes(packet)


def extract_frames(buffer):
    while True:
        start = buffer.find(b'\xd9\x9d')
        if start < 0:
            if len(buffer)>1: del buffer[:-1]
            return
        if start: del buffer[:start]
        if len(buffer)<4: return
        kind,status=buffer[2:4]
        length = 20 if kind==10 and status==127 else 291 if kind==10 else 42 if kind==11 else 0
        if not length:
            del buffer[0]
            continue
        if len(buffer)<length: return
        wire=bytes(buffer[:length])
        del buffer[:length]
        yield wire


def decode(wire, tag, table_crc, first_delay_us, spacing_us,
           expected_firmware=FIRMWARE):
    u16=lambda i:int.from_bytes(wire[i:i+2],'big')
    u32=lambda i:int.from_bytes(wire[i:i+4],'big')
    if wire[:4]==b'\xd9\x9d\x0a\x7f':
        if len(wire)!=20 or wire[-2:]!=b'\x9d\xd9' or u32(4)!=tag:
            raise ValueError('Candidate rejection identity/trailer mismatch')
        raise RuntimeError(f'Candidate rejected by board: flags={u16(8):04x}')
    if len(wire) not in (291,42) or wire[-2:] != b'\x9d\xd9' or u32(len(wire)-6)!=binascii.crc32(wire[:-6]):
        raise ValueError('Candidate frame length/CRC/trailer mismatch')
    if wire[:2]!=b'\xd9\x9d' or u32(4)!=tag or u32(8)!=expected_firmware:
        raise ValueError('Candidate tag/firmware mismatch')
    if wire[2]==11 and len(wire)==42:
        if u32(32)!=table_crc or wire[18:22]!=bytes((1,1,0,2)):
            raise ValueError('Candidate terminal route/shutter/feedback mismatch')
        return dict(kind='terminal', status=wire[3], completed_cycles=u16(12), requested_cycles=u16(14),
            stop_reason=u16(16), firmware_shutter_asserted=bool(wire[19]),
            i2c_error_delta=u16(22), spi_error_delta=u32(24), elapsed_us=u32(28), wire_hex=wire.hex())
    if wire[2]!=10 or len(wire)!=291 or wire[3]&~1 or u32(24)!=table_crc:
        raise ValueError('Candidate data type/flags/route CRC mismatch')
    if wire[28:34]!=bytes((first_delay_us//25,27,16,spacing_us//25,0,2)) or wire[40:67]!=bytes(range(27)):
        raise ValueError('Candidate timing/count/feedback/local order mismatch')
    sequence=u32(12)
    if bool(wire[3]&1)!=(sequence>=16):
        raise ValueError('Noncausal warm-up flag')
    records=[]
    previous=-1
    for local in range(27):
        at=67+local*8
        first,second,write,end=(u16(at+j) for j in (0,2,4,6))
        if max(first,second)>4095 or not previous<write<end<=u32(20)+1 or end-write < first_delay_us+spacing_us:
            raise ValueError('Invalid ADC code or conversion timing')
        previous=end
        records.append(dict(row=local,first_code=first,second_code=second,write_end_us=write,second_end_us=end))
    return dict(kind='data',sequence=sequence,cycle_start_us=u32(16),elapsed_us=u32(20),
        minimum_warmup_complete=sequence>=16,records=records,i2c_error_delta=u16(34),
        spi_error_delta=u32(36),wire_hex=wire.hex())


def prepare_capture(candidate, probe_path, cycles=256, first_delay_us=600, spacing_us=50):
    """Validate all candidate/teacher inputs before another caller may move CNC."""
    rows,_=load_route(candidate,3)
    probe=json.loads(probe_path.read_text('utf-8'))
    if (probe.get('schema')!='candidate_settled_route_probe_v1'
            or probe.get('protocol')!='previous selected -> target; includes last-to-first wrap; 0.3 s minimum hold'
            or probe.get('cnc_commands')!=[] or not probe.get('complete')
            or not probe.get('disarm_ack') or not probe.get('shutter_ack')
            or probe.get('candidate_sha256')!=sha256_file(candidate) or probe.get('route_rows')!=rows
            or len(probe.get('samples',[]))!=54):
        raise ValueError('Complete matching settled-route optical probe is required')
    if probe.get('feedback_selectors') != [0,2]:
        raise ValueError('Settled route must use the same physical feedback selectors')
    for i,sample in enumerate(probe['samples']):
        point, meter = sample['point'], sample['meter']
        if (sample['order'] != i%27 or sample['repeat'] != i//27+1
                or point['index'] != rows[i%27]['index'] or not point['stable']
                or point['dac_codes'] != rows[i%27]['codes']
                or not math.isfinite(point['ch1_adc_code']) or not 0 <= point['ch1_adc_code'] < 4000
                or not math.isfinite(meter['wavelength_nm'])
                or abs(meter['wavelength_nm']-rows[i%27]['target_nm']) > .05
                or not math.isfinite(meter['power_mw']) or meter['power_mw'] <= 0
                or meter['side_mode_suppression_db'] is None
                or not math.isfinite(meter['side_mode_suppression_db'])
                or meter['side_mode_suppression_db'] < 20):
            raise ValueError('Settled optical qualification record is invalid')
    tag=int(time.time_ns()&0x7fffffff) or 1
    packet=command(rows,cycles,tag,first_delay_us,spacing_us)
    table_crc=binascii.crc32(packet[24:348])
    return dict(schema='candidate27_stream_v1',complete=False,created=now_text(),route_rows=rows,
        candidate_sha256=sha256_file(candidate),probe_sha256=sha256_file(probe_path),
        cycles_requested=cycles,first_delay_us=first_delay_us,spacing_us=spacing_us,
        route_crc32=table_crc,expected_firmware=FIRMWARE,command_hex=packet.hex(),frames=[],
        cnc_commands=[],deployed=False,physical_15hz_verified=False,disarm_ack=False,shutter_ack=False)


def execute_device(device, output, *, prepared, on_frame=None, should_stop=None):
    """Stream on an already owned port; hooks do not change protocol checks."""
    if output.exists(): raise ValueError('Output must be new')
    report=deepcopy(prepared)
    packet=bytes.fromhex(report['command_hex'])
    tag=int.from_bytes(packet[8:12],'big')
    cycles=report['cycles_requested']
    first_delay_us,spacing_us=report['first_delay_us'],report['spacing_us']
    if packet != command(report['route_rows'],cycles,tag,first_delay_us,spacing_us):
        raise ValueError('Prepared command was modified')
    table_crc=binascii.crc32(packet[24:348])
    if report['frames'] or report['complete'] or report['route_crc32']!=table_crc or report['expected_firmware']!=FIRMWARE:
        raise ValueError('Prepared capture has invalid identity/state')
    try:
        device.dtr=True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError('Initial shutdown not confirmed')
        worker=app.EqualIntervalWorker(device,(),{},settle_s=0,feedback_selectors=(0,2))
        if not worker._feedback_command_and_ack(): raise RuntimeError('Feedback ACK missing')
        device.reset_input_buffer()
        device.write(packet)
        device.flush()
        deadline=time.monotonic()+min(50,5+cycles*.05)
        buffer=bytearray()
        terminal=None
        while time.monotonic()<deadline and terminal is None:
            if should_stop is not None and should_stop():
                raise RuntimeError('External capture stop requested')
            buffer.extend(device.read(8192))
            for wire in extract_frames(buffer):
                row=decode(wire,tag,table_crc,first_delay_us,spacing_us)
                if row['i2c_error_delta'] or row['spi_error_delta']: raise RuntimeError('Board I2C/SPI error')
                if row['kind']=='terminal':
                    terminal=row
                else:
                    if row['sequence']!=len(report['frames']): raise ValueError('Missing/reordered cycle')
                    if report['frames'] and row['cycle_start_us']<=report['frames'][-1]['cycle_start_us']:
                        raise ValueError('Nonmonotonic board time')
                    report['frames'].append(row)
                    if on_frame is not None: on_frame(row)
                    if len(report['frames'])%64==0: save_checkpoint(output,report)
        if terminal is None: raise TimeoutError('Candidate terminal missing')
        report['terminal']=terminal
        if terminal['status'] or terminal['stop_reason'] or terminal['completed_cycles']!=cycles or terminal['requested_cycles']!=cycles or len(report['frames'])!=cycles:
            raise RuntimeError('Candidate did not finish cleanly')
        report['analysis']=analyze_rate(report['frames'])
        report['complete']=True
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        errors=[]
        try: report['disarm_ack']=_safe_disarm(device)
        except Exception as exc: errors.append('disarm: '+repr(exc))
        try: report['shutter_ack']=_close_shutter(device)
        except Exception as exc: errors.append('shutter: '+repr(exc))
        report['cleanup_errors']=errors
        save_checkpoint(output,report)
    if not report['disarm_ack'] or not report['shutter_ack']: raise RuntimeError('Final shutdown not confirmed')
    print(json.dumps(report['analysis'],indent=2))
    return report


def execute(candidate, probe_path, output, cycles=256, first_delay_us=600, spacing_us=50):
    prepared=prepare_capture(candidate,probe_path,cycles,first_delay_us,spacing_us)
    with serial.Serial('COM6',2_000_000,timeout=.03,write_timeout=3) as device:
        return execute_device(device,output,prepared=prepared)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--probe',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cycles',type=int,default=256)
    parser.add_argument('--first-delay-us',type=int,default=600)
    parser.add_argument('--spacing-us',type=int,default=50)
    args=parser.parse_args()
    execute(args.candidate,args.probe,args.output,args.cycles,args.first_delay_us,args.spacing_us)
