"""Send only zero-current malformed guarded-C27 requests and prove rejection."""
import argparse
import binascii
import json
from pathlib import Path
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_candidate27_guarded_stream import command, extract_frames
from capture_ch1_teacher_spectrum import save_checkpoint, now_text


def malformed(name, tag):
    rows = [dict(index=local * 10, codes=[0] * 5) for local in range(27)]
    guard = dict(index=205, codes=[0] * 5, before_order=21, hold_us=1000)
    packet = bytearray(command(rows, guard, 32, tag))
    if name == 'bad_crc':
        packet[367] ^= 1
    elif name == 'guard_not_between':
        packet[352:354] = (200).to_bytes(2, 'big')
    elif name == 'nonzero_tail':
        packet[500] = 1
    elif name == 'short_guard_hold':
        packet[23] = 19
    elif name == 'nonzero_v1_crc_slot':
        packet[348] = 1
    else:
        raise ValueError(name)
    if name != 'bad_crc':
        packet[364:368] = binascii.crc32(packet[:364]).to_bytes(4, 'big')
    assert all(packet[26 + local * 12:36 + local * 12] == bytes(10)
               for local in range(27))
    assert packet[354:364] == bytes(10)
    return bytes(packet)


def verify(output):
    if output.exists():
        raise FileExistsError(output)
    report = dict(schema='guarded_candidate_rejection_v2', created=now_text(),
                  complete=False, cases=[], cnc_commands=[],
                  all_requested_dac_codes_zero=True, disarm_ack=False,
                  shutter_ack=False)
    with serial.Serial('COM6', 2_000_000, timeout=.03, write_timeout=2) as device:
        try:
            device.dtr = True
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError('Initial shutdown not confirmed')
            worker = app.EqualIntervalWorker(device, (), {}, settle_s=0,
                                             feedback_selectors=(0, 2))
            if not worker._feedback_command_and_ack():
                raise RuntimeError('Gain ACK missing')
            for offset, name in enumerate(('bad_crc', 'guard_not_between', 'nonzero_tail',
                                           'short_guard_hold', 'nonzero_v1_crc_slot')):
                tag = 0x27480001 + offset
                device.reset_input_buffer()
                device.write(malformed(name, tag))
                device.flush()
                deadline = time.monotonic() + 2
                buffer = bytearray()
                reply = None
                while time.monotonic() < deadline and reply is None:
                    buffer.extend(device.read(4096))
                    for wire in extract_frames(buffer):
                        if wire[:4] != b'\xd9\x9d\x0c\x7f':
                            raise RuntimeError('Malformed v2 request produced a data frame')
                        reply = wire
                        break
                if (reply is None or len(reply) != 20 or reply[-2:] != b'\x9d\xd9'
                        or int.from_bytes(reply[4:8], 'big') != tag
                        or not (int.from_bytes(reply[8:10], 'big') & 256)
                        or reply[10:14] != bytes((0, 2, 2, app.PI11210_SOA_SHUTTER_MODE))):
                    raise RuntimeError('Exact rejected-in-shutter response missing: ' + name)
                report['cases'].append(dict(name=name, wire_hex=reply.hex(), rejected=True))
            report['complete'] = True
        except BaseException as exc:
            report['error'] = repr(exc)
            raise
        finally:
            errors = []
            try:
                report['disarm_ack'] = _safe_disarm(device)
            except Exception as exc:
                errors.append('disarm: ' + repr(exc))
            try:
                report['shutter_ack'] = _close_shutter(device)
            except Exception as exc:
                errors.append('shutter: ' + repr(exc))
            report['cleanup_errors'] = errors
            save_checkpoint(output, report)
    if not report['disarm_ack'] or not report['shutter_ack']:
        raise RuntimeError('Final shutdown not confirmed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    verify(parser.parse_args().output)
