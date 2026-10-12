"""Audit a non-moving C27 control at predeclared time windows, without hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_candidate27_contact import rms, shape
from capture_candidate27_stream import decode, FIRMWARE
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file


def summarize(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 27 or len(values) != len(times)
            or len(values) < 180 or not np.all(np.isfinite(values))
            or np.any(values < 0) or np.any(values > 4095)
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError('Ordered finite CH1 frames required')
    mask = (times >= 4.85) & (times < 5.60)
    if mask.sum() < 20 or times[79] >= 4.85:
        raise ValueError('Control baseline/test time coverage is insufficient')
    baseline = np.median(values[16:80], axis=0)
    held = np.median(values[mask], axis=0)
    return dict(test_frames=int(mask.sum()), baseline_noise_rms_codes=rms(np.std(values[16:80], axis=0)),
                delta_rms_codes=rms(held-baseline), shape_delta_rms=rms(shape(held)-shape(baseline)),
                group_delta_rms_codes=[rms(g) for g in (held-baseline).reshape(9, 3)])


def audit(path):
    d = json.loads(path.read_text('utf-8'))
    if (not d.get('complete') or not d.get('disarm_ack') or not d.get('shutter_ack')
            or d.get('commissioning') is not None or d.get('cnc_commands') != []
            or d.get('expected_firmware') != FIRMWARE):
        raise ValueError('Complete optical-only C27 control required')
    tag = int.from_bytes(bytes.fromhex(d['command_hex'])[8:12], 'big')
    for i, frame in enumerate(d['frames']):
        original = decode(bytes.fromhex(frame['wire_hex']), tag, d['route_crc32'],
                          d['first_delay_us'], d['spacing_us'])
        if frame != original or frame['sequence'] != i or frame['i2c_error_delta'] or frame['spi_error_delta']:
            raise ValueError('Invalid raw control frame')
    terminal = decode(bytes.fromhex(d['terminal']['wire_hex']), tag, d['route_crc32'],
                      d['first_delay_us'], d['spacing_us'])
    if (terminal != d['terminal'] or terminal['status'] or terminal['stop_reason']
            or terminal['completed_cycles'] != len(d['frames'])
            or terminal['i2c_error_delta'] or terminal['spi_error_delta']):
        raise ValueError('Abnormal control termination')
    frames = d['frames']
    times = [((f['cycle_start_us']-frames[0]['cycle_start_us']) & 0xffffffff)/1e6 for f in frames]
    values = [[r['second_code'] for r in f['records']] for f in frames]
    return dict(source=str(path.resolve()), sha256=sha256_file(path),
                route_crc32=d['route_crc32'], first_delay_us=d['first_delay_us'],
                spacing_us=d['spacing_us'], **summarize(values, times))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = dict(schema='candidate27_null_control_audit_v1',
        baseline_sequences=[16, 79], test_seconds=[4.85, 5.60],
        observations=[audit(path) for path in args.captures],
        scope='Previously captured optical-only controls; not proof against drift in every later contact.')
    save_checkpoint(args.output, result)
    print(json.dumps(result, indent=2))
