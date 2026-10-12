"""Read-only temporal-skew audit. Synthetic sinusoid is NOT physical response."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_f45_contact_transition import validated_gain_identity
from capture_fast_flank_stream import decode_data as decode_flank, decode_end as decode_flank_end


def sinusoid_skew(offsets_us, frequency_hz):
    offsets = np.asarray(offsets_us, float)
    if offsets.ndim != 1 or len(offsets) < 2 or not np.isfinite(offsets).all() or np.any(np.diff(offsets) <= 0):
        raise ValueError('Expected increasing finite completion times')
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError('Expected positive frequency')
    relative = (offsets-offsets[0])*1e-6
    phasors = np.exp(2j*np.pi*frequency_hz*relative)
    return dict(span_us=float(offsets[-1]-offsets[0]),
        phase_span_degrees=float(relative[-1]*frequency_hz*360),
        first_last_max_difference_per_unit_amplitude=float(abs(phasors[-1]-phasors[0])),
        hypothetical_equal_average_amplitude_ratio=float(abs(phasors.mean())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Output exists')
    raw = a.capture.read_bytes(); data = json.loads(raw)
    if data.get('schema') == 'ch1_fast_flank_stream_capture_v1':
        terminal_raw = bytes.fromhex(data['terminal']['wire_hex'])
        tag = int.from_bytes(terminal_raw[4:8], 'big')
        decode_flank_end(terminal_raw, tag=tag, table_crc32=data['table_codes_crc32'])
        for saved in data['frames']:
            decoded = decode_flank(bytes.fromhex(saved['wire_hex']), tag=tag, table_crc32=data['table_codes_crc32'])
            for key in ('sequence', 'cycle_start_us', 'elapsed_us', 'records'):
                if decoded[key] != saved[key]: raise ValueError('Decoded F18 metadata mismatch')
    else:
        validated_gain_identity(data)
    frames = [f for f in data['frames'] if f['sequence'] >= 80]
    if len(frames) < 32: raise ValueError('Insufficient frames')
    sequence = np.array([f['sequence'] for f in frames], dtype=np.int64)
    if np.any(np.diff(sequence) != 1): raise ValueError('Missing or duplicate frames')
    starts = np.array([f['cycle_start_us'] for f in frames], dtype=np.int64)
    intervals = np.diff(starts)
    if np.any(intervals <= 0): raise ValueError('Clock wrap or non-monotonic clock needs explicit decoding')
    times = np.array([[r['second_end_us'] for r in f['records']] for f in frames])
    frequency_rows = []
    for frequency in (1, 5, 10, 15):
        measures = [sinusoid_skew(t, frequency) for t in times]
        frequency_rows.append(dict(frequency_hz=frequency,
            median={key: float(np.median([m[key] for m in measures])) for key in measures[0]}))
    groups = np.array([r['row']//5 for r in frames[0]['records']])
    report = dict(capture_sha256=hashlib.sha256(raw).hexdigest(), evaluated_frames=len(frames),
        sample_count_per_frame=times.shape[1],
        rate_from_board_clock_hz=float(1e6/np.median(intervals)),
        cycle_interval_p95_us=float(np.percentile(intervals, 95)),
        samples_per_15hz_cycle=float(1e6/np.median(intervals)/15),
        group_second_completion_median_us=[float(np.median(times[:, groups == i])) for i in range(9)],
        hypothetical_common_sinusoid=frequency_rows,
        acquisition_completion_not_exact_adc_aperture=True,
        physical_15hz_verified=False, correction_applied=False,
        interpretation='All sampled wavelengths hypothetically carry the same unit-amplitude sinusoid. Differences arise from measured timing only; this is not observed pressure data or the real classifier transfer function. Current raw135 features discard sample offsets; static tests do not verify dynamic localization.')
    a.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
