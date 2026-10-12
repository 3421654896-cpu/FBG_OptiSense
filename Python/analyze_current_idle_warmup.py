"""Offline fixed-window idle drift audit, never a pressure bandwidth test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_current_three_positions import spectral_feature


def analyze(path):
    data = json.loads(path.read_text('utf-8'))
    safety = data.get('safety_cleanup', {})
    if (not data.get('complete') or not safety.get('exact_disarm_ack')
            or not safety.get('exact_soa_shutter_ack') or data.get('cnc_motion_commands') != 0):
        raise ValueError('Requires complete optical-only run with confirmed shutdown')
    frames = data['frames']
    if len(frames) != data['cycles_requested'] or any(
            f['sequence'] != i or f['i2c_error_delta'] or f['spi_error_delta']
            for i, f in enumerate(frames)):
        raise ValueError('Incomplete or erroneous board stream')
    values = np.array([[r['first_code'] for r in f['records']] for f in frames], dtype=float)
    seconds = (np.array([f['cycle_start_us'] for f in frames], dtype=float)
               - frames[0]['cycle_start_us']) / 1e6
    baseline = np.median(values[16:80], axis=0)
    tail = np.median(values[seconds >= seconds[-1]-10], axis=0)
    windows = []
    previous = None
    for start in range(0, int(seconds[-1])-4, 5):
        block = values[(seconds >= start) & (seconds < start+5)]
        if len(block) < 100:
            continue
        median = np.median(block, axis=0)
        diff = median - baseline
        windows.append(dict(start_s=start, count=len(block), median_codes=median.tolist(),
            rms_to_early_baseline_codes=float(np.sqrt(np.mean(diff**2))),
            max_to_early_baseline_codes=float(np.max(np.abs(diff))),
            per_grating_rms_to_early_codes=np.sqrt(np.mean(diff.reshape(9, 5)**2, axis=1)).tolist(),
            rms_to_last10s_codes=float(np.sqrt(np.mean((median-tail)**2))),
            rms_to_previous_window_codes=(None if previous is None else float(np.sqrt(np.mean((median-previous)**2)))),
            shape_rms_to_early=float(np.sqrt(np.mean(spectral_feature(baseline, median, 'shape_delta')**2))),
            within_window_rms_codes=float(np.sqrt(np.mean((block-median)**2)))))
        previous = median
    return dict(schema='idle_drift_audit_v1', source=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), duration_s=float(seconds[-1]),
        capture_analysis=data['analysis'], early_baseline_codes=baseline.tolist(), windows=windows,
        interpretation='Unloaded drift only; does not establish contact accuracy or 15 Hz physical response',
        deployed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    for row in result['windows']:
        print(row['start_s'], 'early/tail/within',
              *(round(row[k], 3) for k in ('rms_to_early_baseline_codes',
                  'rms_to_last10s_codes', 'within_window_rms_codes')))
