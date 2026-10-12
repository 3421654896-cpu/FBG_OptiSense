"""Compare saved RAW benchmark spectra without changing acquisition data."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    batch = json.loads(args.batch.read_text(encoding='utf-8'))
    br, nr = baseline['rows'], batch['rows']
    if [r['index'] for r in br] != [r['index'] for r in nr]:
        raise ValueError('Physical indices differ')
    if [r['dac_codes'] for r in br] != [r['dac_codes'] for r in nr]:
        raise ValueError('Requested machine DAC tables differ')
    x = np.array([r['measured_wavelength_nm'] for r in br])
    old = np.array([r['signal_adc_code'] for r in br])
    new = np.array([r['signal_adc_code'] for r in nr])
    # Smoothing is ONLY for identifying diagnostic peak neighborhoods;
    # plots, records and metrics retain the original unsmoothed ADC samples.
    peaks, _ = find_peaks(gaussian_filter1d(old, 1),
                         prominence=max(25, .04*np.ptp(old)), distance=70)
    peaks = sorted(peaks, key=lambda i: old[i], reverse=True)[:9]
    peaks = sorted(peaks)
    peak_checks = []
    for p in peaks:
        lo, hi = max(0, p-30), min(len(x), p+31)
        old_p = lo + np.argmax(old[lo:hi])
        new_p = lo + np.argmax(new[lo:hi])
        peak_checks.append(dict(baseline_raw_apex_nm=float(x[old_p]),
                                batch_raw_apex_nm=float(x[new_p]),
                                raw_apex_difference_pm=float((x[new_p]-x[old_p])*1000),
                                baseline_raw_max_adc=float(old[old_p]),
                                batch_raw_max_adc=float(new[new_p])))
    result = dict(points=len(x), indices_identical=True, requested_dac_codes_identical=True,
                   baseline_s=baseline['acquisition_elapsed_s'], batch_s=batch['acquisition_elapsed_s'],
                   speedup=baseline['acquisition_elapsed_s']/batch['acquisition_elapsed_s'],
                   requested_dwell_s=batch['requested_point_dwell_s'],
                   batch_board_s=batch['board_acquisition_elapsed_s'],
                   raw_correlation=float(np.corrcoef(old, new)[0, 1]),
                   raw_rms_difference_adc=float(np.sqrt(np.mean((old-new)**2))),
                   diagnostic_peak_apices=peak_checks,
                   optical_accuracy_certified=False,
                   note='Sequential scans, not simultaneous measurements. Raw apex differences are not a wavelength calibration acceptance test.',
                   exact_shutdown_confirmed=all(r['disarm_ack'] and r['shutter_ack'] for r in (baseline, batch)))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    axes[0].plot(x, old, label=f"Host per-point: {result['baseline_s']:.3f}s", linewidth=.8, alpha=.7)
    axes[0].plot(x, new, label=f"Board batch: {result['batch_s']:.3f}s", linewidth=.8, alpha=.7)
    axes[0].set_ylabel('CH1 raw ADC code')
    axes[0].legend()
    axes[0].set_title('Machine 2: 2001 real points, full 2 ms dwell, same DAC table')
    axes[1].plot(x, new-old, linewidth=.7)
    axes[1].set_ylabel('Batch - host\nADC code')
    axes[1].set_xlabel('Meter-calibrated wavelength (nm)')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=130)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
