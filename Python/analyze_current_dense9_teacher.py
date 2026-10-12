"""Audit fresh dense CH1 windows; generate an offline, unqualified 45-point candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adaptive_sampling_optimizer import optimize_ch1_sampling


def validated_groups(data):
    if (not data.get('complete') or not data.get('disarm_ack') or not data.get('shutter_ack')
            or data.get('cnc_commands') != [] or data.get('dense_ch1_kohm') != 40):
        raise ValueError('Requires completed 40k optical-only run with confirmed shutdown')
    groups = data['groups']
    if [g['group'] for g in groups] != list(range(1, 10)):
        raise ValueError('All nine ordered groups required')
    for g, plan in zip(groups, data['plan']):
        if [r['index'] for r in g['dense']] != plan['indices']:
            raise ValueError('Dense index coverage incomplete')
        if len(g['paired_anchors']) != 5 or any(
                [r['selector'] for r in p['samples']] != [2,1] for p in g['paired_anchors']):
            raise ValueError('Paired gain coverage incomplete')
    return groups


def analyze(source, output):
    data = json.loads(source.read_text('utf-8'))
    groups = validated_groups(data)
    rows = [row for g in groups for row in g['dense']]
    x = np.array([r['measured_wavelength_nm'] for r in rows])
    y = np.array([r['ch1_adc_code'] for r in rows])
    noise = np.maximum(.5, [r['ch1_sigma_codes'] for r in rows])
    stable = np.array([r['stable'] for r in rows])
    regions, start = [], 0
    for g in groups:
        regions.append(slice(start, start+len(g['dense'])))
        start += len(g['dense'])
    plan = optimize_ch1_sampling(x, y, regions, points_per_peak=5,
        sample_noise_std=noise, stable_mask=stable)
    old_path = Path(__file__).resolve().parent/'artifacts/ch1_teacher_peaks_20260905_40k.json'
    old = json.loads(old_path.read_text('utf-8'))
    old_rows = old['rows']
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), layout='constrained')
    summaries, candidate = [], []
    for g, template, ax in zip(groups, plan.peaks, axes.flat):
        dense = g['dense']
        gx = np.array([r['measured_wavelength_nm'] for r in dense])
        gy = np.array([r['ch1_adc_code'] for r in dense])
        peak = int(np.argmax(gy))
        ratios = []
        anchor_x, anchor_y = [], []
        for pair in g['paired_anchors']:
            low, high = [s['sample'] for s in pair['samples']]
            if low['ch1_adc_code'] > 5:
                ratios.append(high['ch1_adc_code']/low['ch1_adc_code'])
            anchor_x.append(high['measured_wavelength_nm'])
            anchor_y.append(high['ch1_adc_code'])
        old_values = [old_rows.get(str(r['index'])) for r in dense]
        comparable = [i for i, r in enumerate(old_values) if r is not None
                      and r['dac_codes'] == dense[i]['dac_codes']]
        old_rmse = None
        if comparable:
            old_y = np.array([old_values[i]['ch1_adc_code'] for i in comparable])
            ax.plot(gx[comparable], old_y, color='#aaaeb8', label='Sep 5 same-DAC 40k')
            old_rmse = float(np.sqrt(np.mean((gy[comparable]-old_y)**2)))
        ax.plot(gx, gy, color='#2276d2', label='Fresh dense 40k')
        ax.scatter(anchor_x, anchor_y, color='#d96b24', marker='x', s=32,
                   label='Installed calibrated anchors 40k')
        chosen = list(map(int, template.selected_indices))
        ax.scatter(x[chosen], y[chosen], color='#16986e', s=25, label='Offline candidate')
        ax.set_title(f'Wavelength group {g["group"]}')
        ax.ticklabel_format(axis='x', useOffset=False)
        ax.grid(alpha=.2)
        summary = dict(group=g['group'], dense_points=len(dense),
            short_window_stable_points=sum(r['stable'] for r in dense),
            peak_table_wavelength_nm=float(gx[peak]), peak_adc_code=float(gy[peak]),
            edge_to_peak_ratios=[float(gy[0]/max(gy)), float(gy[-1]/max(gy))],
            paired_40k_over_5k_ratios=ratios,
            same_dac_historical_overlap=len(comparable),
            same_dac_historical_rms_codes=old_rmse,
            new_meter_checks=g['meter_checks'],
            candidate_fullband_indices=[rows[i]['index'] for i in chosen])
        summaries.append(summary)
        for i in chosen:
            candidate.append(dict(group=g['group'], index=rows[i]['index'],
                target_nm=rows[i]['target_wavelength_nm'], historical_measured_nm=float(x[i]),
                codes=rows[i]['dac_codes'], ch1_adc_code=float(y[i]), short_window_stable=bool(stable[i])))
    axes[0,0].legend(fontsize=7)
    figure.supxlabel('Historical table wavelength (nm); live meter checks reported separately')
    figure.supylabel('Direct CH1 ADC code at fixed 40 kOhm (no digital scaling)')
    figure.savefig(output.with_suffix('.png'), dpi=135)
    plt.close(figure)
    report = dict(schema='dense9_sampling_audit_v1', source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), groups=summaries,
        candidate_rows=candidate, candidate_point_count=len(candidate),
        candidate_dac_source_sha256=data['source_table_sha256'],
        method='Existing non-Gaussian measured-template optimizer, five points per wavelength group',
        short_window_stability_is_not_repeatability=True,
        requires_repeat_scan=True, requires_current_meter_calibration=True,
        requires_fast_route_replay=True, requires_fresh_contact_validation=True,
        hardware_point_table_modified=False, deployed=False, physical_15hz_verified=False)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    for s in summaries:
        print(s['group'], s['dense_points'], 'peak', s['peak_table_wavelength_nm'],
              s['peak_adc_code'], 'edge ratios',s['edge_to_peak_ratios'],
              'candidate',s['candidate_fullband_indices'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.png').exists():
        raise ValueError('Use new output paths')
    analyze(args.source, args.output)
