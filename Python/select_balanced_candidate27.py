"""Offline balanced-flank alternative; never smooth or replace acquired ADC data.

The broad envelope is used ONLY to choose physical 2001-table rows. No pressed
capture or position label is read. A new meter probe and held-out contacts are
required before this candidate can be used for a positioning experiment.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from analyze_current_dense9_teacher import validated_groups
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file, now_text
from probe_current_candidate_route import load_route


def select_group(rows, smoothing_nm=0.07):
    x = np.asarray([r['measured_wavelength_nm'] for r in rows], dtype=float)
    y = np.asarray([r['ch1_adc_code'] for r in rows], dtype=float)
    stable = np.asarray([r['stable'] for r in rows], dtype=bool)
    if (len(x) < 20 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0) or np.any(y < 0) or np.any(y >= 4000)
            or not 0.02 <= smoothing_nm <= 0.1):
        raise ValueError('Finite, ordered, unsaturated dense spectrum required')
    weights = np.exp(-0.5 * ((x[:, None] - x[None, :]) / smoothing_nm) ** 2)
    envelope = (weights @ y) / np.sum(weights, axis=1)
    floor = float(min(envelope[0], envelope[-1]))
    peak = int(np.argmax(envelope))
    height = float(envelope[peak] - floor)
    if height < 20 or max(envelope[0], envelope[-1]) > floor + height * 0.40:
        raise ValueError('Flat or truncated peak; acquire a wider dense reference')
    level = floor + height * 0.5
    left = next((i for i in range(peak - 1, -1, -1) if envelope[i] < level), None)
    right = next((i for i in range(peak + 1, len(x)) if envelope[i] < level), None)
    if left is None or right is None:
        raise ValueError('Both half-height flanks must be observed')
    xl = float(np.interp(level, envelope[left:left+2], x[left:left+2]))
    xr = float(np.interp(level, envelope[right-1:right+1][::-1], x[right-1:right+1][::-1]))
    width = xr - xl
    center = (xl + xr) / 2
    if width < 0.40:
        raise ValueError('Broad-envelope candidate requires at least 0.40 nm width')
    # Symmetric inner samples straddle the broad peak. Outer rows are retained
    # only for compatibility with the five-row-per-group candidate format.
    aims = [float(x[0]), center - .45 * width, center,
            center + .45 * width, float(x[-1])]
    pools = []
    for aim in aims:
        candidates = np.where(stable & (np.abs(x-aim) <= .041))[0]
        if not len(candidates):
            raise ValueError('No stable measured row within 41 pm of balanced target')
        pools.append(candidates)
    # Joint selection matters when the closest centre row is unstable: three
    # independent nearest neighbours can create an unnecessarily skewed triple.
    triples = []
    for triple in product(*pools[1:4]):
        spans = np.diff(x[list(triple)])
        if min(spans) >= .15 and abs(spans[0]-spans[1]) <= .061:
            score = float(np.sum((x[list(triple)]-aims[1:4])**2))
            triples.append((score, tuple(map(int, triple))))
    if not triples:
        raise ValueError('No stable, sufficiently separated symmetric core triple')
    chosen = [int(pools[0][np.argmin(np.abs(x[pools[0]]-aims[0]))]),
              *min(triples)[1],
              int(pools[4][np.argmin(np.abs(x[pools[4]]-aims[4]))])]
    if any(b <= a for a, b in zip(chosen, chosen[1:])):
        raise ValueError('Balanced candidate lacks five distinct ordered rows')
    spans = np.diff(x[chosen[1:4]])
    if min(spans) < .15 or abs(spans[0] - spans[1]) > .061:
        raise ValueError('Balanced core separation/symmetry gate failed')
    return chosen, dict(envelope_center_nm=center, envelope_fwhm_nm=width,
                       requested_wavelengths_nm=aims, core_spans_nm=spans.tolist(),
                       envelope_adc_codes=envelope.tolist())


def build(source, output, excluded_indices=()):
    if output.exists() or output.with_suffix('.png').exists():
        raise FileExistsError(output)
    data = json.loads(source.read_text('utf-8'))
    groups = validated_groups(data)
    excluded_indices = sorted(set(excluded_indices))
    if any(type(i) is not int or not 0 <= i <= 2000 for i in excluded_indices):
        raise ValueError('Only physical 2001-table indices can be excluded')
    candidate, summaries = [], []
    for group in groups:
        eligible_rows = [dict(r, stable=bool(r['stable'] and r['index'] not in excluded_indices))
                         for r in group['dense']]
        chosen, summary = select_group(eligible_rows)
        summary['group'] = group['group']
        summary['candidate_fullband_indices'] = [group['dense'][i]['index'] for i in chosen]
        summaries.append(summary)
        for i in chosen:
            r = group['dense'][i]
            candidate.append(dict(group=group['group'], index=r['index'],
                target_nm=r['target_wavelength_nm'], historical_measured_nm=r['measured_wavelength_nm'],
                codes=r['dac_codes'], ch1_adc_code=r['ch1_adc_code'], short_window_stable=r['stable']))
    report = dict(schema='balanced_envelope_candidate_v1', created=now_text(),
        source=str(source.resolve()), source_sha256=sha256_file(source), groups=summaries,
        candidate_rows=candidate, candidate_point_count=45, fast_core_point_count=27,
        candidate_dac_source_sha256=data['source_table_sha256'], smoothing_sigma_nm=.07,
        excluded_fullband_indices=excluded_indices,
        method='Half-height broad-envelope centre; symmetric core +/- 0.45 FWHM; nearest stable physical table rows.',
        smoothing_scope='Offline selection only; captured ADC is not smoothed or substituted.',
        requires_current_meter_calibration=True, requires_fresh_contact_validation=True,
        short_window_stability_is_not_repeatability=True, deployed=False,
        hardware_point_table_modified=False, physical_15hz_verified=False)
    save_checkpoint(output, report)
    load_route(output, 3)  # Reuse conservative table/current identity validation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), layout='constrained')
    for group, summary, ax in zip(groups, summaries, axes.flat):
        x = [r['measured_wavelength_nm'] for r in group['dense']]
        y = [r['ch1_adc_code'] for r in group['dense']]
        ax.plot(x, y, color='#a1aaba', label='Measured ADC (unchanged)')
        ax.plot(x, summary['envelope_adc_codes'], color='#3180bf', label='Selection envelope only')
        rows = [r for r in candidate if r['group'] == group['group']][1:4]
        ax.scatter([r['historical_measured_nm'] for r in rows], [r['ch1_adc_code'] for r in rows],
                   color='#d06637', label='Three physical core rows')
        ax.set_title(f"Wavelength group {group['group']}")
        ax.ticklabel_format(axis='x', useOffset=False)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    figure.supxlabel('Table wavelength (nm); fresh meter checks required')
    figure.supylabel('CH1 ADC code at 40 kohm')
    figure.savefig(output.with_suffix('.png'), dpi=130)
    plt.close(figure)
    print('candidate', sha256_file(output))
    for s in summaries:
        print(s['group'], s['candidate_fullband_indices'][1:4], 'spans', s['core_spans_nm'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-index', type=int, action='append', default=[],
                   help='Exclude a physical row rejected by independent optical checks')
    args = p.parse_args()
    build(args.source, args.output, args.exclude_index)
