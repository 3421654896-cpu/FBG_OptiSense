"""Offline, full-route A/B assessment, separate from customer display/gates.

Relative Makima centre consistency is NOT absolute wavelength accuracy. Two
bracketing slow captures check stationary conditions; dark/protocol-only runs
and a lone 10ms sample cannot certify an accurate-fast wait configuration.
"""

import numpy as np

from temporary_peak_fit_makima import fit_temporary_frame
from temporary_test_mode import frame_adc_values, points_per_peak_for_rows


def _scope(report):
    plan = report['plan']
    identity = report.get('hardware_identity', {})
    if (not identity.get('serial_number') or type(identity.get('boot_id')) is not int
            or type(identity.get('firmware_version')) is not int
            or not report.get('stationary_setup_confirmed')):
        raise ValueError('Runtime USB identity and stationary optical setup confirmation required')
    return ([(r['index'], tuple(r['codes']), r['measured_nm']) for r in plan['rows']],
            report['signal_channel'], report['feedback_selector'],
            bool(report.get('single_sample', False)), bool(report.get('fast_dac_writes', False)),
            identity['serial_number'], identity['boot_id'], identity['firmware_version'])


def _observations(report):
    if (not report.get('complete') or not report.get('disarm_ack')
            or not report.get('shutter_ack') or report.get('error')
            or report.get('continuous')):
        raise ValueError('Full finite capture and both shutdown ACKs required')
    frames = report['frames']
    if len(frames) < 32 or report.get('received_frames') != len(frames):
        raise ValueError('Need every frame, including warmup, and >=16 measured frames')
    for sequence, frame in enumerate(frames):
        if (frame.get('sequence') != sequence or frame.get('kind') != 'data'
                or frame.get('i2c_errors') or frame.get('spi_errors')
                or frame.get('adc_channel') != report['signal_channel']
                or frame.get('feedback_selector') != report['feedback_selector']
                or frame.get('firmware') != frames[0]['firmware']
                or frame.get('firmware') != report['hardware_identity']['firmware_version']):
            raise ValueError('Invalid frame sequence, firmware, channel or bus status')
    measured = frames[16:]
    values = np.asarray([frame_adc_values(frame) for frame in measured], dtype=float)
    route = report['plan']['rows']
    if (values.shape != (len(measured), len(route)) or not np.all(np.isfinite(values))
            or np.any(values < 0) or np.any(values >= 4080)):
        raise ValueError('Missing, nonfinite or clipped ADC data')
    times = np.asarray([frame['cycle_start_us'] for frame in measured], dtype=float)
    periods = np.diff(times)
    if not np.all(np.isfinite(times)) or np.any(periods <= 0):
        raise ValueError('Invalid board cycle timestamps')
    fits = [fit_temporary_frame(route, sample) for sample in values]
    centers = np.asarray([[f.get('center_nm') if f.get('center_nm') is not None else np.nan
                           for f in group] for group in fits], dtype=float)
    shapes_valid = all(f.get('valid', False) for group in fits for f in group)
    return values, centers, shapes_valid, {
        'mean_hz': float(1e6 / np.mean(periods)),
        'p99_period_ms': float(np.percentile(periods, 99) / 1000),
        'measured_frames': len(measured),
        'firmware': measured[0]['firmware'],
    }


def compare_full_route(before, candidate, after, *, waveform_fraction=.05,
                       reference_drift_fraction=.01,
                       adc_noise_floor_codes=5.0):
    """Assess all points/all measured frames; never correct/mask the raw values."""
    if (_scope(before) != _scope(candidate) or _scope(after) != _scope(candidate)):
        raise ValueError('Reference/candidate route, coordinates, channel, gain or DAC/ADC style differ')
    return _compare_route_values(before, candidate, after, waveform_fraction,
                                 reference_drift_fraction, adc_noise_floor_codes)


def compare_pipeline_styles(before, candidate, after, *, waveform_fraction=.05,
                            reference_drift_fraction=.01,
                            adc_noise_floor_codes=5.0):
    """Qualify optimized sampling against bracketing legacy slow references.

    Only this explicit diagnostic permits a DAC/ADC style difference. Route,
    gain, channel, coordinates, board and boot remain identical; EVERY capture
    must wait 15ms at every point. It is still relative, not absolute accuracy.
    """
    old, new, final = _scope(before), _scope(candidate), _scope(after)
    if old != final or old[:3] != new[:3] or old[5:] != new[5:]:
        raise ValueError('Pipeline comparison route/identity differs or legacy references differ')
    if old[3] or old[4] or not new[3]:
        raise ValueError('Need legacy dual ADC/unmerged references and optimized single ADC candidate')
    count = len(candidate['plan']['rows'])
    waits = candidate.get('point_delays_us')
    if (not isinstance(waits, list) or len(waits) != count
            or any(type(wait) is not int or wait != 15000 for wait in waits)):
        raise ValueError('Pipeline style candidate must also use 15000us at EVERY point')
    result = _compare_route_values(before, candidate, after, waveform_fraction,
                                   reference_drift_fraction, adc_noise_floor_codes)
    result.update(schema='temporary_sampling_style_comparison_v1',
                  relative_sampling_style_consistency_passed=result['relative_accuracy_passed'],
                  reference_style=dict(single_sample=False, fast_dac_writes=False),
                  candidate_style=dict(single_sample=True, fast_dac_writes=new[4]))
    return result


def _compare_route_values(before, candidate, after, waveform_fraction,
                          reference_drift_fraction, adc_noise_floor_codes):
    for value in (waveform_fraction, reference_drift_fraction, adc_noise_floor_codes):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Accuracy criteria must be finite and positive')
    pre, pre_centers, pre_shape, pre_rate = _observations(before)
    fast, fast_centers, fast_shape, fast_rate = _observations(candidate)
    post, post_centers, post_shape, post_rate = _observations(after)
    if pre_rate['firmware'] != fast_rate['firmware'] or post_rate['firmware'] != fast_rate['firmware']:
        raise ValueError('Firmware changed between reference and candidate')
    for report in (before, after):
        waits = report.get('point_delays_us')
        if (not isinstance(waits, list) or len(waits) != pre.shape[1]
                or any(type(w) is not int or w != 15000 for w in waits)):
            raise ValueError('Slow reference must use 15000us at EVERY route point')
    med_pre, med_post = np.median(pre, axis=0), np.median(post, axis=0)
    reference = (med_pre + med_post) * .5
    tolerance = np.maximum(adc_noise_floor_codes, waveform_fraction * np.abs(reference))
    drift_tolerance = np.maximum(adc_noise_floor_codes, reference_drift_fraction * np.abs(reference))
    adc_drift = float(np.max(np.abs(med_post-med_pre) / drift_tolerance))
    ref_scatter = max(float(np.max(np.abs(pre-med_pre) / drift_tolerance)),
                      float(np.max(np.abs(post-med_post) / drift_tolerance)))
    center_pre = np.median(pre_centers, axis=0)
    center_post = np.median(post_centers, axis=0)
    centers_finite = bool(np.all(np.isfinite(pre_centers)) and
                          np.all(np.isfinite(post_centers)) and np.all(np.isfinite(fast_centers)))
    center_drift_pm = float(np.max(np.abs(center_pre-center_post)) * 1000) if centers_finite else None
    # Check the candidate against BOTH bracketing levels; an average must not
    # conceal a one-sided mismatch or drift during the comparison.
    raw_error_ratio = max(float(np.max(np.abs(fast-med_pre) / tolerance)),
                          float(np.max(np.abs(fast-med_post) / tolerance)))
    center_error_pm = (max(float(np.max(np.abs(fast_centers-center_pre))*1000),
                           float(np.max(np.abs(fast_centers-center_post))*1000))
                       if centers_finite else None)
    reasons = []
    if not (pre_shape and post_shape): reasons.append('reference_peak_shape_unresolved')
    if not fast_shape: reasons.append('candidate_peak_shape_unresolved')
    if adc_drift > 1 or ref_scatter > 1: reasons.append('reference_moving_or_noisy')
    # Centre differences remain reported, never compared against a pm cutoff.
    if center_drift_pm is None:
        reasons.append('reference_center_unresolved')
    if raw_error_ratio > 1: reasons.append('raw_waveform_differs_from_slow_reference')
    if center_error_pm is None:
        reasons.append('candidate_center_unresolved')
    route = candidate['plan']['rows']
    group = points_per_peak_for_rows(route)
    return {
        'schema': 'temporary_full_route_wait_comparison_v1',
        'relative_accuracy_passed': not reasons,
        'absolute_wavelength_accuracy_verified': False,
        'criteria': {'waveform_fraction': waveform_fraction, 'center_difference_is_report_only': True,
                     'reference_drift_fraction': reference_drift_fraction,
                     'adc_noise_floor_codes': adc_noise_floor_codes,
                     'scope': 'relative ADC/Makima consistency, all post-warmup frames, no correction'},
        'reference_adc_drift_tolerance_ratio': adc_drift,
        'reference_scatter_tolerance_ratio': ref_scatter,
        'reference_center_drift_pm': center_drift_pm,
        'max_raw_waveform_tolerance_ratio': raw_error_ratio,
        'max_relative_center_error_pm': center_error_pm,
        'rate': fast_rate,
        'point_delays_us': candidate['point_delays_us'],
        'point_count': len(route), 'points_per_peak': group,
        'first_and_wrap_samples_included': True,
        'reasons': reasons,
    }


def choose_fastest_comparison(comparisons):
    eligible = [c for c in comparisons if c['relative_accuracy_passed']]
    # Prefer measured P99 throughput, not a high mean that hides long stalls.
    return max(eligible, key=lambda c: min(c['rate']['mean_hz'],
               1000 / c['rate']['p99_period_ms'])) if eligible else None
