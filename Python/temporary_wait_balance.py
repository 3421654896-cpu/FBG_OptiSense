"""Conservative ADC wait candidates from individual board transient repeats.

This is not a wavelength accuracy certificate. A held-source diagnostic differs
from a continuously scanned route; every candidate needs full-route A/B testing.
No RC correction, inferred intermediate samples or median-hidden failures.
"""

import math
import numpy as np

from temporary_test_mode import command, points_per_peak_for_rows


ADC_GROUP_MARGIN_US = 55
WAIT_QUANTUM_US = 25
MAX_WAIT_US = 15000


def _repeat_metric(record, target_level, tolerance, channel, *, reference_baseline=None):
    times = np.asarray(record['sample_start_after_target_write_us'], dtype=float)
    samples = np.asarray(record['direct_adc'], dtype=float)
    baseline = np.asarray(record['baseline_adc'], dtype=float)
    if (times.shape != (33,) or samples.shape != (33, 6) or baseline.shape != (6,)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(samples))
            or not np.all(np.isfinite(baseline)) or np.any(times < 0)
            or np.any(np.diff(times) <= 0) or times[-1] < 80000
            or np.any(samples < 0) or np.any(samples > 4095)
            or np.any(baseline < 0) or np.any(baseline > 4095)):
        raise ValueError('Invalid or incomplete board transient')
    values = samples[:, channel]
    tail = values[-3:]
    reasons = []
    # A zero calculation reference is not a fabricated raw baseline. Keep the
    # actual source ADC, but don't use it as the target's amplitude or clipping
    # criterion when the caller explicitly chooses point-relative-to-zero math.
    calculation_start = float(baseline[channel]) if reference_baseline is None else float(reference_baseline)
    if not np.isfinite(calculation_start):
        raise ValueError('Invalid calculation baseline')
    if np.any(values >= 4080) or (reference_baseline is None and baseline[channel] >= 4080):
        reasons.append('adc_clipped')
    if (np.ptp(tail) > tolerance or np.max(np.abs(tail - target_level)) > tolerance
            or times[-1] - times[-3] < 20000):
        reasons.append('tail_not_repeatably_flat')
    bad = np.flatnonzero(np.abs(values - target_level) > tolerance)
    start = int(bad[-1] + 1) if len(bad) else 0
    if start > len(times) - 3 or (start < len(times) and times[-1] - times[start] < 20000):
        reasons.append('insufficient_stable_tail')
    settle = None if reasons else float(times[start])
    return {
        'settled_after_target_write_us': settle,
        'tail_median_adc': float(np.median(tail)),
        'tail_range_adc': float(np.ptp(tail)),
        # Diagnostic values only: none of these change acceptance or waits.
        'tail_adc': [float(value) for value in tail],
        'tail_sample_times_after_target_write_us': [float(value) for value in times[-3:]],
        'tail_max_deviation_from_target_adc': float(np.max(np.abs(tail - target_level))),
        'tolerance_adc': float(tolerance),
        'tail_span_us': float(times[-1] - times[-3]),
        'observed_stable_sample_count': int(len(times) - start),
        'observed_stable_span_us': float(times[-1] - times[start]) if start < len(times) else 0.0,
        'signal_peak_adc': float(np.max(values)),
        # Even a source==target level may have a large intermediate excursion.
        'max_excursion_from_target_adc': float(np.max(np.abs(values - target_level))),
        'meaningful_step': bool(abs(target_level - calculation_start) >= 20),
        'raw_baseline_adc': float(baseline[channel]),
        'calculation_start_adc': calculation_start,
        'reasons': reasons,
    }


def analyze_wait_balance(report, current_waits):
    """Return candidates without silently shortening, capping or applying waits."""
    if not (report.get('complete') and report.get('exact_disarm_ack')
            and report.get('exact_soa_shutter_ack')):
        raise ValueError('Incomplete measurement or unconfirmed shutdown')
    route = report['route']
    count = len(route)
    group = points_per_peak_for_rows(route)
    command(route, 32, 1)  # Same lossless route and 145 mA validation as scanning.
    if (len(current_waits) != count or any(type(x) is not int or
            not 50 <= x <= MAX_WAIT_US or x % WAIT_QUANTUM_US for x in current_waits)):
        raise ValueError('Invalid current point waits')
    repeats = report['repeats']
    if type(repeats) is not int or repeats < 3:
        raise ValueError('At least three individual repeats are required')
    records = report['captures']
    if len(records) != repeats * (count - 1):
        raise ValueError('Missing forward transient repeats')
    fast = bool(report.get('fast_dac_writes', False))
    channel = report.get('signal_channel', 1)
    if type(channel) is not int or channel not in range(4):
        raise ValueError('Invalid signal channel')
    selector = report.get('feedback_selector', report.get('ch1_feedback_selector'))
    if type(selector) is not int or selector not in range(4) or (channel >= 2 and selector != 0):
        raise ValueError('Invalid feedback selector for signal channel')
    by_target = [[] for _ in route]
    for order, record in enumerate(records):
        repeat, offset = divmod(order, count - 1)
        target = offset + 1
        if (record.get('repeat') != repeat or record.get('route_source_order') != offset
                or record.get('route_target_order') != target
                or record.get('source_fullband_index') != route[offset]['index']
                or record.get('target_fullband_index') != route[target]['index']
                or bool(record.get('fast_dac_writes', False)) != fast):
            raise ValueError('Stale, reordered or DAC-style-mismatched transient')
        by_target[target].append(record)
    candidates = [current_waits[0]] + [None] * (count - 1)
    points = [None]
    for target in range(1, count):
        selected = by_target[target]
        # Do not let a quiet median conceal an outlier repeat. Its samples are
        # checked against this common target level, and it can veto a candidate.
        try:
            tails = [np.asarray(r['direct_adc'], dtype=float)[-3:, channel] for r in selected]
            target_level = float(np.median([np.median(tail) for tail in tails]))
        except (ValueError, IndexError, TypeError) as exc:
            raise ValueError('Invalid transient samples') from exc
        tolerance = max(5.0, 0.05 * abs(target_level))
        metrics = [_repeat_metric(r, target_level, tolerance, channel) for r in selected]
        reasons = sorted({reason for m in metrics for reason in m['reasons']})
        required = None
        if not reasons:
            latest = max(m['settled_after_target_write_us'] for m in metrics)
            required = max(50, math.ceil((latest + ADC_GROUP_MARGIN_US) / WAIT_QUANTUM_US)
                           * WAIT_QUANTUM_US)
            if required > MAX_WAIT_US:
                reasons.append('required_wait_exceeds_protocol_limit')
            else:
                candidates[target] = required
        points.append({
            'target_order': target,
            'target_index': route[target]['index'],
            'category': 'cross_peak' if target % group == 0 else 'within_peak',
            'target_adc': target_level,
            'tolerance_adc': tolerance,
            'required_wait_us': required,
            'candidate_wait_us': candidates[target],
            'individual_repeats': metrics,
            'reasons': reasons,
        })
    available = all(x is not None for x in candidates)
    return {
        'schema': 'temporary_adc_wait_balance_v1',
        'criterion': 'all_repeat_observed_suffix_within_max_5_codes_or_5pct_target_adc',
        'route_key': [[r['index'], list(r['codes'])] for r in route],
        'signal_channel': channel,
        'signal_channel_name': f'CH{channel}',
        'feedback_selector': selector,
        'fast_dac_writes': fast,
        'repeats': repeats,
        'adc_group_margin_us': ADC_GROUP_MARGIN_US,
        'candidate_available': available,
        'candidate_point_delays_us': candidates,
        'current_wait_sum_us': sum(current_waits),
        'candidate_wait_sum_us': sum(candidates) if available else None,
        'points': points,
        'first_point_unmeasured': True,
        'reverse_wrap_tested': False,
        'closed_loop_accuracy_verified': False,
        'absolute_wavelength_accuracy_verified': False,
        'scope': 'held-source forward transient; sampled suffix, not continuous settling proof',
    }
