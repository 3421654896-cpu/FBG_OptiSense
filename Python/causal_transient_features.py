"""Causal CH1 transient features. No teacher, row identity or CNC labels as input.

Diagnostic six-channel acquisition differs from production sampling. These
features require independent path-matched validation before deployment.
"""
import math


def extract(curve, horizon_us):
    if not math.isfinite(horizon_us) or horizon_us <= 0:
        raise ValueError('Positive finite horizon required')
    eligible = []
    last_end = -1
    for sample in curve:
        start, end = sample['start_us'], sample['end_us']
        if not last_end < start < end:
            raise ValueError('Nonmonotonic conversion intervals')
        last_end = end
        if end > horizon_us:
            continue
        adc = sample['direct_adc']
        if len(adc) != 6 or any(type(v) is not int or not 0 <= v <= 4095 for v in adc):
            raise ValueError('Invalid ADC sample')
        eligible.append(sample)
    if len(eligible) < 3:
        raise ValueError('At least three completed samples required')
    first, mid, last = eligible[0], eligible[len(eligible)//2], eligible[-1]
    result = {}
    for channel, name in ((1, 'ch1'), (4, 'pdt'), (5, 'pdr')):
        a, b, c = (s['direct_adc'][channel] for s in (first, mid, last))
        result.update({name+'_first': a, name+'_middle': b, name+'_last': c,
                       name+'_early_slope_codes_per_us': (b-a)/(mid['end_us']-first['end_us']),
                       name+'_late_slope_codes_per_us': (c-b)/(last['end_us']-mid['end_us'])})
    result.update(first_end_us=first['end_us'], middle_end_us=mid['end_us'],
                  available_us=last['end_us'], completed_samples=len(eligible))
    return result
