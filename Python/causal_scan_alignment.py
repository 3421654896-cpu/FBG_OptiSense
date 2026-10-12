"""Experimental two-scan interpolation to an explicitly delayed common time.

No extrapolation, hardware access or model deployment. ADC completion timestamps
approximate acquisition time, not exact aperture time. Linear interpolation is
not bandwidth-neutral; qualify frequency response before use.
"""
import numpy as np


def align(previous_times_us, previous_values, current_times_us, current_values, *, max_gap_us):
    pt, pv, ct, cv = [np.asarray(v, dtype=float) for v in
                      (previous_times_us, previous_values, current_times_us, current_values)]
    if not (pt.ndim == 1 and len(pt) >= 2 and pt.shape == pv.shape == ct.shape == cv.shape):
        raise ValueError('Expected matching nonempty scan arrays')
    if not all(np.isfinite(v).all() for v in (pt, pv, ct, cv)):
        raise ValueError('Nonfinite scan')
    if not np.isfinite(max_gap_us) or max_gap_us <= 0:
        raise ValueError('Invalid gap limit')
    if np.any(np.diff(ct) <= 0) or np.any(np.diff(pt) <= 0):
        raise ValueError('Non-monotonic within-scan timestamps')
    gap = ct-pt
    if np.any(gap <= 0) or np.any(gap > max_gap_us):
        raise ValueError('Missing scan or excessive interval')
    target = float(ct[0])
    if np.any(pt > target) or np.any(ct < target):
        raise ValueError('Target not bracketed; extrapolation forbidden')
    fraction = (target-pt)/gap
    values = pv+(cv-pv)*fraction
    return dict(values=values, target_us=target,
                available_us=float(ct[-1]), inherent_delay_us=float(ct[-1]-target),
                interpolation_fraction=fraction)
