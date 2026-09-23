"""Fast, segment-aware FBG peak fitting and display helpers.

The laser table is made of short, continuous wavelength sections separated by
large gaps.  Each section contains one FBG reflection peak.  Treating those
sections independently prevents a fitting routine from inventing data across
unscanned wavelength gaps.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, peak_widths


@dataclass
class FBGPeakFit:
    center_nm: float = np.nan
    amplitude_v: float = 0.0
    display_amplitude_v: float = np.nan
    sigma_nm: float = np.nan
    baseline_v: float = 0.0
    display_baseline_v: float = np.nan
    baseline_slope_v_per_nm: float = 0.0
    x_reference_nm: float = 0.0
    r_squared: float = 0.0
    rmse_v: float = np.inf
    center_std_pm: float = np.inf
    valid: bool = False

    def baseline(self, wavelength_nm):
        x = np.asarray(wavelength_nm, dtype=float)
        return self.baseline_v + self.baseline_slope_v_per_nm * (
            x - self.x_reference_nm
        )

    def evaluate(self, wavelength_nm):
        x = np.asarray(wavelength_nm, dtype=float)
        baseline = self.baseline(x)
        if not np.isfinite(self.center_nm) or not np.isfinite(self.sigma_nm):
            return baseline
        gaussian = self.amplitude_v * np.exp(
            -0.5 * ((x - self.center_nm) / self.sigma_nm) ** 2
        )
        return baseline + gaussian


@dataclass(frozen=True)
class DenseFBGSpectrumFit:
    """Automatic peak-fit result for a continuous dense reflection spectrum."""

    fits: tuple[FBGPeakFit, ...]
    candidate_count: int
    rejected_count: int
    prominence_threshold_v: float
    noise_sigma_v: float


def build_segments(point_count, gap_after):
    """Convert indices after which a gap occurs into inclusive/exclusive slices."""
    if point_count <= 0:
        return []
    starts = [0] + [int(index) + 1 for index in sorted(gap_after)]
    ends = [int(index) + 1 for index in sorted(gap_after)] + [point_count]
    return [slice(start, end) for start, end in zip(starts, ends) if end > start]


def detect_wavelength_gaps(wavelength_nm, minimum_scale_ratio=3.0):
    """Find real scan discontinuities without splitting nonuniform peak samples.

    The sampling step inside a peak is allowed to vary.  True unscanned gaps form
    a separate, much larger step-size population, so the split threshold is taken
    from the largest multiplicative jump between sorted positive step sizes.
    """
    values = np.asarray(wavelength_nm, dtype=float)
    if values.size < 2:
        return set()

    steps = np.diff(values)
    gap_after = set(np.flatnonzero(steps < 0).tolist())
    positive = np.sort(steps[steps > 0])
    if positive.size < 2:
        return gap_after

    ratios = positive[1:] / positive[:-1]
    split_index = int(np.argmax(ratios))
    if float(ratios[split_index]) < float(minimum_scale_ratio):
        return gap_after

    threshold = float(np.sqrt(positive[split_index] * positive[split_index + 1]))
    gap_after.update(np.flatnonzero(steps >= threshold).tolist())
    return gap_after


def _invalid_fit(x, y):
    x_ref = float(np.mean(x)) if len(x) else 0.0
    baseline = float(np.median(y)) if len(y) else 0.0
    return FBGPeakFit(baseline_v=baseline, x_reference_nm=x_ref)


def repair_isolated_outliers(values, minimum_absolute_v=0.005):
    """Replace a one-point switching glitch without smoothing a real peak."""
    source = np.asarray(values, dtype=float)
    repaired = source.copy()
    if source.size < 5:
        return repaired
    full_span = float(np.ptp(source))
    threshold = max(0.30 * full_span, float(minimum_absolute_v))
    for index in range(1, source.size - 1):
        is_isolated_extreme = (
            source[index] < min(source[index - 1], source[index + 1])
            or source[index] > max(source[index - 1], source[index + 1])
        )
        if not is_isolated_extreme:
            continue
        left = source[max(0, index - 2):index]
        right = source[index + 1:min(source.size, index + 3)]
        neighbors = np.concatenate((left, right))
        expected = float(np.median(neighbors))
        if abs(source[index] - expected) > threshold:
            repaired[index] = expected
    return repaired


def _fit_monotonic_gaussian(
    x,
    y,
    initial_center,
    initial_sigma,
    allow_edge_peak=False,
):
    """Fit baseline + one Gaussian using all points and a vectorized grid.

    The baseline is deliberately constant: the resulting model has exactly one
    summit, is monotonic rising to its left, and monotonic falling to its right.
    Two vectorized grid stages avoid a slow nonlinear optimizer in every frame.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(x[-1] - x[0])
    step = float(np.median(np.diff(x)))

    if allow_edge_peak:
        # A rapidly strained grating may leave a scan section between two
        # frames.  Keep the last observable half-peak useful by allowing its
        # summit to sit on either section boundary.
        center_low, center_high = float(x[0]), float(x[-1])
    else:
        center_low = max(float(x[1]), float(initial_center - 2.5 * step))
        center_high = min(float(x[-2]), float(initial_center + 2.5 * step))
        if center_high <= center_low:
            center_low, center_high = float(x[1]), float(x[-2])
    sigma_low = max(0.65 * step, min(float(initial_sigma) * 0.45, 0.25 * span))
    sigma_high = min(
        1.5 * span,
        max(float(initial_sigma) * 1.8, 2.0 * step, 0.45 * span),
    )

    def solve(centers, sigmas):
        center_grid, sigma_grid = np.meshgrid(centers, sigmas, indexing="ij")
        center_values = center_grid.ravel()
        sigma_values = sigma_grid.ravel()
        gaussian = np.exp(
            -0.5
            * ((x[None, :] - center_values[:, None]) / sigma_values[:, None]) ** 2
        )
        gaussian_mean = np.mean(gaussian, axis=1)
        centered_gaussian = gaussian - gaussian_mean[:, None]
        centered_y = y - float(np.mean(y))
        denominator = np.sum(centered_gaussian**2, axis=1)
        amplitude = (centered_gaussian @ centered_y) / np.maximum(
            denominator, 1e-15
        )
        baseline = float(np.mean(y)) - amplitude * gaussian_mean
        predicted = baseline[:, None] + amplitude[:, None] * gaussian
        residual = y[None, :] - predicted
        sse = np.sum(residual**2, axis=1)
        sse[amplitude <= 0.0] = np.inf
        best = int(np.argmin(sse))
        return (
            float(center_values[best]),
            float(sigma_values[best]),
            float(amplitude[best]),
            float(baseline[best]),
            predicted[best],
            float(sse[best]),
        )

    coarse_centers = np.linspace(center_low, center_high, 61)
    coarse_sigmas = np.linspace(sigma_low, sigma_high, 24)
    coarse = solve(coarse_centers, coarse_sigmas)
    coarse_step = max(float(coarse_centers[1] - coarse_centers[0]), step / 100.0)

    refined_centers = np.linspace(
        max(center_low, coarse[0] - 1.5 * coarse_step),
        min(center_high, coarse[0] + 1.5 * coarse_step),
        31,
    )
    refined_sigmas = np.linspace(
        max(sigma_low, coarse[1] * 0.80),
        min(sigma_high, coarse[1] * 1.20),
        17,
    )
    return solve(refined_centers, refined_sigmas)


def fit_fbg_segment(
    wavelength_nm,
    voltage_v,
    min_prominence_v=0.02,
    allow_edge_peak=False,
):
    """Locate a peak with a fast robust quadratic fit around its summit.

    A local quadratic is the standard sub-sample peak locator for a smooth,
    broad FBG peak.  It is deterministic and much faster than running a
    multi-parameter nonlinear optimizer for every channel and scan section.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(voltage_v, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5 or np.any(np.diff(x) <= 0):
        return _invalid_fit(x, y)

    span = float(x[-1] - x[0])
    y_range = float(np.ptp(y))
    min_prominence_v = float(min_prominence_v)
    if span <= 0.0 or y_range < min_prominence_v:
        return _invalid_fit(x, y)

    y = repair_isolated_outliers(y)
    y_range = float(np.ptp(y))

    typical_step = float(np.median(np.diff(x)))
    peak_index = int(np.argmax(y))
    edge_peak = peak_index < 2 or peak_index > x.size - 3
    if edge_peak and not allow_edge_peak:
        return _invalid_fit(x, y)

    edge_baseline = float(np.median(np.partition(y, min(2, y.size - 1))[:3]))
    if edge_peak:
        # There are not enough points for a two-sided quadratic.  Seed the
        # one-summit Gaussian directly from the boundary sample instead.
        center = float(x[peak_index])
        amplitude = max(float(y[peak_index]) - edge_baseline, 0.0)
        sigma = float(np.clip(0.35 * span, 2.0 * typical_step, span))
    else:
        half_window = 3
        start = max(0, peak_index - half_window)
        stop = min(x.size, peak_index + half_window + 1)
        if stop - start < 5:
            return _invalid_fit(x, y)
        local_x = x[start:stop]
        local_y = y[start:stop]
        x_origin = float(x[peak_index])
        dx = local_x - x_origin
        design = np.column_stack((dx * dx, dx, np.ones_like(dx)))

        try:
            coefficients = np.linalg.lstsq(design, local_y, rcond=None)[0]
            for _ in range(2):
                residual = local_y - design @ coefficients
                mad = float(np.median(np.abs(residual - np.median(residual))))
                robust_scale = max(1.4826 * mad, 0.0004)
                normalized = np.abs(residual) / (1.5 * robust_scale)
                weights = np.ones_like(normalized)
                large = normalized > 1.0
                weights[large] = 1.0 / normalized[large]
                weighted_design = design * np.sqrt(weights)[:, None]
                weighted_y = local_y * np.sqrt(weights)
                coefficients = np.linalg.lstsq(
                    weighted_design, weighted_y, rcond=None
                )[0]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return _invalid_fit(x, y)

        quadratic, linear, constant = coefficients
        if quadratic >= -1e-12:
            return _invalid_fit(x, y)
        vertex_offset = float(-linear / (2.0 * quadratic))
        center = x_origin + vertex_offset
        if center < local_x[0] or center > local_x[-1]:
            return _invalid_fit(x, y)

        peak_voltage = float(
            quadratic * vertex_offset**2 + linear * vertex_offset + constant
        )
        amplitude = max(peak_voltage - edge_baseline, 0.0)
        sigma = float(
            np.clip(
                np.sqrt(max(-amplitude / (2.0 * quadratic), 0.0)),
                typical_step * 0.6,
                span,
            )
        )

    try:
        (
            center,
            sigma,
            amplitude,
            edge_baseline,
            fitted_full,
            residual_sum_squares,
        ) = _fit_monotonic_gaussian(
            x,
            y,
            center,
            sigma,
            allow_edge_peak=allow_edge_peak,
        )
        residual = y - fitted_full
        total_sum_squares = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual_sum_squares / max(total_sum_squares, 1e-12)
        rmse = float(np.sqrt(np.mean(residual**2)))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return _invalid_fit(x, y)

    center_std_pm = float(
        np.clip(
            sigma * rmse / max(amplitude, 1e-9)
            * np.sqrt(2.0 / max(float(x.size), 1.0))
            * 1000.0,
            0.2,
            120.0,
        )
    )

    edge_margin = max(0.08 * span, typical_step * 0.5)
    signal_to_error = amplitude / max(rmse, 1e-6)
    center_is_observable = (
        x[0] <= center <= x[-1]
        if allow_edge_peak
        else x[0] + edge_margin <= center <= x[-1] - edge_margin
    )
    # A strong FBG reflection can be visibly asymmetric or flattened by the
    # analogue path and therefore miss the strict Gaussian residual ratio by a
    # small amount.  Presence and centre are still trustworthy when the peak
    # excursion is large, R² is at least 0.60 and the estimated centre remains
    # bounded.  Keep the original strict gate for weak bumps so ADC noise is
    # never promoted to an extra grating.
    strong_peak = amplitude >= max(0.15, 5.0 * min_prominence_v)
    shape_quality_ok = (
        r_squared >= 0.80 and signal_to_error >= 5.0
    ) or (
        strong_peak and r_squared >= 0.60 and signal_to_error >= 4.0
    )
    valid = bool(
        amplitude >= min_prominence_v
        and center_is_observable
        and shape_quality_ok
        and center_std_pm <= 120.0
    )
    x_ref = float(np.mean(x))
    display_baseline = max(
        0.0,
        float(np.median(np.partition(y, min(2, y.size - 1))[:3])),
    )
    display_amplitude = min(
        float(amplitude),
        max(float(np.max(y)) + 0.15 * y_range - display_baseline, 0.0),
    )
    return FBGPeakFit(
        center_nm=float(center),
        amplitude_v=amplitude,
        display_amplitude_v=display_amplitude,
        sigma_nm=sigma,
        baseline_v=edge_baseline,
        display_baseline_v=display_baseline,
        baseline_slope_v_per_nm=0.0,
        x_reference_nm=x_ref,
        r_squared=float(r_squared),
        rmse_v=rmse,
        center_std_pm=center_std_pm,
        valid=valid,
    )


def fit_channel_segments(
    wavelength_nm,
    voltage_v,
    segments,
    min_prominence_v=0.02,
    allow_edge_peak=False,
):
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(voltage_v, dtype=float)
    return [
        fit_fbg_segment(
            x[segment],
            y[segment],
            min_prominence_v,
            allow_edge_peak=allow_edge_peak,
        )
        for segment in segments
    ]


def fit_dense_reflection_spectrum(
    wavelength_nm,
    voltage_v,
    *,
    minimum_prominence_v=0.003,
    minimum_separation_nm=0.30,
    maximum_peaks=64,
):
    """Detect and precisely fit every resolvable FBG peak in a dense scan.

    Detection uses a noise-adaptive prominence threshold.  Each candidate is
    then isolated between neighbouring valleys and fitted by the existing
    monotonic one-summit Gaussian model, yielding a sub-sample centre without
    smoothing the entire measured spectrum.
    """

    x = np.asarray(wavelength_nm, dtype=float).reshape(-1)
    y = np.asarray(voltage_v, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 9 or y.size != x.size or np.any(np.diff(x) <= 0.0):
        return DenseFBGSpectrumFit((), 0, 0, float(minimum_prominence_v), np.nan)

    repaired = repair_isolated_outliers(y, minimum_absolute_v=0.002)
    differences = np.diff(repaired)
    difference_median = float(np.median(differences))
    difference_mad = float(
        np.median(np.abs(differences - difference_median))
    )
    # Adjacent-point differences contain the noise of two samples.
    noise_sigma = 1.4826 * difference_mad / math.sqrt(2.0)
    if not math.isfinite(noise_sigma):
        noise_sigma = 0.0
    prominence = max(float(minimum_prominence_v), 7.0 * noise_sigma)

    typical_step = float(np.median(np.diff(x)))
    minimum_distance = max(
        3, int(round(float(minimum_separation_nm) / typical_step))
    )
    candidates, properties = find_peaks(
        repaired,
        prominence=prominence,
        distance=minimum_distance,
        width=2.0,
    )
    if candidates.size == 0:
        return DenseFBGSpectrumFit((), 0, 0, prominence, noise_sigma)

    if candidates.size > int(maximum_peaks):
        strongest = np.argsort(properties["prominences"])[-int(maximum_peaks):]
        candidates = np.sort(candidates[strongest])
    widths = peak_widths(repaired, candidates, rel_height=0.70)[0]

    accepted = []
    for candidate_number, (peak_index, width_points) in enumerate(
        zip(candidates, widths, strict=True)
    ):
        centre_seed = float(x[peak_index])
        half_span_nm = max(0.22, 2.2 * float(width_points) * typical_step)
        left_nm = centre_seed - half_span_nm
        right_nm = centre_seed + half_span_nm
        if candidate_number > 0:
            left_nm = max(
                left_nm,
                0.5 * (float(x[candidates[candidate_number - 1]]) + centre_seed),
            )
        if candidate_number + 1 < candidates.size:
            right_nm = min(
                right_nm,
                0.5 * (centre_seed + float(x[candidates[candidate_number + 1]])),
            )
        start = max(0, int(np.searchsorted(x, left_nm, side="left")))
        stop = min(x.size, int(np.searchsorted(x, right_nm, side="right")))
        if stop - start < 9:
            half_points = 5
            start = max(0, int(peak_index) - half_points)
            stop = min(x.size, int(peak_index) + half_points + 1)
        fit = fit_fbg_segment(
            x[start:stop],
            repaired[start:stop],
            min_prominence_v=max(float(minimum_prominence_v), 0.55 * prominence),
            allow_edge_peak=False,
        )
        if fit.valid:
            accepted.append(fit)

    accepted.sort(key=lambda item: item.center_nm)
    return DenseFBGSpectrumFit(
        fits=tuple(accepted),
        candidate_count=int(candidates.size),
        rejected_count=int(candidates.size - len(accepted)),
        prominence_threshold_v=prominence,
        noise_sigma_v=noise_sigma,
    )


def mask_unconnected_fbg_segments(
    fits,
    channel,
    temperature_mode=False,
    stress_ch1_all_segments=False,
    stress_all_segments=False,
):
    """Reject optical sections that are not connected to this ADC channel.

    The legacy stress routing has CH2 on peaks 0..2, CH0 on 3..5 and CH1 on
    6..8.  A locally acquired adaptive spectrum must instead set
    ``stress_all_segments`` so every active optical channel can report every
    peak that is actually present.  The legacy default remains available for
    old remote frames whose physical routing is not self-describing.
    """
    if temperature_mode:
        # Precision mode scans the complete table.  CH0 and CH1 independently
        # keep every section whose own fit is valid; CH2/CH3 are not sampled.
        allowed = (
            range(0, len(fits)) if int(channel) in (0, 1) else range(0, 0)
        )
    elif stress_all_segments:
        allowed = range(0, len(fits)) if int(channel) in (0, 1, 2, 3) else range(0, 0)
    else:
        allowed = {
            2: range(0, 3),
            0: range(3, 6),
            1: range(0, len(fits)) if stress_ch1_all_segments else range(6, 9),
        }.get(int(channel), range(0, 0))
    allowed = set(allowed)
    for index, fit in enumerate(fits):
        if index not in allowed:
            fit.valid = False
    return fits


def required_display_gain(
    baseline_v,
    amplitude_v,
    *,
    hardware_is_maximum,
    target_peak_v=1.05,
    maximum_gain=64.0,
    minimum_signal_v=0.003,
):
    """Return a display-only gain that lifts a real peak above ``target_peak_v``.

    Hardware gain always has priority.  A 20 kOhm channel therefore returns
    unity gain; digital gain is permitted only when the front end is already at
    its maximum 40 kOhm setting.  Very small amplitudes are treated as missing
    signal so ADC noise is never promoted into a one-volt-looking peak.
    """
    if not hardware_is_maximum:
        return 1.0
    baseline = float(baseline_v)
    amplitude = float(amplitude_v)
    if not np.isfinite(baseline) or not np.isfinite(amplitude):
        return 1.0
    if amplitude < float(minimum_signal_v):
        return 1.0
    required = (float(target_peak_v) - baseline) / amplitude
    return float(np.clip(required, 1.0, float(maximum_gain)))


class AdaptivePeakTracker:
    """Reduce static jitter while immediately following a real wavelength step."""

    def __init__(
        self,
        peak_count,
        slow_alpha=0.12,
        fast_alpha=0.88,
        transient_threshold_nm=0.090,
        confirmation_threshold_nm=0.025,
        immediate_response=False,
    ):
        self.peak_count = int(peak_count)
        self.slow_alpha = float(slow_alpha)
        self.fast_alpha = float(fast_alpha)
        self.transient_threshold_nm = float(transient_threshold_nm)
        self.confirmation_threshold_nm = float(confirmation_threshold_nm)
        self.immediate_response = bool(immediate_response)
        self.reset()

    def reset(self):
        self.centers_nm = np.full(self.peak_count, np.nan, dtype=float)
        self.noise_nm = np.full(self.peak_count, 0.002, dtype=float)
        self.fast_hold = np.zeros(self.peak_count, dtype=np.int16)
        self.missing_frames = np.zeros(self.peak_count, dtype=np.int16)
        self.candidate_centers_nm = np.full(self.peak_count, np.nan, dtype=float)
        self.best_quality = np.full(self.peak_count, np.nan, dtype=float)
        self.quality_rejection_frames = np.zeros(self.peak_count, dtype=np.int16)
        self.measurement_accepted = np.zeros(self.peak_count, dtype=bool)

    def update(self, fits):
        self.measurement_accepted[:] = False
        output = self.centers_nm.copy()
        for index in range(self.peak_count):
            fit = fits[index] if index < len(fits) else None
            measurement = (
                float(fit.center_nm)
                if fit is not None and fit.valid and np.isfinite(fit.center_nm)
                else np.nan
            )
            if not np.isfinite(measurement):
                self.missing_frames[index] += 1
                self.candidate_centers_nm[index] = np.nan
                if self.missing_frames[index] > 30:
                    self.centers_nm[index] = np.nan
                    self.best_quality[index] = np.nan
                    self.quality_rejection_frames[index] = 0
                output[index] = self.centers_nm[index]
                continue

            self.missing_frames[index] = 0
            if not np.isfinite(self.centers_nm[index]):
                self.centers_nm[index] = measurement
                self.best_quality[index] = float(fit.r_squared)
                self.quality_rejection_frames[index] = 0
                self.measurement_accepted[index] = True
                output[index] = measurement
                continue

            innovation = measurement - self.centers_nm[index]
            quality = float(fit.r_squared)
            if self.immediate_response:
                # Stress is the signal, not an outlier.  Publish every valid
                # current-frame estimate immediately: no two-frame candidate,
                # no quality-based hold, and no interpolation with the past.
                self.noise_nm[index] = float(
                    np.clip(
                        0.92 * self.noise_nm[index] + 0.08 * abs(innovation),
                        0.0005,
                        0.050,
                    )
                )
                self.centers_nm[index] = measurement
                self.best_quality[index] = quality
                self.candidate_centers_nm[index] = np.nan
                self.quality_rejection_frames[index] = 0
                self.fast_hold[index] = 0
                self.measurement_accepted[index] = True
                output[index] = measurement
                continue
            previous_quality = self.best_quality[index]
            quality_jump = bool(
                np.isfinite(previous_quality)
                and quality >= previous_quality + 0.060
            )
            if (
                np.isfinite(previous_quality)
                and quality < previous_quality - 0.060
            ):
                # Some weak CH2 scans alternate between a well-formed peak and
                # a visibly distorted optical state.  Prefer the recent clean
                # shape, but decay the gate so a genuine long-lived change can
                # be reacquired in a few frames.
                self.quality_rejection_frames[index] += 1
                if self.quality_rejection_frames[index] <= 15:
                    output[index] = self.centers_nm[index]
                    continue
                # If the clean shape truly disappears, allow a new baseline
                # after about two seconds instead of holding forever.
                previous_quality = np.nan
            self.quality_rejection_frames[index] = 0
            self.best_quality[index] = (
                quality if not np.isfinite(previous_quality)
                else max(quality, previous_quality)
            )
            fit_uncertainty_nm = float(
                np.clip(fit.center_std_pm * 0.001, 0.0, 0.025)
            )
            threshold = max(
                self.transient_threshold_nm,
                4.0 * self.noise_nm[index],
                min(0.050, 2.0 * fit_uncertainty_nm),
            )
            if quality_jump or abs(innovation) >= threshold:
                alpha = 1.0
                self.fast_hold[index] = 2
                self.candidate_centers_nm[index] = np.nan
            elif abs(innovation) >= max(
                self.confirmation_threshold_nm,
                5.0 * self.noise_nm[index],
                min(0.050, 2.0 * fit_uncertainty_nm),
            ):
                candidate = self.candidate_centers_nm[index]
                confirmation_tolerance = max(
                    0.015,
                    2.0 * self.noise_nm[index],
                    min(0.040, 2.0 * fit_uncertainty_nm),
                )
                if (
                    np.isfinite(candidate)
                    and abs(measurement - candidate) <= confirmation_tolerance
                ):
                    alpha = 1.0
                    self.fast_hold[index] = 2
                    self.candidate_centers_nm[index] = np.nan
                else:
                    # One-frame 25--90 pm excursions are usually a distorted
                    # weak peak.  Keep them as a candidate; a real load that
                    # persists into the next 133 ms frame is accepted quickly.
                    self.candidate_centers_nm[index] = measurement
                    output[index] = self.centers_nm[index]
                    continue
            else:
                # A clean peak below the outlier-confirmation threshold is the
                # current physical result, so publish it in this same frame.
                alpha = 1.0
                self.candidate_centers_nm[index] = np.nan
                self.noise_nm[index] = float(
                    np.clip(
                        0.92 * self.noise_nm[index] + 0.08 * abs(innovation),
                        0.0005,
                        0.020,
                    )
                )

            self.centers_nm[index] += alpha * innovation
            self.measurement_accepted[index] = True
            output[index] = self.centers_nm[index]
        return output


class TemperaturePeakTracker:
    """High-stability tracker for slowly varying temperature measurements.

    Temperature changes are slow compared with the one-hertz scan.  A robust
    seven-frame center estimate and a 1 pm deadband suppress optical/ADC jitter
    while still publishing once per completed precision spectrum.
    """

    def __init__(self, peak_count, window=7, deadband_pm=1.0, tracking_alpha=0.30):
        self.peak_count = int(peak_count)
        self.window = int(window)
        self.deadband_nm = float(deadband_pm) * 0.001
        self.tracking_alpha = float(tracking_alpha)
        self.reset()

    def reset(self):
        self.centers_nm = np.full(self.peak_count, np.nan, dtype=float)
        self.histories = [deque(maxlen=self.window) for _ in range(self.peak_count)]
        self.measurement_accepted = np.zeros(self.peak_count, dtype=bool)
        self.missing_frames = np.zeros(self.peak_count, dtype=np.int16)

    def update(self, fits):
        self.measurement_accepted[:] = False
        output = self.centers_nm.copy()
        for index in range(self.peak_count):
            fit = fits[index] if index < len(fits) else None
            measurement = (
                float(fit.center_nm)
                if fit is not None and fit.valid and np.isfinite(fit.center_nm)
                else np.nan
            )
            if not np.isfinite(measurement):
                self.missing_frames[index] += 1
                if self.missing_frames[index] > 10:
                    self.centers_nm[index] = np.nan
                    self.histories[index].clear()
                output[index] = self.centers_nm[index]
                continue

            self.missing_frames[index] = 0
            history = self.histories[index]
            history.append(measurement)
            values = np.asarray(history, dtype=float)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            tolerance = max(0.003, 3.0 * 1.4826 * mad)
            inliers = values[np.abs(values - median) <= tolerance]
            estimate = float(np.mean(inliers)) if inliers.size else median

            if not np.isfinite(self.centers_nm[index]):
                self.centers_nm[index] = estimate
            else:
                innovation = estimate - self.centers_nm[index]
                if abs(innovation) > self.deadband_nm:
                    alpha = 0.55 if abs(innovation) >= 0.020 else self.tracking_alpha
                    effective = innovation - np.sign(innovation) * self.deadband_nm
                    self.centers_nm[index] += alpha * effective

            self.measurement_accepted[index] = True
            output[index] = self.centers_nm[index]
        return output


class PeakDisplayNormalizer:
    """Match low-gain CH2/CH3 peak heights to the high-gain reference channel."""

    def __init__(self, channel_count=4, smoothing=0.08, max_scale=40.0):
        self.channel_count = int(channel_count)
        self.smoothing = float(smoothing)
        self.max_scale = float(max_scale)
        self.reset()

    def reset(self):
        self.scales = np.ones(self.channel_count, dtype=float)
        self.initialized = np.zeros(self.channel_count, dtype=bool)
        self.initialized[: min(2, self.channel_count)] = True

    @staticmethod
    def _typical_amplitude(fits):
        amplitudes = [
            (
                fit.display_amplitude_v
                if np.isfinite(fit.display_amplitude_v)
                else fit.amplitude_v
            )
            for fit in fits
            if fit.valid
            and (
                fit.display_amplitude_v
                if np.isfinite(fit.display_amplitude_v)
                else fit.amplitude_v
            ) > 0
        ]
        return float(np.median(amplitudes)) if amplitudes else np.nan

    def update(self, channel_fits):
        amplitudes = [self._typical_amplitude(fits) for fits in channel_fits]
        high_gain = [value for value in amplitudes[:2] if np.isfinite(value)]
        reference = float(np.median(high_gain)) if high_gain else np.nan
        if not np.isfinite(reference) or reference <= 0:
            return self.scales.copy()

        for channel in range(2, min(self.channel_count, len(amplitudes))):
            amplitude = amplitudes[channel]
            if not np.isfinite(amplitude) or amplitude <= 0:
                continue
            target = float(np.clip(reference / amplitude, 1.0, self.max_scale))
            if not self.initialized[channel]:
                self.scales[channel] = target
                self.initialized[channel] = True
            else:
                self.scales[channel] += self.smoothing * (
                    target - self.scales[channel]
                )
        return self.scales.copy()


def scale_segment_samples(wavelength_nm, voltage_v, segments, fits, scale):
    """Scale peak excursions around their local baseline, not the DC offset."""
    wavelengths = np.asarray(wavelength_nm, dtype=float)
    values = np.asarray(voltage_v, dtype=float)
    output = values.copy()
    scales = np.asarray(scale, dtype=float)
    if scales.ndim == 0:
        scales = np.full(len(segments), float(scales), dtype=float)
    elif scales.shape != (len(segments),):
        raise ValueError("scale must be scalar or one value per segment")
    if np.all(np.abs(scales - 1.0) < 1e-9):
        return output
    for segment_index, (segment, fit) in enumerate(zip(segments, fits)):
        section = values[segment]
        section_wavelengths = wavelengths[segment]
        if section.size == 0:
            continue
        if fit.valid:
            if np.isfinite(fit.display_baseline_v):
                baseline = np.full(section.size, fit.display_baseline_v, dtype=float)
            else:
                baseline = fit.baseline(section_wavelengths)
        else:
            baseline = np.linspace(section[0], section[-1], section.size)
        output[segment] = baseline + scales[segment_index] * (section - baseline)
    return output


def rounded_display_curve(
    wavelength_nm,
    voltage_v,
    segments,
    fits,
    display_scale=1.0,
    points_per_interval=8,
    fitted=True,
    normalize_segment_heights=True,
):
    """Return a smooth curve with NaN separators between real scan sections."""
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(voltage_v, dtype=float)
    plot_x = []
    plot_y = []
    display_scales = np.asarray(display_scale, dtype=float)
    scalar_display_scale = display_scales.ndim == 0
    if scalar_display_scale:
        display_scales = np.full(len(segments), float(display_scales), dtype=float)
    elif display_scales.shape != (len(segments),):
        raise ValueError("display_scale must be scalar or one value per segment")
    valid_display_amplitudes = [
        float(fit.display_amplitude_v)
        for fit in fits
        if fit.valid
        and np.isfinite(fit.display_amplitude_v)
        and fit.display_amplitude_v > 0.0
    ]
    normalized_low_gain_amplitude = (
        float(display_scales[0]) * float(np.median(valid_display_amplitudes))
        if normalize_segment_heights
        and scalar_display_scale
        and float(display_scales[0]) > 1.0
        and valid_display_amplitudes
        else np.nan
    )
    for segment_index, (segment, fit) in enumerate(zip(segments, fits)):
        segment_scale = float(display_scales[segment_index])
        section_x = x[segment]
        section_y = y[segment]
        if section_x.size < 2:
            continue
        dense_count = max(
            int((section_x.size - 1) * int(points_per_interval) + 1),
            section_x.size,
        )
        dense_x = np.linspace(section_x[0], section_x[-1], dense_count)
        if fitted and fit.valid:
            # Use the fitted single-peak model rather than interpolating through
            # noisy ADC points.  Inserting the fitted center gives one exact
            # summit; with the constant fitted baseline, the left side is
            # strictly rising and the right side strictly falling.
            if section_x[0] < fit.center_nm < section_x[-1]:
                dense_x = np.unique(np.append(dense_x, fit.center_nm))
            display_baseline = (
                float(fit.display_baseline_v)
                if np.isfinite(fit.display_baseline_v)
                else float(fit.baseline(np.asarray([fit.center_nm]))[0])
            )
            baseline = np.full(dense_x.size, display_baseline, dtype=float)
            gaussian = np.exp(
                -0.5 * ((dense_x - fit.center_nm) / fit.sigma_nm) ** 2
            )
            # Limit only the rendered height of a clipped/broad fit.  Peak
            # center, quality and uncertainty continue to use the unconstrained
            # estimator, so this visual guard cannot slow or destabilize strain
            # tracking.  Low-gain channel normalization is applied afterwards.
            display_amplitude = (
                float(fit.display_amplitude_v)
                if np.isfinite(fit.display_amplitude_v)
                else float(fit.amplitude_v)
            )
            rendered_amplitude = (
                normalized_low_gain_amplitude
                if np.isfinite(normalized_low_gain_amplitude)
                else segment_scale * display_amplitude
            )
            dense_y = baseline + rendered_amplitude * gaussian
        elif fitted:
            display_y = repair_isolated_outliers(section_y)
            dense_y = PchipInterpolator(section_x, display_y)(dense_x)
            edge_baseline = np.interp(
                dense_x,
                [section_x[0], section_x[-1]],
                [section_y[0], section_y[-1]],
            )
            dense_y = edge_baseline + segment_scale * (
                dense_y - edge_baseline
            )
        else:
            dense_x = section_x
            edge_baseline = np.interp(
                dense_x,
                [section_x[0], section_x[-1]],
                [section_y[0], section_y[-1]],
            )
            dense_y = edge_baseline + segment_scale * (
                section_y - edge_baseline
            )
        plot_x.extend(dense_x.tolist())
        plot_y.extend(np.asarray(dense_y, dtype=float).tolist())
        plot_x.append(np.nan)
        plot_y.append(np.nan)
    if plot_x:
        plot_x.pop()
        plot_y.pop()
    return np.asarray(plot_x, dtype=float), np.asarray(plot_y, dtype=float)
