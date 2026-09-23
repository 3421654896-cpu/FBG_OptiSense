"""Sparse, shift-sensitive sampling for the nine CH1 FBG reflections.

The module deliberately has no UI, serial-port, network, or firmware dependency.
It consumes a wavelength-meter calibrated dense teacher spectrum and nine
non-overlapping peak regions.  The default plan contains five samples per peak
(45 samples in total): a peak/amplitude anchor, a baseline anchor on each side,
and shift-sensitive slope coverage on both sides.

Unlike equally spaced or Gaussian-assumed sampling, selection uses the measured
template derivative.  The score is the Fisher information for a wavelength
translation after projecting out unknown offset and amplitude.  Consequently,
asymmetric and rippled (but repeatable) peak shapes remain useful.

The shift estimator fits a translated copy of the dense template together with
an affine intensity correction.  It returns an uncertainty and explicit reject
reason; it never invents contact area or any other mechanical quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


DEFAULT_PEAK_COUNT = 9
DEFAULT_POINTS_PER_PEAK = 5


@dataclass(frozen=True)
class PeakRegion:
    """Inclusive wavelength limits for one FBG reflection."""

    start_nm: float
    stop_nm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_nm) or not np.isfinite(self.stop_nm):
            raise ValueError("peak region limits must be finite")
        if self.stop_nm <= self.start_nm:
            raise ValueError("peak region stop_nm must be greater than start_nm")


@dataclass(frozen=True)
class PeakSamplingTemplate:
    """Dense template and selected sparse samples for a single peak."""

    peak_number: int
    region_indices: np.ndarray
    dense_wavelength_nm: np.ndarray
    dense_signal: np.ndarray
    selected_indices: np.ndarray
    selected_wavelength_nm: np.ndarray
    selected_template_signal: np.ndarray
    selected_template_slope: np.ndarray
    selected_reliability_weight: np.ndarray
    peak_index: int
    projected_shift_information: float


@dataclass(frozen=True)
class SamplingPlan:
    """An ordered sparse plan for all nine CH1 peaks."""

    peaks: tuple[PeakSamplingTemplate, ...]
    points_per_peak: tuple[int, ...]

    @property
    def selected_indices(self) -> np.ndarray:
        result = np.concatenate([peak.selected_indices for peak in self.peaks])
        result.setflags(write=False)
        return result

    @property
    def selected_wavelength_nm(self) -> np.ndarray:
        result = np.concatenate(
            [peak.selected_wavelength_nm for peak in self.peaks]
        )
        result.setflags(write=False)
        return result

    @property
    def total_points(self) -> int:
        return int(sum(self.points_per_peak))


@dataclass(frozen=True)
class ShiftEstimate:
    """Result of fitting one sparse peak against its dense teacher template."""

    shift_nm: float
    uncertainty_nm: float
    accepted: bool
    reason: str
    offset: float
    amplitude_scale: float
    rmse: float
    normalized_rmse: float
    projected_fisher_information: float


def _readonly_float(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=float)
    result.setflags(write=False)
    return result


def _readonly_int(values: Iterable[int]) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.int64)
    result.setflags(write=False)
    return result


def _smooth(values: np.ndarray, window: int | None) -> np.ndarray:
    n = int(values.size)
    if window is None:
        # Three samples suppress isolated ADC noise while retaining narrow and
        # non-Gaussian structure in a 0.02 nm teacher scan.
        window = 3 if n >= 7 else 1
    window = int(window)
    if window <= 1:
        return values.astype(float, copy=True)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 else n - 1)
    if window <= 1:
        return values.astype(float, copy=True)
    half = window // 2
    # A triangular kernel has less ringing than a rectangular moving average.
    rising = np.arange(1, half + 2, dtype=float)
    kernel = np.r_[rising, rising[-2::-1]]
    kernel /= np.sum(kernel)
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _region_to_indices(
    wavelength_nm: np.ndarray,
    region: PeakRegion | Sequence[float] | slice,
) -> np.ndarray:
    if isinstance(region, slice):
        indices = np.arange(wavelength_nm.size, dtype=np.int64)[region]
    else:
        if isinstance(region, PeakRegion):
            start, stop = region.start_nm, region.stop_nm
        else:
            if len(region) != 2:
                raise ValueError("a peak region must contain exactly two limits")
            start, stop = float(region[0]), float(region[1])
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            raise ValueError("invalid peak wavelength region")
        indices = np.flatnonzero((wavelength_nm >= start) & (wavelength_nm <= stop))
    if indices.size < DEFAULT_POINTS_PER_PEAK:
        raise ValueError("each peak region must contain at least five dense samples")
    if np.any(np.diff(indices) != 1):
        raise ValueError("peak regions must be contiguous")
    return indices.astype(np.int64, copy=False)


def _projected_shift_information(
    signal: np.ndarray,
    slope: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Translation information after removing offset and amplitude nuisance."""

    nuisance = np.column_stack((np.ones(signal.size), signal))
    if weights is None:
        root_weight = np.ones(signal.size, dtype=float)
    else:
        root_weight = np.sqrt(np.asarray(weights, dtype=float))
    weighted_nuisance = nuisance * root_weight[:, None]
    weighted_slope = slope * root_weight
    try:
        nuisance_component = weighted_nuisance @ np.linalg.lstsq(
            weighted_nuisance, weighted_slope, rcond=None
        )[0]
    except np.linalg.LinAlgError:
        return 0.0
    residual = weighted_slope - nuisance_component
    return float(np.dot(residual, residual))


def _strategic_candidates(
    signal: np.ndarray,
    slope: np.ndarray,
    peak_local: int,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, set[int], set[int], set[int], set[int]]:
    """Return a compact pool plus left/right tail and slope constraint sets."""

    n = signal.size
    local = np.arange(n, dtype=np.int64)
    available = (
        np.ones(n, dtype=bool)
        if allowed is None
        else np.asarray(allowed, dtype=bool)
    )
    left = local[:peak_local][available[:peak_local]]
    right = local[peak_local + 1 :][available[peak_local + 1 :]]
    if left.size < 2 or right.size < 2:
        raise ValueError("peak maximum needs at least two teacher samples per side")

    left_tail_limit = max(1, int(np.ceil(0.24 * n)))
    right_tail_start = min(n - 1, int(np.floor(0.76 * n)))
    left_tail = set(
        int(i) for i in local[:left_tail_limit]
        if i < peak_local and available[i]
    )
    right_tail = set(
        int(i) for i in local[right_tail_start:]
        if i > peak_local and available[i]
    )
    if not left_tail:
        left_tail = {int(left[0])}
    if not right_tail:
        right_tail = {int(right[-1])}

    # Slope candidates are chosen from the measured template, not a Gaussian.
    # Keep several high-gradient sites on each side because rippled peaks may
    # have more than one informative flank.
    def top_slope(side: np.ndarray) -> set[int]:
        if side.size == 0:
            return set()
        ordered = side[np.argsort(np.abs(slope[side]))[::-1]]
        count = min(5, max(2, int(np.ceil(side.size * 0.12))))
        chosen = set(int(i) for i in ordered[:count])
        maximum = float(np.max(np.abs(slope[side])))
        if maximum > 0:
            above = side[np.abs(slope[side]) >= 0.55 * maximum]
            # Bound the combinatorial pool even on broad flat-topped peaks.
            chosen.update(int(i) for i in above[: min(6, above.size)])
        return chosen

    left_slope = top_slope(left)
    right_slope = top_slope(right)

    pool: set[int] = {0, n - 1, peak_local}
    # Tail *constraint sets* may be large.  Only a few representatives belong
    # in the exhaustive-search pool, keeping the optimizer deterministic and
    # fast for 2001-point scans.
    for tail in (sorted(left_tail), sorted(right_tail)):
        if tail:
            for position in np.linspace(0, len(tail) - 1, min(4, len(tail))):
                pool.add(int(tail[int(round(position))]))
    pool.update(left_slope)
    pool.update(right_slope)
    for fraction in np.linspace(0.0, 1.0, 9):
        pool.add(int(round(fraction * (n - 1))))
    # Include low-intensity tail representatives even for sloping baselines.
    for side in (left, right):
        if side.size:
            lowest = side[np.argsort(signal[side])[: min(2, side.size)]]
            pool.update(int(i) for i in lowest)
    return (
        np.asarray(sorted(pool), dtype=np.int64),
        left_tail,
        right_tail,
        left_slope,
        right_slope,
    )


def _selection_score(
    chosen: np.ndarray,
    x: np.ndarray,
    signal: np.ndarray,
    slope: np.ndarray,
    reliability_weight: np.ndarray | None = None,
) -> float:
    selected_signal = signal[chosen]
    selected_slope = slope[chosen]
    selected_weight = (
        None if reliability_weight is None else reliability_weight[chosen]
    )
    information = _projected_shift_information(
        selected_signal, selected_slope, selected_weight
    )
    span = max(float(x[-1] - x[0]), np.finfo(float).eps)
    amplitude = max(float(np.ptp(signal)), np.finfo(float).eps)
    # Dimensionless normalization makes the tie-breakers stable across ADC
    # gain settings.  Information remains the dominant term.
    info_scale = (amplitude / span) ** 2
    normalized_information = information / max(info_scale, np.finfo(float).eps)
    dynamic = float(np.ptp(selected_signal)) / amplitude
    selected_x = x[chosen]
    spacing = np.diff(selected_x) / span
    separation = float(np.min(spacing)) if spacing.size else 0.0
    conditioning = np.linalg.cond(
        np.column_stack((np.ones(chosen.size), selected_signal / amplitude))
    )
    condition_bonus = 0.0 if not np.isfinite(conditioning) else 1.0 / conditioning
    return (
        normalized_information
        + 0.025 * dynamic
        + 0.010 * min(separation, 0.20)
        + 0.005 * condition_bonus
    )


def _select_five(
    x: np.ndarray,
    signal: np.ndarray,
    slope: np.ndarray,
    peak_local: int,
    allowed: np.ndarray | None = None,
    reliability_weight: np.ndarray | None = None,
) -> np.ndarray:
    pool, left_tail, right_tail, left_slope, right_slope = _strategic_candidates(
        signal, slope, peak_local, allowed=allowed
    )
    if allowed is not None:
        pool = pool[np.asarray(allowed, dtype=bool)[pool]]
    remaining = [int(i) for i in pool if int(i) != peak_local]
    span = max(float(x[-1] - x[0]), np.finfo(float).eps)
    minimum_separation = 0.012 * span
    best: tuple[float, np.ndarray] | None = None
    for extra in combinations(remaining, 4):
        selected_set = set(extra)
        if not (selected_set & left_tail and selected_set & right_tail):
            continue
        if not (selected_set & left_slope and selected_set & right_slope):
            continue
        chosen = np.asarray(sorted((*extra, peak_local)), dtype=np.int64)
        if np.min(np.diff(x[chosen])) < minimum_separation:
            continue
        score = _selection_score(
            chosen, x, signal, slope, reliability_weight=reliability_weight
        )
        if best is None or score > best[0]:
            best = (score, chosen)
    if best is None:
        raise ValueError("peak region is too narrow to form a valid five-point plan")
    return best[1]


def _expand_selection(
    base: np.ndarray,
    count: int,
    x: np.ndarray,
    signal: np.ndarray,
    slope: np.ndarray,
    allowed: np.ndarray | None = None,
    reliability_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Greedily add informative points for a configurable budget above five."""

    chosen = list(int(i) for i in base)
    candidates = [
        i for i in range(x.size)
        if i not in chosen and (allowed is None or bool(allowed[i]))
    ]
    while len(chosen) < count:
        best_index = None
        best_score = -np.inf
        for candidate in candidates:
            trial = np.asarray(sorted((*chosen, candidate)), dtype=np.int64)
            score = _selection_score(
                trial, x, signal, slope, reliability_weight=reliability_weight
            )
            if score > best_score:
                best_score = score
                best_index = candidate
        if best_index is None:
            raise ValueError("not enough unique dense samples for requested budget")
        chosen.append(best_index)
        candidates.remove(best_index)
    return np.asarray(sorted(chosen), dtype=np.int64)


def build_peak_template(
    wavelength_nm: Sequence[float],
    teacher_signal: Sequence[float],
    region: PeakRegion | Sequence[float] | slice,
    selected_indices: Sequence[int],
    *,
    peak_number: int = 1,
    smoothing_window: int | None = None,
    reliability_weight: Sequence[float] | None = None,
) -> PeakSamplingTemplate:
    """Build an estimator template from a caller-supplied sparse selection.

    This is useful for benchmarking an optimized plan against an equal-spacing
    baseline with exactly the same shift estimator.
    """

    x_all = np.asarray(wavelength_nm, dtype=float)
    y_all = np.asarray(teacher_signal, dtype=float)
    if x_all.ndim != 1 or y_all.shape != x_all.shape:
        raise ValueError("wavelength and teacher arrays must be equally sized 1-D arrays")
    region_indices = _region_to_indices(x_all, region)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size < 4:
        raise ValueError("at least four unique selected indices are required")
    if np.any(np.diff(selected) <= 0) or np.unique(selected).size != selected.size:
        raise ValueError("selected indices must be unique and strictly increasing")
    if not set(int(i) for i in selected).issubset(
        set(int(i) for i in region_indices)
    ):
        raise ValueError("selected indices must lie inside the peak region")
    local_x = x_all[region_indices]
    local_y = _smooth(y_all[region_indices], smoothing_window)
    local_slope = np.gradient(local_y, local_x, edge_order=2)
    local_selected = np.searchsorted(region_indices, selected)
    if reliability_weight is None:
        selected_weight = np.ones(selected.size, dtype=float)
    else:
        weights_all = np.asarray(reliability_weight, dtype=float)
        if weights_all.shape != x_all.shape:
            raise ValueError("reliability_weight must match the dense spectrum")
        if np.any(~np.isfinite(weights_all)) or np.any(weights_all < 0):
            raise ValueError("reliability weights must be finite and nonnegative")
        selected_weight = weights_all[selected]
    peak_local = int(np.argmax(local_y))
    info = _projected_shift_information(
        local_y[local_selected], local_slope[local_selected], selected_weight
    )
    return PeakSamplingTemplate(
        peak_number=int(peak_number),
        region_indices=_readonly_int(region_indices),
        dense_wavelength_nm=_readonly_float(local_x),
        dense_signal=_readonly_float(local_y),
        selected_indices=_readonly_int(selected),
        selected_wavelength_nm=_readonly_float(x_all[selected]),
        selected_template_signal=_readonly_float(local_y[local_selected]),
        selected_template_slope=_readonly_float(local_slope[local_selected]),
        selected_reliability_weight=_readonly_float(selected_weight),
        peak_index=int(region_indices[peak_local]),
        projected_shift_information=float(info),
    )


def optimize_ch1_sampling(
    wavelength_nm: Sequence[float],
    teacher_signal: Sequence[float],
    peak_regions: Sequence[PeakRegion | Sequence[float] | slice],
    *,
    points_per_peak: int | Sequence[int] = DEFAULT_POINTS_PER_PEAK,
    expected_peak_count: int = DEFAULT_PEAK_COUNT,
    smoothing_window: int | None = None,
    sample_noise_std: Sequence[float] | None = None,
    stable_mask: Sequence[bool] | None = None,
) -> SamplingPlan:
    """Select a sparse, shift-sensitive plan from a dense CH1 teacher scan.

    ``points_per_peak=5`` produces the intended 45-point, nine-FBG real-time
    plan.  Larger budgets retain the mandatory five anchors and greedily add
    Fisher-informative samples.  Budgets below five are rejected because offset,
    amplitude, translation, two-sided coverage, and basic fault detection would
    otherwise be underdetermined.
    """

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(teacher_signal, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 5:
        raise ValueError("wavelength and teacher arrays must be equally sized 1-D arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("teacher spectrum must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelengths must be strictly increasing")
    if len(peak_regions) != int(expected_peak_count):
        raise ValueError(
            f"expected {expected_peak_count} peak regions, got {len(peak_regions)}"
        )
    if isinstance(points_per_peak, (int, np.integer)):
        budgets = (int(points_per_peak),) * len(peak_regions)
    else:
        budgets = tuple(int(value) for value in points_per_peak)
        if len(budgets) != len(peak_regions):
            raise ValueError("points_per_peak sequence must match peak region count")
    if any(value < DEFAULT_POINTS_PER_PEAK for value in budgets):
        raise ValueError("at least five samples per peak are required")

    if sample_noise_std is None:
        reliability = np.ones(x.size, dtype=float)
        noise_valid = np.ones(x.size, dtype=bool)
    else:
        noise = np.asarray(sample_noise_std, dtype=float)
        if noise.shape != x.shape:
            raise ValueError("sample_noise_std must match the dense spectrum")
        noise_valid = np.isfinite(noise) & (noise > 0)
        if not np.any(noise_valid):
            raise ValueError("sample_noise_std has no valid entries")
        reference_noise = float(np.median(noise[noise_valid]))
        reliability = np.zeros(x.size, dtype=float)
        reliability[noise_valid] = np.clip(
            (reference_noise / noise[noise_valid]) ** 2, 1e-4, 1e4
        )
    if stable_mask is None:
        stable = np.ones(x.size, dtype=bool)
    else:
        stable = np.asarray(stable_mask, dtype=bool)
        if stable.shape != x.shape:
            raise ValueError("stable_mask must match the dense spectrum")
    preferred = stable & noise_valid

    templates: list[PeakSamplingTemplate] = []
    used: set[int] = set()
    previous_stop = -1
    for peak_number, (region, budget) in enumerate(
        zip(peak_regions, budgets, strict=True), start=1
    ):
        region_indices = _region_to_indices(x, region)
        if int(region_indices[0]) <= previous_stop:
            raise ValueError("peak regions must be ordered and non-overlapping")
        previous_stop = int(region_indices[-1])
        if budget > region_indices.size:
            raise ValueError("point budget exceeds dense samples in a peak region")
        local_x = x[region_indices]
        local_signal = _smooth(y[region_indices], smoothing_window)
        if float(np.ptp(local_signal)) <= 64.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(local_signal)))
        ):
            raise ValueError(f"peak region {peak_number} has no measurable structure")
        local_slope = np.gradient(local_signal, local_x, edge_order=2)
        local_preferred = preferred[region_indices]
        # Use only stable/noise-qualified samples when there are enough usable
        # neighbours on both flanks.  Otherwise retain all points but give the
        # flagged ones a tiny score weight, making fallback explicit and rare.
        nominal_peak = int(np.argmax(local_signal))
        enough_preferred = (
            np.count_nonzero(local_preferred) >= budget
            and np.count_nonzero(local_preferred[:nominal_peak]) >= 2
            and np.count_nonzero(local_preferred[nominal_peak + 1 :]) >= 2
        )
        if enough_preferred:
            allowed = local_preferred
            peak_candidates = np.flatnonzero(allowed)
            peak_local = int(
                peak_candidates[np.argmax(local_signal[peak_candidates])]
            )
        else:
            allowed = np.ones(region_indices.size, dtype=bool)
            peak_local = nominal_peak
            reliability[region_indices[~local_preferred]] = np.minimum(
                reliability[region_indices[~local_preferred]], 1e-4
            )
        local_reliability = reliability[region_indices]
        selected_local = _select_five(
            local_x,
            local_signal,
            local_slope,
            peak_local,
            allowed=allowed,
            reliability_weight=local_reliability,
        )
        if budget > DEFAULT_POINTS_PER_PEAK:
            selected_local = _expand_selection(
                selected_local,
                budget,
                local_x,
                local_signal,
                local_slope,
                allowed=allowed,
                reliability_weight=local_reliability,
            )
        selected_global = region_indices[selected_local]
        if used.intersection(int(i) for i in selected_global):
            raise ValueError("selected indices overlap between peak regions")
        used.update(int(i) for i in selected_global)
        templates.append(
            build_peak_template(
                x,
                y,
                slice(int(region_indices[0]), int(region_indices[-1]) + 1),
                selected_global,
                peak_number=peak_number,
                smoothing_window=smoothing_window,
                reliability_weight=reliability,
            )
        )

    all_indices = np.concatenate([item.selected_indices for item in templates])
    if np.any(np.diff(all_indices) <= 0):
        raise AssertionError("internal error: sparse plan is not strictly increasing")
    return SamplingPlan(peaks=tuple(templates), points_per_peak=budgets)


def equally_spaced_peak_indices(
    wavelength_nm: Sequence[float],
    region: PeakRegion | Sequence[float] | slice,
    count: int = DEFAULT_POINTS_PER_PEAK,
) -> np.ndarray:
    """Return a deterministic equal-spacing baseline for evaluation."""

    x = np.asarray(wavelength_nm, dtype=float)
    region_indices = _region_to_indices(x, region)
    if count < 2 or count > region_indices.size:
        raise ValueError("invalid equal-spacing point count")
    positions = np.linspace(0, region_indices.size - 1, count)
    selected_local = np.rint(positions).astype(np.int64)
    # Rounding can only duplicate when count approaches region size; repair by
    # choosing the closest currently unused dense point.
    if np.unique(selected_local).size != count:
        chosen: list[int] = []
        for position in positions:
            candidates = np.argsort(np.abs(np.arange(region_indices.size) - position))
            chosen.append(next(int(i) for i in candidates if int(i) not in chosen))
        selected_local = np.asarray(sorted(chosen), dtype=np.int64)
    result = region_indices[selected_local]
    result.setflags(write=False)
    return result


def _evaluate_shift_grid(
    shifts: np.ndarray,
    sample_x: np.ndarray,
    observed: np.ndarray,
    dense_x: np.ndarray,
    dense_y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = sample_x[None, :] - shifts[:, None]
    templates = np.interp(query.ravel(), dense_x, dense_y).reshape(query.shape)
    if weights is None:
        weights = np.ones(observed.size, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    t_mean = (templates @ weights) / weight_sum
    y_mean = float(np.dot(observed, weights) / weight_sum)
    centered_t = templates - t_mean[:, None]
    centered_y = observed - y_mean
    denominator = np.sum(centered_t * centered_t * weights[None, :], axis=1)
    scale = np.divide(
        (centered_t * weights[None, :]) @ centered_y,
        denominator,
        out=np.full(shifts.size, np.nan),
        where=denominator > np.finfo(float).eps,
    )
    offset = y_mean - scale * t_mean
    residual = observed[None, :] - (
        offset[:, None] + scale[:, None] * templates
    )
    rss = np.sum(residual * residual * weights[None, :], axis=1)
    return rss, offset, scale


def estimate_template_shift(
    template: PeakSamplingTemplate,
    sampled_signal: Sequence[float],
    *,
    noise_std: float | Sequence[float] | None = None,
    shift_bounds_nm: tuple[float, float] | None = None,
    max_uncertainty_nm: float | None = None,
    max_normalized_rmse: float = 0.25,
    min_amplitude_scale: float = 0.10,
    max_amplitude_scale: float = 10.0,
) -> ShiftEstimate:
    """Estimate a peak translation from sparse CH1 samples.

    The observed values must be in the same order as
    ``template.selected_indices``.  Positive shift means the reflection moved
    toward longer wavelength.  The fit allows an unknown offset and amplitude
    scale, so the wavelength estimate is not confused by moderate intensity
    drift.  ``noise_std`` should be the ADC-domain standard deviation measured
    from stable frames whenever available.
    """

    observed_all = np.asarray(sampled_signal, dtype=float)
    if observed_all.shape != template.selected_template_signal.shape:
        raise ValueError("sampled_signal length does not match sparse template")
    if noise_std is None or np.isscalar(noise_std):
        noise_all = None
    else:
        noise_all = np.asarray(noise_std, dtype=float)
        if noise_all.shape != observed_all.shape:
            raise ValueError("per-sample noise_std must match sampled_signal")
    finite = np.isfinite(observed_all)
    if noise_all is not None:
        finite &= np.isfinite(noise_all) & (noise_all > 0)
    if np.count_nonzero(finite) < 4:
        return ShiftEstimate(
            np.nan, np.inf, False, "insufficient_finite_samples", np.nan, np.nan,
            np.inf, np.inf, 0.0,
        )
    sample_x = np.asarray(template.selected_wavelength_nm)[finite]
    observed = observed_all[finite]
    if noise_all is None:
        search_weights = np.ones(observed.size, dtype=float)
    else:
        selected_noise = noise_all[finite]
        search_weights = 1.0 / (selected_noise * selected_noise)
        search_weights /= float(np.mean(search_weights))
    dense_x = np.asarray(template.dense_wavelength_nm)
    dense_y = np.asarray(template.dense_signal)
    span = float(dense_x[-1] - dense_x[0])
    if shift_bounds_nm is None:
        margin = min(0.18 * span, 2.5 * float(np.median(np.diff(sample_x))))
        shift_bounds_nm = (-margin, margin)
    lower, upper = (float(shift_bounds_nm[0]), float(shift_bounds_nm[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("shift_bounds_nm must be finite and increasing")
    # ``np.interp`` holds the edge baseline constant outside the region.  That
    # is intentional: a conventional equal-spacing benchmark often includes
    # both region endpoints, while those endpoints are baseline anchors rather
    # than shape-bearing samples.
    safe_lower = lower
    safe_upper = upper

    shifts = np.linspace(safe_lower, safe_upper, 401)
    rss, offsets, scales = _evaluate_shift_grid(
        shifts, sample_x, observed, dense_x, dense_y, search_weights
    )
    for _ in range(3):
        best = int(np.nanargmin(rss))
        spacing = float(shifts[1] - shifts[0])
        centre = float(shifts[best])
        local_lower = max(safe_lower, centre - 2.0 * spacing)
        local_upper = min(safe_upper, centre + 2.0 * spacing)
        shifts = np.linspace(local_lower, local_upper, 101)
        rss, offsets, scales = _evaluate_shift_grid(
            shifts, sample_x, observed, dense_x, dense_y, search_weights
        )
    best = int(np.nanargmin(rss))
    shift = float(shifts[best])
    offset = float(offsets[best])
    scale = float(scales[best])
    best_template = np.interp(sample_x - shift, dense_x, dense_y)
    best_residual = observed - (offset + scale * best_template)
    rmse = float(np.sqrt(np.mean(best_residual * best_residual)))
    observed_range = max(
        float(np.ptp(observed)),
        abs(scale) * float(np.ptp(dense_y)) * 0.25,
        np.finfo(float).eps,
    )
    normalized_rmse = rmse / observed_range

    shifted_template = np.interp(sample_x - shift, dense_x, dense_y)
    dense_slope = np.gradient(dense_y, dense_x, edge_order=2)
    derivative = -scale * np.interp(sample_x - shift, dense_x, dense_slope)
    nuisance = np.column_stack((np.ones(observed.size), shifted_template))
    try:
        projected = derivative - nuisance @ np.linalg.lstsq(
            nuisance, derivative, rcond=None
        )[0]
        derivative_energy = float(np.dot(projected, projected))
    except np.linalg.LinAlgError:
        derivative_energy = 0.0

    if noise_std is None:
        degrees = max(1, observed.size - 3)
        estimated_noise = float(np.sqrt(np.sum(best_residual ** 2) / degrees))
        noise = max(estimated_noise, observed_range * 1e-4)
        fisher = derivative_energy / (noise * noise) if derivative_energy > 0 else 0.0
    elif noise_all is not None:
        whitened_derivative = derivative / selected_noise
        weighted_nuisance = nuisance / selected_noise[:, None]
        try:
            projected_white = whitened_derivative - weighted_nuisance @ np.linalg.lstsq(
                weighted_nuisance, whitened_derivative, rcond=None
            )[0]
            fisher = float(np.dot(projected_white, projected_white))
        except np.linalg.LinAlgError:
            fisher = 0.0
    else:
        noise = float(noise_std)
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("noise_std must be finite and positive")
        fisher = derivative_energy / (noise * noise) if derivative_energy > 0 else 0.0
    uncertainty = float(1.0 / np.sqrt(fisher)) if fisher > 0 else np.inf
    if max_uncertainty_nm is None:
        max_uncertainty_nm = max(0.002, 0.08 * span)

    boundary_tolerance = max(2e-6, 2e-3 * (safe_upper - safe_lower))
    reason = "ok"
    if not np.isfinite(scale) or scale < min_amplitude_scale:
        reason = "nonpositive_or_too_small_amplitude"
    elif scale > max_amplitude_scale:
        reason = "amplitude_scale_out_of_range"
    elif derivative_energy <= np.finfo(float).eps:
        reason = "unidentifiable_shift"
    elif uncertainty > float(max_uncertainty_nm):
        reason = "uncertainty_too_large"
    elif normalized_rmse > float(max_normalized_rmse):
        reason = "template_mismatch"
    elif (
        shift - safe_lower <= boundary_tolerance
        or safe_upper - shift <= boundary_tolerance
    ):
        reason = "shift_at_search_boundary"
    return ShiftEstimate(
        shift_nm=shift,
        uncertainty_nm=uncertainty,
        accepted=(reason == "ok"),
        reason=reason,
        offset=offset,
        amplitude_scale=scale,
        rmse=rmse,
        normalized_rmse=normalized_rmse,
        projected_fisher_information=fisher,
    )


def estimate_plan_shifts(
    plan: SamplingPlan,
    sampled_signal: Sequence[float],
    *,
    noise_std: float | Sequence[float] | None = None,
    **estimate_kwargs: object,
) -> tuple[ShiftEstimate, ...]:
    """Estimate every peak shift from one ordered sparse CH1 frame.

    ``sampled_signal`` follows ``plan.selected_indices`` order.  A scalar noise
    value is shared by all peaks; a sequence supplies one noise estimate per
    sparse point.  Results stay in G1..G9 order, including rejected estimates.
    """

    values = np.asarray(sampled_signal, dtype=float)
    if values.ndim != 1 or values.size != plan.total_points:
        raise ValueError("sampled_signal length does not match sampling plan")
    if noise_std is None or np.isscalar(noise_std):
        noises: np.ndarray | None = None
    else:
        noises = np.asarray(noise_std, dtype=float)
        if noises.shape != values.shape:
            raise ValueError("per-sample noise_std must match sampled_signal")
    estimates: list[ShiftEstimate] = []
    start = 0
    for template, count in zip(plan.peaks, plan.points_per_peak, strict=True):
        stop = start + count
        local_noise: float | np.ndarray | None
        if noises is None:
            local_noise = noise_std  # scalar or None
        else:
            local_noise = noises[start:stop]
        estimates.append(
            estimate_template_shift(
                template,
                values[start:stop],
                noise_std=local_noise,
                **estimate_kwargs,
            )
        )
        start = stop
    return tuple(estimates)


__all__ = [
    "DEFAULT_PEAK_COUNT",
    "DEFAULT_POINTS_PER_PEAK",
    "PeakRegion",
    "PeakSamplingTemplate",
    "SamplingPlan",
    "ShiftEstimate",
    "build_peak_template",
    "equally_spaced_peak_indices",
    "estimate_plan_shifts",
    "estimate_template_shift",
    "optimize_ch1_sampling",
]
