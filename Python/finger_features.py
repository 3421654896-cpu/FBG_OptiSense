"""Causal CH1 features for the nine-FBG mechanical finger.

The real-time table is expected to contain five ordered wavelengths per
grating by default (45 points total).  All numerical features start from raw
12-bit ADC codes.  Software/display gain is accepted as audit metadata only and
never participates in voltage, peak fitting, or baseline normalization.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntFlag
from typing import Any

import numpy as np

from adaptive_sampling_optimizer import ShiftEstimate, estimate_template_shift
from finger_dataset import (
    ADC_CODE_COUNT,
    ADC_MAX_CODE,
    ADC_REFERENCE_V,
    DEFAULT_GRATING_COUNT,
    DEFAULT_POINTS_PER_PEAK,
    DEFAULT_SENSOR_CHANNEL,
    SpectrumFrame,
)
from finger_sampling_template import (
    FingerSamplingTemplate,
    load_finger_sampling_template,
)

FEATURE_NAMES = (
    "center_shift_pm",
    "amplitude_relative",
    "width_relative",
    "left_slope_relative",
    "right_slope_relative",
    "asymmetry_delta",
)
CAUSAL_FRAME_COUNT = 3


class PeakQuality(IntFlag):
    GOOD = 0
    NONFINITE_SAMPLE = 1 << 0
    ADC_OUT_OF_RANGE = 1 << 1
    NONMONOTONIC_WAVELENGTH = 1 << 2
    LOW_AMPLITUDE = 1 << 3
    PEAK_AT_EDGE = 1 << 4
    WIDTH_UNRESOLVED = 1 << 5
    BASELINE_INVALID = 1 << 6
    FRAME_CRC_FAILED = 1 << 7
    TABLE_CRC_MISMATCH = 1 << 8
    TEMPLATE_FIT_REJECTED = 1 << 9
    TEMPLATE_UNCERTAINTY_HIGH = 1 << 10
    TEMPLATE_MISMATCH = 1 << 11
    TEMPLATE_SHIFT_AT_BOUNDARY = 1 << 12
    TEMPLATE_AMPLITUDE_INVALID = 1 << 13
    TEMPLATE_SHIFT_UNIDENTIFIABLE = 1 << 14


@dataclass(frozen=True)
class PeakFeature:
    peak_index: int
    center_nm: float
    center_shift_pm: float
    amplitude_v: float
    amplitude_relative: float
    width_nm: float
    width_relative: float
    left_slope_v_per_nm: float
    left_slope_relative: float
    right_slope_v_per_nm: float
    right_slope_relative: float
    asymmetry: float
    asymmetry_delta: float
    normalized_profile: tuple[float, ...]
    quality_mask: int
    center_uncertainty_pm: float = math.nan
    template_fit_normalized_rmse: float = math.nan
    template_fit_reason: str = "legacy_local_shape"

    @property
    def valid(self) -> bool:
        fatal = (
            PeakQuality.NONFINITE_SAMPLE
            | PeakQuality.ADC_OUT_OF_RANGE
            | PeakQuality.NONMONOTONIC_WAVELENGTH
            | PeakQuality.LOW_AMPLITUDE
            | PeakQuality.BASELINE_INVALID
            | PeakQuality.FRAME_CRC_FAILED
            | PeakQuality.TABLE_CRC_MISMATCH
            | PeakQuality.TEMPLATE_FIT_REJECTED
        )
        return not bool(PeakQuality(self.quality_mask) & fatal)

    def normalized_vector(self) -> np.ndarray:
        values = np.asarray(
            (
                self.center_shift_pm,
                self.amplitude_relative,
                self.width_relative,
                self.left_slope_relative,
                self.right_slope_relative,
                self.asymmetry_delta,
            ),
            dtype=float,
        )
        # The quality mask is supplied separately.  A finite fill prevents a
        # single bad peak from poisoning the complete model input tensor.
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class FingerFeatureFrame:
    sequence: int
    monotonic_ns: int
    channel: int
    peaks: tuple[PeakFeature, ...]
    table_crc32: int | None = None
    digital_gain_audit: float | None = None

    def __post_init__(self) -> None:
        if self.channel != DEFAULT_SENSOR_CHANNEL:
            raise ValueError("mechanical-finger features must come from CH1")
        if len(self.peaks) != DEFAULT_GRATING_COUNT:
            raise ValueError("mechanical-finger feature frame must contain 9 peaks")

    @property
    def values(self) -> np.ndarray:
        return np.stack([peak.normalized_vector() for peak in self.peaks], axis=0)

    @property
    def quality_masks(self) -> np.ndarray:
        return np.asarray([peak.quality_mask for peak in self.peaks], dtype=np.uint16)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.asarray([peak.valid for peak in self.peaks], dtype=bool)


@dataclass(frozen=True)
class CausalFeatureWindow:
    """A three-frame, past-to-present model input with no future leakage."""

    values: np.ndarray
    quality_masks: np.ndarray
    monotonic_ns: tuple[int, int, int]
    sequences: tuple[int, int, int]
    history_size: int

    @property
    def ready(self) -> bool:
        return self.history_size == CAUSAL_FRAME_COUNT

    @property
    def current(self) -> np.ndarray:
        return self.values[-1]

    @property
    def first_difference(self) -> np.ndarray:
        return self.values[-1] - self.values[-2]

    @property
    def second_difference(self) -> np.ndarray:
        return self.values[-1] - 2.0 * self.values[-2] + self.values[-3]

    def model_vector(self) -> np.ndarray:
        """Return current, causal velocity and causal acceleration per FBG."""

        combined = np.concatenate(
            (self.current, self.first_difference, self.second_difference), axis=1
        )
        return combined.reshape(-1)


class CausalFeatureExtractor:
    """Maintain exactly three input frames, left-padding only during warm-up."""

    def __init__(self) -> None:
        self._history = deque(maxlen=CAUSAL_FRAME_COUNT)

    def reset(self) -> None:
        self._history.clear()

    def update(self, frame: FingerFeatureFrame) -> CausalFeatureWindow:
        if self._history and frame.monotonic_ns < self._history[-1].monotonic_ns:
            raise ValueError("feature frames must be time ordered")
        self._history.append(frame)
        entries = list(self._history)
        padded = [entries[0]] * (CAUSAL_FRAME_COUNT - len(entries)) + entries
        return CausalFeatureWindow(
            values=np.stack([item.values for item in padded], axis=0),
            quality_masks=np.stack([item.quality_masks for item in padded], axis=0),
            monotonic_ns=tuple(item.monotonic_ns for item in padded),
            sequences=tuple(item.sequence for item in padded),
            history_size=len(entries),
        )


@dataclass(frozen=True)
class _LocalShape:
    center_nm: float
    amplitude: float
    width_nm: float
    left_slope: float
    right_slope: float
    asymmetry: float
    corrected: np.ndarray
    quality: PeakQuality


def _adc_voltage(codes: Sequence[int]) -> tuple[np.ndarray, PeakQuality]:
    raw = np.asarray(codes)
    if raw.ndim != 1:
        raise ValueError("ADC codes must be one-dimensional")
    try:
        values = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ADC codes must be numeric") from exc
    quality = PeakQuality.GOOD
    if not np.all(np.isfinite(values)):
        quality |= PeakQuality.NONFINITE_SAMPLE
    if np.any((values < 0.0) | (values > ADC_MAX_CODE)):
        quality |= PeakQuality.ADC_OUT_OF_RANGE
    values[(values < 0.0) | (values > ADC_MAX_CODE)] = np.nan
    return values * ADC_REFERENCE_V / ADC_CODE_COUNT, quality


def _crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = min(max((level - y0) / (y1 - y0), 0.0), 1.0)
    return x0 + fraction * (x1 - x0)


def _local_shape(
    wavelength_nm: np.ndarray,
    voltage_v: np.ndarray,
    minimum_amplitude_v: float,
) -> _LocalShape:
    quality = PeakQuality.GOOD
    if not np.all(np.isfinite(voltage_v)):
        quality |= PeakQuality.NONFINITE_SAMPLE
        return _LocalShape(*(math.nan,) * 6, np.full_like(voltage_v, np.nan), quality)
    if np.any(np.diff(wavelength_nm) <= 0.0):
        quality |= PeakQuality.NONMONOTONIC_WAVELENGTH
        return _LocalShape(*(math.nan,) * 6, np.full_like(voltage_v, np.nan), quality)

    baseline = np.interp(
        wavelength_nm,
        (wavelength_nm[0], wavelength_nm[-1]),
        (voltage_v[0], voltage_v[-1]),
    )
    corrected = voltage_v - baseline
    peak_sample = int(np.argmax(corrected))
    amplitude = float(corrected[peak_sample])
    if amplitude < float(minimum_amplitude_v):
        quality |= PeakQuality.LOW_AMPLITUDE
    if peak_sample == 0 or peak_sample == len(voltage_v) - 1:
        quality |= PeakQuality.PEAK_AT_EDGE

    weights = np.clip(corrected, 0.0, None)
    weight_sum = float(np.sum(weights))
    center = (
        float(np.dot(wavelength_nm, weights) / weight_sum)
        if weight_sum > 0.0
        else math.nan
    )
    # A local parabola substantially improves sub-step center accuracy while
    # the positive-moment center remains the safe fallback near a boundary.
    if 0 < peak_sample < len(voltage_v) - 1:
        local_x = wavelength_nm[peak_sample - 1 : peak_sample + 2]
        # Fit the raw local voltages for the vertex.  Subtracting a line drawn
        # through only the two window edges can bias the center when a shifted
        # peak contributes unequal tails to those edge samples.
        local_y = voltage_v[peak_sample - 1 : peak_sample + 2]
        x_origin = float(local_x[1])
        coefficients = np.polyfit(local_x - x_origin, local_y, 2)
        if coefficients[0] < 0.0:
            vertex = x_origin - float(coefficients[1] / (2.0 * coefficients[0]))
            if float(local_x[0]) <= vertex <= float(local_x[-1]):
                center = vertex
                vertex_baseline = float(np.interp(vertex, wavelength_nm, baseline))
                amplitude = max(
                    amplitude,
                    float(np.polyval(coefficients, vertex - x_origin))
                    - vertex_baseline,
                )

    left_slope = math.nan
    right_slope = math.nan
    if peak_sample > 0:
        left_slope = float(
            (corrected[peak_sample] - corrected[0])
            / (wavelength_nm[peak_sample] - wavelength_nm[0])
        )
    if peak_sample < len(voltage_v) - 1:
        right_slope = float(
            (corrected[-1] - corrected[peak_sample])
            / (wavelength_nm[-1] - wavelength_nm[peak_sample])
        )
    slope_denominator = abs(left_slope) + abs(right_slope)
    asymmetry = (
        (abs(left_slope) - abs(right_slope)) / slope_denominator
        if math.isfinite(slope_denominator) and slope_denominator > 0.0
        else math.nan
    )

    width = math.nan
    if amplitude > 0.0 and 0 < peak_sample < len(voltage_v) - 1:
        half = 0.5 * amplitude
        left = None
        right = None
        for index in range(peak_sample - 1, -1, -1):
            if corrected[index] <= half <= corrected[index + 1]:
                left = _crossing(
                    float(wavelength_nm[index]),
                    float(corrected[index]),
                    float(wavelength_nm[index + 1]),
                    float(corrected[index + 1]),
                    half,
                )
                break
        for index in range(peak_sample, len(voltage_v) - 1):
            if corrected[index] >= half >= corrected[index + 1]:
                right = _crossing(
                    float(wavelength_nm[index]),
                    float(corrected[index]),
                    float(wavelength_nm[index + 1]),
                    float(corrected[index + 1]),
                    half,
                )
                break
        if left is not None and right is not None and right > left:
            width = right - left
        else:
            quality |= PeakQuality.WIDTH_UNRESOLVED
    else:
        quality |= PeakQuality.WIDTH_UNRESOLVED

    return _LocalShape(
        center, amplitude, width, left_slope, right_slope, asymmetry, corrected, quality
    )


def _relative(current: float, baseline: float, *, offset_one: bool = False) -> float:
    if (
        not math.isfinite(current)
        or not math.isfinite(baseline)
        or abs(baseline) <= 1e-15
    ):
        return math.nan
    ratio = current / baseline
    return ratio - 1.0 if offset_one else ratio


def _frame_values(
    frame_or_codes: Any, channel: int
) -> tuple[Sequence[float], Sequence[int]]:
    if isinstance(frame_or_codes, SpectrumFrame):
        return frame_or_codes.wavelengths_nm, frame_or_codes.raw_adc_codes[channel]
    raise TypeError("expected SpectrumFrame when wavelengths_nm is omitted")


def _template_estimate_quality(estimate: ShiftEstimate) -> PeakQuality:
    """Translate an optimizer rejection into stable, serializable quality bits."""

    if estimate.accepted:
        return PeakQuality.GOOD
    quality = PeakQuality.TEMPLATE_FIT_REJECTED
    if estimate.reason == "uncertainty_too_large":
        quality |= PeakQuality.TEMPLATE_UNCERTAINTY_HIGH
    elif estimate.reason == "template_mismatch":
        quality |= PeakQuality.TEMPLATE_MISMATCH
    elif estimate.reason == "shift_at_search_boundary":
        quality |= PeakQuality.TEMPLATE_SHIFT_AT_BOUNDARY
    elif estimate.reason in {
        "nonpositive_or_too_small_amplitude",
        "amplitude_scale_out_of_range",
    }:
        quality |= PeakQuality.TEMPLATE_AMPLITUDE_INVALID
    else:
        quality |= PeakQuality.TEMPLATE_SHIFT_UNIDENTIFIABLE
    return quality


def _template_peak_center_nm(template: Any) -> float:
    dense_x = np.asarray(template.dense_wavelength_nm, dtype=float)
    dense_y = np.asarray(template.dense_signal, dtype=float)
    return float(dense_x[int(np.argmax(dense_y))])


def extract_ch1_peak_features(
    wavelengths_or_frame: Any,
    raw_ch1_adc_codes: Sequence[int] | None = None,
    *,
    baseline_adc_codes: Sequence[int] | None = None,
    baseline_frame: SpectrumFrame | None = None,
    grating_count: int = DEFAULT_GRATING_COUNT,
    points_per_peak: int = DEFAULT_POINTS_PER_PEAK,
    transimpedance_ohm: float | None = None,
    baseline_transimpedance_ohm: float | None = None,
    minimum_amplitude_v: float = 0.003,
    digital_gain_audit: float | None = None,
    sampling_template: FingerSamplingTemplate | Any | None = None,
    template_max_uncertainty_pm: float = 5.0,
    template_max_normalized_rmse: float = 0.25,
    template_shift_bounds_nm: tuple[float, float] | None = None,
) -> FingerFeatureFrame:
    """Extract nine local peak features from unscaled CH1 ADC codes.

    ``digital_gain_audit`` is copied to the result only.  Supplying a different
    value cannot change any numerical feature.  When ``sampling_template`` is
    supplied, ``center_shift_pm`` comes from a translated fit of the measured
    (possibly asymmetric or rippled) dense peak rather than a three-point
    parabola.  The remaining amplitude/width/slope/shape features are retained
    for backwards-compatible model vectors.
    """

    if int(grating_count) != DEFAULT_GRATING_COUNT:
        raise ValueError("this finger has exactly 9 gratings on CH1")
    if int(points_per_peak) < 3:
        raise ValueError("points_per_peak must be at least 3")
    if (
        not math.isfinite(float(template_max_uncertainty_pm))
        or float(template_max_uncertainty_pm) <= 0.0
    ):
        raise ValueError("template_max_uncertainty_pm must be finite and positive")
    if (
        not math.isfinite(float(template_max_normalized_rmse))
        or float(template_max_normalized_rmse) <= 0.0
    ):
        raise ValueError("template_max_normalized_rmse must be finite and positive")

    current_frame = (
        wavelengths_or_frame
        if isinstance(wavelengths_or_frame, SpectrumFrame)
        else None
    )
    if current_frame is not None:
        if raw_ch1_adc_codes is not None:
            raise ValueError("raw_ch1_adc_codes must be omitted with SpectrumFrame")
        wavelengths_nm, raw_codes = _frame_values(current_frame, DEFAULT_SENSOR_CHANNEL)
        sequence = current_frame.sequence
        monotonic_ns = current_frame.monotonic_ns
        table_crc32 = current_frame.table_crc32
        if transimpedance_ohm is None:
            transimpedance_ohm = current_frame.transimpedance_ohm.get(
                DEFAULT_SENSOR_CHANNEL
            )
        if digital_gain_audit is None:
            digital_gain_audit = current_frame.digital_gain.get(DEFAULT_SENSOR_CHANNEL)
    else:
        wavelengths_nm = wavelengths_or_frame
        if raw_ch1_adc_codes is None:
            raise ValueError("raw_ch1_adc_codes is required")
        raw_codes = raw_ch1_adc_codes
        sequence = 0
        monotonic_ns = 0
        table_crc32 = None

    if baseline_frame is not None:
        if baseline_adc_codes is not None:
            raise ValueError("choose baseline_frame or baseline_adc_codes, not both")
        if tuple(float(v) for v in baseline_frame.wavelengths_nm) != tuple(
            float(v) for v in wavelengths_nm
        ):
            raise ValueError("baseline and current wavelengths must match")
        baseline_adc_codes = baseline_frame.raw_adc_codes[DEFAULT_SENSOR_CHANNEL]
        if baseline_transimpedance_ohm is None:
            baseline_transimpedance_ohm = baseline_frame.transimpedance_ohm.get(
                DEFAULT_SENSOR_CHANNEL
            )
    if baseline_adc_codes is None:
        baseline_adc_codes = raw_codes

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    expected_points = int(grating_count) * int(points_per_peak)
    if wavelengths.ndim != 1 or len(wavelengths) != expected_points:
        raise ValueError(
            f"expected {expected_points} points (9 peaks x {points_per_peak}), got {len(wavelengths)}"
        )
    if len(raw_codes) != expected_points or len(baseline_adc_codes) != expected_points:
        raise ValueError(
            "current and baseline ADC arrays must match the wavelength table"
        )
    current_v, _ = _adc_voltage(raw_codes)
    baseline_v, _ = _adc_voltage(baseline_adc_codes)

    template_bundle = None
    template_axis_matches = True
    if sampling_template is not None:
        template_bundle = load_finger_sampling_template(sampling_template)
        if expected_points != template_bundle.point_count:
            raise ValueError(
                "sampling template requires the strict 9 peak x 5 point layout"
            )
        template_axis_matches = template_bundle.wavelength_axis_matches(wavelengths)

    current_tia = 1.0 if transimpedance_ohm is None else float(transimpedance_ohm)
    baseline_tia = (
        current_tia
        if baseline_transimpedance_ohm is None
        else float(baseline_transimpedance_ohm)
    )
    if not math.isfinite(current_tia) or current_tia <= 0.0:
        raise ValueError("transimpedance_ohm must be finite and positive")
    if not math.isfinite(baseline_tia) or baseline_tia <= 0.0:
        raise ValueError("baseline_transimpedance_ohm must be finite and positive")

    frame_quality = PeakQuality.GOOD
    if current_frame is not None and current_frame.packet_crc_ok is False:
        frame_quality |= PeakQuality.FRAME_CRC_FAILED
    if (
        current_frame is not None
        and baseline_frame is not None
        and current_frame.table_crc32 is not None
        and baseline_frame.table_crc32 is not None
        and current_frame.table_crc32 != baseline_frame.table_crc32
    ):
        frame_quality |= PeakQuality.TABLE_CRC_MISMATCH
    if template_bundle is not None:
        if not template_axis_matches:
            frame_quality |= PeakQuality.TABLE_CRC_MISMATCH
        if current_frame is not None and (
            current_frame.table_crc32 is None
            or int(current_frame.table_crc32) != int(template_bundle.table_crc32)
        ):
            frame_quality |= PeakQuality.TABLE_CRC_MISMATCH
        if baseline_frame is not None and (
            baseline_frame.table_crc32 is None
            or int(baseline_frame.table_crc32) != int(template_bundle.table_crc32)
        ):
            frame_quality |= PeakQuality.TABLE_CRC_MISMATCH

    peaks = []
    for peak_index in range(DEFAULT_GRATING_COUNT):
        start = peak_index * int(points_per_peak)
        stop = start + int(points_per_peak)
        x = wavelengths[start:stop]
        _, current_adc_quality = _adc_voltage(raw_codes[start:stop])
        _, baseline_adc_quality = _adc_voltage(baseline_adc_codes[start:stop])
        current = _local_shape(x, current_v[start:stop], minimum_amplitude_v)
        baseline = _local_shape(x, baseline_v[start:stop], minimum_amplitude_v)
        quality = (
            current.quality | current_adc_quality | baseline_adc_quality | frame_quality
        )
        baseline_fatal = baseline.quality & (
            PeakQuality.NONFINITE_SAMPLE
            | PeakQuality.NONMONOTONIC_WAVELENGTH
            | PeakQuality.LOW_AMPLITUDE
            | PeakQuality.PEAK_AT_EDGE
        )
        if baseline_fatal:
            quality |= PeakQuality.BASELINE_INVALID

        center_nm = current.center_nm
        center_shift_pm = (current.center_nm - baseline.center_nm) * 1000.0
        center_uncertainty_pm = math.nan
        template_fit_normalized_rmse = math.nan
        template_fit_reason = "legacy_local_shape"
        if template_bundle is not None:
            if not template_axis_matches:
                quality |= (
                    PeakQuality.TEMPLATE_FIT_REJECTED | PeakQuality.TABLE_CRC_MISMATCH
                )
                center_nm = math.nan
                center_shift_pm = math.nan
                template_fit_reason = "wavelength_table_mismatch"
            else:
                measured_template = template_bundle.plan.peaks[peak_index]
                noise_std = template_bundle.selected_noise_std_v[peak_index]
                estimate_options = {
                    "noise_std": noise_std,
                    "shift_bounds_nm": template_shift_bounds_nm,
                    "max_uncertainty_nm": float(template_max_uncertainty_pm) / 1000.0,
                    "max_normalized_rmse": float(template_max_normalized_rmse),
                    # The analogue transimpedance is user-selectable and can
                    # differ from the teacher scan by 20x.  Absolute signal
                    # validity remains guarded by ADC range and LOW_AMPLITUDE.
                    "min_amplitude_scale": 0.01,
                    "max_amplitude_scale": 100.0,
                }
                current_estimate = estimate_template_shift(
                    measured_template,
                    current_v[start:stop],
                    **estimate_options,
                )
                baseline_estimate = estimate_template_shift(
                    measured_template,
                    baseline_v[start:stop],
                    **estimate_options,
                )
                quality |= _template_estimate_quality(current_estimate)
                baseline_template_quality = _template_estimate_quality(
                    baseline_estimate
                )
                quality |= baseline_template_quality
                if baseline_template_quality:
                    quality |= PeakQuality.BASELINE_INVALID
                template_fit_reason = (
                    f"current:{current_estimate.reason};"
                    f"baseline:{baseline_estimate.reason}"
                )
                template_fit_normalized_rmse = max(
                    float(current_estimate.normalized_rmse),
                    float(baseline_estimate.normalized_rmse),
                )
                if current_estimate.accepted and baseline_estimate.accepted:
                    template_center = _template_peak_center_nm(measured_template)
                    center_nm = template_center + current_estimate.shift_nm
                    center_shift_pm = (
                        current_estimate.shift_nm - baseline_estimate.shift_nm
                    ) * 1000.0
                    center_uncertainty_pm = 1000.0 * math.hypot(
                        current_estimate.uncertainty_nm,
                        baseline_estimate.uncertainty_nm,
                    )
                else:
                    center_nm = math.nan
                    center_shift_pm = math.nan

        current_amplitude_physical = current.amplitude / current_tia
        baseline_amplitude_physical = baseline.amplitude / baseline_tia
        current_left_physical = current.left_slope / current_tia
        baseline_left_physical = baseline.left_slope / baseline_tia
        current_right_physical = current.right_slope / current_tia
        baseline_right_physical = baseline.right_slope / baseline_tia
        normalized_profile = tuple(
            np.nan_to_num(
                (current.corrected / current_tia)
                / max(abs(baseline_amplitude_physical), 1e-15),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).tolist()
        )
        peaks.append(
            PeakFeature(
                peak_index=peak_index,
                center_nm=center_nm,
                center_shift_pm=center_shift_pm,
                amplitude_v=current.amplitude,
                amplitude_relative=_relative(
                    current_amplitude_physical,
                    baseline_amplitude_physical,
                    offset_one=True,
                ),
                width_nm=current.width_nm,
                width_relative=_relative(
                    current.width_nm, baseline.width_nm, offset_one=True
                ),
                left_slope_v_per_nm=current.left_slope,
                left_slope_relative=_relative(
                    current_left_physical, baseline_left_physical, offset_one=True
                ),
                right_slope_v_per_nm=current.right_slope,
                right_slope_relative=_relative(
                    current_right_physical, baseline_right_physical, offset_one=True
                ),
                asymmetry=current.asymmetry,
                asymmetry_delta=current.asymmetry - baseline.asymmetry,
                normalized_profile=normalized_profile,
                quality_mask=int(quality),
                center_uncertainty_pm=center_uncertainty_pm,
                template_fit_normalized_rmse=template_fit_normalized_rmse,
                template_fit_reason=template_fit_reason,
            )
        )

    return FingerFeatureFrame(
        sequence=sequence,
        monotonic_ns=monotonic_ns,
        channel=DEFAULT_SENSOR_CHANNEL,
        peaks=tuple(peaks),
        table_crc32=table_crc32,
        digital_gain_audit=(
            None if digital_gain_audit is None else float(digital_gain_audit)
        ),
    )


# Concise aliases for callers that do not need to repeat the channel name.
extract_peak_features = extract_ch1_peak_features
extract_features = extract_ch1_peak_features


def build_causal_features(
    frames: Iterable[FingerFeatureFrame],
) -> CausalFeatureWindow:
    extractor = CausalFeatureExtractor()
    result = None
    for frame in frames:
        result = extractor.update(frame)
    if result is None:
        raise ValueError("at least one feature frame is required")
    return result


__all__ = [
    "CAUSAL_FRAME_COUNT",
    "FEATURE_NAMES",
    "CausalFeatureExtractor",
    "CausalFeatureWindow",
    "FingerFeatureFrame",
    "FingerSamplingTemplate",
    "PeakFeature",
    "PeakQuality",
    "build_causal_features",
    "extract_ch1_peak_features",
    "extract_features",
    "extract_peak_features",
    "load_finger_sampling_template",
]
