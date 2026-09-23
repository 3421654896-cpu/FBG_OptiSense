"""Auditable end-to-end acceptance for the mechanical-finger objective.

This module deliberately does not equate a fast ADC stream with a verified
15 Hz pressure response.  Bandwidth acceptance needs a synchronized physical
pressure reference and a separately measured low-frequency sensitivity.
Position and contact-area acceptance need an independent held-out truth set.

The functions are pure/offline: they never open a device or command Mach3.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VALID_POSITION_TRUTH = {
    "mach3_calibrated",
    "camera_calibrated",
    "cmm_measured",
}
VALID_AREA_TRUTH = {
    "measured",
    "calibrated",
    "pressure_film",
    "camera_contact_patch_calibrated",
    "known_indenter_contact",
}


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite samples")
    return result


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


@dataclass(frozen=True)
class ToneFit:
    frequency_hz: float
    amplitude: float
    phase_deg: float
    offset: float
    linear_drift_per_s: float
    r_squared: float


@dataclass(frozen=True)
class PhysicalReferenceTrace:
    """A traceable reference signal on the same host monotonic clock as ADC.

    Setting ``synchronized`` alone is deliberately insufficient.  The source,
    calibration record, synchronization method, and a finite clock-error bound
    are required so a saved report cannot silently turn an unsynchronised load
    cell/pressure trace into 15 Hz evidence.
    """

    monotonic_ns: Sequence[int]
    values: Sequence[float]
    quantity: str
    unit: str
    source_device: str
    calibration_id: str
    synchronization_method: str
    maximum_clock_error_ms: float
    timebase: str = "host_monotonic_ns"
    synchronized: bool = False

    def __post_init__(self) -> None:
        times = np.asarray(self.monotonic_ns, dtype=np.int64).reshape(-1)
        values = _vector(self.values, "reference values")
        if times.size != values.size or times.size < 12:
            raise ValueError("physical reference needs at least 12 paired samples")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("reference monotonic_ns must increase strictly")
        for name in (
            "quantity",
            "unit",
            "source_device",
            "calibration_id",
            "synchronization_method",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        clock_error = float(self.maximum_clock_error_ms)
        if not math.isfinite(clock_error) or clock_error < 0.0:
            raise ValueError("maximum_clock_error_ms must be finite and non-negative")
        if type(self.synchronized) is not bool:
            raise ValueError("synchronized must be a boolean")
        object.__setattr__(self, "monotonic_ns", tuple(int(value) for value in times))
        object.__setattr__(self, "values", tuple(float(value) for value in values))
        for name in (
            "quantity",
            "unit",
            "source_device",
            "calibration_id",
            "synchronization_method",
            "timebase",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        object.__setattr__(self, "maximum_clock_error_ms", clock_error)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PhysicalReferenceTrace":
        if payload.get("schema") != "fbg-physical-reference/v1":
            raise ValueError("unsupported physical-reference schema")
        samples = payload.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise ValueError("physical reference samples must be an array")
        try:
            times = [sample["monotonic_ns"] for sample in samples]
            values = [sample["value"] for sample in samples]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "each physical reference sample needs monotonic_ns and value"
            ) from exc
        return cls(
            monotonic_ns=times,
            values=values,
            quantity=payload.get("quantity", ""),
            unit=payload.get("unit", ""),
            source_device=payload.get("source_device", ""),
            calibration_id=payload.get("calibration_id", ""),
            synchronization_method=payload.get("synchronization_method", ""),
            maximum_clock_error_ms=payload.get("maximum_clock_error_ms", math.nan),
            timebase=payload.get("timebase", ""),
            synchronized=payload.get("synchronized", False),
        )


def load_physical_reference(path: str | Path) -> PhysicalReferenceTrace:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("physical reference document must be a JSON object")
    return PhysicalReferenceTrace.from_mapping(payload)


@dataclass(frozen=True)
class ContactAreaTruthRecord:
    """One independently measured contact area linked to one press event."""

    event_id: str
    area_mm2: float
    source: str
    measurement_id: str
    calibration_id: str
    independent: bool

    def __post_init__(self) -> None:
        for name in ("event_id", "source", "measurement_id", "calibration_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        area = float(self.area_mm2)
        if not math.isfinite(area) or area < 0.0:
            raise ValueError("area_mm2 must be finite and non-negative")
        object.__setattr__(self, "area_mm2", area)
        object.__setattr__(self, "source", self.source.lower())
        if self.source not in VALID_AREA_TRUTH:
            raise ValueError(f"untrusted contact-area truth source: {self.source}")
        if type(self.independent) is not bool:
            raise ValueError("independent must be a boolean")
        if not self.independent:
            raise ValueError("contact-area truth must be independently measured")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContactAreaTruthRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("each contact-area record must be an object")
        return cls(
            event_id=payload.get("event_id", ""),
            area_mm2=payload.get("area_mm2", math.nan),
            source=payload.get("source", ""),
            measurement_id=payload.get("measurement_id", ""),
            calibration_id=payload.get("calibration_id", ""),
            independent=payload.get("independent", False),
        )


@dataclass(frozen=True)
class ContactAreaTruthSet:
    """A fail-closed set of traceable area labels for model training/testing."""

    records: tuple[ContactAreaTruthRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if len(records) < 12:
            raise ValueError("contact-area truth needs at least 12 records")
        event_ids = [record.event_id for record in records]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("contact-area truth event_id values must be unique")
        distinct_areas = len({round(record.area_mm2, 3) for record in records})
        if distinct_areas < 3:
            raise ValueError("contact-area truth needs at least 3 distinct area levels")
        object.__setattr__(self, "records", records)

    @property
    def distinct_area_count(self) -> int:
        return len({round(record.area_mm2, 3) for record in self.records})

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.calibration_id for record in self.records}))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ContactAreaTruthSet":
        if payload.get("schema") != "fbg-contact-area-truth/v1":
            raise ValueError("unsupported contact-area truth schema")
        records = payload.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("contact-area truth records must be an array")
        return cls(tuple(ContactAreaTruthRecord.from_mapping(item) for item in records))


def load_contact_area_truth(path: str | Path) -> ContactAreaTruthSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contact-area truth document must be a JSON object")
    return ContactAreaTruthSet.from_mapping(payload)


@dataclass(frozen=True)
class AlignedPhysicalReference:
    times_s: tuple[float, ...]
    measured_response: tuple[float, ...]
    reference_values: tuple[float, ...]
    original_frame_count: int
    discarded_outside_reference: int
    discarded_reference_gap: int
    maximum_interpolation_gap_ms: float
    reference: PhysicalReferenceTrace


def align_physical_reference(
    frame_monotonic_ns: Sequence[int],
    measured_response: Sequence[float],
    reference: PhysicalReferenceTrace,
    *,
    maximum_reference_gap_ms: float = 20.0,
    maximum_allowed_clock_error_ms: float = 2.0,
) -> AlignedPhysicalReference:
    """Interpolate a qualified physical reference onto received ADC times.

    Frames outside the reference interval or bracketed by a data gap are
    excluded and counted.  Fewer than 12 surviving frames fail closed.
    """

    frame_times = np.asarray(frame_monotonic_ns, dtype=np.int64).reshape(-1)
    response = _vector(measured_response, "measured_response")
    if frame_times.size != response.size or frame_times.size < 12:
        raise ValueError("alignment needs at least 12 paired ADC frames")
    if np.any(frame_times < 0) or np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame monotonic_ns must increase strictly")
    maximum_gap = float(maximum_reference_gap_ms)
    maximum_clock_error = float(maximum_allowed_clock_error_ms)
    if not math.isfinite(maximum_gap) or maximum_gap <= 0.0:
        raise ValueError("maximum_reference_gap_ms must be finite and positive")
    if not math.isfinite(maximum_clock_error) or maximum_clock_error < 0.0:
        raise ValueError(
            "maximum_allowed_clock_error_ms must be finite and non-negative"
        )
    if not reference.synchronized:
        raise ValueError("physical reference is not declared synchronized")
    if reference.timebase != "host_monotonic_ns":
        raise ValueError("physical reference must use host_monotonic_ns")
    if reference.maximum_clock_error_ms > maximum_clock_error:
        raise ValueError(
            "physical-reference clock error exceeds the allowed bound: "
            f"{reference.maximum_clock_error_ms:.3f} > {maximum_clock_error:.3f} ms"
        )

    reference_times = np.asarray(reference.monotonic_ns, dtype=np.int64)
    reference_values = np.asarray(reference.values, dtype=float)
    right = np.searchsorted(reference_times, frame_times, side="left")
    outside = (right == 0) & (frame_times != reference_times[0])
    outside |= right == reference_times.size
    exact = np.zeros(frame_times.size, dtype=bool)
    in_range = ~outside
    exact[in_range] = reference_times[right[in_range]] == frame_times[in_range]

    left = np.maximum(right - 1, 0)
    bracket_gap_ns = np.zeros(frame_times.size, dtype=np.int64)
    interpolated = in_range & ~exact
    bracket_gap_ns[interpolated] = (
        reference_times[right[interpolated]] - reference_times[left[interpolated]]
    )
    gap_limit_ns = maximum_gap * 1_000_000.0
    bad_gap = interpolated & (bracket_gap_ns > gap_limit_ns)
    keep = in_range & ~bad_gap
    if int(np.count_nonzero(keep)) < 12:
        raise ValueError("fewer than 12 ADC frames have a qualified reference bracket")

    aligned_values = np.empty(frame_times.size, dtype=float)
    aligned_values[exact] = reference_values[right[exact]]
    good_interp = interpolated & ~bad_gap
    fraction = (
        (frame_times[good_interp] - reference_times[left[good_interp]])
        / bracket_gap_ns[good_interp]
    )
    aligned_values[good_interp] = reference_values[left[good_interp]] + fraction * (
        reference_values[right[good_interp]] - reference_values[left[good_interp]]
    )
    kept_times = frame_times[keep]
    times_s = (kept_times - kept_times[0]).astype(float) / 1_000_000_000.0
    observed_gap_ms = (
        0.0
        if not np.any(good_interp)
        else float(np.max(bracket_gap_ns[good_interp])) / 1_000_000.0
    )
    return AlignedPhysicalReference(
        times_s=tuple(float(value) for value in times_s),
        measured_response=tuple(float(value) for value in response[keep]),
        reference_values=tuple(float(value) for value in aligned_values[keep]),
        original_frame_count=int(frame_times.size),
        discarded_outside_reference=int(np.count_nonzero(outside)),
        discarded_reference_gap=int(np.count_nonzero(bad_gap)),
        maximum_interpolation_gap_ms=observed_gap_ms,
        reference=reference,
    )


def fit_tone(
    times_s: Sequence[float], values: Sequence[float], frequency_hz: float
) -> ToneFit:
    """Least-squares sine fit for irregular, strictly ordered sample times."""

    times = _vector(times_s, "times_s")
    signal = _vector(values, "values")
    if times.size != signal.size or times.size < 12:
        raise ValueError("tone fit requires at least 12 paired samples")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must increase strictly")
    frequency = float(frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")

    centred = times - float(np.mean(times))
    angle = 2.0 * math.pi * frequency * times
    design = np.column_stack(
        (np.sin(angle), np.cos(angle), np.ones(times.size), centred)
    )
    coefficients, *_ = np.linalg.lstsq(design, signal, rcond=None)
    fitted = design @ coefficients
    residual = signal - fitted
    total = signal - float(np.mean(signal))
    total_power = float(np.dot(total, total))
    residual_power = float(np.dot(residual, residual))
    r_squared = 0.0 if total_power <= 0.0 else 1.0 - residual_power / total_power
    sine, cosine, offset, drift = (float(value) for value in coefficients)
    return ToneFit(
        frequency_hz=frequency,
        amplitude=math.hypot(sine, cosine),
        phase_deg=math.degrees(math.atan2(cosine, sine)),
        offset=offset,
        linear_drift_per_s=drift,
        r_squared=float(r_squared),
    )


def _wrapped_phase_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class FrequencyResponseEvidence:
    frequency_hz: float
    sample_count: int
    duration_s: float
    observed_cycles: float
    median_sample_rate_hz: float
    maximum_sample_gap_ms: float
    nyquist_timing_pass: bool
    reference_fit: ToneFit
    response_fit: ToneFit
    low_frequency_gain_output_per_input: float | None
    gain_relative_to_low_frequency: float | None
    amplitude_error_db: float | None
    phase_difference_deg: float
    synchronized_physical_reference: bool
    reference_provenance_valid: bool = False
    reference_timebase: str | None = None
    reference_source_device: str | None = None
    reference_calibration_id: str | None = None
    reference_synchronization_method: str | None = None
    maximum_clock_error_ms: float | None = None
    maximum_interpolation_gap_ms: float | None = None
    discarded_unaligned_samples: int = 0
    physical_15hz_verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def assess_frequency_response(
    times_s: Sequence[float],
    reference_pressure: Sequence[float],
    measured_response: Sequence[float],
    *,
    frequency_hz: float = 15.0,
    low_frequency_gain_output_per_input: float | None,
    synchronized_physical_reference: bool,
    minimum_cycles: float = 8.0,
) -> FrequencyResponseEvidence:
    """Measure 15 Hz gain/phase evidence without deciding user tolerances.

    ``low_frequency_gain_output_per_input`` must come from a separate slow or
    static calibration in the same units.  Without it, a large-looking ADC
    tone cannot establish amplitude fidelity.
    """

    times = _vector(times_s, "times_s")
    reference = _vector(reference_pressure, "reference_pressure")
    response = _vector(measured_response, "measured_response")
    if not (times.size == reference.size == response.size):
        raise ValueError("frequency-response arrays must have equal length")
    gaps = np.diff(times)
    if np.any(gaps <= 0.0):
        raise ValueError("times_s must increase strictly")
    frequency = float(frequency_hz)
    duration = float(times[-1] - times[0])
    cycles = duration * frequency
    if cycles < float(minimum_cycles):
        raise ValueError(
            f"only {cycles:.2f} cycles observed; at least {minimum_cycles:.2f} required"
        )
    maximum_gap = float(np.max(gaps))
    nyquist_pass = maximum_gap <= 1.0 / (2.0 * frequency)
    reference_fit = fit_tone(times, reference, frequency)
    response_fit = fit_tone(times, response, frequency)
    if reference_fit.amplitude <= 0.0:
        raise ValueError("physical reference has zero fitted amplitude")

    calibration_gain = (
        None
        if low_frequency_gain_output_per_input is None
        else float(low_frequency_gain_output_per_input)
    )
    if calibration_gain is not None and (
        not math.isfinite(calibration_gain) or calibration_gain <= 0.0
    ):
        raise ValueError("low-frequency gain must be finite and positive")
    relative_gain = None
    amplitude_db = None
    if calibration_gain is not None:
        relative_gain = response_fit.amplitude / (
            reference_fit.amplitude * calibration_gain
        )
        if relative_gain > 0.0:
            amplitude_db = 20.0 * math.log10(relative_gain)

    return FrequencyResponseEvidence(
        frequency_hz=frequency,
        sample_count=int(times.size),
        duration_s=duration,
        observed_cycles=cycles,
        median_sample_rate_hz=1.0 / float(np.median(gaps)),
        maximum_sample_gap_ms=maximum_gap * 1000.0,
        nyquist_timing_pass=nyquist_pass,
        reference_fit=reference_fit,
        response_fit=response_fit,
        low_frequency_gain_output_per_input=calibration_gain,
        gain_relative_to_low_frequency=relative_gain,
        amplitude_error_db=amplitude_db,
        phase_difference_deg=_wrapped_phase_degrees(
            response_fit.phase_deg - reference_fit.phase_deg
        ),
        synchronized_physical_reference=bool(synchronized_physical_reference),
    )


def assess_aligned_frequency_response(
    aligned: AlignedPhysicalReference,
    *,
    frequency_hz: float = 15.0,
    low_frequency_gain_output_per_input: float | None,
    minimum_cycles: float = 8.0,
) -> FrequencyResponseEvidence:
    """Assess bandwidth from a provenance-checked, clock-aligned trace."""

    evidence = assess_frequency_response(
        aligned.times_s,
        aligned.reference_values,
        aligned.measured_response,
        frequency_hz=frequency_hz,
        low_frequency_gain_output_per_input=low_frequency_gain_output_per_input,
        synchronized_physical_reference=True,
        minimum_cycles=minimum_cycles,
    )
    reference = aligned.reference
    return replace(
        evidence,
        reference_provenance_valid=True,
        reference_timebase=reference.timebase,
        reference_source_device=reference.source_device,
        reference_calibration_id=reference.calibration_id,
        reference_synchronization_method=reference.synchronization_method,
        maximum_clock_error_ms=reference.maximum_clock_error_ms,
        maximum_interpolation_gap_ms=aligned.maximum_interpolation_gap_ms,
        discarded_unaligned_samples=(
            aligned.discarded_outside_reference + aligned.discarded_reference_gap
        ),
    )


@dataclass(frozen=True)
class SpatialEvidence:
    independent_held_out_test: bool
    sample_count: int
    distinct_positions: int
    distinct_areas: int
    position_truth_source: str
    area_truth_source: str
    position_median_error_mm: float
    position_p95_error_mm: float
    area_mae_mm2: float
    area_p95_abs_error_mm2: float
    position_truth_valid: bool
    area_truth_valid: bool
    position_and_area_verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def assess_spatial_accuracy(
    true_xy_mm: Sequence[Sequence[float]],
    predicted_xy_mm: Sequence[Sequence[float]],
    true_area_mm2: Sequence[float],
    predicted_area_mm2: Sequence[float],
    *,
    independent_held_out_test: bool,
    position_truth_source: str,
    area_truth_source: str,
) -> SpatialEvidence:
    """Score location and true contact area on an independent test set."""

    truth_xy = np.asarray(true_xy_mm, dtype=float)
    prediction_xy = np.asarray(predicted_xy_mm, dtype=float)
    truth_area = _vector(true_area_mm2, "true_area_mm2")
    prediction_area = _vector(predicted_area_mm2, "predicted_area_mm2")
    if truth_xy.ndim != 2 or truth_xy.shape[1] != 2:
        raise ValueError("true_xy_mm must have shape (n, 2)")
    if prediction_xy.shape != truth_xy.shape:
        raise ValueError("predicted_xy_mm must match true_xy_mm")
    if not np.all(np.isfinite(truth_xy)) or not np.all(np.isfinite(prediction_xy)):
        raise ValueError("position arrays must contain finite coordinates")
    count = truth_xy.shape[0]
    if count < 12 or truth_area.size != count or prediction_area.size != count:
        raise ValueError("spatial evidence requires at least 12 aligned samples")
    if np.any(truth_area < 0.0) or np.any(prediction_area < 0.0):
        raise ValueError("contact areas must be non-negative")

    position_error = np.linalg.norm(prediction_xy - truth_xy, axis=1)
    area_error = np.abs(prediction_area - truth_area)
    distinct_positions = int(np.unique(np.round(truth_xy, 3), axis=0).shape[0])
    distinct_areas = int(np.unique(np.round(truth_area, 3)).size)
    position_source = str(position_truth_source).strip().lower()
    area_source = str(area_truth_source).strip().lower()
    return SpatialEvidence(
        independent_held_out_test=bool(independent_held_out_test),
        sample_count=int(count),
        distinct_positions=distinct_positions,
        distinct_areas=distinct_areas,
        position_truth_source=position_source,
        area_truth_source=area_source,
        position_median_error_mm=float(np.median(position_error)),
        position_p95_error_mm=_percentile(position_error, 95.0),
        area_mae_mm2=float(np.mean(area_error)),
        area_p95_abs_error_mm2=_percentile(area_error, 95.0),
        position_truth_valid=(
            position_source in VALID_POSITION_TRUTH and distinct_positions >= 5
        ),
        area_truth_valid=(area_source in VALID_AREA_TRUTH and distinct_areas >= 3),
    )


@dataclass(frozen=True)
class AcceptanceLimits:
    maximum_amplitude_error_db: float
    maximum_phase_error_deg: float
    minimum_tone_r_squared: float
    maximum_position_p95_error_mm: float
    maximum_area_p95_error_mm2: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in asdict(self).values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("acceptance limits must be finite and non-negative")
        if not 0.0 <= self.minimum_tone_r_squared <= 1.0:
            raise ValueError("minimum_tone_r_squared must be within 0..1")


@dataclass(frozen=True)
class PhysicalAcceptanceSpatialRecord:
    """Predictions and independent position truth for one held-out event."""

    event_id: str
    true_x_mm: float
    true_y_mm: float
    predicted_x_mm: float
    predicted_y_mm: float
    predicted_area_mm2: float

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("spatial event_id must not be empty")
        object.__setattr__(self, "event_id", event_id)
        for name in (
            "true_x_mm",
            "true_y_mm",
            "predicted_x_mm",
            "predicted_y_mm",
            "predicted_area_mm2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "predicted_area_mm2" and value < 0.0:
                raise ValueError("predicted_area_mm2 must be non-negative")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any]
    ) -> "PhysicalAcceptanceSpatialRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("each spatial record must be an object")
        return cls(
            event_id=payload.get("event_id", ""),
            true_x_mm=payload.get("true_x_mm", math.nan),
            true_y_mm=payload.get("true_y_mm", math.nan),
            predicted_x_mm=payload.get("predicted_x_mm", math.nan),
            predicted_y_mm=payload.get("predicted_y_mm", math.nan),
            predicted_area_mm2=payload.get("predicted_area_mm2", math.nan),
        )


@dataclass(frozen=True)
class PhysicalAcceptanceSession:
    """CH1 frequency and held-out spatial predictions awaiting truth pairing."""

    frame_monotonic_ns: tuple[int, ...]
    measured_response: tuple[float, ...]
    frequency_hz: float
    low_frequency_gain_output_per_input: float | None
    minimum_cycles: float
    maximum_reference_gap_ms: float
    maximum_allowed_clock_error_ms: float
    independent_held_out_test: bool
    position_truth_source: str
    spatial_records: tuple[PhysicalAcceptanceSpatialRecord, ...]
    limits: AcceptanceLimits

    def __post_init__(self) -> None:
        times = np.asarray(self.frame_monotonic_ns, dtype=np.int64).reshape(-1)
        response = _vector(self.measured_response, "measured_response")
        if times.size != response.size or times.size < 12:
            raise ValueError("acceptance session needs at least 12 paired CH1 frames")
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError("CH1 frame monotonic_ns must increase strictly")
        frequency = float(self.frequency_hz)
        if not math.isfinite(frequency) or abs(frequency - 15.0) > 0.05:
            raise ValueError("physical acceptance frequency must be 15.00 +/- 0.05 Hz")
        gain = self.low_frequency_gain_output_per_input
        if gain is not None:
            gain = float(gain)
            if not math.isfinite(gain) or gain <= 0.0:
                raise ValueError("low-frequency gain must be finite and positive")
        minimum_cycles = float(self.minimum_cycles)
        maximum_gap = float(self.maximum_reference_gap_ms)
        maximum_clock_error = float(self.maximum_allowed_clock_error_ms)
        if not math.isfinite(minimum_cycles) or minimum_cycles < 8.0:
            raise ValueError("acceptance session must cover at least 8 cycles")
        if not math.isfinite(maximum_gap) or maximum_gap <= 0.0:
            raise ValueError("maximum_reference_gap_ms must be finite and positive")
        if not math.isfinite(maximum_clock_error) or maximum_clock_error < 0.0:
            raise ValueError(
                "maximum_allowed_clock_error_ms must be finite and non-negative"
            )
        if type(self.independent_held_out_test) is not bool:
            raise ValueError("independent_held_out_test must be a boolean")
        records = tuple(self.spatial_records)
        if len(records) < 12:
            raise ValueError("acceptance session needs at least 12 spatial records")
        event_ids = [record.event_id for record in records]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("spatial event_id values must be unique")
        position_source = str(self.position_truth_source).strip().lower()
        if not position_source:
            raise ValueError("position_truth_source must not be empty")
        object.__setattr__(self, "frame_monotonic_ns", tuple(int(v) for v in times))
        object.__setattr__(self, "measured_response", tuple(float(v) for v in response))
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "low_frequency_gain_output_per_input", gain)
        object.__setattr__(self, "minimum_cycles", minimum_cycles)
        object.__setattr__(self, "maximum_reference_gap_ms", maximum_gap)
        object.__setattr__(
            self, "maximum_allowed_clock_error_ms", maximum_clock_error
        )
        object.__setattr__(self, "position_truth_source", position_source)
        object.__setattr__(self, "spatial_records", records)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PhysicalAcceptanceSession":
        if payload.get("schema") != "fbg-physical-acceptance-session/v1":
            raise ValueError("unsupported physical-acceptance session schema")
        frames = payload.get("ch1_frames")
        records = payload.get("spatial_records")
        limits = payload.get("limits")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError("ch1_frames must be an array")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError("spatial_records must be an array")
        if not isinstance(limits, Mapping):
            raise ValueError("limits must be an object")
        try:
            times = [frame["monotonic_ns"] for frame in frames]
            response = [frame["response_value"] for frame in frames]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "each CH1 frame needs monotonic_ns and response_value"
            ) from exc
        try:
            acceptance_limits = AcceptanceLimits(**dict(limits))
        except TypeError as exc:
            raise ValueError("limits fields do not match AcceptanceLimits") from exc
        return cls(
            frame_monotonic_ns=tuple(times),
            measured_response=tuple(response),
            frequency_hz=payload.get("frequency_hz", math.nan),
            low_frequency_gain_output_per_input=payload.get(
                "low_frequency_gain_output_per_input"
            ),
            minimum_cycles=payload.get("minimum_cycles", 8.0),
            maximum_reference_gap_ms=payload.get(
                "maximum_reference_gap_ms", 20.0
            ),
            maximum_allowed_clock_error_ms=payload.get(
                "maximum_allowed_clock_error_ms", 2.0
            ),
            independent_held_out_test=payload.get(
                "independent_held_out_test", False
            ),
            position_truth_source=payload.get("position_truth_source", ""),
            spatial_records=tuple(
                PhysicalAcceptanceSpatialRecord.from_mapping(record)
                for record in records
            ),
            limits=acceptance_limits,
        )


def load_physical_acceptance_session(
    path: str | Path,
) -> PhysicalAcceptanceSession:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("physical-acceptance session must be a JSON object")
    return PhysicalAcceptanceSession.from_mapping(payload)


def evaluate_end_to_end_acceptance(
    frequency: FrequencyResponseEvidence,
    spatial: SpatialEvidence,
    limits: AcceptanceLimits,
) -> dict:
    """Return a fail-closed requirement-by-requirement acceptance record."""

    amplitude_available = frequency.amplitude_error_db is not None
    checks = {
        "synchronized_physical_reference": frequency.synchronized_physical_reference,
        "reference_provenance_valid": frequency.reference_provenance_valid,
        "nyquist_timing": frequency.nyquist_timing_pass,
        "reference_tone_quality": (
            frequency.reference_fit.r_squared >= limits.minimum_tone_r_squared
        ),
        "response_tone_quality": (
            frequency.response_fit.r_squared >= limits.minimum_tone_r_squared
        ),
        "amplitude_calibration_available": amplitude_available,
        "amplitude_fidelity": bool(
            amplitude_available
            and abs(float(frequency.amplitude_error_db))
            <= limits.maximum_amplitude_error_db
        ),
        "phase_fidelity": (
            abs(frequency.phase_difference_deg) <= limits.maximum_phase_error_deg
        ),
        "independent_spatial_test": spatial.independent_held_out_test,
        "position_truth_valid": spatial.position_truth_valid,
        "area_truth_valid": spatial.area_truth_valid,
        "position_accuracy": (
            spatial.position_p95_error_mm <= limits.maximum_position_p95_error_mm
        ),
        "area_accuracy": (
            spatial.area_p95_abs_error_mm2 <= limits.maximum_area_p95_error_mm2
        ),
    }
    frequency_passed = all(
        checks[name]
        for name in (
            "synchronized_physical_reference",
            "reference_provenance_valid",
            "nyquist_timing",
            "reference_tone_quality",
            "response_tone_quality",
            "amplitude_calibration_available",
            "amplitude_fidelity",
            "phase_fidelity",
        )
    )
    position_passed = all(
        checks[name]
        for name in (
            "independent_spatial_test",
            "position_truth_valid",
            "position_accuracy",
        )
    )
    area_passed = all(
        checks[name]
        for name in (
            "independent_spatial_test",
            "area_truth_valid",
            "area_accuracy",
        )
    )
    passed = frequency_passed and position_passed and area_passed
    return {
        "schema": "fbg-physical-pressure-acceptance/v1",
        "physical_15hz_verified": frequency_passed,
        "position_accuracy_verified": position_passed,
        "physical_contact_area_verified": area_passed,
        "end_to_end_goal_verified": passed,
        "checks": checks,
        "limits": asdict(limits),
        "frequency_evidence": frequency.to_dict(),
        "spatial_evidence": spatial.to_dict(),
    }


def evaluate_physical_acceptance_session(
    session: PhysicalAcceptanceSession,
    pressure_reference: PhysicalReferenceTrace,
    area_truth: ContactAreaTruthSet,
) -> dict:
    """Pair all three evidence streams and execute the fail-closed gates."""

    aligned = align_physical_reference(
        session.frame_monotonic_ns,
        session.measured_response,
        pressure_reference,
        maximum_reference_gap_ms=session.maximum_reference_gap_ms,
        maximum_allowed_clock_error_ms=session.maximum_allowed_clock_error_ms,
    )
    frequency = assess_aligned_frequency_response(
        aligned,
        frequency_hz=session.frequency_hz,
        low_frequency_gain_output_per_input=(
            session.low_frequency_gain_output_per_input
        ),
        minimum_cycles=session.minimum_cycles,
    )

    area_by_event = {record.event_id: record for record in area_truth.records}
    session_event_ids = {record.event_id for record in session.spatial_records}
    area_event_ids = set(area_by_event)
    if session_event_ids != area_event_ids:
        missing = sorted(session_event_ids - area_event_ids)
        extra = sorted(area_event_ids - session_event_ids)
        raise ValueError(
            "spatial and area event IDs must match exactly; "
            f"missing_area={missing[:3]}, unused_area={extra[:3]}"
        )
    area_sources = {record.source for record in area_truth.records}
    if len(area_sources) != 1:
        raise ValueError("one acceptance session must use one area truth source")

    records = session.spatial_records
    spatial = assess_spatial_accuracy(
        [(record.true_x_mm, record.true_y_mm) for record in records],
        [(record.predicted_x_mm, record.predicted_y_mm) for record in records],
        [area_by_event[record.event_id].area_mm2 for record in records],
        [record.predicted_area_mm2 for record in records],
        independent_held_out_test=session.independent_held_out_test,
        position_truth_source=session.position_truth_source,
        area_truth_source=next(iter(area_sources)),
    )
    report = evaluate_end_to_end_acceptance(frequency, spatial, session.limits)
    report["session_schema"] = "fbg-physical-acceptance-session/v1"
    report["paired_spatial_event_count"] = len(records)
    report["area_calibration_ids"] = list(area_truth.calibration_ids)
    return report


__all__ = [
    "AlignedPhysicalReference",
    "AcceptanceLimits",
    "ContactAreaTruthRecord",
    "ContactAreaTruthSet",
    "FrequencyResponseEvidence",
    "PhysicalReferenceTrace",
    "PhysicalAcceptanceSession",
    "PhysicalAcceptanceSpatialRecord",
    "SpatialEvidence",
    "ToneFit",
    "align_physical_reference",
    "assess_aligned_frequency_response",
    "assess_frequency_response",
    "assess_spatial_accuracy",
    "evaluate_end_to_end_acceptance",
    "evaluate_physical_acceptance_session",
    "fit_tone",
    "load_contact_area_truth",
    "load_physical_acceptance_session",
    "load_physical_reference",
]
