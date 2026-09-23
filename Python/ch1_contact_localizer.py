"""Causal CH1 nine-FBG contact localization and response-footprint estimates.

The module is intentionally independent of the Qt/3-D viewer and of all device
control code.  It accepts either the nine fitted Bragg-peak shifts or one
strict 45-point CH1 spectrum (five samples per FBG).  Geometry comes from the
same ``finger_sensor_layout.yaml`` and ``t31_touch_calibration.json`` files as
the viewer, so an estimate can be reported both in normalized CAD coordinates
and in the calibrated Mach3 work-coordinate plane.

``effective_response_area_mm2`` is the spatial participation of the nine FBG
responses expressed on the calibrated geometry.  It is *not* physical contact
area.  No physical-contact-area field is populated until independently traced
area labels (pressure film, a calibrated contact mechanics setup, etc.) exist.

The multi-rate tracker is causal.  MAP/SURVEY measurements may refresh all
peaks while TRACK may refresh only a subset; stale peaks decay explicitly and
never masquerade as fresh measurements.  Prediction between acquisitions is
short-horizon interpolation/extrapolation only and cannot recover a 15 Hz
signal from an acquisition stream below Nyquist (30 Hz; 60 Hz is recommended).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from finger_dataset import (
    ADC_CODE_COUNT,
    ADC_MAX_CODE,
    ADC_REFERENCE_V,
    MAX_INDENTATION_MM,
    SpectrumFrame,
)
from finger_features import PeakQuality, extract_ch1_peak_features
from finger_sampling_template import (
    FingerSamplingTemplate,
    load_finger_sampling_template,
)

HERE = Path(__file__).resolve().parent
DEFAULT_LAYOUT_PATH = HERE / "finger_sensor_layout.yaml"
DEFAULT_TOUCH_CALIBRATION_PATH = HERE / "t31_touch_calibration.json"
DEFAULT_CERTIFIED_TEMPLATE_PATH = (
    HERE / "artifacts" / "ch1_measured_template_selection_v4_certified_20260905.json"
)

SENSOR_COUNT = 9
POINTS_PER_PEAK = 5
POINT_COUNT = SENSOR_COUNT * POINTS_PER_PEAK
CH1 = 1

# These are the same normalized silicone outline and true-size CAD spans used
# by mechanical_finger_3d.py.  Keeping the values here avoids importing Qt.
SILICONE_X_RADIUS = 0.53
SILICONE_Y_CENTER = 0.32
SILICONE_Y_RADIUS = 0.50
DEFAULT_MODEL_X_HALF_SPAN_MM = 17.832737
DEFAULT_MODEL_Y_HALF_SPAN_MM = 24.3053895

AREA_KIND = "effective_fbg_response_area_not_physical_contact_area"
AREA_METHOD = "response_participation_ratio_x_calibrated_sensor_cell"
ALLOWED_PROFILES = (
    "MAP",
    "SURVEY",
    "TRACK",
    "TRACK13_WIDE",
    "TRACK11_SINGLE",
)
PROFILE_FRESH_LOCAL_INDICES = {
    "MAP": frozenset(range(POINTS_PER_PEAK)),
    "SURVEY": frozenset((1, 3)),
    "TRACK": frozenset((1, 2, 3)),
    "TRACK13_WIDE": frozenset((1, 2, 3)),
    "TRACK11_SINGLE": frozenset((1, 2, 3)),
}
SPECTRUM_EVIDENCE_KIND = "normalized_45point_residual_plus_local_shape"
SHIFT_EVIDENCE_KIND = "nine_peak_shift_pm"

_PRIMARY_FATAL_QUALITY = (
    PeakQuality.NONFINITE_SAMPLE
    | PeakQuality.ADC_OUT_OF_RANGE
    | PeakQuality.NONMONOTONIC_WAVELENGTH
    | PeakQuality.LOW_AMPLITUDE
    | PeakQuality.BASELINE_INVALID
    | PeakQuality.FRAME_CRC_FAILED
    | PeakQuality.TABLE_CRC_MISMATCH
)
_SOFT_SHAPE_QUALITY = PeakQuality.PEAK_AT_EDGE | PeakQuality.WIDTH_UNRESOLVED


def _normalize_profile_name(profile: str) -> str:
    """Return a stable machine name while accepting UI-style hyphens."""

    profile_name = str(profile).strip().upper().replace("-", "_")
    if profile_name not in ALLOWED_PROFILES:
        raise ValueError(f"profile must be one of {ALLOWED_PROFILES}")
    return profile_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(
    values: Sequence[float], name: str, *, length: int = SENSOR_COUNT
) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size != int(length):
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _physical_adc_values(
    codes: Sequence[int],
    *,
    transimpedance_ohm: float | None,
    expected_length: int = POINT_COUNT,
) -> np.ndarray:
    """Convert raw ADC codes to a gain-independent proportional PD signal."""

    try:
        raw = np.asarray(codes, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("ADC array must be numeric") from exc
    if raw.size != int(expected_length):
        raise ValueError(f"ADC array must contain exactly {expected_length} values")
    invalid = ~np.isfinite(raw) | (raw < 0.0) | (raw > ADC_MAX_CODE)
    raw = raw.copy()
    raw[invalid] = np.nan
    tia = 1.0 if transimpedance_ohm is None else float(transimpedance_ohm)
    if not math.isfinite(tia) or tia <= 0.0:
        raise ValueError("transimpedance_ohm must be finite and positive")
    return raw * ADC_REFERENCE_V / ADC_CODE_COUNT / tia


def _normalized_peak_profiles(
    wavelengths_nm: Sequence[float],
    codes: Sequence[int],
    *,
    transimpedance_ohm: float | None,
) -> np.ndarray:
    """Return endpoint-baseline-corrected, amplitude-normalized 9x5 profiles."""

    wavelength = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    if wavelength.size != POINT_COUNT or not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength table must contain 45 finite values")
    values = _physical_adc_values(
        codes,
        transimpedance_ohm=transimpedance_ohm,
    )
    profiles = np.full((SENSOR_COUNT, POINTS_PER_PEAK), np.nan, dtype=float)
    for peak_index in range(SENSOR_COUNT):
        start = peak_index * POINTS_PER_PEAK
        stop = start + POINTS_PER_PEAK
        x = wavelength[start:stop]
        y = values[start:stop]
        if np.any(np.diff(x) <= 0.0):
            continue
        line = np.interp(x, (x[0], x[-1]), (y[0], y[-1]))
        corrected = y - line
        if not np.all(np.isfinite(corrected)):
            continue
        amplitude = float(np.max(corrected))
        if not math.isfinite(amplitude) or amplitude <= 1e-18:
            continue
        profiles[peak_index] = corrected / amplitude
    return profiles


def _point_acquisition_metadata(
    *,
    profile: str,
    fresh_point_indices: Iterable[int] | None,
    point_age_frames: Sequence[float] | None,
    sample_offsets_2us: Sequence[int] | None,
    map_age_frames: int | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int | None, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate raw multi-rate metadata and reduce it to nine peak updates."""

    profile_name = _normalize_profile_name(profile)
    if fresh_point_indices is not None and sample_offsets_2us is not None:
        raise ValueError(
            "choose fresh_point_indices or sample_offsets_2us freshness, not both"
        )

    offsets: tuple[int | None, ...]
    if sample_offsets_2us is not None:
        raw_offsets = np.asarray(sample_offsets_2us).reshape(-1)
        if raw_offsets.size != POINT_COUNT:
            raise ValueError("sample_offsets_2us must contain exactly 45 values")
        parsed_offsets: list[int | None] = []
        fresh = np.zeros(POINT_COUNT, dtype=bool)
        for point_index, raw in enumerate(raw_offsets):
            value = int(raw)
            if not 0 <= value <= 0xFFFF:
                raise ValueError("sample offsets must be within 0..65535")
            if value == 0xFFFF:
                parsed_offsets.append(None)
            else:
                parsed_offsets.append(value)
                fresh[point_index] = True
        offsets = tuple(parsed_offsets)
    elif fresh_point_indices is not None:
        fresh = np.zeros(POINT_COUNT, dtype=bool)
        for raw in fresh_point_indices:
            point_index = int(raw)
            if not 0 <= point_index < POINT_COUNT:
                raise ValueError("fresh point index must be within 0..44")
            fresh[point_index] = True
        offsets = (None,) * POINT_COUNT
    else:
        fresh = np.ones(POINT_COUNT, dtype=bool)
        offsets = (None,) * POINT_COUNT

    if map_age_frames is not None:
        map_age = int(map_age_frames)
        if map_age < 0:
            raise ValueError("map_age_frames must be non-negative")
    else:
        map_age = 1
    if point_age_frames is None:
        ages = np.where(fresh, 0.0, float(map_age))
    else:
        ages = _finite_vector(
            point_age_frames,
            "point_age_frames",
            length=POINT_COUNT,
        )
        if np.any(~np.isfinite(ages)) or np.any(ages < 0.0):
            raise ValueError("point_age_frames must be finite and non-negative")
        if np.any(fresh & (ages != 0.0)):
            raise ValueError("fresh points must have zero point_age_frames")

    expected_local = PROFILE_FRESH_LOCAL_INDICES[profile_name]
    peak_fresh = np.zeros(SENSOR_COUNT, dtype=bool)
    peak_activity_fresh = np.zeros(SENSOR_COUNT, dtype=bool)
    peak_ages = np.zeros(SENSOR_COUNT, dtype=float)
    for peak_index in range(SENSOR_COUNT):
        start = peak_index * POINTS_PER_PEAK
        local_fresh = fresh[start : start + POINTS_PER_PEAK]
        peak_activity_fresh[peak_index] = bool(np.any(local_fresh))
        peak_fresh[peak_index] = all(local_fresh[index] for index in expected_local)
        if peak_fresh[peak_index]:
            peak_ages[peak_index] = 0.0
        else:
            peak_ages[peak_index] = float(np.max(ages[start : start + POINTS_PER_PEAK]))
    return fresh, ages, offsets, peak_fresh, peak_activity_fresh, peak_ages


def _tuple_matrix(value: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 2) or not np.all(np.isfinite(array)):
        raise ValueError("coordinate transform must be a finite 2x2 matrix")
    return (
        (float(array[0, 0]), float(array[0, 1])),
        (float(array[1, 0]), float(array[1, 1])),
    )


@dataclass(frozen=True)
class SensorSite:
    sensor_id: int
    peak_index: int
    name: str
    x_norm: float
    y_norm: float


@dataclass(frozen=True)
class ContactGeometry:
    """Nine sensor sites and the normalized-CAD to millimetre transform."""

    sensors: tuple[SensorSite, ...]
    normalized_to_mm_matrix: tuple[tuple[float, float], tuple[float, float]]
    mm_intercept: tuple[float, float]
    coordinate_frame: str
    position_calibrated: bool
    calibration_path: str | None
    calibration_sha256: str | None
    effective_sensor_cell_area_mm2: float
    silicone_planform_area_mm2: float
    baseline_frames: int = 20
    activation_pm: float = 5.0
    full_scale_pm: float = 100.0

    def __post_init__(self) -> None:
        if len(self.sensors) != SENSOR_COUNT:
            raise ValueError("contact geometry must contain exactly nine sensors")
        if sorted(site.sensor_id for site in self.sensors) != list(range(1, 10)):
            raise ValueError("sensor ids must be exactly 1..9")
        if sorted(site.peak_index for site in self.sensors) != list(range(9)):
            raise ValueError("peak indices must be exactly 0..8")
        matrix = np.asarray(self.normalized_to_mm_matrix, dtype=float)
        intercept = np.asarray(self.mm_intercept, dtype=float)
        if matrix.shape != (2, 2) or abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise ValueError("contact geometry transform is singular")
        if intercept.shape != (2,) or not np.all(np.isfinite(intercept)):
            raise ValueError("contact geometry intercept is invalid")
        if self.effective_sensor_cell_area_mm2 <= 0.0:
            raise ValueError("effective sensor cell area must be positive")
        if self.silicone_planform_area_mm2 <= 0.0:
            raise ValueError("silicone planform area must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.normalized_to_mm_matrix, dtype=float)

    @property
    def intercept(self) -> np.ndarray:
        return np.asarray(self.mm_intercept, dtype=float)

    @property
    def sensor_positions_norm_by_peak(self) -> np.ndarray:
        result = np.empty((SENSOR_COUNT, 2), dtype=float)
        for site in self.sensors:
            result[site.peak_index] = (site.x_norm, site.y_norm)
        return result

    @property
    def sensor_positions_mm_by_peak(self) -> np.ndarray:
        return self.normalized_to_mm(self.sensor_positions_norm_by_peak)

    def normalized_to_mm(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(point, dtype=float)
        if values.shape[-1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError("normalized point must end in two finite coordinates")
        return values @ self.matrix + self.intercept

    def mm_to_normalized(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(point, dtype=float)
        if values.shape[-1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError("millimetre point must end in two finite coordinates")
        return (values - self.intercept) @ np.linalg.inv(self.matrix)


def _two_point_similarity(
    normalized: np.ndarray,
    measured_mm: np.ndarray,
    half_spans_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_mm = normalized * half_spans_mm
    model_delta = model_mm[1] - model_mm[0]
    measured_delta = measured_mm[1] - measured_mm[0]
    model_length = float(np.linalg.norm(model_delta))
    measured_length = float(np.linalg.norm(measured_delta))
    if model_length <= 1e-9 or measured_length <= 1e-9:
        raise ValueError("two-point finger calibration anchors are coincident")
    scale = measured_length / model_length
    angle = math.atan2(measured_delta[1], measured_delta[0]) - math.atan2(
        model_delta[1], model_delta[0]
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    metric_to_measured = scale * np.asarray(
        ((cosine, sine), (-sine, cosine)), dtype=float
    )
    matrix = np.diag(half_spans_mm) @ metric_to_measured
    intercept = measured_mm[0] - normalized[0] @ matrix
    return matrix, intercept


def _sensor_cell_area_mm2(sites: Sequence[SensorSite], matrix: np.ndarray) -> float:
    xs = sorted({round(float(site.x_norm), 12) for site in sites})
    ys = sorted({round(float(site.y_norm), 12) for site in sites})
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("sensor layout must span at least two rows and columns")
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    return abs(float(np.linalg.det(matrix))) * abs(dx * dy)


def _silicone_area_mm2(matrix: np.ndarray) -> float:
    # Area of |x/a|^4 + |y/b|^4 <= 1 in normalized coordinates.
    normalized_area = (
        4.0
        * SILICONE_X_RADIUS
        * SILICONE_Y_RADIUS
        * math.gamma(1.25) ** 2
        / math.gamma(1.5)
    )
    return normalized_area * abs(float(np.linalg.det(matrix)))


def load_contact_geometry(
    layout_path: Path | str = DEFAULT_LAYOUT_PATH,
    touch_calibration_path: Path | str | None = DEFAULT_TOUCH_CALIBRATION_PATH,
) -> ContactGeometry:
    """Load the CH1 3x3 layout and optional CNC work-coordinate calibration."""

    layout_source = Path(layout_path).resolve()
    payload = yaml.safe_load(layout_source.read_text(encoding="utf-8")) or {}
    if int(payload.get("channel", -1)) != CH1:
        raise ValueError("mechanical-finger localization is pinned to CH1")
    sites = tuple(
        SensorSite(
            sensor_id=int(item["id"]),
            peak_index=int(item["peak_index"]),
            name=str(item.get("name", f"G{item['id']}")),
            x_norm=float(item["x"]),
            y_norm=float(item["y"]),
        )
        for item in payload.get("sensors", ())
    )
    if any(
        not math.isfinite(value) or abs(value) > 1.0
        for site in sites
        for value in (site.x_norm, site.y_norm)
    ):
        raise ValueError(
            "sensor normalized coordinates must be finite and within -1..1"
        )

    half_spans = np.asarray(
        (DEFAULT_MODEL_X_HALF_SPAN_MM, DEFAULT_MODEL_Y_HALF_SPAN_MM), dtype=float
    )
    matrix = np.diag(half_spans)
    intercept = np.zeros(2, dtype=float)
    coordinate_frame = "cad_model_mm"
    calibrated = False
    calibration_path = None
    calibration_sha = None

    if touch_calibration_path is not None:
        calibration_source = Path(touch_calibration_path).resolve()
        calibration = json.loads(calibration_source.read_text(encoding="utf-8"))
        if str(calibration.get("coordinate_units", "mm")).lower() != "mm":
            raise ValueError("touch calibration coordinates must use millimetres")
        mapping = calibration.get("model_mapping", {})
        if isinstance(mapping, Mapping):
            half_spans = np.asarray(
                (
                    float(
                        mapping.get("cad_half_span_x_mm", DEFAULT_MODEL_X_HALF_SPAN_MM)
                    ),
                    float(
                        mapping.get("cad_half_span_y_mm", DEFAULT_MODEL_Y_HALF_SPAN_MM)
                    ),
                ),
                dtype=float,
            )
        anchors = []
        for item in calibration.get("samples", ()):
            if not isinstance(item, Mapping):
                continue
            if not bool(item.get("active", True)) or not bool(item.get("contact")):
                continue
            required = ("model_x_norm", "model_y_norm", "work_x", "work_y")
            if any(item.get(name) is None for name in required):
                continue
            anchors.append(
                (
                    (float(item["model_x_norm"]), float(item["model_y_norm"])),
                    (float(item["work_x"]), float(item["work_y"])),
                )
            )
        if len(anchors) == 2:
            normalized = np.asarray([item[0] for item in anchors], dtype=float)
            measured = np.asarray([item[1] for item in anchors], dtype=float)
            matrix, intercept = _two_point_similarity(normalized, measured, half_spans)
            calibrated = True
        elif len(anchors) >= 3:
            normalized = np.asarray([item[0] for item in anchors], dtype=float)
            measured = np.asarray([item[1] for item in anchors], dtype=float)
            design = np.column_stack((normalized, np.ones(len(normalized))))
            if np.linalg.matrix_rank(design) < 3:
                raise ValueError("active touch calibration anchors are collinear")
            coefficients = np.linalg.lstsq(design, measured, rcond=None)[0]
            matrix = coefficients[:2]
            intercept = coefficients[2]
            calibrated = True
        else:
            matrix = np.diag(half_spans)
            intercept = np.zeros(2, dtype=float)
        if calibrated:
            coordinate_frame = "mach3_work_mm"
            calibration_path = str(calibration_source)
            calibration_sha = _sha256(calibration_source)

    matrix_tuple = _tuple_matrix(matrix)
    return ContactGeometry(
        sensors=sites,
        normalized_to_mm_matrix=matrix_tuple,
        mm_intercept=(float(intercept[0]), float(intercept[1])),
        coordinate_frame=coordinate_frame,
        position_calibrated=calibrated,
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha,
        effective_sensor_cell_area_mm2=_sensor_cell_area_mm2(sites, matrix),
        silicone_planform_area_mm2=_silicone_area_mm2(matrix),
        baseline_frames=max(3, int(payload.get("baseline_frames", 20))),
        activation_pm=max(0.0, float(payload.get("activation_pm", 5.0))),
        full_scale_pm=max(1.0, float(payload.get("full_scale_pm", 100.0))),
    )


@dataclass(frozen=True)
class ContactLocalizerConfig:
    activation_pm: float = 5.0
    full_scale_pm: float = 100.0
    remove_common_mode: bool = False
    position_power: float = 1.35
    minimum_valid_peaks: int = 3
    freshness_half_life_frames: float = 3.0
    maximum_peak_age_frames: int = 12
    confidence_floor: float = 0.05
    normalized_residual_activation_rms: float = 0.035
    amplitude_relative_activation: float = 0.10
    width_relative_activation: float = 0.15
    slope_relative_activation: float = 0.30
    asymmetry_activation: float = 0.15
    residual_evidence_weight: float = 0.65
    template_audit_disagreement_pm: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.activation_pm) or self.activation_pm < 0.0:
            raise ValueError("activation_pm must be finite and non-negative")
        if (
            not math.isfinite(self.full_scale_pm)
            or self.full_scale_pm <= self.activation_pm
        ):
            raise ValueError("full_scale_pm must exceed activation_pm")
        if not math.isfinite(self.position_power) or self.position_power <= 0.0:
            raise ValueError("position_power must be finite and positive")
        if not 1 <= int(self.minimum_valid_peaks) <= SENSOR_COUNT:
            raise ValueError("minimum_valid_peaks must be within 1..9")
        if (
            not math.isfinite(self.freshness_half_life_frames)
            or self.freshness_half_life_frames <= 0.0
        ):
            raise ValueError("freshness_half_life_frames must be positive")
        if int(self.maximum_peak_age_frames) < 0:
            raise ValueError("maximum_peak_age_frames must be non-negative")
        positive_scales = (
            self.normalized_residual_activation_rms,
            self.amplitude_relative_activation,
            self.width_relative_activation,
            self.slope_relative_activation,
            self.asymmetry_activation,
            self.template_audit_disagreement_pm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_scales):
            raise ValueError("spectrum evidence scales must be finite and positive")
        if not 0.0 <= self.residual_evidence_weight <= 1.0:
            raise ValueError("residual_evidence_weight must be within 0..1")


@dataclass(frozen=True)
class SpectrumContactEvidence:
    """Traceable 45-point evidence; template displacement is audit-only."""

    timestamp_ns: int
    profile: str
    map_age_frames: int | None
    fresh_point_mask_45: tuple[bool, ...]
    point_age_frames_45: tuple[float, ...]
    sample_offsets_2us_45: tuple[int | None, ...]
    # ``peak_fresh_mask`` means enough new points for a local peak-shape/
    # centre update.  TRACK11/13 additionally carry one certified high-slope
    # sentinel on each non-ROI peak; those fresh activity observations are
    # represented separately and must not be advertised as fitted centres.
    peak_fresh_mask: tuple[bool, ...]
    peak_activity_fresh_mask: tuple[bool, ...]
    peak_age_frames: tuple[float, ...]
    peak_response_scores: tuple[float, ...]
    normalized_residual_rms: tuple[float | None, ...]
    normalized_profile_residual_45: tuple[float | None, ...]
    local_shape_features: tuple[tuple[float | None, ...], ...]
    local_shape_shifts_pm: tuple[float | None, ...]
    primary_quality_masks: tuple[int, ...]
    valid_peak_mask: tuple[bool, ...]
    template_audit_shifts_pm: tuple[float | None, ...]
    template_audit_uncertainty_pm: tuple[float | None, ...]
    template_audit_quality_masks: tuple[int, ...]
    template_audit_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "profile": self.profile,
            "map_age_frames": self.map_age_frames,
            "fresh_point_mask_45": list(self.fresh_point_mask_45),
            "point_age_frames_45": list(self.point_age_frames_45),
            "sample_offsets_2us_45": list(self.sample_offsets_2us_45),
            "peak_fresh_mask": list(self.peak_fresh_mask),
            "peak_activity_fresh_mask": list(self.peak_activity_fresh_mask),
            "peak_age_frames": list(self.peak_age_frames),
            "evidence_kind": SPECTRUM_EVIDENCE_KIND,
            "primary_feature_policy": (
                "baseline-normalized 45-point residual and local shape"
            ),
            "template_policy": "audit_only_not_localization_weight",
            "peak_response_scores": list(self.peak_response_scores),
            "normalized_residual_rms": list(self.normalized_residual_rms),
            "normalized_profile_residual_45": list(self.normalized_profile_residual_45),
            "local_shape_features": [list(row) for row in self.local_shape_features],
            "local_shape_shifts_pm": list(self.local_shape_shifts_pm),
            "primary_quality_masks": list(self.primary_quality_masks),
            "valid_peak_mask": list(self.valid_peak_mask),
            "template_audit_shifts_pm": list(self.template_audit_shifts_pm),
            "template_audit_uncertainty_pm": list(self.template_audit_uncertainty_pm),
            "template_audit_quality_masks": list(self.template_audit_quality_masks),
            "template_audit_reasons": list(self.template_audit_reasons),
        }


@dataclass(frozen=True)
class ContactEstimate:
    timestamp_ns: int
    contact_detected: bool
    x_norm: float | None
    y_norm: float | None
    x_mm: float | None
    y_mm: float | None
    coordinate_frame: str
    position_calibrated: bool
    confidence: float
    dominant_sensor_id: int | None
    peak_shifts_pm: tuple[float | None, ...]
    common_mode_pm: float
    response_weights: tuple[float, ...]
    valid_peak_count: int
    fresh_peak_count: int
    peak_response_pm: float
    rms_response_pm: float
    effective_sensor_count: float
    effective_response_area_mm2: float
    effective_response_area_kind: str = AREA_KIND
    effective_response_area_method: str = AREA_METHOD
    physical_contact_area_mm2: None = None
    physical_contact_area_calibrated: bool = False
    peak_quality_masks: tuple[int, ...] = (0,) * SENSOR_COUNT
    quality_flags: tuple[str, ...] = ()
    localization_evidence_kind: str = SHIFT_EVIDENCE_KIND
    peak_localization_scores: tuple[float, ...] = ()
    normalized_residual_rms: tuple[float | None, ...] = ()
    normalized_profile_residual_45: tuple[float | None, ...] = ()
    local_shape_features: tuple[tuple[float | None, ...], ...] = ()
    local_shape_shifts_pm: tuple[float | None, ...] = ()
    template_audit_shifts_pm: tuple[float | None, ...] = ()
    template_audit_uncertainty_pm: tuple[float | None, ...] = ()
    template_audit_quality_masks: tuple[int, ...] = ()
    template_audit_reasons: tuple[str, ...] = ()
    acquisition_profile: str | None = None
    map_age_frames: int | None = None
    fresh_point_mask_45: tuple[bool, ...] = ()
    point_age_frames_45: tuple[float, ...] = ()
    sample_offsets_2us_45: tuple[int | None, ...] = ()
    peak_age_frames: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "contact_detected": self.contact_detected,
            "position": {
                "x_norm": self.x_norm,
                "y_norm": self.y_norm,
                "x_mm": self.x_mm,
                "y_mm": self.y_mm,
                "coordinate_frame": self.coordinate_frame,
                "calibrated": self.position_calibrated,
            },
            "confidence": self.confidence,
            "dominant_sensor_id": self.dominant_sensor_id,
            "peak_shifts_pm": list(self.peak_shifts_pm),
            "common_mode_pm": self.common_mode_pm,
            "response_weights": list(self.response_weights),
            "valid_peak_count": self.valid_peak_count,
            "fresh_peak_count": self.fresh_peak_count,
            "peak_response_pm": self.peak_response_pm,
            "rms_response_pm": self.rms_response_pm,
            "effective_sensor_count": self.effective_sensor_count,
            "effective_response_area_mm2": self.effective_response_area_mm2,
            "effective_response_area_kind": self.effective_response_area_kind,
            "effective_response_area_method": self.effective_response_area_method,
            "physical_contact_area_mm2": None,
            "physical_contact_area_calibrated": False,
            "area_semantics": {
                "display_label_zh": "有效响应面积（估计影响范围）",
                "is_physical_contact_area": False,
                "warning_zh": (
                    "未做独立面积标定；固定8 mm压头不能据此反演真实接触面积"
                ),
            },
            "peak_quality_masks": list(self.peak_quality_masks),
            "quality_flags": list(self.quality_flags),
            "localization_evidence_kind": self.localization_evidence_kind,
            "peak_localization_scores": list(self.peak_localization_scores),
            "normalized_residual_rms": list(self.normalized_residual_rms),
            "normalized_profile_residual_45": list(self.normalized_profile_residual_45),
            "local_shape_features": [list(row) for row in self.local_shape_features],
            "local_shape_shifts_pm": list(self.local_shape_shifts_pm),
            "template_audit": {
                "policy": "audit_only_not_localization_weight",
                "shifts_pm": list(self.template_audit_shifts_pm),
                "uncertainty_pm": list(self.template_audit_uncertainty_pm),
                "quality_masks": list(self.template_audit_quality_masks),
                "reasons": list(self.template_audit_reasons),
            },
            "acquisition": {
                "profile": self.acquisition_profile,
                "map_age_frames": self.map_age_frames,
                "fresh_point_mask_45": list(self.fresh_point_mask_45),
                "point_age_frames_45": list(self.point_age_frames_45),
                "sample_offsets_2us_45": list(self.sample_offsets_2us_45),
                "peak_age_frames": list(self.peak_age_frames),
                "sample_offset_unit_us": 2,
                "cached_sample_offset_wire_value": 0xFFFF,
            },
        }


class CH1NinePeakLocalizer:
    """Fast deterministic contact estimator for the strict CH1 9x5 table."""

    def __init__(
        self,
        geometry: ContactGeometry | None = None,
        *,
        config: ContactLocalizerConfig | None = None,
        sampling_template: FingerSamplingTemplate
        | Path
        | str
        | Mapping[str, Any]
        | None = None,
    ) -> None:
        self.geometry = geometry or load_contact_geometry()
        self.config = config or ContactLocalizerConfig(
            activation_pm=self.geometry.activation_pm,
            full_scale_pm=self.geometry.full_scale_pm,
        )
        self.sampling_template = (
            None
            if sampling_template is None
            else load_finger_sampling_template(sampling_template)
        )

    @classmethod
    def from_project_defaults(
        cls,
        *,
        use_certified_template: bool = True,
        config: ContactLocalizerConfig | None = None,
    ) -> CH1NinePeakLocalizer:
        template: Path | None = None
        if use_certified_template:
            if not DEFAULT_CERTIFIED_TEMPLATE_PATH.exists():
                raise FileNotFoundError(
                    "certified CH1 9x5 sampling template is not installed"
                )
            template = DEFAULT_CERTIFIED_TEMPLATE_PATH
        return cls(config=config, sampling_template=template)

    def estimate_peak_shifts(
        self,
        peak_shifts_pm: Sequence[float],
        *,
        timestamp_ns: int | None = None,
        valid_mask: Sequence[bool] | None = None,
        fresh_peak_mask: Sequence[bool] | None = None,
        peak_age_frames: Sequence[float] | None = None,
        peak_uncertainty_pm: Sequence[float] | None = None,
        peak_quality_masks: Sequence[int] | None = None,
        peak_response_scores: Sequence[float] | None = None,
    ) -> ContactEstimate:
        shifts = _finite_vector(peak_shifts_pm, "peak_shifts_pm")
        valid = np.isfinite(shifts)
        if valid_mask is not None:
            supplied = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if supplied.size != SENSOR_COUNT:
                raise ValueError("valid_mask must contain exactly 9 values")
            valid &= supplied

        quality_masks = np.zeros(SENSOR_COUNT, dtype=np.uint16)
        if peak_quality_masks is not None:
            raw_quality = np.asarray(peak_quality_masks).reshape(-1)
            if raw_quality.size != SENSOR_COUNT:
                raise ValueError("peak_quality_masks must contain exactly 9 values")
            quality_masks = raw_quality.astype(np.uint16)
        fatal_quality = (quality_masks & int(_PRIMARY_FATAL_QUALITY)) != 0
        valid &= ~fatal_quality

        ages = np.zeros(SENSOR_COUNT, dtype=float)
        if peak_age_frames is not None:
            ages = _finite_vector(peak_age_frames, "peak_age_frames")
            if np.any(ages < 0.0):
                raise ValueError("peak_age_frames must be non-negative")
        valid &= ages <= float(self.config.maximum_peak_age_frames)

        fresh = (
            valid.copy()
            if fresh_peak_mask is None
            else np.asarray(fresh_peak_mask, dtype=bool).reshape(-1)
        )
        if fresh.size != SENSOR_COUNT:
            raise ValueError("fresh_peak_mask must contain exactly 9 values")
        fresh &= valid

        uncertainty_reliability = np.ones(SENSOR_COUNT, dtype=float)
        if peak_uncertainty_pm is not None:
            uncertainty = _finite_vector(peak_uncertainty_pm, "peak_uncertainty_pm")
            finite_uncertainty = np.isfinite(uncertainty) & (uncertainty >= 0.0)
            scale = max(self.config.activation_pm, 1.0)
            uncertainty_reliability[finite_uncertainty] = 1.0 / (
                1.0 + (uncertainty[finite_uncertainty] / scale) ** 2
            )
            uncertainty_reliability[~finite_uncertainty] = 0.65

        common_mode = 0.0
        corrected = shifts.copy()
        if self.config.remove_common_mode and np.any(valid):
            common_mode = float(np.median(shifts[valid]))
            corrected[valid] -= common_mode

        response = np.zeros(SENSOR_COUNT, dtype=float)
        response[valid] = np.maximum(
            np.abs(corrected[valid]) - self.config.activation_pm, 0.0
        )
        freshness = np.zeros(SENSOR_COUNT, dtype=float)
        freshness[valid] = np.exp2(
            -ages[valid] / self.config.freshness_half_life_frames
        )
        soft_quality = (quality_masks & int(_SOFT_SHAPE_QUALITY)) != 0
        shape_quality_reliability = np.where(soft_quality, 0.65, 1.0)
        reliability = freshness * uncertainty_reliability * shape_quality_reliability
        if peak_response_scores is None:
            activation_scale = max(self.config.activation_pm, 1e-9)
            localization_scores = np.zeros(SENSOR_COUNT, dtype=float)
            localization_scores[valid] = np.abs(corrected[valid]) / activation_scale
        else:
            localization_scores = _finite_vector(
                peak_response_scores, "peak_response_scores"
            )
            if np.any(
                ~np.isfinite(localization_scores[valid])
                | (localization_scores[valid] < 0.0)
            ):
                raise ValueError(
                    "valid peak_response_scores must be finite and non-negative"
                )
            # A score of one is the configured evidence activation boundary.
            # Scaling the excess by activation_pm keeps the historical weight
            # dynamic range without pretending that the score itself is pm.
            response = np.zeros(SENSOR_COUNT, dtype=float)
            response[valid] = np.maximum(localization_scores[valid] - 1.0, 0.0) * max(
                self.config.activation_pm, 1.0
            )
        weights = np.power(response, self.config.position_power) * reliability
        total_weight = float(np.sum(weights))
        contact = total_weight > 0.0

        flags: list[str] = []
        valid_count = int(np.count_nonzero(valid))
        fresh_count = int(np.count_nonzero(fresh))
        if valid_count < self.config.minimum_valid_peaks:
            flags.append("insufficient_valid_peaks")
        if np.any(fatal_quality):
            flags.append("fatal_peak_quality_excluded")
        if np.any(soft_quality & valid):
            flags.append("edge_or_width_quality_downweighted")
        if fresh_count < SENSOR_COUNT:
            flags.append("partial_or_stale_peak_update")
        if self.config.remove_common_mode:
            flags.append("median_common_mode_removed")
        if not self.geometry.position_calibrated:
            flags.append("position_in_cad_model_frame_not_mach3_calibrated")

        peak_response = (
            float(np.max(np.abs(corrected[valid]))) if np.any(valid) else 0.0
        )
        rms_response = (
            float(np.sqrt(np.mean(np.square(corrected[valid]))))
            if np.any(valid)
            else 0.0
        )

        if not contact:
            flags.append("below_contact_activation")
            return ContactEstimate(
                timestamp_ns=time.monotonic_ns()
                if timestamp_ns is None
                else int(timestamp_ns),
                contact_detected=False,
                x_norm=None,
                y_norm=None,
                x_mm=None,
                y_mm=None,
                coordinate_frame=self.geometry.coordinate_frame,
                position_calibrated=self.geometry.position_calibrated,
                confidence=0.0,
                dominant_sensor_id=None,
                peak_shifts_pm=tuple(
                    float(value) if math.isfinite(value) else None for value in shifts
                ),
                common_mode_pm=common_mode,
                response_weights=tuple(float(value) for value in weights),
                valid_peak_count=valid_count,
                fresh_peak_count=fresh_count,
                peak_response_pm=peak_response,
                rms_response_pm=rms_response,
                effective_sensor_count=0.0,
                effective_response_area_mm2=0.0,
                peak_quality_masks=tuple(int(value) for value in quality_masks),
                quality_flags=tuple(flags),
                peak_localization_scores=tuple(
                    float(value) if math.isfinite(value) else 0.0
                    for value in localization_scores
                ),
            )

        positions_norm = self.geometry.sensor_positions_norm_by_peak
        centroid_norm = np.sum(positions_norm * weights[:, None], axis=0) / total_weight
        centroid_mm = self.geometry.normalized_to_mm(centroid_norm)
        dominant_peak = int(np.argmax(weights))
        dominant_site = next(
            site for site in self.geometry.sensors if site.peak_index == dominant_peak
        )

        squared_sum = float(np.dot(weights, weights))
        effective_count = (
            total_weight * total_weight / squared_sum if squared_sum > 0.0 else 0.0
        )
        effective_area = min(
            self.geometry.silicone_planform_area_mm2,
            effective_count * self.geometry.effective_sensor_cell_area_mm2,
        )

        cell_x = max(float(np.ptp(positions_norm[:, 0])) / 2.0, 1e-9)
        cell_y = max(float(np.ptp(positions_norm[:, 1])) / 2.0, 1e-9)
        offsets = positions_norm - centroid_norm
        spread_cells = float(
            np.sum(
                weights
                * ((offsets[:, 0] / cell_x) ** 2 + (offsets[:, 1] / cell_y) ** 2)
            )
            / total_weight
        )
        coherence = 1.0 / math.sqrt(1.0 + spread_cells)
        if peak_response_scores is None:
            maximum_excess = max(0.0, peak_response - self.config.activation_pm)
            signal_score = 1.0 - math.exp(
                -maximum_excess / max(self.config.activation_pm, 1.0)
            )
        else:
            maximum_excess_score = max(
                0.0, float(np.max(localization_scores[valid])) - 1.0
            )
            signal_score = 1.0 - math.exp(-maximum_excess_score)
        coverage_score = min(1.0, valid_count / self.config.minimum_valid_peaks)
        fresh_support = float(np.dot(weights, freshness) / total_weight)
        uncertainty_score = float(
            np.dot(weights, uncertainty_reliability) / total_weight
        )
        confidence = float(
            np.clip(
                signal_score
                * coverage_score
                * math.sqrt(max(0.0, fresh_support))
                * uncertainty_score
                * (0.65 + 0.35 * coherence),
                0.0,
                1.0,
            )
        )
        if confidence < self.config.confidence_floor:
            flags.append("low_localization_confidence")

        return ContactEstimate(
            timestamp_ns=time.monotonic_ns()
            if timestamp_ns is None
            else int(timestamp_ns),
            contact_detected=True,
            x_norm=float(centroid_norm[0]),
            y_norm=float(centroid_norm[1]),
            x_mm=float(centroid_mm[0]),
            y_mm=float(centroid_mm[1]),
            coordinate_frame=self.geometry.coordinate_frame,
            position_calibrated=self.geometry.position_calibrated,
            confidence=confidence,
            dominant_sensor_id=dominant_site.sensor_id,
            peak_shifts_pm=tuple(
                float(value) if math.isfinite(value) else None for value in shifts
            ),
            common_mode_pm=common_mode,
            response_weights=tuple(float(value) for value in weights),
            valid_peak_count=valid_count,
            fresh_peak_count=fresh_count,
            peak_response_pm=peak_response,
            rms_response_pm=rms_response,
            effective_sensor_count=float(effective_count),
            effective_response_area_mm2=float(effective_area),
            peak_quality_masks=tuple(int(value) for value in quality_masks),
            quality_flags=tuple(flags),
            peak_localization_scores=tuple(
                float(value) if math.isfinite(value) else 0.0
                for value in localization_scores
            ),
        )

    def estimate_spectrum(
        self,
        wavelengths_or_frame: Sequence[float] | SpectrumFrame,
        raw_ch1_adc_codes: Sequence[int] | None = None,
        *,
        baseline_adc_codes: Sequence[int] | None = None,
        baseline_frame: SpectrumFrame | None = None,
        transimpedance_ohm: float | None = None,
        baseline_transimpedance_ohm: float | None = None,
        timestamp_ns: int | None = None,
        frame_start_timestamp_ns: int | None = None,
        profile: str = "MAP",
        fresh_peak_mask: Sequence[bool] | None = None,
        fresh_point_indices: Iterable[int] | None = None,
        point_age_frames: Sequence[float] | None = None,
        sample_offsets_2us: Sequence[int] | None = None,
        map_age_frames: int | None = None,
        peak_age_frames: Sequence[float] | None = None,
    ) -> ContactEstimate:
        """Localize a strict 45-point CH1 frame from robust primary evidence.

        The measured-template displacement is deliberately not used as the
        localization weight.  It is returned as an audit feature because a
        five-point nonlinear template fit can jump between local minima even
        when the original five-sample outline remains smooth.
        """

        evidence = self.extract_spectrum_evidence(
            wavelengths_or_frame,
            raw_ch1_adc_codes,
            baseline_adc_codes=baseline_adc_codes,
            baseline_frame=baseline_frame,
            transimpedance_ohm=transimpedance_ohm,
            baseline_transimpedance_ohm=baseline_transimpedance_ohm,
            timestamp_ns=timestamp_ns,
            frame_start_timestamp_ns=frame_start_timestamp_ns,
            profile=profile,
            fresh_point_indices=fresh_point_indices,
            point_age_frames=point_age_frames,
            sample_offsets_2us=sample_offsets_2us,
            map_age_frames=map_age_frames,
        )
        if fresh_peak_mask is not None and fresh_point_indices is not None:
            raise ValueError("choose fresh_peak_mask or fresh_point_indices, not both")
        estimate = self.estimate_peak_shifts(
            [
                math.nan if value is None else value
                for value in evidence.local_shape_shifts_pm
            ],
            timestamp_ns=evidence.timestamp_ns,
            valid_mask=evidence.valid_peak_mask,
            fresh_peak_mask=(
                evidence.peak_fresh_mask if fresh_peak_mask is None else fresh_peak_mask
            ),
            peak_age_frames=(
                evidence.peak_age_frames if peak_age_frames is None else peak_age_frames
            ),
            # The template uncertainty belongs to the audit branch and must
            # not silently attenuate robust residual/local-shape evidence.
            peak_quality_masks=evidence.primary_quality_masks,
            peak_response_scores=evidence.peak_response_scores,
        )
        flags = list(estimate.quality_flags)
        disagreements = []
        for primary, audit in zip(
            evidence.local_shape_shifts_pm,
            evidence.template_audit_shifts_pm,
            strict=True,
        ):
            if primary is None or audit is None:
                continue
            disagreements.append(abs(float(primary) - float(audit)))
        if (
            disagreements
            and max(disagreements) > self.config.template_audit_disagreement_pm
        ):
            flags.append("template_audit_disagrees_with_primary_shape")
        if self.sampling_template is not None and not any(
            value is not None for value in evidence.template_audit_shifts_pm
        ):
            flags.append("template_audit_unavailable_primary_evidence_retained")
        return replace(
            estimate,
            quality_flags=tuple(dict.fromkeys(flags)),
            localization_evidence_kind=SPECTRUM_EVIDENCE_KIND,
            normalized_residual_rms=evidence.normalized_residual_rms,
            normalized_profile_residual_45=evidence.normalized_profile_residual_45,
            local_shape_features=evidence.local_shape_features,
            local_shape_shifts_pm=evidence.local_shape_shifts_pm,
            template_audit_shifts_pm=evidence.template_audit_shifts_pm,
            template_audit_uncertainty_pm=evidence.template_audit_uncertainty_pm,
            template_audit_quality_masks=evidence.template_audit_quality_masks,
            template_audit_reasons=evidence.template_audit_reasons,
            acquisition_profile=evidence.profile,
            map_age_frames=evidence.map_age_frames,
            fresh_point_mask_45=evidence.fresh_point_mask_45,
            point_age_frames_45=evidence.point_age_frames_45,
            sample_offsets_2us_45=evidence.sample_offsets_2us_45,
            peak_age_frames=evidence.peak_age_frames,
        )

    def extract_spectrum_evidence(
        self,
        wavelengths_or_frame: Sequence[float] | SpectrumFrame,
        raw_ch1_adc_codes: Sequence[int] | None = None,
        *,
        baseline_adc_codes: Sequence[int] | None = None,
        baseline_frame: SpectrumFrame | None = None,
        transimpedance_ohm: float | None = None,
        baseline_transimpedance_ohm: float | None = None,
        timestamp_ns: int | None = None,
        frame_start_timestamp_ns: int | None = None,
        profile: str = "MAP",
        fresh_point_indices: Iterable[int] | None = None,
        point_age_frames: Sequence[float] | None = None,
        sample_offsets_2us: Sequence[int] | None = None,
        map_age_frames: int | None = None,
    ) -> SpectrumContactEvidence:
        """Extract model-ready primary evidence and a separate template audit."""

        if timestamp_ns is not None and frame_start_timestamp_ns is not None:
            raise ValueError(
                "choose timestamp_ns or frame_start_timestamp_ns with sample offsets"
            )

        primary = extract_ch1_peak_features(
            wavelengths_or_frame,
            raw_ch1_adc_codes,
            baseline_adc_codes=baseline_adc_codes,
            baseline_frame=baseline_frame,
            transimpedance_ohm=transimpedance_ohm,
            baseline_transimpedance_ohm=baseline_transimpedance_ohm,
            sampling_template=None,
        )

        if isinstance(wavelengths_or_frame, SpectrumFrame):
            current_frame = wavelengths_or_frame
            wavelengths = current_frame.wavelengths_nm
            current_codes = current_frame.raw_adc_codes[CH1]
            current_tia = (
                current_frame.transimpedance_ohm.get(CH1)
                if transimpedance_ohm is None
                else transimpedance_ohm
            )
        else:
            current_frame = None
            wavelengths = wavelengths_or_frame
            if raw_ch1_adc_codes is None:
                raise ValueError("raw_ch1_adc_codes is required")
            current_codes = raw_ch1_adc_codes
            current_tia = transimpedance_ohm

        if baseline_frame is not None:
            baseline_codes = baseline_frame.raw_adc_codes[CH1]
            baseline_tia = (
                baseline_frame.transimpedance_ohm.get(CH1)
                if baseline_transimpedance_ohm is None
                else baseline_transimpedance_ohm
            )
        else:
            baseline_codes = (
                current_codes if baseline_adc_codes is None else baseline_adc_codes
            )
            baseline_tia = (
                current_tia
                if baseline_transimpedance_ohm is None
                else baseline_transimpedance_ohm
            )

        (
            fresh_points,
            point_ages,
            sample_offsets,
            peak_fresh,
            peak_activity_fresh,
            peak_ages,
        ) = _point_acquisition_metadata(
            profile=profile,
            fresh_point_indices=fresh_point_indices,
            point_age_frames=point_age_frames,
            sample_offsets_2us=sample_offsets_2us,
            map_age_frames=map_age_frames,
        )
        profile_name = _normalize_profile_name(profile)

        current_profiles = _normalized_peak_profiles(
            wavelengths,
            current_codes,
            transimpedance_ohm=current_tia,
        )
        baseline_profiles = _normalized_peak_profiles(
            wavelengths,
            baseline_codes,
            transimpedance_ohm=baseline_tia,
        )
        residual = current_profiles - baseline_profiles
        residual_rms = np.full(SENSOR_COUNT, np.nan, dtype=float)
        for peak_index in range(SENSOR_COUNT):
            row = residual[peak_index]
            start = peak_index * POINTS_PER_PEAK
            local_fresh = fresh_points[start : start + POINTS_PER_PEAK]
            expected_local = PROFILE_FRESH_LOCAL_INDICES[profile_name]
            selected = np.asarray(
                [index for index in expected_local if local_fresh[index]],
                dtype=int,
            )
            # A wholly cached peak still carries an explicitly aged MAP value.
            # A partially refreshed peak uses only the profile-defined new
            # samples, avoiding a false full-frame timestamp.
            values = row if selected.size == 0 else row[selected]
            if np.all(np.isfinite(values)):
                residual_rms[peak_index] = float(np.sqrt(np.mean(np.square(values))))

        scores = np.zeros(SENSOR_COUNT, dtype=float)
        local_feature_rows: list[tuple[float | None, ...]] = []
        local_shifts: list[float | None] = []
        primary_masks: list[int] = []
        valid_mask: list[bool] = []
        residual_weight = self.config.residual_evidence_weight
        for peak_index, peak in enumerate(primary.peaks):
            raw_features = (
                peak.center_shift_pm,
                peak.amplitude_relative,
                peak.width_relative,
                peak.left_slope_relative,
                peak.right_slope_relative,
                peak.asymmetry_delta,
            )
            local_feature_rows.append(
                tuple(
                    float(value) if math.isfinite(value) else None
                    for value in raw_features
                )
            )
            local_shifts.append(
                float(peak.center_shift_pm)
                if math.isfinite(peak.center_shift_pm)
                else None
            )
            primary_masks.append(int(peak.quality_mask))
            activity_only = bool(
                peak_activity_fresh[peak_index] and not peak_fresh[peak_index]
            )
            is_valid = math.isfinite(residual_rms[peak_index]) and (
                bool(peak.valid) or activity_only
            )
            valid_mask.append(is_valid)
            if not is_valid:
                continue

            local_terms = np.asarray(
                (
                    abs(peak.center_shift_pm) / max(self.config.activation_pm, 1.0),
                    abs(peak.amplitude_relative)
                    / self.config.amplitude_relative_activation,
                    abs(peak.width_relative) / self.config.width_relative_activation,
                    abs(peak.left_slope_relative)
                    / self.config.slope_relative_activation,
                    abs(peak.right_slope_relative)
                    / self.config.slope_relative_activation,
                    abs(peak.asymmetry_delta) / self.config.asymmetry_activation,
                ),
                dtype=float,
            )
            local_terms = local_terms[np.isfinite(local_terms)]
            local_shape_score = (
                float(np.sqrt(np.mean(np.square(np.clip(local_terms, 0.0, 12.0)))))
                if local_terms.size
                else 0.0
            )
            fresh_fraction = float(
                np.count_nonzero(
                    fresh_points[
                        peak_index * POINTS_PER_PEAK : (peak_index + 1)
                        * POINTS_PER_PEAK
                    ]
                )
                / POINTS_PER_PEAK
            )
            if fresh_fraction > 0.0:
                local_shape_score *= fresh_fraction
            residual_score = min(
                12.0,
                float(residual_rms[peak_index])
                / self.config.normalized_residual_activation_rms,
            )
            if activity_only:
                # One certified high-slope sentinel is a valid *activity*
                # measurement but cannot support a new centre/width/asymmetry
                # fit.  Use only its normalized residual and retain the last
                # complete centre in the causal tracker.
                scores[peak_index] = residual_score
            else:
                scores[peak_index] = (
                    residual_weight * residual_score
                    + (1.0 - residual_weight) * local_shape_score
                )

        audit_shifts: list[float | None] = [None] * SENSOR_COUNT
        audit_uncertainties: list[float | None] = [None] * SENSOR_COUNT
        audit_masks: list[int] = [0] * SENSOR_COUNT
        audit_reasons: list[str] = ["template_not_configured"] * SENSOR_COUNT
        if self.sampling_template is not None:
            audit = extract_ch1_peak_features(
                wavelengths_or_frame,
                raw_ch1_adc_codes,
                baseline_adc_codes=baseline_adc_codes,
                baseline_frame=baseline_frame,
                transimpedance_ohm=transimpedance_ohm,
                baseline_transimpedance_ohm=baseline_transimpedance_ohm,
                sampling_template=self.sampling_template,
            )
            audit_shifts = [
                float(peak.center_shift_pm)
                if math.isfinite(peak.center_shift_pm)
                else None
                for peak in audit.peaks
            ]
            audit_uncertainties = [
                float(peak.center_uncertainty_pm)
                if math.isfinite(peak.center_uncertainty_pm)
                else None
                for peak in audit.peaks
            ]
            audit_masks = [int(peak.quality_mask) for peak in audit.peaks]
            audit_reasons = [str(peak.template_fit_reason) for peak in audit.peaks]

        evidence_timestamp = (
            int(timestamp_ns) if timestamp_ns is not None else int(primary.monotonic_ns)
        )
        if frame_start_timestamp_ns is not None:
            frame_start = int(frame_start_timestamp_ns)
            finite_offsets = [value for value in sample_offsets if value is not None]
            evidence_timestamp = frame_start + (
                0 if not finite_offsets else max(finite_offsets) * 2_000
            )
        return SpectrumContactEvidence(
            timestamp_ns=evidence_timestamp,
            profile=profile_name,
            map_age_frames=(None if map_age_frames is None else int(map_age_frames)),
            fresh_point_mask_45=tuple(bool(value) for value in fresh_points),
            point_age_frames_45=tuple(float(value) for value in point_ages),
            sample_offsets_2us_45=sample_offsets,
            peak_fresh_mask=tuple(bool(value) for value in peak_fresh),
            peak_activity_fresh_mask=tuple(
                bool(value) for value in peak_activity_fresh
            ),
            peak_age_frames=tuple(float(value) for value in peak_ages),
            peak_response_scores=tuple(float(value) for value in scores),
            normalized_residual_rms=tuple(
                float(value) if math.isfinite(value) else None for value in residual_rms
            ),
            normalized_profile_residual_45=tuple(
                float(value) if math.isfinite(value) else None
                for value in residual.reshape(-1)
            ),
            local_shape_features=tuple(local_feature_rows),
            local_shape_shifts_pm=tuple(local_shifts),
            primary_quality_masks=tuple(primary_masks),
            valid_peak_mask=tuple(valid_mask),
            template_audit_shifts_pm=tuple(audit_shifts),
            template_audit_uncertainty_pm=tuple(audit_uncertainties),
            template_audit_quality_masks=tuple(audit_masks),
            template_audit_reasons=tuple(audit_reasons),
        )


def peak_fresh_mask_from_point_indices(
    point_indices: Iterable[int],
    *,
    minimum_points_per_peak: int = 1,
) -> tuple[bool, ...]:
    """Convert fresh 0..44 sparse-table indices into a nine-peak mask."""

    minimum = int(minimum_points_per_peak)
    if not 1 <= minimum <= POINTS_PER_PEAK:
        raise ValueError("minimum_points_per_peak must be within 1..5")
    counts = np.zeros(SENSOR_COUNT, dtype=int)
    for raw in point_indices:
        index = int(raw)
        if not 0 <= index < POINT_COUNT:
            raise ValueError("fresh point index must be within 0..44")
        counts[index // POINTS_PER_PEAK] += 1
    return tuple(bool(value >= minimum) for value in counts)


@dataclass(frozen=True)
class TrackedContactEstimate:
    timestamp_ns: int
    profile: str
    sequence: int | None
    track_state: str
    measurement: ContactEstimate
    x_norm: float | None
    y_norm: float | None
    x_mm: float | None
    y_mm: float | None
    confidence: float
    effective_response_area_mm2: float
    velocity_mm_s: tuple[float, float]
    effective_area_rate_mm2_s: float
    fresh_peak_count: int
    measurement_rate_hz: float | None
    signal_bandwidth_hz: float
    nyquist_requirement_hz: float
    recommended_measurement_rate_hz: float
    nyquist_met: bool | None
    recommended_rate_met: bool | None
    quality_flags: tuple[str, ...] = ()


class MultiRateContactTracker:
    """Causal MAP/SURVEY/TRACK fusion with explicit freshness and cadence."""

    def __init__(
        self,
        localizer: CH1NinePeakLocalizer,
        *,
        signal_bandwidth_hz: float = 15.0,
        recommended_oversample: float = 4.0,
        position_bandwidth_hz: float = 15.0,
        maximum_prediction_s: float = 0.10,
    ) -> None:
        if signal_bandwidth_hz <= 0.0 or not math.isfinite(signal_bandwidth_hz):
            raise ValueError("signal_bandwidth_hz must be finite and positive")
        if recommended_oversample < 2.0 or not math.isfinite(recommended_oversample):
            raise ValueError("recommended_oversample must be at least 2")
        if position_bandwidth_hz <= 0.0 or not math.isfinite(position_bandwidth_hz):
            raise ValueError("position_bandwidth_hz must be finite and positive")
        if maximum_prediction_s <= 0.0 or not math.isfinite(maximum_prediction_s):
            raise ValueError("maximum_prediction_s must be finite and positive")
        self.localizer = localizer
        self.signal_bandwidth_hz = float(signal_bandwidth_hz)
        self.recommended_oversample = float(recommended_oversample)
        self.position_bandwidth_hz = float(position_bandwidth_hz)
        self.maximum_prediction_s = float(maximum_prediction_s)
        self.reset()

    def reset(self) -> None:
        self._peak_shifts = np.full(SENSOR_COUNT, np.nan, dtype=float)
        self._peak_response_scores = np.full(SENSOR_COUNT, np.nan, dtype=float)
        self._peak_uncertainty_pm = np.full(SENSOR_COUNT, np.nan, dtype=float)
        self._peak_quality_masks = np.zeros(SENSOR_COUNT, dtype=np.uint16)
        self._peak_ages = np.full(SENSOR_COUNT, np.inf, dtype=float)
        self._last_timestamp_ns: int | None = None
        self._last_sequence: int | None = None
        self._rate_hz: float | None = None
        self._position_norm: np.ndarray | None = None
        self._position_mm: np.ndarray | None = None
        self._velocity_mm_s = np.zeros(2, dtype=float)
        self._area_mm2 = 0.0
        self._area_rate_mm2_s = 0.0
        self._last_track: TrackedContactEstimate | None = None

    def _cadence_fields(self) -> tuple[float | None, bool | None, bool | None]:
        if self._rate_hz is None:
            return None, None, None
        # Nanosecond timestamps make an ideal 60 Hz interval alternate by one
        # nanosecond; use a ppm-scale comparison tolerance rather than marking
        # that physically exact cadence as below its recommendation.
        tolerance = 1.0 - 1e-6
        return (
            self._rate_hz,
            bool(self._rate_hz >= tolerance * 2.0 * self.signal_bandwidth_hz),
            bool(
                self._rate_hz
                >= tolerance * self.recommended_oversample * self.signal_bandwidth_hz
            ),
        )

    def update(
        self,
        peak_shifts_pm: Sequence[float],
        *,
        timestamp_ns: int,
        profile: str = "MAP",
        sequence: int | None = None,
        fresh_peak_mask: Sequence[bool] | None = None,
        activity_fresh_peak_mask: Sequence[bool] | None = None,
        fresh_point_indices: Iterable[int] | None = None,
        minimum_fresh_points_per_peak: int = 1,
        peak_uncertainty_pm: Sequence[float] | None = None,
        peak_quality_masks: Sequence[int] | None = None,
        peak_response_scores: Sequence[float] | None = None,
    ) -> TrackedContactEstimate:
        profile_name = _normalize_profile_name(profile)
        timestamp = int(timestamp_ns)
        if timestamp < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._last_timestamp_ns is not None and timestamp <= self._last_timestamp_ns:
            raise ValueError("TRACK timestamps must be strictly increasing")
        if fresh_peak_mask is not None and fresh_point_indices is not None:
            raise ValueError("choose fresh_peak_mask or fresh_point_indices, not both")

        supplied = _finite_vector(peak_shifts_pm, "peak_shifts_pm")
        if fresh_point_indices is not None:
            fresh_peak_mask = peak_fresh_mask_from_point_indices(
                fresh_point_indices,
                minimum_points_per_peak=minimum_fresh_points_per_peak,
            )
        shape_fresh = (
            np.isfinite(supplied)
            if fresh_peak_mask is None
            else np.asarray(fresh_peak_mask, dtype=bool).reshape(-1)
        )
        if shape_fresh.size != SENSOR_COUNT:
            raise ValueError("fresh_peak_mask must contain exactly 9 values")
        shape_fresh &= np.isfinite(supplied)
        activity_fresh = (
            shape_fresh.copy()
            if activity_fresh_peak_mask is None
            else np.asarray(activity_fresh_peak_mask, dtype=bool).reshape(-1)
        )
        if activity_fresh.size != SENSOR_COUNT:
            raise ValueError(
                "activity_fresh_peak_mask must contain exactly 9 values"
            )
        activity_fresh |= shape_fresh
        if not np.any(activity_fresh):
            raise ValueError("a tracker update must contain at least one fresh peak")

        supplied_scores = None
        if peak_response_scores is not None:
            supplied_scores = _finite_vector(
                peak_response_scores, "peak_response_scores"
            )
            if np.any(
                ~np.isfinite(supplied_scores[activity_fresh])
                | (supplied_scores[activity_fresh] < 0.0)
            ):
                raise ValueError(
                    "fresh peak_response_scores must be finite and non-negative"
                )
        elif np.any(activity_fresh & ~shape_fresh):
            raise ValueError(
                "activity-only sentinel updates require peak_response_scores"
            )
        supplied_uncertainty = None
        if peak_uncertainty_pm is not None:
            supplied_uncertainty = _finite_vector(
                peak_uncertainty_pm, "peak_uncertainty_pm"
            )
            if np.any(
                np.isfinite(supplied_uncertainty[shape_fresh])
                & (supplied_uncertainty[shape_fresh] < 0.0)
            ):
                raise ValueError("fresh peak uncertainty must be non-negative")
        supplied_quality = None
        if peak_quality_masks is not None:
            supplied_quality = np.asarray(peak_quality_masks).reshape(-1)
            if supplied_quality.size != SENSOR_COUNT:
                raise ValueError("peak_quality_masks must contain exactly 9 values")
            supplied_quality = supplied_quality.astype(np.uint16)

        sequence_gap = False
        elapsed_frames = 1
        if sequence is not None:
            sequence = int(sequence) & 0xFFFFFFFF
            if self._last_sequence is not None:
                elapsed_frames = (sequence - self._last_sequence) & 0xFFFFFFFF
                if elapsed_frames == 0 or elapsed_frames > 0x7FFFFFFF:
                    raise ValueError("TRACK sequence is duplicate or reversed")
                sequence_gap = elapsed_frames != 1
        if sequence is not None:
            self._last_sequence = sequence

        self._peak_ages += float(elapsed_frames)
        self._peak_shifts[shape_fresh] = supplied[shape_fresh]
        # Ages describe the response evidence used for localization.  A
        # A TRACK11/13 sentinel refreshes activity, but does not overwrite
        # the cached centre/shape estimate above.
        self._peak_ages[activity_fresh] = 0.0
        if supplied_scores is None:
            self._peak_response_scores[shape_fresh] = np.nan
        else:
            self._peak_response_scores[activity_fresh] = supplied_scores[
                activity_fresh
            ]
        if supplied_uncertainty is None:
            self._peak_uncertainty_pm[shape_fresh] = np.nan
        else:
            self._peak_uncertainty_pm[shape_fresh] = supplied_uncertainty[
                shape_fresh
            ]
        if supplied_quality is not None:
            self._peak_quality_masks[shape_fresh] = supplied_quality[shape_fresh]

        dt = None
        if self._last_timestamp_ns is not None:
            dt = (timestamp - self._last_timestamp_ns) / 1_000_000_000.0
            instantaneous_rate = 1.0 / dt
            self._rate_hz = (
                instantaneous_rate
                if self._rate_hz is None
                else 0.80 * self._rate_hz + 0.20 * instantaneous_rate
            )
        self._last_timestamp_ns = timestamp

        currently_valid = np.isfinite(self._peak_shifts)
        cached_scores = (
            self._peak_response_scores
            if np.all(np.isfinite(self._peak_response_scores[currently_valid]))
            else None
        )
        cached_uncertainty = (
            self._peak_uncertainty_pm
            if np.any(np.isfinite(self._peak_uncertainty_pm[currently_valid]))
            else None
        )
        measurement = self.localizer.estimate_peak_shifts(
            self._peak_shifts,
            timestamp_ns=timestamp,
            fresh_peak_mask=activity_fresh,
            peak_age_frames=self._peak_ages,
            peak_uncertainty_pm=cached_uncertainty,
            peak_quality_masks=self._peak_quality_masks,
            peak_response_scores=cached_scores,
        )
        flags = list(measurement.quality_flags)
        if np.any(activity_fresh & ~shape_fresh):
            flags.append("fresh_activity_sentinel_without_fitted_peak_center")
        if sequence_gap:
            flags.append("sequence_gap_aged_stale_peaks")

        if measurement.contact_detected:
            raw_norm = np.asarray((measurement.x_norm, measurement.y_norm), dtype=float)
            raw_mm = np.asarray((measurement.x_mm, measurement.y_mm), dtype=float)
            if self._position_mm is None or dt is None:
                self._position_norm = raw_norm
                self._position_mm = raw_mm
                self._velocity_mm_s[:] = 0.0
                self._area_mm2 = measurement.effective_response_area_mm2
                self._area_rate_mm2_s = 0.0
            else:
                predicted_mm = self._position_mm + self._velocity_mm_s * dt
                alpha = 1.0 - math.exp(-2.0 * math.pi * self.position_bandwidth_hz * dt)
                gain = float(
                    np.clip(alpha * (0.35 + 0.65 * measurement.confidence), 0.05, 1.0)
                )
                innovation = raw_mm - predicted_mm
                updated_mm = predicted_mm + gain * innovation
                velocity_gain = min(0.75, gain * gain / max(2.0 - gain, 1e-9))
                instantaneous_velocity = innovation / dt
                self._velocity_mm_s = (
                    1.0 - velocity_gain
                ) * self._velocity_mm_s + velocity_gain * instantaneous_velocity
                previous_area = self._area_mm2
                area_alpha = 1.0 - math.exp(
                    -2.0 * math.pi * self.signal_bandwidth_hz * dt
                )
                self._area_mm2 += area_alpha * (
                    measurement.effective_response_area_mm2 - self._area_mm2
                )
                self._area_rate_mm2_s = (self._area_mm2 - previous_area) / dt
                self._position_mm = updated_mm
                self._position_norm = self.localizer.geometry.mm_to_normalized(
                    updated_mm
                )
            track_state = "MEASURED"
        else:
            self._position_norm = None
            self._position_mm = None
            self._velocity_mm_s[:] = 0.0
            self._area_mm2 = 0.0
            self._area_rate_mm2_s = 0.0
            track_state = "NO_CONTACT"

        rate, nyquist, recommended = self._cadence_fields()
        if nyquist is False:
            flags.append("measurement_rate_below_15hz_signal_nyquist")
        elif recommended is False:
            flags.append("measurement_rate_has_low_15hz_oversampling_margin")
        confidence = measurement.confidence
        if nyquist is False:
            confidence *= max(
                0.1, float(rate or 0.0) / (2.0 * self.signal_bandwidth_hz)
            )

        result = TrackedContactEstimate(
            timestamp_ns=timestamp,
            profile=profile_name,
            sequence=sequence,
            track_state=track_state,
            measurement=measurement,
            x_norm=(
                None if self._position_norm is None else float(self._position_norm[0])
            ),
            y_norm=(
                None if self._position_norm is None else float(self._position_norm[1])
            ),
            x_mm=(None if self._position_mm is None else float(self._position_mm[0])),
            y_mm=(None if self._position_mm is None else float(self._position_mm[1])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            effective_response_area_mm2=float(self._area_mm2),
            velocity_mm_s=(
                float(self._velocity_mm_s[0]),
                float(self._velocity_mm_s[1]),
            ),
            effective_area_rate_mm2_s=float(self._area_rate_mm2_s),
            fresh_peak_count=int(np.count_nonzero(activity_fresh)),
            measurement_rate_hz=rate,
            signal_bandwidth_hz=self.signal_bandwidth_hz,
            nyquist_requirement_hz=2.0 * self.signal_bandwidth_hz,
            recommended_measurement_rate_hz=(
                self.recommended_oversample * self.signal_bandwidth_hz
            ),
            nyquist_met=nyquist,
            recommended_rate_met=recommended,
            quality_flags=tuple(dict.fromkeys(flags)),
        )
        self._last_track = result
        return result

    def update_evidence(
        self,
        evidence: SpectrumContactEvidence,
        *,
        profile: str | None = None,
        sequence: int | None = None,
        fresh_peak_mask: Sequence[bool] | None = None,
        fresh_point_indices: Iterable[int] | None = None,
        minimum_fresh_points_per_peak: int = 1,
    ) -> TrackedContactEstimate:
        """Feed robust 45-point evidence into the same causal tracker."""

        if fresh_peak_mask is None and fresh_point_indices is None:
            fresh_peak_mask = evidence.peak_fresh_mask
            activity_fresh_peak_mask = evidence.peak_activity_fresh_mask
        else:
            activity_fresh_peak_mask = fresh_peak_mask
        result = self.update(
            [
                math.nan if value is None else value
                for value in evidence.local_shape_shifts_pm
            ],
            timestamp_ns=evidence.timestamp_ns,
            profile=evidence.profile if profile is None else profile,
            sequence=sequence,
            fresh_peak_mask=fresh_peak_mask,
            activity_fresh_peak_mask=activity_fresh_peak_mask,
            fresh_point_indices=fresh_point_indices,
            minimum_fresh_points_per_peak=minimum_fresh_points_per_peak,
            peak_quality_masks=evidence.primary_quality_masks,
            peak_response_scores=evidence.peak_response_scores,
        )
        measurement = replace(
            result.measurement,
            localization_evidence_kind=SPECTRUM_EVIDENCE_KIND,
            normalized_residual_rms=evidence.normalized_residual_rms,
            normalized_profile_residual_45=evidence.normalized_profile_residual_45,
            local_shape_features=evidence.local_shape_features,
            local_shape_shifts_pm=evidence.local_shape_shifts_pm,
            template_audit_shifts_pm=evidence.template_audit_shifts_pm,
            template_audit_uncertainty_pm=evidence.template_audit_uncertainty_pm,
            template_audit_quality_masks=evidence.template_audit_quality_masks,
            template_audit_reasons=evidence.template_audit_reasons,
            acquisition_profile=evidence.profile,
            map_age_frames=evidence.map_age_frames,
            fresh_point_mask_45=evidence.fresh_point_mask_45,
            point_age_frames_45=evidence.point_age_frames_45,
            sample_offsets_2us_45=evidence.sample_offsets_2us_45,
            peak_age_frames=evidence.peak_age_frames,
        )
        result = replace(result, measurement=measurement)
        self._last_track = result
        return result

    def predict(self, timestamp_ns: int) -> TrackedContactEstimate:
        """Return a short causal prediction without marking it as measured."""

        if self._last_track is None or self._last_timestamp_ns is None:
            raise RuntimeError("TRACK has no measurement to predict from")
        timestamp = int(timestamp_ns)
        if timestamp < self._last_timestamp_ns:
            raise ValueError("prediction timestamp precedes the last measurement")
        horizon = (timestamp - self._last_timestamp_ns) / 1_000_000_000.0
        flags = list(self._last_track.quality_flags)
        if horizon > self.maximum_prediction_s:
            flags.append("prediction_horizon_exceeded")
            return replace(
                self._last_track,
                timestamp_ns=timestamp,
                track_state="LOST",
                confidence=0.0,
                quality_flags=tuple(dict.fromkeys(flags)),
            )
        if self._position_mm is None or self._position_norm is None:
            return replace(
                self._last_track,
                timestamp_ns=timestamp,
                track_state="NO_CONTACT",
            )
        predicted_mm = self._position_mm + self._velocity_mm_s * horizon
        predicted_norm = self.localizer.geometry.mm_to_normalized(predicted_mm)
        confidence = self._last_track.confidence * math.exp(
            -horizon / self.maximum_prediction_s
        )
        flags.append("predicted_not_new_spectrum")
        return replace(
            self._last_track,
            timestamp_ns=timestamp,
            track_state="PREDICTED",
            x_norm=float(predicted_norm[0]),
            y_norm=float(predicted_norm[1]),
            x_mm=float(predicted_mm[0]),
            y_mm=float(predicted_mm[1]),
            confidence=float(confidence),
            effective_response_area_mm2=max(
                0.0, self._area_mm2 + self._area_rate_mm2_s * horizon
            ),
            quality_flags=tuple(dict.fromkeys(flags)),
        )


def make_machine_training_record(
    estimate: ContactEstimate,
    *,
    session_id: str,
    event_id: str,
    labelled_work_x_mm: float,
    labelled_work_y_mm: float,
    indentation_mm: float,
    profile: str,
    table_crc32: int | None = None,
) -> dict[str, Any]:
    """Create a traceable CNC label record for later grouped model training."""

    indentation = float(indentation_mm)
    if not 0.0 <= indentation <= MAX_INDENTATION_MM:
        raise ValueError(f"indentation_mm must be within 0..{MAX_INDENTATION_MM}")
    label = np.asarray((labelled_work_x_mm, labelled_work_y_mm), dtype=float)
    if not np.all(np.isfinite(label)):
        raise ValueError("labelled work coordinates must be finite")
    profile_name = _normalize_profile_name(profile)
    return {
        "schema": "ch1_finger_localization_training_record/v1",
        "session_id": str(session_id),
        "event_id": str(event_id),
        "timestamp_ns": int(estimate.timestamp_ns),
        "profile": profile_name,
        "table_crc32": None if table_crc32 is None else int(table_crc32) & 0xFFFFFFFF,
        "sensor_channel": CH1,
        "grating_count": SENSOR_COUNT,
        "points_per_peak": POINTS_PER_PEAK,
        "peak_shifts_pm": list(estimate.peak_shifts_pm),
        "response_weights": list(estimate.response_weights),
        "analytic_estimate": estimate.to_dict(),
        "machine_label": {
            "coordinate_frame": "mach3_work_mm",
            "x_mm": float(label[0]),
            "y_mm": float(label[1]),
            "indentation_mm": indentation,
            "source": "cnc_operator_verified_contact",
        },
        "area_label": {
            "physical_contact_area_mm2": None,
            "available": False,
            "reason": (
                "CNC XY/indentation labels do not measure physical contact area; "
                "effective_response_area_mm2 remains a sensor-spread feature"
            ),
        },
    }


def benchmark_localizer(
    localizer: CH1NinePeakLocalizer,
    peak_shifts_pm: Sequence[float],
    *,
    repetitions: int = 500,
) -> dict[str, float | bool | int | str]:
    """Benchmark only the localizer computation against the 15 Hz budget."""

    count = int(repetitions)
    if count < 10:
        raise ValueError("benchmark repetitions must be at least 10")
    for _ in range(5):
        localizer.estimate_peak_shifts(peak_shifts_pm, timestamp_ns=0)
    elapsed_ms = []
    for _ in range(count):
        start = time.perf_counter_ns()
        localizer.estimate_peak_shifts(peak_shifts_pm, timestamp_ns=0)
        elapsed_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    values = np.asarray(elapsed_ms, dtype=float)
    budget_ms = 1000.0 / 15.0
    return {
        "repetitions": count,
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(np.max(values)),
        "budget_ms_at_15hz": budget_ms,
        "meets_15hz_compute_budget": bool(np.percentile(values, 99) <= budget_ms),
        "scope": "localization only; optical scan, transport, fitting and UI are separate",
    }


__all__ = [
    "ALLOWED_PROFILES",
    "AREA_KIND",
    "AREA_METHOD",
    "DEFAULT_CERTIFIED_TEMPLATE_PATH",
    "DEFAULT_LAYOUT_PATH",
    "DEFAULT_TOUCH_CALIBRATION_PATH",
    "SHIFT_EVIDENCE_KIND",
    "SPECTRUM_EVIDENCE_KIND",
    "CH1NinePeakLocalizer",
    "ContactEstimate",
    "ContactGeometry",
    "ContactLocalizerConfig",
    "MultiRateContactTracker",
    "SensorSite",
    "SpectrumContactEvidence",
    "TrackedContactEstimate",
    "benchmark_localizer",
    "load_contact_geometry",
    "make_machine_training_record",
    "peak_fresh_mask_from_point_indices",
]
