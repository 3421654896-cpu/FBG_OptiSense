"""Serializable measured-template contract for the nine-FBG CH1 table.

The sparse stress table contains exactly five samples for each of the nine
gratings.  Five isolated samples are not a spectrum model, so this module binds
them to the measured dense shape used by :mod:`adaptive_sampling_optimizer`.
The binding also includes the firmware wavelength-table CRC; a template can
therefore never be silently reused with a different sparse table.

This module is deliberately offline-only.  It has no serial, network, laser or
machine-control dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_sampling_optimizer import PeakSamplingTemplate, SamplingPlan
from finger_dataset import (
    ADC_CODE_COUNT,
    ADC_REFERENCE_V,
    DEFAULT_GRATING_COUNT,
    DEFAULT_POINTS_PER_PEAK,
    DEFAULT_SENSOR_CHANNEL,
)

TEMPLATE_SCHEMA = "fbg-finger-sampling-template/v1"
AUTO_SELECTION_SCHEMA = "equal_interval_auto_mode_selection_v4"
SPARSE_PLAN_SCHEMA = "ch1_sparse_sampling_plan/v1"
DEFAULT_MODE = "stress"


def wavelength_table_crc32(wavelengths_nm: Sequence[float]) -> int:
    """Return the exact CRC used by firmware metadata for a wavelength axis."""

    values = np.asarray(wavelengths_nm, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("wavelength table must be a non-empty finite 1-D array")
    wavelength_pm = np.rint(values * 1000.0)
    if np.any((wavelength_pm < 0.0) | (wavelength_pm > 0xFFFFFFFF)):
        raise ValueError("wavelength table cannot be represented as uint32 picometres")
    encoded = wavelength_pm.astype(">u4", copy=False).tobytes()
    return zlib.crc32(encoded) & 0xFFFFFFFF


def _sparse_stress_table_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Mirror the installer fingerprint without importing its UI dependencies."""

    canonical = []
    for mode_index, raw in enumerate(rows):
        codes = tuple(int(value) for value in raw.get("codes", ()))
        if len(codes) != 5:
            raise ValueError(
                f"selection report stress row {mode_index + 1} must contain 5 DAC codes"
            )
        canonical.append(
            {
                "mode_index": int(mode_index),
                "peak_number": int(raw.get("peak_number", 0)),
                "fullband_index": int(raw["fullband_index"]),
                "codes": list(codes),
            }
        )
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _readonly(values: Sequence[Any], dtype: Any) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=dtype)
    if result.ndim != 1:
        raise ValueError("template arrays must be one-dimensional")
    result.setflags(write=False)
    return result


def _parse_crc(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        converted = int(value.strip(), 0)
    else:
        converted = int(value)
    if not 0 <= converted <= 0xFFFFFFFF:
        raise ValueError("table_crc32 must fit uint32")
    return converted


def _validate_peak(template: PeakSamplingTemplate, expected_number: int) -> None:
    if int(template.peak_number) != expected_number:
        raise ValueError("template peaks must be ordered G1 through G9")
    dense_x = np.asarray(template.dense_wavelength_nm, dtype=float)
    dense_y = np.asarray(template.dense_signal, dtype=float)
    selected_indices = np.asarray(template.selected_indices, dtype=np.int64)
    selected_x = np.asarray(template.selected_wavelength_nm, dtype=float)
    selected_y = np.asarray(template.selected_template_signal, dtype=float)
    selected_slope = np.asarray(template.selected_template_slope, dtype=float)
    selected_weight = np.asarray(template.selected_reliability_weight, dtype=float)
    if dense_x.ndim != 1 or dense_x.size < DEFAULT_POINTS_PER_PEAK:
        raise ValueError(f"G{expected_number} dense template has too few samples")
    if dense_y.shape != dense_x.shape:
        raise ValueError(f"G{expected_number} dense wavelength/signal sizes differ")
    if not np.all(np.isfinite(dense_x)) or not np.all(np.isfinite(dense_y)):
        raise ValueError(f"G{expected_number} dense template must be finite")
    if np.any(np.diff(dense_x) <= 0.0):
        raise ValueError(f"G{expected_number} dense wavelengths are not increasing")
    expected_shape = (DEFAULT_POINTS_PER_PEAK,)
    for name, values in (
        ("selected_indices", selected_indices),
        ("selected_wavelength_nm", selected_x),
        ("selected_template_signal", selected_y),
        ("selected_template_slope", selected_slope),
        ("selected_reliability_weight", selected_weight),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"G{expected_number} {name} must contain exactly 5 values")
    if np.any(np.diff(selected_indices) <= 0) or np.any(np.diff(selected_x) <= 0.0):
        raise ValueError(f"G{expected_number} sparse samples are not increasing")
    if not np.all(np.isfinite(selected_x)) or not np.all(np.isfinite(selected_y)):
        raise ValueError(f"G{expected_number} sparse template must be finite")
    if not np.all(np.isfinite(selected_slope)):
        raise ValueError(f"G{expected_number} template slopes must be finite")
    if np.any(~np.isfinite(selected_weight)) or np.any(selected_weight <= 0.0):
        raise ValueError(f"G{expected_number} reliability weights must be positive")
    tolerance = 1e-9
    if (
        selected_x[0] < dense_x[0] - tolerance
        or selected_x[-1] > dense_x[-1] + tolerance
    ):
        raise ValueError(
            f"G{expected_number} sparse wavelengths leave its dense region"
        )


@dataclass(frozen=True)
class FingerSamplingTemplate:
    """Strict nine-peak/45-point measured-template bundle."""

    plan: SamplingPlan
    table_crc32: int | None = None
    selected_noise_std_v: tuple[tuple[float, ...], ...] = ()
    source_schema: str = TEMPLATE_SCHEMA
    source_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.plan.peaks) != DEFAULT_GRATING_COUNT:
            raise ValueError("mechanical-finger template must contain exactly 9 peaks")
        counts = tuple(int(value) for value in self.plan.points_per_peak)
        if counts != (DEFAULT_POINTS_PER_PEAK,) * DEFAULT_GRATING_COUNT:
            raise ValueError(
                "mechanical-finger template must contain 5 points per peak"
            )
        for number, peak in enumerate(self.plan.peaks, start=1):
            _validate_peak(peak, number)
        selected_indices = np.concatenate(
            [np.asarray(peak.selected_indices) for peak in self.plan.peaks]
        )
        selected_x = np.asarray(self.plan.selected_wavelength_nm, dtype=float)
        if selected_indices.size != 45 or np.any(np.diff(selected_indices) <= 0):
            raise ValueError(
                "mechanical-finger template must contain 45 ordered unique rows"
            )
        if selected_x.size != 45 or np.any(np.diff(selected_x) <= 0.0):
            raise ValueError(
                "mechanical-finger template must contain 45 ordered wavelengths"
            )

        computed_crc = wavelength_table_crc32(selected_x)
        declared_crc = _parse_crc(self.table_crc32)
        if declared_crc is not None and declared_crc != computed_crc:
            raise ValueError(
                "template table CRC does not match its 45 selected wavelengths: "
                f"declared=0x{declared_crc:08X}, calculated=0x{computed_crc:08X}"
            )
        object.__setattr__(self, "table_crc32", computed_crc)

        noises = self.selected_noise_std_v
        if not noises:
            quantum_v = ADC_REFERENCE_V / ADC_CODE_COUNT
            derived = []
            for peak in self.plan.peaks:
                weights = np.asarray(peak.selected_reliability_weight, dtype=float)
                normalized = weights / max(float(np.median(weights)), 1e-12)
                values = quantum_v / np.sqrt(np.clip(normalized, 1e-4, 1e4))
                derived.append(tuple(float(value) for value in values))
            noises = tuple(derived)
        else:
            noises = tuple(tuple(float(value) for value in peak) for peak in noises)
        if len(noises) != DEFAULT_GRATING_COUNT:
            raise ValueError("selected_noise_std_v must contain exactly 9 peak arrays")
        for number, values in enumerate(noises, start=1):
            array = np.asarray(values, dtype=float)
            if array.shape != (DEFAULT_POINTS_PER_PEAK,):
                raise ValueError(f"G{number} noise array must contain exactly 5 values")
            if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(
                    f"G{number} noise estimates must be finite and positive"
                )
        object.__setattr__(self, "selected_noise_std_v", noises)
        object.__setattr__(self, "source_schema", str(self.source_schema))

    @property
    def channel(self) -> int:
        return DEFAULT_SENSOR_CHANNEL

    @property
    def grating_count(self) -> int:
        return DEFAULT_GRATING_COUNT

    @property
    def points_per_peak(self) -> int:
        return DEFAULT_POINTS_PER_PEAK

    @property
    def point_count(self) -> int:
        return self.plan.total_points

    @property
    def wavelengths_nm(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.plan.selected_wavelength_nm)

    def wavelength_axis_matches(self, wavelengths_nm: Sequence[float]) -> bool:
        actual = np.asarray(wavelengths_nm, dtype=float)
        expected = np.asarray(self.wavelengths_nm, dtype=float)
        return bool(
            actual.shape == expected.shape
            and np.all(np.isfinite(actual))
            and np.array_equal(np.rint(actual * 1000.0), np.rint(expected * 1000.0))
        )

    def to_dict(self) -> dict[str, Any]:
        peaks = []
        for peak, noise in zip(self.plan.peaks, self.selected_noise_std_v, strict=True):
            peaks.append(
                {
                    "peak_number": int(peak.peak_number),
                    "region_indices": [int(value) for value in peak.region_indices],
                    "dense_wavelength_nm": [
                        float(value) for value in peak.dense_wavelength_nm
                    ],
                    "dense_template_signal_v": [
                        float(value) for value in peak.dense_signal
                    ],
                    "selected_fullband_indices": [
                        int(value) for value in peak.selected_indices
                    ],
                    "selected_wavelength_nm": [
                        float(value) for value in peak.selected_wavelength_nm
                    ],
                    "selected_template_signal_v": [
                        float(value) for value in peak.selected_template_signal
                    ],
                    "selected_template_slope_v_per_nm": [
                        float(value) for value in peak.selected_template_slope
                    ],
                    "selected_reliability_weight": [
                        float(value) for value in peak.selected_reliability_weight
                    ],
                    "selected_noise_std_v": [float(value) for value in noise],
                    "teacher_peak_fullband_index": int(peak.peak_index),
                    "projected_shift_information": float(
                        peak.projected_shift_information
                    ),
                }
            )
        return {
            "schema": TEMPLATE_SCHEMA,
            "source_schema": self.source_schema,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "channel": self.channel,
            "grating_count": self.grating_count,
            "points_per_peak": self.points_per_peak,
            "point_count": self.point_count,
            "table_crc32": int(self.table_crc32),
            "table_crc32_hex": f"0x{int(self.table_crc32):08X}",
            "peaks": peaks,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FingerSamplingTemplate:
        if payload.get("schema") != TEMPLATE_SCHEMA:
            raise ValueError("not a serialized mechanical-finger sampling template")
        peaks_payload = payload.get("peaks")
        if not isinstance(peaks_payload, list):
            raise TypeError("serialized template peaks must be a list")
        peaks = tuple(_peak_from_serialized(item) for item in peaks_payload)
        noises = tuple(
            tuple(float(value) for value in item.get("selected_noise_std_v", ()))
            for item in peaks_payload
        )
        return cls(
            plan=SamplingPlan(
                peaks=peaks,
                points_per_peak=(DEFAULT_POINTS_PER_PEAK,) * len(peaks),
            ),
            table_crc32=_parse_crc(payload.get("table_crc32")),
            selected_noise_std_v=noises if all(noises) else (),
            source_schema=str(payload.get("source_schema", TEMPLATE_SCHEMA)),
            source_fingerprint_sha256=payload.get("source_fingerprint_sha256"),
        )

    def save(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return path


def _peak_from_serialized(item: Mapping[str, Any]) -> PeakSamplingTemplate:
    dense_x = _readonly(item["dense_wavelength_nm"], float)
    dense_y = _readonly(item["dense_template_signal_v"], float)
    selected_x = _readonly(item["selected_wavelength_nm"], float)
    selected_y = _readonly(item["selected_template_signal_v"], float)
    slopes = item.get("selected_template_slope_v_per_nm")
    if slopes is None:
        dense_slope = np.gradient(dense_y, dense_x, edge_order=2)
        slopes = np.interp(selected_x, dense_x, dense_slope)
    weights = item.get("selected_reliability_weight", [1.0] * DEFAULT_POINTS_PER_PEAK)
    region_indices = item.get("region_indices")
    peak_index = int(item.get("teacher_peak_fullband_index", 0))
    if region_indices is None:
        local_peak = int(np.argmax(dense_y))
        region_start = peak_index - local_peak
        region_indices = range(region_start, region_start + dense_x.size)
    return PeakSamplingTemplate(
        peak_number=int(item["peak_number"]),
        region_indices=_readonly(region_indices, np.int64),
        dense_wavelength_nm=dense_x,
        dense_signal=dense_y,
        selected_indices=_readonly(item["selected_fullband_indices"], np.int64),
        selected_wavelength_nm=selected_x,
        selected_template_signal=selected_y,
        selected_template_slope=_readonly(slopes, float),
        selected_reliability_weight=_readonly(weights, float),
        peak_index=peak_index,
        projected_shift_information=float(item.get("projected_shift_information", 0.0)),
    )


def _report_rows_for_peak(
    rows: Sequence[Mapping[str, Any]], peak_number: int, selected_indices: Sequence[int]
) -> list[Mapping[str, Any]]:
    candidates = {
        int(row["fullband_index"]): row
        for row in rows
        if int(row.get("peak_number", -1)) == peak_number
    }
    try:
        return [candidates[int(index)] for index in selected_indices]
    except KeyError as exc:
        raise ValueError(
            f"G{peak_number} selected index {int(exc.args[0])} has no matching row"
        ) from None


def _template_from_auto_selection_report(
    payload: Mapping[str, Any], mode: str
) -> FingerSamplingTemplate:
    if payload.get("schema") != AUTO_SELECTION_SCHEMA:
        raise ValueError("only strict v4 automatic-selection reports are supported")
    if int(payload.get("detected_peak_count", 0)) != DEFAULT_GRATING_COUNT:
        raise ValueError("selection report must contain exactly 9 detected peaks")
    if int(payload.get("configured_peak_count", 0)) != DEFAULT_GRATING_COUNT:
        raise ValueError("selection report must be configured for exactly 9 peaks")
    if tuple(int(value) for value in payload.get("fbg_channels", ())) != (
        DEFAULT_SENSOR_CHANNEL,
    ):
        raise ValueError("selection report must use CH1 only")
    selection = payload.get("selection", {})
    if selection.get("method") != "measured_template_fisher_v1":
        raise ValueError(
            "selection report does not use the measured-template optimizer"
        )
    mode_payload = payload.get(mode)
    if not isinstance(mode_payload, Mapping):
        raise TypeError(f"selection report {mode!r} mode must be a mapping")
    if int(mode_payload.get("point_count", 0)) != 45:
        raise ValueError(
            "mechanical-finger stress template must contain exactly 45 points"
        )
    counts = tuple(int(value) for value in mode_payload.get("points_per_peak", ()))
    if counts != (DEFAULT_POINTS_PER_PEAK,) * DEFAULT_GRATING_COUNT:
        raise ValueError(
            "selection report must contain five points for each of 9 peaks"
        )
    peaks_payload = mode_payload.get("peaks")
    rows = mode_payload.get("rows")
    if not isinstance(peaks_payload, list) or not isinstance(rows, list):
        raise TypeError("selection report template peaks and sparse rows must be lists")
    if len(peaks_payload) != 9 or len(rows) != 45:
        raise ValueError("selection report must contain 9 templates and 45 rows")

    calculated_fingerprint = _sparse_stress_table_fingerprint(rows)
    validation = payload.get("sparse_sequence_validation", {})
    declared_fingerprint = (
        validation.get("stress_table_fingerprint_sha256")
        if isinstance(validation, Mapping) and mode == "stress"
        else None
    )
    if declared_fingerprint is not None:
        declared_fingerprint = str(declared_fingerprint).strip().lower()
        if declared_fingerprint and declared_fingerprint != calculated_fingerprint:
            raise ValueError(
                "selection report sparse-table fingerprint does not match its "
                "ordered indices and DAC codes"
            )

    templates: list[PeakSamplingTemplate] = []
    noise_arrays: list[tuple[float, ...]] = []
    selected_axis: list[float] = []
    for peak_number, item in enumerate(peaks_payload, start=1):
        if int(item.get("peak_number", 0)) != peak_number:
            raise ValueError("selection report peaks must be ordered G1 through G9")
        if (
            int(item.get("source_channel", DEFAULT_SENSOR_CHANNEL))
            != DEFAULT_SENSOR_CHANNEL
        ):
            raise ValueError(f"G{peak_number} measured template must come from CH1")
        selected_indices = tuple(
            int(value) for value in item.get("selected_fullband_indices", ())
        )
        if len(selected_indices) != DEFAULT_POINTS_PER_PEAK:
            raise ValueError(f"G{peak_number} must select exactly five fullband rows")
        peak_rows = _report_rows_for_peak(rows, peak_number, selected_indices)
        selected_x = tuple(
            float(row.get("measured_wavelength_nm", row["target_wavelength_nm"]))
            for row in peak_rows
        )
        dense_x = item.get("dense_wavelength_nm")
        dense_y = item.get("dense_template_signal_v")
        if dense_x is None or dense_y is None:
            raise ValueError(
                f"G{peak_number} report lacks the dense measured template required "
                "for translation fitting"
            )
        selected_y = item.get("selected_template_signal_v")
        if selected_y is None:
            selected_y = [row.get("template_signal_v", math.nan) for row in peak_rows]
        slopes = item.get("selected_template_slope_v_per_nm")
        if slopes is None:
            row_slopes = [
                row.get("template_slope_v_per_nm", math.nan) for row in peak_rows
            ]
            if np.all(np.isfinite(row_slopes)):
                slopes = row_slopes
        weights = item.get("selected_reliability_weight")
        if weights is None:
            weights = [row.get("template_reliability_weight", 1.0) for row in peak_rows]
        peak_mapping = {
            "peak_number": peak_number,
            "dense_wavelength_nm": dense_x,
            "dense_template_signal_v": dense_y,
            "selected_fullband_indices": selected_indices,
            "selected_wavelength_nm": selected_x,
            "selected_template_signal_v": selected_y,
            "selected_template_slope_v_per_nm": slopes,
            "selected_reliability_weight": weights,
            "teacher_peak_fullband_index": item.get(
                "teacher_peak_fullband_index",
                selected_indices[int(np.argmax(np.asarray(selected_y, dtype=float)))],
            ),
            "projected_shift_information": item.get("projected_shift_information", 0.0),
        }
        if item.get("dense_fullband_indices") is not None:
            peak_mapping["region_indices"] = item["dense_fullband_indices"]
        templates.append(_peak_from_serialized(peak_mapping))
        selected_axis.extend(selected_x)

        explicit_noise = [
            row.get("template_noise_std_v", row.get("selection_noise_floor_40k_v"))
            for row in peak_rows
        ]
        if all(value is not None and float(value) > 0.0 for value in explicit_noise):
            noise_arrays.append(tuple(float(value) for value in explicit_noise))
        else:
            weight_values = np.asarray(weights, dtype=float)
            normalized = weight_values / max(float(np.median(weight_values)), 1e-12)
            quantum_v = ADC_REFERENCE_V / ADC_CODE_COUNT
            derived = quantum_v / np.sqrt(np.clip(normalized, 1e-4, 1e4))
            noise_arrays.append(tuple(float(value) for value in derived))

    expected_row_indices = tuple(
        int(value)
        for template in templates
        for value in template.selected_indices
    )
    actual_row_indices = tuple(int(row["fullband_index"]) for row in rows)
    expected_peak_numbers = tuple(
        peak_number
        for peak_number in range(1, DEFAULT_GRATING_COUNT + 1)
        for _ in range(DEFAULT_POINTS_PER_PEAK)
    )
    actual_peak_numbers = tuple(int(row.get("peak_number", 0)) for row in rows)
    if (
        actual_row_indices != expected_row_indices
        or actual_peak_numbers != expected_peak_numbers
    ):
        raise ValueError(
            "selection report rows must be the exact ordered G1..G9, five-point "
            "sparse table"
        )

    calculated_crc = wavelength_table_crc32(selected_axis)
    declared_crc = _parse_crc(
        mode_payload.get("table_crc32", payload.get("table_crc32"))
    )
    if declared_crc is not None and declared_crc != calculated_crc:
        raise ValueError("selection report table CRC does not match its sparse rows")
    return FingerSamplingTemplate(
        plan=SamplingPlan(peaks=tuple(templates), points_per_peak=counts),
        table_crc32=calculated_crc,
        selected_noise_std_v=tuple(noise_arrays),
        source_schema=AUTO_SELECTION_SCHEMA,
        source_fingerprint_sha256=(
            declared_fingerprint or calculated_fingerprint
            if mode == "stress"
            else None
        ),
    )


def _smooth_three(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.astype(float, copy=True)
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.asarray((0.25, 0.5, 0.25)), mode="valid")


def _resolve_referenced_path(reference: str, report_path: Path | None) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    roots = [] if report_path is None else [report_path.parent, *report_path.parents]
    for root in roots:
        joined = root / candidate
        if joined.exists():
            return joined
    raise FileNotFoundError(f"cannot resolve referenced teacher spectrum: {reference}")


def _teacher_row_map(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = payload.get("rows")
    if isinstance(rows, Mapping):
        return {int(key): value for key, value in rows.items()}
    if isinstance(rows, list):
        return {int(row["index"]): row for row in rows}
    raise ValueError("teacher spectrum is missing indexed rows")


def _template_from_sparse_plan_artifact(
    payload: Mapping[str, Any],
    report_path: Path | None,
    teacher_path: str | Path | None,
) -> FingerSamplingTemplate:
    if int(payload.get("channel", -1)) != DEFAULT_SENSOR_CHANNEL:
        raise ValueError("sparse plan must use CH1")
    if int(payload.get("expected_fbg_count", 0)) != 9:
        raise ValueError("sparse plan must contain exactly 9 FBGs")
    if (
        int(payload.get("points_per_peak", 0)) != 5
        or int(payload.get("total_points", 0)) != 45
    ):
        raise ValueError("sparse plan must contain exactly 45 points")
    if teacher_path is None:
        source = payload.get("source", {})
        teacher_path = (
            source.get("teacher_40k") if isinstance(source, Mapping) else None
        )
    if teacher_path is None:
        raise ValueError("legacy sparse plan requires its dense teacher spectrum")
    resolved_teacher = _resolve_referenced_path(str(teacher_path), report_path)
    teacher_payload = json.loads(resolved_teacher.read_text(encoding="utf-8"))
    teacher_rows = _teacher_row_map(teacher_payload)

    templates: list[PeakSamplingTemplate] = []
    noises: list[tuple[float, ...]] = []
    selected_axis: list[float] = []
    peaks_payload = payload.get("peaks")
    if not isinstance(peaks_payload, list) or len(peaks_payload) != 9:
        raise ValueError("legacy sparse plan must contain 9 peak entries")
    for peak_number, item in enumerate(peaks_payload, start=1):
        start, stop = (int(value) for value in item["source_index_region"])
        region_indices = np.arange(start, stop + 1, dtype=np.int64)
        try:
            rows = [teacher_rows[int(index)] for index in region_indices]
        except KeyError as exc:
            raise ValueError(
                f"dense teacher lacks fullband row {int(exc.args[0])}"
            ) from None
        dense_x = np.asarray(
            [float(row["measured_wavelength_nm"]) for row in rows], dtype=float
        )
        dense_y = _smooth_three(
            np.asarray([float(row["ch1_voltage_v"]) for row in rows], dtype=float)
        )
        dense_slope = np.gradient(dense_y, dense_x, edge_order=2)
        points = item.get("points")
        if not isinstance(points, list) or len(points) != 5:
            raise ValueError(f"G{peak_number} legacy plan must contain five points")
        points = sorted(points, key=lambda point: int(point["within_peak_order"]))
        selected_indices = np.asarray(
            [int(point["source_index"]) for point in points], dtype=np.int64
        )
        local = selected_indices - start
        selected_x = np.asarray(
            [float(point["calibrated_wavelength_nm"]) for point in points], dtype=float
        )
        selected_y = dense_y[local]
        raw_noise = np.asarray(
            [
                max(
                    float(point.get("selection_noise_floor_40k_v", 0.0)),
                    ADC_REFERENCE_V / ADC_CODE_COUNT,
                )
                for point in points
            ],
            dtype=float,
        )
        reference_noise = float(np.median(raw_noise))
        weights = np.clip((reference_noise / raw_noise) ** 2, 1e-4, 1e4)
        template = PeakSamplingTemplate(
            peak_number=peak_number,
            region_indices=_readonly(region_indices, np.int64),
            dense_wavelength_nm=_readonly(dense_x, float),
            dense_signal=_readonly(dense_y, float),
            selected_indices=_readonly(selected_indices, np.int64),
            selected_wavelength_nm=_readonly(selected_x, float),
            selected_template_signal=_readonly(selected_y, float),
            selected_template_slope=_readonly(dense_slope[local], float),
            selected_reliability_weight=_readonly(weights, float),
            peak_index=int(item["template_peak_source_index"]),
            projected_shift_information=float(
                item.get("projected_fisher_information_per_nm2", 0.0)
            ),
        )
        templates.append(template)
        noises.append(tuple(float(value) for value in raw_noise))
        selected_axis.extend(float(value) for value in selected_x)

    return FingerSamplingTemplate(
        plan=SamplingPlan(
            peaks=tuple(templates),
            points_per_peak=(DEFAULT_POINTS_PER_PEAK,) * DEFAULT_GRATING_COUNT,
        ),
        table_crc32=wavelength_table_crc32(selected_axis),
        selected_noise_std_v=tuple(noises),
        source_schema=SPARSE_PLAN_SCHEMA,
    )


def load_finger_sampling_template(
    source: FingerSamplingTemplate | SamplingPlan | Mapping[str, Any] | str | Path,
    *,
    mode: str = DEFAULT_MODE,
    expected_table_crc32: int | None = None,
    teacher_path: str | Path | None = None,
) -> FingerSamplingTemplate:
    """Load/validate a strict template or a v4 45-point selection report."""

    report_path: Path | None = None
    if isinstance(source, FingerSamplingTemplate):
        result = source
    elif isinstance(source, SamplingPlan):
        result = FingerSamplingTemplate(plan=source)
    else:
        if isinstance(source, (str, Path)):
            report_path = Path(source).resolve()
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        elif isinstance(source, Mapping):
            payload = source
        else:
            raise TypeError(
                "sampling template must be a template, plan, mapping or JSON path"
            )
        schema = payload.get("schema")
        if schema == TEMPLATE_SCHEMA:
            result = FingerSamplingTemplate.from_dict(payload)
        elif schema == AUTO_SELECTION_SCHEMA:
            result = _template_from_auto_selection_report(payload, mode)
        elif schema == SPARSE_PLAN_SCHEMA:
            result = _template_from_sparse_plan_artifact(
                payload, report_path, teacher_path
            )
        else:
            raise ValueError(f"unsupported sampling-template schema: {schema!r}")

    expected_crc = _parse_crc(expected_table_crc32)
    if expected_crc is not None and result.table_crc32 != expected_crc:
        raise ValueError(
            "sampling-template/table CRC mismatch: "
            f"template=0x{int(result.table_crc32):08X}, "
            f"expected=0x{expected_crc:08X}"
        )
    return result


__all__ = [
    "AUTO_SELECTION_SCHEMA",
    "SPARSE_PLAN_SCHEMA",
    "TEMPLATE_SCHEMA",
    "FingerSamplingTemplate",
    "load_finger_sampling_template",
    "wavelength_table_crc32",
]
