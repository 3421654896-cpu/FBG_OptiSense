"""Offline speed/accuracy Pareto audit for the nine-FBG CH1 finger.

This module is intentionally unable to talk to the laser, wavemeter or CNC.
It combines the captured 40 kOhm CH1 teacher spectrum, the audited 2001-point
row identities and an existing stress-frame benchmark.  The output is an
evaluation report only; it never changes a mode table or firmware source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from adaptive_sampling_optimizer import (
    PeakSamplingTemplate,
    SamplingPlan,
    build_peak_template,
    optimize_ch1_sampling,
)


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
DEFAULT_TEACHER = HERE / "artifacts" / "ch1_teacher_peaks_20260905_40k.json"
DEFAULT_BENCHMARK = HERE / "stress_realtime_benchmark_latest.json"
DEFAULT_FIRMWARE = REPOSITORY / "JDSU" / "Core" / "Src" / "ms5614t.c"
DEFAULT_STRESS_HEADER = REPOSITORY / "JDSU" / "Core" / "Inc" / "stress_table.h"
DEFAULT_OUTPUT = HERE / "artifacts" / "ch1_sampling_pareto_latest.json"

EXPECTED_PEAK_COUNT = 9
FULL_MAP_TARGET_HZ = 15.0
PRESSURE_BANDWIDTH_HZ = 15.0
PRESSURE_NYQUIST_MIN_HZ = 2.0 * PRESSURE_BANDWIDTH_HZ
ADC_REFERENCE_V = 2.5
ADC_CODE_COUNT = 4096.0
ADC_QUANTUM_V = ADC_REFERENCE_V / ADC_CODE_COUNT
DAC_CODE_LIMITS = (58981, 58981, 32767, 24575, 24575)


@dataclass(frozen=True)
class TeacherCapture:
    wavelength_nm: np.ndarray
    signal_v: np.ndarray
    noise_std_v: np.ndarray
    stable_mask: np.ndarray
    reliability_weight: np.ndarray
    source_indices: np.ndarray
    dac_codes: np.ndarray
    peak_regions: tuple[slice, ...]
    source: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _load_normalized_audited_dac_rows(path: Path) -> tuple[tuple[int, ...], ...]:
    """Load the authoritative 2001 rows using the runtime safety normalization."""

    rows: list[tuple[int, ...]] = []
    code_fields = (
        "gain_code",
        "soa_code",
        "phase_code",
        "wavelength_a_code",
        "wavelength_b_code",
    )
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"index", "single_mode_pass", *code_fields}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("2001-point audit table has incomplete DAC columns")
        for expected_index, raw in enumerate(reader):
            if int(raw["index"]) != expected_index:
                raise ValueError("2001-point audit table indices are not contiguous")
            if int(raw["single_mode_pass"]) != 1:
                raise ValueError(f"audit row {expected_index} is not single mode")
            codes = []
            for field, limit in zip(code_fields, DAC_CODE_LIMITS):
                code = int(raw[field])
                # The runtime maps the one-code quantization overflow at the
                # physical current ceiling back to the certified safe maximum.
                if code == limit + 1:
                    code = limit
                if not 0 <= code <= limit:
                    raise ValueError(f"audit row {expected_index} exceeds DAC limits")
                codes.append(code)
            rows.append(tuple(codes))
    if len(rows) != 2001:
        raise ValueError(f"audit table must contain 2001 rows, got {len(rows)}")
    return tuple(rows)


def load_teacher_capture(path: Path = DEFAULT_TEACHER) -> TeacherCapture:
    """Load nine ordered 81-point measured CH1 peak windows."""

    path = Path(path)
    payload = _read_json(path)
    if payload.get("schema") != "fbg_ch1_teacher_spectrum_v1":
        raise ValueError("teacher spectrum schema is not supported")
    if int(payload.get("channel", -1)) != 1:
        raise ValueError("Pareto evaluation requires the CH1 teacher")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, dict):
        raise ValueError("teacher spectrum has no indexed rows")
    rows = sorted(raw_rows.values(), key=lambda item: int(item["index"]))
    source_indices = np.asarray([int(row["index"]) for row in rows], dtype=np.int64)
    if source_indices.size == 0 or np.unique(source_indices).size != source_indices.size:
        raise ValueError("teacher row indices must be unique")
    wavelength = np.asarray(
        [float(row["measured_wavelength_nm"]) for row in rows], dtype=float
    )
    signal = np.asarray([float(row["ch1_voltage_v"]) for row in rows], dtype=float)
    sigma = np.asarray(
        [float(row.get("ch1_sigma_v", math.nan)) for row in rows], dtype=float
    )
    stable = np.asarray(
        [
            bool(row.get("stable", False))
            and bool(row.get("all_monitor_channels_stable", False))
            and not bool(row.get("ch1_saturated", False))
            for row in rows
        ],
        dtype=bool,
    )
    codes = np.asarray([row["dac_codes"] for row in rows], dtype=np.int64)
    if (
        np.any(~np.isfinite(wavelength))
        or np.any(~np.isfinite(signal))
        or np.any(np.diff(wavelength) <= 0.0)
        or codes.shape != (source_indices.size, 5)
    ):
        raise ValueError("teacher spectrum axis, signal or DAC rows are invalid")

    source_metadata = payload.get("source", {})
    audit_name = Path(str(source_metadata.get("table", ""))).name
    if not audit_name:
        raise ValueError("teacher spectrum does not identify its 2001-point audit table")
    audit_path = HERE / audit_name
    expected_audit_hash = str(source_metadata.get("table_sha256", ""))
    actual_audit_hash = _sha256(audit_path)
    if not expected_audit_hash or actual_audit_hash != expected_audit_hash:
        raise ValueError("teacher spectrum audit-table fingerprint does not match")
    audited_dac_rows = _load_normalized_audited_dac_rows(audit_path)
    for position, source_index in enumerate(source_indices):
        if tuple(int(value) for value in codes[position]) != audited_dac_rows[
            int(source_index)
        ]:
            raise ValueError(
                f"teacher row {int(source_index)} is not an exact normalized audit row"
            )
    noise = np.where(np.isfinite(sigma) & (sigma >= 0.0), sigma, ADC_QUANTUM_V)
    noise = np.maximum(noise, ADC_QUANTUM_V)
    reference_noise = float(np.median(noise[stable])) if np.any(stable) else ADC_QUANTUM_V
    reliability = np.clip((reference_noise / noise) ** 2, 1e-4, 1e4)
    reliability[~stable] = np.minimum(reliability[~stable], 1e-4)

    sessions = [
        session
        for session in payload.get("sessions", ())
        if str(session.get("status", "")).lower() == "complete"
    ]
    if len(sessions) != EXPECTED_PEAK_COUNT:
        raise ValueError(
            f"teacher must contain exactly {EXPECTED_PEAK_COUNT} complete peak windows"
        )
    peak_regions: list[slice] = []
    previous_stop = -1
    for session in sessions:
        start_index = int(session["actual_start_index"])
        stop_index = int(session["last_saved_index"])
        positions = np.flatnonzero(
            (source_indices >= start_index) & (source_indices <= stop_index)
        )
        expected_size = stop_index - start_index + 1
        if (
            positions.size != expected_size
            or np.any(np.diff(source_indices[positions]) != 1)
            or int(positions[0]) <= previous_stop
        ):
            raise ValueError("teacher peak windows must be complete and non-overlapping")
        peak_regions.append(slice(int(positions[0]), int(positions[-1]) + 1))
        previous_stop = int(positions[-1])

    return TeacherCapture(
        wavelength_nm=wavelength,
        signal_v=signal,
        noise_std_v=noise,
        stable_mask=stable,
        reliability_weight=reliability,
        source_indices=source_indices,
        dac_codes=codes,
        peak_regions=tuple(peak_regions),
        source={
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "audit_table_path": str(audit_path.resolve()),
            "audit_table_sha256": actual_audit_hash,
            "dac_rows_verified_against_normalized_2001_audit": True,
            "captured_points": int(payload.get("summary", {}).get("captured_points", 0)),
            "stable_points": int(payload.get("summary", {}).get("stable_points", 0)),
            "feedback_kohm": int(payload.get("settings", {}).get("ch1_feedback_kohm", 0)),
        },
    )


def _four_point_plan(
    teacher: TeacherCapture,
    five_point_plan: SamplingPlan,
) -> SamplingPlan:
    """Drop one five-point anchor while retaining peak and two-sided coverage."""

    peaks: list[PeakSamplingTemplate] = []
    for peak_number, five in enumerate(five_point_plan.peaks, start=1):
        best: PeakSamplingTemplate | None = None
        best_score: tuple[float, float, float] | None = None
        for selected in combinations((int(value) for value in five.selected_indices), 4):
            if int(five.peak_index) not in selected:
                continue
            if not any(index < int(five.peak_index) for index in selected):
                continue
            if not any(index > int(five.peak_index) for index in selected):
                continue
            candidate = build_peak_template(
                teacher.wavelength_nm,
                teacher.signal_v,
                teacher.peak_regions[peak_number - 1],
                selected,
                peak_number=peak_number,
                reliability_weight=teacher.reliability_weight,
            )
            score = _template_identifiability_score(candidate, teacher)
            if best_score is None or score < best_score:
                best = candidate
                best_score = score
        if best is None:
            raise ValueError(f"G{peak_number} has no identifiable four-point subset")
        peaks.append(best)
    return SamplingPlan(peaks=tuple(peaks), points_per_peak=(4,) * len(peaks))


def _noise_limited_sigma_pm(
    template: PeakSamplingTemplate,
    teacher: TeacherCapture,
) -> float:
    signal = np.asarray(template.selected_template_signal, dtype=float)
    derivative = -np.asarray(template.selected_template_slope, dtype=float)
    noise = teacher.noise_std_v[np.asarray(template.selected_indices, dtype=int)]
    design = np.column_stack((np.ones(signal.size), signal))
    whitened_design = design / noise[:, None]
    whitened_derivative = derivative / noise
    projection = whitened_design @ np.linalg.lstsq(
        whitened_design, whitened_derivative, rcond=None
    )[0]
    fisher = float(np.sum((whitened_derivative - projection) ** 2))
    return 1000.0 / math.sqrt(fisher) if fisher > 0.0 else math.inf


def _template_identifiability_score(
    template: PeakSamplingTemplate,
    teacher: TeacherCapture,
) -> tuple[float, float, float]:
    """Penalize ambiguous shift fits that a local Fisher score cannot see."""

    true_shift = np.linspace(-0.040, 0.040, 81)
    sample_x = np.asarray(template.selected_wavelength_nm, dtype=float)
    dense_x = np.asarray(template.dense_wavelength_nm, dtype=float)
    dense_y = np.asarray(template.dense_signal, dtype=float)
    observed = np.interp(
        (sample_x[None, :] - true_shift[:, None]).ravel(), dense_x, dense_y
    ).reshape(true_shift.size, sample_x.size)
    noise = teacher.noise_std_v[np.asarray(template.selected_indices, dtype=int)]
    estimated, _accepted, _uncertainty, _mismatch = _estimate_shift_batch(
        template, observed, noise
    )
    absolute_error_pm = np.abs(estimated - true_shift) * 1000.0
    sigma_pm = _noise_limited_sigma_pm(template, teacher)
    return (
        float(np.percentile(absolute_error_pm, 95.0) + 3.0 * sigma_pm),
        float(np.max(absolute_error_pm)),
        -float(template.projected_shift_information),
    )


def build_candidate_plans(teacher: TeacherCapture) -> dict[str, SamplingPlan]:
    common = {
        "wavelength_nm": teacher.wavelength_nm,
        "teacher_signal": teacher.signal_v,
        "peak_regions": teacher.peak_regions,
        "expected_peak_count": EXPECTED_PEAK_COUNT,
        "sample_noise_std": teacher.noise_std_v,
        "stable_mask": teacher.stable_mask,
    }
    five = optimize_ch1_sampling(points_per_peak=5, **common)
    four = _four_point_plan(teacher, five)
    six = optimize_ch1_sampling(points_per_peak=6, **common)

    # Start from the identifiable four-point floor and spend nine additional
    # samples where they reduce the worst noise-limited peak uncertainty most.
    options: dict[int, tuple[PeakSamplingTemplate, ...]] = {
        4: four.peaks,
        5: five.peaks,
        6: six.peaks,
    }
    for count in range(7, 14):
        options[count] = optimize_ch1_sampling(
            points_per_peak=count, **common
        ).peaks
    option_scores = {
        count: tuple(
            _template_identifiability_score(template, teacher)[0]
            for template in templates
        )
        for count, templates in options.items()
    }
    counts = [4] * EXPECTED_PEAK_COUNT
    for _ in range(9):
        best_peak = None
        best_objective = None
        for peak_index in range(EXPECTED_PEAK_COUNT):
            trial_counts = list(counts)
            trial_counts[peak_index] += 1
            scores = [
                option_scores[count][index]
                for index, count in enumerate(trial_counts)
            ]
            objective = (max(scores), float(np.mean(scores)))
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_peak = peak_index
        if best_peak is None:
            raise AssertionError("adaptive allocation failed")
        counts[best_peak] += 1
    adaptive = SamplingPlan(
        peaks=tuple(options[count][index] for index, count in enumerate(counts)),
        points_per_peak=tuple(counts),
    )
    return {
        "uniform_4x9_36": four,
        "uniform_5x9_45": five,
        "uniform_6x9_54": six,
        "adaptive_45": adaptive,
    }


def _estimate_shift_batch(
    template: PeakSamplingTemplate,
    observed: np.ndarray,
    noise_std: np.ndarray,
    *,
    lower_nm: float = -0.08,
    upper_nm: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized affine-template fit equivalent to the runtime estimator."""

    sample_x = np.asarray(template.selected_wavelength_nm, dtype=float)
    dense_x = np.asarray(template.dense_wavelength_nm, dtype=float)
    dense_y = np.asarray(template.dense_signal, dtype=float)
    observed = np.asarray(observed, dtype=float)
    noise = np.asarray(noise_std, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != sample_x.size:
        raise ValueError("observed batch shape does not match template")
    if noise.shape != sample_x.shape or np.any(noise <= 0.0):
        raise ValueError("noise vector does not match template")
    shifts = np.linspace(lower_nm, upper_nm, 1601)
    query = sample_x[None, :] - shifts[:, None]
    grid = np.interp(query.ravel(), dense_x, dense_y).reshape(query.shape)
    weights = 1.0 / (noise * noise)
    weight_sum = float(np.sum(weights))
    template_mean = (grid @ weights) / weight_sum
    centered_template = grid - template_mean[:, None]
    denominator = np.sum(centered_template**2 * weights[None, :], axis=1)
    observed_mean = (observed @ weights) / weight_sum
    centered_observed = observed - observed_mean[:, None]
    numerator = centered_observed @ (centered_template * weights[None, :]).T
    scale_grid = np.divide(
        numerator,
        denominator[None, :],
        out=np.full_like(numerator, np.nan),
        where=denominator[None, :] > np.finfo(float).eps,
    )
    weighted_energy = np.sum(
        centered_observed**2 * weights[None, :], axis=1
    )
    rss = weighted_energy[:, None] - np.divide(
        numerator**2,
        denominator[None, :],
        out=np.zeros_like(numerator),
        where=denominator[None, :] > np.finfo(float).eps,
    )
    best_index = np.nanargmin(rss, axis=1)
    estimated_shift = shifts[best_index].copy()
    step = float(shifts[1] - shifts[0])
    interior = (best_index > 0) & (best_index + 1 < shifts.size)
    rows = np.flatnonzero(interior)
    if rows.size:
        center_rss = rss[rows, best_index[rows]]
        left_rss = rss[rows, best_index[rows] - 1]
        right_rss = rss[rows, best_index[rows] + 1]
        curvature = left_rss - 2.0 * center_rss + right_rss
        offset_steps = np.divide(
            0.5 * (left_rss - right_rss),
            curvature,
            out=np.zeros_like(curvature),
            where=np.abs(curvature) > np.finfo(float).eps,
        )
        estimated_shift[rows] += np.clip(offset_steps, -1.0, 1.0) * step

    fitted_template = np.interp(
        (sample_x[None, :] - estimated_shift[:, None]).ravel(),
        dense_x,
        dense_y,
    ).reshape(observed.shape)
    fitted_mean = (fitted_template @ weights) / weight_sum
    fitted_centered = fitted_template - fitted_mean[:, None]
    fitted_denominator = np.sum(fitted_centered**2 * weights[None, :], axis=1)
    fitted_numerator = np.sum(
        centered_observed * fitted_centered * weights[None, :], axis=1
    )
    scale = np.divide(
        fitted_numerator,
        fitted_denominator,
        out=np.full(observed.shape[0], np.nan),
        where=fitted_denominator > np.finfo(float).eps,
    )
    offset = observed_mean - scale * fitted_mean
    residual = observed - (offset[:, None] + scale[:, None] * fitted_template)
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    observed_range = np.maximum(
        np.ptp(observed, axis=1),
        np.abs(scale) * float(np.ptp(dense_y)) * 0.25,
    )
    normalized_rmse = rmse / np.maximum(observed_range, np.finfo(float).eps)

    dense_slope = np.gradient(dense_y, dense_x, edge_order=2)
    uncertainty = np.full(observed.shape[0], math.inf, dtype=float)
    for row in range(observed.shape[0]):
        shifted_slope = np.interp(
            sample_x - estimated_shift[row], dense_x, dense_slope
        )
        derivative = -scale[row] * shifted_slope
        design = np.column_stack((np.ones(sample_x.size), fitted_template[row]))
        whitened_design = design / noise[:, None]
        whitened_derivative = derivative / noise
        projected = whitened_derivative - whitened_design @ np.linalg.lstsq(
            whitened_design, whitened_derivative, rcond=None
        )[0]
        fisher = float(np.dot(projected, projected))
        if fisher > 0.0:
            uncertainty[row] = 1.0 / math.sqrt(fisher)
    accepted = (
        np.isfinite(scale)
        & (scale >= 0.10)
        & (scale <= 10.0)
        & (uncertainty <= 0.010)
        & (normalized_rmse <= 0.30)
        & (estimated_shift > lower_nm + 0.0002)
        & (estimated_shift < upper_nm - 0.0002)
    )
    return estimated_shift, accepted, uncertainty, normalized_rmse


def _simulate_peak(
    template: PeakSamplingTemplate,
    teacher: TeacherCapture,
    true_shift_nm: np.ndarray,
    rng: np.random.Generator,
    *,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_x = np.asarray(template.selected_wavelength_nm, dtype=float)
    dense_x = np.asarray(template.dense_wavelength_nm, dtype=float)
    dense_y = np.asarray(template.dense_signal, dtype=float)
    center = float(dense_x[int(np.argmax(dense_y))])
    count = true_shift_nm.size
    amplitude = rng.uniform(0.75, 1.25, count)
    peak_range = max(float(np.ptp(dense_y)), ADC_QUANTUM_V)
    baseline = rng.uniform(-0.05, 0.05, count) * peak_range
    if scenario == "model_mismatch_stress":
        width_scale = np.clip(rng.normal(1.0, 0.025, count), 0.94, 1.06)
        tilt_fraction = rng.normal(0.0, 0.015, count)
        noise_multiplier = 3.0
    elif scenario == "noise_stress":
        width_scale = np.ones(count, dtype=float)
        tilt_fraction = np.zeros(count, dtype=float)
        noise_multiplier = 3.0
    elif scenario == "nominal":
        width_scale = np.ones(count, dtype=float)
        tilt_fraction = np.zeros(count, dtype=float)
        noise_multiplier = 1.0
    else:
        raise ValueError(f"unknown simulation scenario: {scenario}")
    query = center + (
        sample_x[None, :] - center - true_shift_nm[:, None]
    ) / width_scale[:, None]
    shape = np.interp(query.ravel(), dense_x, dense_y).reshape(query.shape)
    span = max(float(dense_x[-1] - dense_x[0]), 1e-9)
    tilt_axis = (sample_x - center) / (0.5 * span)
    distortion = tilt_fraction[:, None] * tilt_axis[None, :] * peak_range
    selected_noise = (
        teacher.noise_std_v[np.asarray(template.selected_indices, dtype=int)]
        * noise_multiplier
    )
    observed = (
        baseline[:, None]
        + amplitude[:, None] * shape
        + distortion
        + rng.normal(0.0, selected_noise[None, :], shape.shape)
    )
    return _estimate_shift_batch(template, observed, selected_noise)


def simulate_plan_accuracy(
    plan: SamplingPlan,
    teacher: TeacherCapture,
    *,
    trials: int,
    seed: int,
    scenario: str,
) -> dict:
    if trials < 8:
        raise ValueError("at least eight Monte Carlo trials are required")
    rng = np.random.default_rng(seed)
    errors = np.empty((trials, len(plan.peaks)), dtype=float)
    accepted = np.empty((trials, len(plan.peaks)), dtype=bool)
    uncertainty = np.empty((trials, len(plan.peaks)), dtype=float)
    mismatch = np.empty((trials, len(plan.peaks)), dtype=float)
    per_peak = []
    for peak_index, template in enumerate(plan.peaks):
        true_shift = rng.uniform(-0.040, 0.040, trials)
        estimate, valid, sigma, normalized_rmse = _simulate_peak(
            template, teacher, true_shift, rng, scenario=scenario
        )
        error_pm = (estimate - true_shift) * 1000.0
        errors[:, peak_index] = error_pm
        accepted[:, peak_index] = valid
        uncertainty[:, peak_index] = sigma * 1000.0
        mismatch[:, peak_index] = normalized_rmse
        per_peak.append({
            "fbg": f"G{peak_index + 1}",
            "point_count": int(template.selected_indices.size),
            "rmse_error_pm": float(np.sqrt(np.mean(error_pm**2))),
            "p95_absolute_error_pm": float(np.percentile(np.abs(error_pm), 95.0)),
            "acceptance_rate": float(np.mean(valid)),
            "median_reported_uncertainty_pm": float(np.median(sigma) * 1000.0),
            "median_normalized_template_mismatch": float(np.median(normalized_rmse)),
        })
    return {
        "scenario": scenario,
        "trials_per_peak": int(trials),
        "shift_range_pm": [-40.0, 40.0],
        "all_nine_acceptance_rate": float(np.mean(np.all(accepted, axis=1))),
        "mean_point_acceptance_rate": float(np.mean(accepted)),
        "all_peak_rmse_error_pm": float(np.sqrt(np.mean(errors**2))),
        "worst_peak_rmse_error_pm": float(
            max(item["rmse_error_pm"] for item in per_peak)
        ),
        "worst_peak_p95_absolute_error_pm": float(
            max(item["p95_absolute_error_pm"] for item in per_peak)
        ),
        "per_peak": per_peak,
    }


def simulate_single_contact_localization(
    plan: SamplingPlan,
    teacher: TeacherCapture,
    *,
    trials: int,
    seed: int,
) -> dict:
    """Measure spectral G1..G9 identification, not mechanical contact area."""

    rng = np.random.default_rng(seed)
    active_peak = rng.integers(0, len(plan.peaks), size=trials)
    active_shift = rng.uniform(0.008, 0.040, size=trials)
    estimates = np.empty((trials, len(plan.peaks)), dtype=float)
    accepted = np.empty((trials, len(plan.peaks)), dtype=bool)
    for peak_index, template in enumerate(plan.peaks):
        true_shift = np.where(active_peak == peak_index, active_shift, 0.0)
        estimate, valid, _sigma, _mismatch = _simulate_peak(
            template, teacher, true_shift, rng, scenario="noise_stress"
        )
        estimates[:, peak_index] = estimate
        accepted[:, peak_index] = valid
    predicted = np.argmax(np.abs(estimates), axis=1)
    return {
            "scenario": "one_active_FBG_8_to_40_pm_at_3x_noise",
        "trials": int(trials),
        "correct_peak_rate": float(np.mean(predicted == active_peak)),
        "all_nine_estimates_accepted_rate": float(np.mean(np.all(accepted, axis=1))),
        "scope_limit": (
            "This checks spectral G1..G9 identity only. Contact area requires CNC-labelled "
            "multi-position pressure training and is not inferred by this report."
        ),
    }


def load_timing_reference(
    benchmark_path: Path = DEFAULT_BENCHMARK,
    firmware_path: Path = DEFAULT_FIRMWARE,
) -> dict:
    benchmark_path = Path(benchmark_path)
    firmware_path = Path(firmware_path)
    stress_header_path = DEFAULT_STRESS_HEADER
    benchmark = _read_json(benchmark_path)
    point_count = int(benchmark["point_count"])
    period = benchmark["period_ms"]
    if point_count <= 0 or float(period["mean"]) <= 0.0:
        raise ValueError("stress timing benchmark is invalid")
    firmware_text = (
        firmware_path.read_text(encoding="utf-8")
        + "\n"
        + stress_header_path.read_text(encoding="utf-8")
    )

    def macro(name: str) -> int:
        match = re.search(
            rf"^#define\s+{re.escape(name)}\s+(\d+)U?\b",
            firmware_text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise ValueError(f"firmware timing macro {name} was not found")
        return int(match.group(1))

    return {
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": _sha256(benchmark_path),
        "benchmark_schema": benchmark.get("schema"),
        "reference_point_count": point_count,
        "reference_measured_hz": float(benchmark["measured_hz"]),
        "reference_period_ms": {
            key: float(period[key]) for key in ("mean", "median", "p95", "p99")
        },
        "reference_peak_segments": 8,
        "reference_hidden_predecessors": 1,
        "firmware_path": str(firmware_path.resolve()),
        "firmware_sha256": _sha256(firmware_path),
        "stress_table_header_path": str(stress_header_path.resolve()),
        "stress_table_header_sha256": _sha256(stress_header_path),
        "firmware_delays_us": {
            "first_adc": macro("FAST_ADC_FIRST_DELAY_US"),
            "two_sample_spacing": macro("FAST_ADC_SPACING_US"),
            "section_boundary_extra": macro("FAST_BOUNDARY_EXTRA_DELAY_US"),
            "certified_hidden_predecessor": macro(
                "STRESS_PATH_PRECONDITION_HOLD_US"
            ),
        },
        "projection_method": (
            "conservative direct scaling of the measured 63-point host-arrival periods; "
            "the old frame's 5 ms hidden predecessor is deliberately not subtracted"
        ),
    }


def project_timing(
    point_count: int,
    timing: dict,
    *,
    output_frames_per_cycle: int = 1,
) -> dict:
    """Project timing from measured effective cost; never label it as measured."""

    reference_count = int(timing["reference_point_count"])
    ratio = float(point_count) / float(reference_count)
    periods = {
        key: float(value) * ratio
        for key, value in timing["reference_period_ms"].items()
    }
    mean_hz = 1000.0 / periods["mean"]
    p99_hz = 1000.0 / periods["p99"]
    firmware_delays = timing["firmware_delays_us"]
    theoretical_delay_floor_ms = (
        point_count
        * (
            int(firmware_delays["first_adc"])
            + int(firmware_delays["two_sample_spacing"])
        )
        + EXPECTED_PEAK_COUNT * int(firmware_delays["section_boundary_extra"])
    ) / 1000.0
    return {
        "point_count_equivalent": int(point_count),
        "output_frames_per_cycle": int(output_frames_per_cycle),
        "predicted_period_ms": periods,
        "predicted_mean_cycle_hz": mean_hz,
        "predicted_p99_cycle_hz": p99_hz,
        "firmware_explicit_delay_floor_ms": theoretical_delay_floor_ms,
        "predicted_full_map_meets_15_hz_mean": mean_hz >= FULL_MAP_TARGET_HZ,
        "predicted_full_map_meets_15_hz_p99": p99_hz >= FULL_MAP_TARGET_HZ,
        "is_measured_for_this_point_count": False,
        "requires_live_board_timing_benchmark": True,
    }


def project_global_plus_roi_schedule(roi_point_count: int, timing: dict) -> dict:
    """Project one 36-point map plus one active-peak ROI estimate per cycle."""

    if roi_point_count not in (4, 5, 6):
        raise ValueError("ROI point count must be 4, 5 or 6")
    projection = project_timing(
        36 + roi_point_count, timing, output_frames_per_cycle=2
    )
    mean_cycle_hz = float(projection["predicted_mean_cycle_hz"])
    p99_cycle_hz = float(projection["predicted_p99_cycle_hz"])
    projection.update({
        "schedule": (
            "36-point complete map followed by one "
            f"{roi_point_count}-point active-peak ROI estimate"
        ),
        "global_points_per_peak": 4,
        "roi_point_count": int(roi_point_count),
        "predicted_mean_complete_map_hz": mean_cycle_hz,
        "predicted_p99_complete_map_hz": p99_cycle_hz,
        # The active FBG is estimated once inside the global map and once in
        # the extra ROI, hence two active-location samples per schedule cycle.
        "predicted_mean_active_location_sample_hz": 2.0 * mean_cycle_hz,
        "predicted_p99_active_location_sample_hz": 2.0 * p99_cycle_hz,
        "mean_complete_map_15_hz_gate": mean_cycle_hz >= FULL_MAP_TARGET_HZ,
        "p99_complete_map_15_hz_gate": p99_cycle_hz >= FULL_MAP_TARGET_HZ,
        "mean_pressure_nyquist_gate": (
            2.0 * mean_cycle_hz >= PRESSURE_NYQUIST_MIN_HZ
        ),
        "p99_pressure_nyquist_gate": (
            2.0 * p99_cycle_hz >= PRESSURE_NYQUIST_MIN_HZ
        ),
        "joint_mean_projection_gate": (
            mean_cycle_hz >= FULL_MAP_TARGET_HZ
            and 2.0 * mean_cycle_hz >= PRESSURE_NYQUIST_MIN_HZ
        ),
        "joint_p99_projection_gate": (
            p99_cycle_hz >= FULL_MAP_TARGET_HZ
            and 2.0 * p99_cycle_hz >= PRESSURE_NYQUIST_MIN_HZ
        ),
        "warning": (
            "Direct point-count projection omits extra frame-boundary and ROI path "
            "overhead. Treat even a passing mean estimate as a design candidate, "
            "not demonstrated performance."
        ),
    })
    return projection


def project_roi_only_schedule(roi_point_count: int, timing: dict) -> dict:
    """Project an already-localized active-peak tracker."""

    if roi_point_count not in (4, 5, 6):
        raise ValueError("ROI point count must be 4, 5 or 6")
    projection = project_timing(roi_point_count, timing)
    projection.update({
        "scope": "one already-localized active FBG; not a complete position map",
        "roi_point_count": int(roi_point_count),
        "predicted_mean_active_location_sample_hz": projection[
            "predicted_mean_cycle_hz"
        ],
        "predicted_p99_active_location_sample_hz": projection[
            "predicted_p99_cycle_hz"
        ],
        "mean_pressure_nyquist_gate": (
            projection["predicted_mean_cycle_hz"] >= PRESSURE_NYQUIST_MIN_HZ
        ),
        "p99_pressure_nyquist_gate": (
            projection["predicted_p99_cycle_hz"] >= PRESSURE_NYQUIST_MIN_HZ
        ),
        "complete_map_15_hz_gate": False,
    })
    return projection


def project_periodic_global_roi_schedule(
    global_point_count: int,
    global_refresh_hz: float,
    roi_point_count: int,
    timing: dict,
) -> dict:
    """Mean-throughput budget for an accuracy-first periodic global refresh."""

    if global_point_count not in (36, 45, 54):
        raise ValueError("global point count must be 36, 45 or 54")
    if roi_point_count not in (4, 5, 6):
        raise ValueError("ROI point count must be 4, 5 or 6")
    if not math.isfinite(global_refresh_hz) or global_refresh_hz <= 0.0:
        raise ValueError("global refresh rate must be positive")
    mean_point_throughput = (
        float(timing["reference_point_count"])
        * float(timing["reference_measured_hz"])
    )
    global_point_budget = float(global_point_count) * global_refresh_hz
    remaining_point_budget = max(0.0, mean_point_throughput - global_point_budget)
    extra_roi_hz = remaining_point_budget / float(roi_point_count)
    active_hz = global_refresh_hz + extra_roi_hz
    return {
        "global_point_count": int(global_point_count),
        "global_refresh_hz": float(global_refresh_hz),
        "roi_point_count": int(roi_point_count),
        "reference_mean_point_throughput_per_s": mean_point_throughput,
        "predicted_additional_roi_sample_hz": extra_roi_hz,
        "predicted_active_location_total_sample_hz": active_hz,
        "complete_map_15_hz_gate": global_refresh_hz >= FULL_MAP_TARGET_HZ,
        "pressure_nyquist_gate": active_hz >= PRESSURE_NYQUIST_MIN_HZ,
        "projection_limit": (
            "Mean point-budget calculation only; it excludes path, frame and "
            "scheduler overhead and therefore requires a board benchmark."
        ),
    }


def plan_path_metrics(plan: SamplingPlan, teacher: TeacherCapture) -> dict:
    positions = np.asarray(plan.selected_indices, dtype=int)
    source_indices = teacher.source_indices[positions]
    wavelengths = teacher.wavelength_nm[positions]
    codes = teacher.dac_codes[positions].astype(float)
    limits = np.asarray((58981.0, 58981.0, 32767.0, 24575.0, 24575.0))
    normalized = codes / limits[None, :]
    cycle_indices = np.r_[source_indices, source_indices[0]]
    cycle_normalized = np.vstack((normalized, normalized[0]))
    index_jumps = np.abs(np.diff(cycle_indices))
    dac_l1 = np.sum(np.abs(np.diff(cycle_normalized, axis=0)), axis=1)
    dac_linf = np.max(np.abs(np.diff(cycle_normalized, axis=0)), axis=1)
    skipped = np.count_nonzero(np.diff(cycle_indices) != 1)
    return {
        "first_source_index": int(source_indices[0]),
        "last_source_index": int(source_indices[-1]),
        "wavelength_span_nm": float(wavelengths[-1] - wavelengths[0]),
        "maximum_forward_wavelength_gap_nm": float(np.max(np.diff(wavelengths))),
        "maximum_fullband_index_jump_including_frame_wrap": int(np.max(index_jumps)),
        "cyclic_normalized_DAC_L1_path": float(np.sum(dac_l1)),
        "maximum_normalized_DAC_Linf_step": float(np.max(dac_linf)),
        "transitions_not_equal_to_dense_immediate_predecessor": int(skipped),
        "all_sparse_transitions_require_hardware_replay_certification": True,
    }


def _plan_rows(plan: SamplingPlan, teacher: TeacherCapture) -> tuple[list[int], list[list[int]]]:
    positions = np.asarray(plan.selected_indices, dtype=int)
    return (
        [int(value) for value in teacher.source_indices[positions]],
        [[int(code) for code in row] for row in teacher.dac_codes[positions]],
    )


def _pareto_front(candidates: list[dict]) -> list[str]:
    front = []
    for candidate in candidates:
        period = float(candidate["timing"]["predicted_period_ms"]["mean"])
        error = float(
            candidate["accuracy"]["noise_stress"][
                "worst_peak_p95_absolute_error_pm"
            ]
        )
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_period = float(other["timing"]["predicted_period_ms"]["mean"])
            other_error = float(
                other["accuracy"]["noise_stress"][
                    "worst_peak_p95_absolute_error_pm"
                ]
            )
            if (
                other_period <= period
                and other_error <= error
                and (other_period < period or other_error < error)
            ):
                dominated = True
                break
        if not dominated:
            front.append(str(candidate["id"]))
    return front


def evaluate_pareto(
    teacher: TeacherCapture,
    timing: dict,
    *,
    trials: int = 160,
    seed: int = 20260905,
) -> dict:
    plans = build_candidate_plans(teacher)
    candidates = []
    for candidate_number, (name, plan) in enumerate(plans.items()):
        source_indices, dac_rows = _plan_rows(plan, teacher)
        nominal = simulate_plan_accuracy(
            plan,
            teacher,
            trials=trials,
            seed=seed + 1000 * candidate_number,
            scenario="nominal",
        )
        noise_stress = simulate_plan_accuracy(
            plan,
            teacher,
            trials=trials,
            seed=seed + 1000 * candidate_number + 1,
            scenario="noise_stress",
        )
        model_mismatch = simulate_plan_accuracy(
            plan,
            teacher,
            trials=trials,
            seed=seed + 1000 * candidate_number + 2,
            scenario="model_mismatch_stress",
        )
        localization = simulate_single_contact_localization(
            plan,
            teacher,
            trials=trials,
            seed=seed + 1000 * candidate_number + 3,
        )
        timing_projection = project_timing(plan.total_points, timing)
        accuracy_screen = (
            noise_stress["worst_peak_p95_absolute_error_pm"] <= 5.0
            and noise_stress["all_nine_acceptance_rate"] >= 0.90
            and localization["correct_peak_rate"] >= 0.99
        )
        candidates.append({
            "id": name,
            "point_count": int(plan.total_points),
            "points_per_peak": [int(value) for value in plan.points_per_peak],
            "selected_source_indices": source_indices,
            "selected_dac_rows_from_teacher_for_audit_only": dac_rows,
            "accuracy": {
                "nominal": nominal,
                "noise_stress": noise_stress,
                "model_mismatch_stress": model_mismatch,
                "single_contact_peak_identification": localization,
                "engineering_screen": {
                    "robust_worst_peak_p95_error_at_most_5_pm": (
                        noise_stress["worst_peak_p95_absolute_error_pm"] <= 5.0
                    ),
                    "robust_all_nine_acceptance_at_least_90_percent": (
                        noise_stress["all_nine_acceptance_rate"] >= 0.90
                    ),
                    "single_contact_peak_identity_at_least_99_percent": (
                        localization["correct_peak_rate"] >= 0.99
                    ),
                    "pass": accuracy_screen,
                    "thresholds_are_offline_engineering_screens_not_final_metrology_claims": True,
                },
            },
            "timing": timing_projection,
            "path": plan_path_metrics(plan, teacher),
            "offline_joint_gate": {
                "predicted_mean_full_map_at_least_15_hz": timing_projection[
                    "predicted_full_map_meets_15_hz_mean"
                ],
                "predicted_p99_full_map_at_least_15_hz": timing_projection[
                    "predicted_full_map_meets_15_hz_p99"
                ],
                "accuracy_screen_pass": accuracy_screen,
                "mean_only_pass": bool(
                    timing_projection["predicted_full_map_meets_15_hz_mean"]
                    and accuracy_screen
                ),
                "pass": bool(
                    timing_projection["predicted_full_map_meets_15_hz_p99"]
                    and accuracy_screen
                ),
                "hard_timing_gate_uses_p99_projection_not_mean": True,
                "final_hardware_acceptance_still_required": True,
            },
        })

    uniform36 = next(item for item in candidates if item["id"] == "uniform_4x9_36")
    uniform45 = next(item for item in candidates if item["id"] == "uniform_5x9_45")
    uniform54 = next(item for item in candidates if item["id"] == "uniform_6x9_54")
    simultaneous_schedules = {
        f"global36_plus_roi{roi_points}": project_global_plus_roi_schedule(
            roi_points, timing
        )
        for roi_points in (4, 5, 6)
    }
    roi_only_schedules = {
        f"roi{roi_points}": project_roi_only_schedule(roi_points, timing)
        for roi_points in (4, 5, 6)
    }
    periodic_schedules = {
        "speed_first_global36_at_5hz_plus_roi6": (
            project_periodic_global_roi_schedule(36, 5.0, 6, timing)
        ),
        "accuracy_first_global54_at_5hz_plus_roi6": (
            project_periodic_global_roi_schedule(54, 5.0, 6, timing)
        ),
    }

    viable = [
        item
        for item in candidates
        if item["offline_joint_gate"]["pass"]
    ]
    default_id = viable[0]["id"] if viable else None
    return {
        "schema": "ch1_sampling_pareto_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "modifies_mode_table": False,
        "modifies_firmware": False,
        "teacher": teacher.source,
        "requirements": {
            "ch1_fbg_count": EXPECTED_PEAK_COUNT,
            "complete_position_map_target_hz": FULL_MAP_TARGET_HZ,
            "pressure_waveform_bandwidth_hz": PRESSURE_BANDWIDTH_HZ,
            "pressure_waveform_nyquist_min_sample_hz": PRESSURE_NYQUIST_MIN_HZ,
            "distinction": (
                "15 complete maps/s is a display/localization update target. A real 15 Hz "
                "pressure waveform needs at least 30 samples/s at the active location."
            ),
        },
        "timing_reference": timing,
        "simulation": {
            "seed": int(seed),
            "trials_per_peak_and_localization": int(trials),
            "ADC_noise_floor_v": ADC_QUANTUM_V,
            "stress_scenarios": (
                "noise_stress uses 3x measured/quantization noise. The separate "
                "model_mismatch_stress also adds 0.94..1.06 peak-width variation "
                "and a small non-affine tilt. Both include 0.75..1.25 amplitude "
                "and baseline drift."
            ),
        },
        "candidates": candidates,
        "pareto_front": _pareto_front(candidates),
        "screened_out": [
            {
                "id": "shared_baseline_less_than_36",
                "status": "not_admitted",
                "reason": (
                    "Three samples per peak cannot independently identify offset, amplitude "
                    "and shift with a residual for fault detection. Sharing a baseline across "
                    "wavelengths is not certified because laser power and optical path are "
                    "wavelength/state dependent."
                ),
            }
        ],
        "hybrid_schedules": {
            "simultaneous_15hz_map_and_15hz_pressure_candidates": (
                simultaneous_schedules
            ),
            "active_peak_only_after_localization": roi_only_schedules,
            "active_peak_with_periodic_global_refresh": periodic_schedules,
        },
        "recommendation": {
            "production_default_fixed_table": default_id,
            "default_status": (
                "provisional_pending_live_board_benchmark_and_sparse_order_replay"
                if default_id is not None
                else (
                    "no_single_fixed-point mode simultaneously passes the robust "
                    "accuracy screen and the hard p99 15-map/s timing projection"
                )
            ),
            "fastest_complete_map_candidate": {
                "mode": "uniform_4x9_36",
                "predicted_mean_hz": uniform36["timing"][
                    "predicted_mean_cycle_hz"
                ],
                "predicted_p99_hz": uniform36["timing"][
                    "predicted_p99_cycle_hz"
                ],
                "status": (
                    "mean speed passes, but p99 speed and 3x-noise wavelength accuracy "
                    "do not; use only as a coarse localization candidate"
                ),
            },
            "balanced_legacy_default_candidate": {
                "mode": "uniform_5x9_45",
                "predicted_mean_hz": uniform45["timing"][
                    "predicted_mean_cycle_hz"
                ],
                "status": (
                    "keeps the five measured template-sensitive samples per peak, but "
                    "misses 15 maps/s and the 5 pm worst-peak stress-noise screen"
                ),
            },
            "accuracy_first_degraded_rate_mode": {
                "mode": "uniform_6x9_54",
                "predicted_mean_hz": uniform54["timing"][
                    "predicted_mean_cycle_hz"
                ],
                "status": (
                    "only fixed candidate passing the 3x-noise <=5 pm screen; use for "
                    "confirmation or accept a lower full-map rate"
                ),
            },
            "true_15hz_pressure_tracking": (
                "After a global map identifies the active region, use a six-point ROI by "
                "default. Four or five points are permitted only after the particular FBG "
                "passes a peak-specific uncertainty and sparse-order replay gate."
            ),
            "simultaneous_dual_target_candidate": (
                "36-point global plus 6-point ROI per cycle"
            ),
            "simultaneous_dual_target_status": (
                "mean projection is only barely above 15 complete maps/s and 30 active "
                "samples/s; p99 projection fails both, so this is not yet a valid claim"
            ),
            "adaptive_45_status": (
                "The evaluated information-based 45-point allocation is dominated by the "
                "uniform 45-point plan; do not promote it without new labelled validation."
            ),
            "template_shape_model_risk": (
                "No candidate passes the separate non-affine peak-shape mismatch diagnostic. "
                "Online template adaptation or labelled CNC retraining remains necessary."
            ),
            "contact_area_status": (
                "not established by optical-template simulation; requires CNC-labelled "
                "multi-position/multi-area pressure data"
            ),
            "final_acceptance": [
                "measure >=15 complete 36-point maps/s on MCU board telemetry",
                "measure >=30 active-location samples/s for a 15 Hz pressure waveform",
                "pass p99/jitter timing, not only mean throughput",
                "replay the exact sparse DAC order and certify every point",
                "validate G1..G9 location and contact area on held-out CNC presses",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--firmware", type=Path, default=DEFAULT_FIRMWARE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    teacher = load_teacher_capture(args.teacher)
    timing = load_timing_reference(args.benchmark, args.firmware)
    report = evaluate_pareto(
        teacher,
        timing,
        trials=args.trials,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "report": str(args.output.resolve()),
        "pareto_front": report["pareto_front"],
        "recommendation": report["recommendation"],
        "candidate_summary": [
            {
                "id": item["id"],
                "points": item["point_count"],
                "mean_hz": item["timing"]["predicted_mean_cycle_hz"],
                "noise_stress_p95_pm": item["accuracy"]["noise_stress"][
                    "worst_peak_p95_absolute_error_pm"
                ],
                "localization": item["accuracy"][
                    "single_contact_peak_identification"
                ]["correct_peak_rate"],
                "offline_gate": item["offline_joint_gate"]["pass"],
            }
            for item in report["candidates"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
