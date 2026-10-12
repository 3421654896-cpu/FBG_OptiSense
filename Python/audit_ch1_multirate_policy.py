"""Offline audit of the CH1 nine-FBG MAP45/SURVEY9/TRACK13 policy.

The script is intentionally read-only with respect to instruments, firmware and
mode tables.  It quantifies what the certified measured templates can support,
then checks the latest saved timing benchmark.  It does not certify contact
localisation or physical contact area; those require labelled mechanical data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = (
    HERE / "artifacts" / "ch1_measured_template_selection_v4_certified_20260905.json"
)
DEFAULT_BENCHMARK = HERE / "outputs" / "stress_bandwidth_v1.0.24_usb300.json"
DEFAULT_OUTPUT = HERE / "outputs" / "ch1_multirate_policy_audit_20260905.json"

FBG_COUNT = 9
POINTS_PER_FBG = 5
SENTINEL_LOCAL_OFFSETS = (3, 3, 3, 3, 1, 3, 1, 1, 1)
TRACK_LOCAL_OFFSETS = (1, 2, 3)
SHIFT_RANGE_PM = 100.0
LOCAL_LINEAR_RANGE_PM = 20.0
NYQUIST_PERIOD_MS = 1000.0 / 30.0


def _load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _projected_shift_information(
    signal: np.ndarray, slope: np.ndarray, weight: np.ndarray
) -> float:
    """Shift Fisher information after removing offset/amplitude nuisance."""

    signal = np.asarray(signal, dtype=float)
    derivative = -np.asarray(slope, dtype=float)
    weight = np.asarray(weight, dtype=float)
    design = np.column_stack((np.ones(signal.size), signal))
    root_weight = np.sqrt(weight)
    weighted_design = design * root_weight[:, None]
    weighted_derivative = derivative * root_weight
    projected = weighted_design @ np.linalg.lstsq(
        weighted_design, weighted_derivative, rcond=None
    )[0]
    residual = weighted_derivative - projected
    return float(np.dot(residual, residual))


def _linear_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    coefficients = np.polyfit(x, y, 1)
    fitted = np.polyval(coefficients, x)
    residual = float(np.sum(np.square(y - fitted)))
    total = float(np.sum(np.square(y - np.mean(y))))
    return 1.0 if total <= np.finfo(float).eps else 1.0 - residual / total


def _monotonic_window_pm(shifts_nm: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    zero = int(np.argmin(np.abs(shifts_nm)))
    gradient = np.gradient(response, shifts_nm)
    local = gradient[max(0, zero - 2) : zero + 3]
    direction = 1.0 if float(np.median(local)) >= 0.0 else -1.0
    # A tiny numerical tolerance prevents interpolation roundoff from ending a
    # genuinely monotonic interval.  Flat extrapolated tails are not admitted.
    scale = max(float(np.ptp(response)), 1.0)
    tolerance = 1e-12 * scale
    left = zero
    while left > 0 and direction * (response[left] - response[left - 1]) > tolerance:
        left -= 1
    right = zero
    while (
        right + 1 < response.size
        and direction * (response[right + 1] - response[right]) > tolerance
    ):
        right += 1
    return float(shifts_nm[left] * 1000.0), float(shifts_nm[right] * 1000.0)


def _template_metrics(payload: dict) -> list[dict]:
    stress = payload.get("stress", {})
    peaks = stress.get("peaks", ())
    rows = stress.get("rows", ())
    if len(peaks) != FBG_COUNT or len(rows) != FBG_COUNT * POINTS_PER_FBG:
        raise ValueError("certified template must contain 9 peaks and 45 rows")
    rows_by_peak = {
        peak_number: sorted(
            [row for row in rows if int(row["peak_number"]) == peak_number],
            key=lambda row: int(row["within_peak_order"]),
        )
        for peak_number in range(1, FBG_COUNT + 1)
    }
    shifts_nm = np.linspace(-SHIFT_RANGE_PM / 1000.0, SHIFT_RANGE_PM / 1000.0, 801)
    output = []
    for index, peak in enumerate(peaks):
        peak_number = index + 1
        peak_rows = rows_by_peak[peak_number]
        if len(peak_rows) != POINTS_PER_FBG:
            raise ValueError(f"G{peak_number} does not contain five selected rows")
        signal = np.asarray(peak["selected_template_signal_v"], dtype=float)
        slope = np.asarray(peak["selected_template_slope_v_per_nm"], dtype=float)
        weight = np.asarray(peak["selected_reliability_weight"], dtype=float)
        full_information = _projected_shift_information(signal, slope, weight)
        triplet = np.asarray(TRACK_LOCAL_OFFSETS, dtype=int)
        triplet_information = _projected_shift_information(
            signal[triplet], slope[triplet], weight[triplet]
        )

        sentinel = int(SENTINEL_LOCAL_OFFSETS[index])
        sample_nm = float(peak_rows[sentinel]["measured_wavelength_nm"])
        dense_x = np.asarray(peak["dense_wavelength_nm"], dtype=float)
        dense_y = np.asarray(peak["dense_template_signal_v"], dtype=float)
        baseline = float(np.interp(sample_nm, dense_x, dense_y))
        response = np.interp(sample_nm - shifts_nm, dense_x, dense_y) - baseline
        local_mask = np.abs(shifts_nm * 1000.0) <= LOCAL_LINEAR_RANGE_PM + 1e-9
        monotonic_left_pm, monotonic_right_pm = _monotonic_window_pm(
            shifts_nm, response
        )
        local_sensitivity_v_per_nm = float(
            np.polyfit(shifts_nm[local_mask], response[local_mask], 1)[0]
        )
        output.append(
            {
                "grating": f"G{peak_number}",
                "sentinel_local_offset": sentinel,
                "sentinel_wavelength_nm": sample_nm,
                "five_point_projected_shift_information": full_information,
                "track_triplet_projected_shift_information": triplet_information,
                "triplet_information_fraction": (
                    triplet_information / full_information
                    if full_information > 0.0
                    else None
                ),
                "sentinel_local_sensitivity_v_per_nm": local_sensitivity_v_per_nm,
                "sentinel_linear_r_squared_plus_minus_20_pm": _linear_r_squared(
                    shifts_nm[local_mask], response[local_mask]
                ),
                "sentinel_monotonic_shift_window_pm": [
                    monotonic_left_pm,
                    monotonic_right_pm,
                ],
                "sentinel_covers_certified_plus_minus_100_pm_monotonically": bool(
                    monotonic_left_pm <= -SHIFT_RANGE_PM
                    and monotonic_right_pm >= SHIFT_RANGE_PM
                ),
            }
        )
    return output


def _timing_metrics(payload: dict) -> dict:
    multirate = payload.get("multirate_freshness", {})
    per_grating = multirate.get("per_grating", {})
    rows = []
    for grating in (f"G{value}" for value in range(1, FBG_COUNT + 1)):
        period = per_grating.get(grating, {}).get("fresh_inter_sample_period_ms", {})
        rows.append(
            {
                "grating": grating,
                "mean_hz": period.get("mean_hz"),
                "p99_ms": period.get("p99"),
                "max_ms": period.get("max"),
                "p99_meets_30_hz_necessary_gate": (
                    period.get("p99") is not None
                    and float(period["p99"]) <= NYQUIST_PERIOD_MS
                ),
                "max_meets_30_hz_necessary_gate": (
                    period.get("max") is not None
                    and float(period["max"]) <= NYQUIST_PERIOD_MS
                ),
            }
        )
    return {
        "source_schema": payload.get("schema"),
        "firmware_versions": payload.get("firmware_versions"),
        "measured_frame_hz": payload.get("measured_frame_hz"),
        "host_period_ms": payload.get("period_ms"),
        "board_acquisition_ms": payload.get("board_telemetry", {}).get(
            "acquisition_ms"
        ),
        "board_frame_period_ms": payload.get("board_frame_period_ms"),
        "profile_distribution": multirate.get("profile_distribution"),
        "bandwidth_discontinuity_frames": multirate.get(
            "bandwidth_discontinuity_frames"
        ),
        "per_grating": rows,
        "all_gratings_p99_meet_30_hz_necessary_gate": all(
            row["p99_meets_30_hz_necessary_gate"] for row in rows
        ),
        "all_gratings_max_meet_30_hz_necessary_gate": all(
            row["max_meets_30_hz_necessary_gate"] for row in rows
        ),
        "benchmark_pass": payload.get("pass"),
        "interpretation": (
            "30 Hz is only the Nyquist necessary condition for a 15 Hz input. "
            "It does not establish 15 Hz optical/mechanical amplitude or phase fidelity."
        ),
    }


def build_report(template_path: Path, benchmark_path: Path | None) -> dict:
    template = _load_object(template_path)
    peak_metrics = _template_metrics(template)
    timing = None
    if benchmark_path is not None and benchmark_path.exists():
        timing = _timing_metrics(_load_object(benchmark_path))
    return {
        "schema": "ch1_multirate_policy_offline_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "hardware_or_firmware_modified": False,
        "sources": {
            "certified_template": str(template_path.resolve()),
            "saved_timing_benchmark": (
                None if benchmark_path is None else str(benchmark_path.resolve())
            ),
        },
        "requirements": {
            "channel": 1,
            "grating_count": FBG_COUNT,
            "physical_signal_bandwidth_hz": 15.0,
            "nyquist_necessary_fresh_rate_hz": 30.0,
            "preferred_fresh_rate_hz": 60.0,
        },
        "peak_metrics": peak_metrics,
        "summary": {
            "track_triplet_information_fraction_min": float(
                min(row["triplet_information_fraction"] for row in peak_metrics)
            ),
            "track_triplet_information_fraction_median": float(
                np.median(
                    [row["triplet_information_fraction"] for row in peak_metrics]
                )
            ),
            "track_triplet_information_fraction_max": float(
                max(row["triplet_information_fraction"] for row in peak_metrics)
            ),
            "all_sentinels_monotonic_over_plus_minus_100_pm": all(
                row["sentinel_covers_certified_plus_minus_100_pm_monotonically"]
                for row in peak_metrics
            ),
            "smallest_sentinel_monotonic_half_range_pm": float(
                min(
                    min(abs(value) for value in row["sentinel_monotonic_shift_window_pm"])
                    for row in peak_metrics
                )
            ),
        },
        "saved_timing_benchmark": timing,
        "decision": {
            "MAP45": (
                "retain as the certified full-state/reference mode; five samples per "
                "FBG support offset, amplitude, shift, two-sided shape and a residual"
            ),
            "SURVEY9": (
                "retain only as the minimum all-location activity watcher; one sentinel "
                "per FBG is not a unique peak-shift or physical-area measurement"
            ),
            "TRACK13": (
                "retain provisionally as the minimum two-ROI tracker (nine sentinels plus "
                "two extra samples for each of two triplets); it is not demonstrated for "
                "wide or more-than-two-region contact"
            ),
            "TRACK11": (
                "do not use as a global replacement for TRACK13; test it as a "
                "single-contact-first mode with nine fresh sentinels plus one identifiable "
                "triplet. It gains roughly 1.3-1.5 ms from the saved timing data but loses "
                "simultaneous second-region shape evidence"
            ),
            "point_reduction": (
                "TRACK11 can refresh one identifiable triplet but sacrifices simultaneous "
                "second-region shape tracking; SURVEY9-only inference requires labelled "
                "schedule-specific training and strict range/OOD guards"
            ),
            "hierarchical_runtime_policy": (
                "use TRACK11-SINGLE only when the nine-sentinel response is unimodal and "
                "one region is dominant; escalate to TRACK13-FAIR for two comparable "
                "regions, spatial broadening or OOD, and lower confidence for three-or-more "
                "region/wide contact because neither mode refreshes all peak centres"
            ),
            "TRACK11_promotion_gates": (
                "USB and LAN per-grating max fresh gap <=33.333 ms; held-out single-contact "
                "XY p95 degradation versus TRACK13 <=0.5 mm; effective-response-area error "
                "degradation <=5%; missed/false single-contact decisions <1%; every held-out "
                "two-contact or wide-contact case must escalate rather than emit a falsely "
                "precise TRACK11 result"
            ),
            "required_scheduler_fix_before_multicontact_claim": (
                "rank all nine regions using the same normalized sentinel statistic, add "
                "hysteresis and bounded-age/fair ROI rotation; do not compare a three-point "
                "maximum directly with a one-point maximum"
            ),
            "timestamp_requirement": (
                "use each sample_offset_us in a causal asynchronous estimator or validate "
                "deskew; frame-end timestamps alone do not make sequential FBG samples "
                "simultaneous at 15 Hz"
            ),
        },
        "conservative_v4_state_machine": {
            "normalization": (
                "learn per-sentinel median and 1.4826*MAD only while unloaded; use a "
                "minimum noise floor. Detect with z=abs(code-median)/sigma_code, but rank "
                "ROIs by the non-negative lower confidence bound of template-slope-normalized "
                "pseudo-shift. Extra triplet samples must never enter cross-grating ranking"
            ),
            "initial_thresholds_without_labelled_contact_data": {
                "activation_z": 8.0,
                "secondary_candidate_z": 5.0,
                "release_z": 3.0,
                "note": (
                    "conservative starting values, not certified contact thresholds; replace "
                    "only after held-out mechanical data"
                ),
            },
            "SURVEY9_to_TRACK13_WIDE": (
                "on any valid activation, immediately enter TRACK13-WIDE for confirmation; "
                "do not enter TRACK11 directly from an unconfirmed SURVEY sample"
            ),
            "TRACK13_WIDE_to_TRACK11_SINGLE": (
                "only after eight consecutive frames with one connected compact response, "
                "top weight share >=0.60, participation ratio <=1.8, no second candidate >=35% "
                "of the top, no monotonic-range violation and valid primary triplet shape"
            ),
            "TRACK11_SINGLE_to_TRACK13_WIDE": (
                "upgrade on the next frame after any second component/candidate, participation "
                "ratio >1.8, primary ambiguity, triplet-fit failure, sentinel range violation, "
                "missing freshness or OOD; require no persistence before upgrading"
            ),
            "TRACK_to_SURVEY9": (
                "all nine z scores <=3 for four consecutive frames, with no shape residual "
                "alarm; update the baseline only after return to qualified quiet SURVEY"
            ),
            "ROI_hysteresis_and_age": (
                "rank only common sentinels; keep a selected ROI unless a challenger exceeds "
                "its lower-confidence score by 25% for two frames, or switch immediately when "
                "the current ROI falls below release. In TRACK13, any active ROI that has "
                "missed four frames gets deadline priority, so all nine active ROIs are "
                "triplet-refreshed at least once every five frames"
            ),
            "MAP45_rule": (
                "one compulsory startup/reference MAP; no periodic MAP during active contact. "
                "A forced active MAP must declare a bandwidth discontinuity"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.template, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "summary": report["summary"],
                "timing_pass": (
                    None
                    if report["saved_timing_benchmark"] is None
                    else report["saved_timing_benchmark"]["benchmark_pass"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
