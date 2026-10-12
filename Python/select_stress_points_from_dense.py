"""Select informative monotonic stress points from settled single-value truth.

This is an experimental-design step, not a live intensity predictor.  The
settled spectrum is used only to select laser states whose seven samples cover
both slopes, the summit, and both edges of each FBG band.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from fbg_peak_fitting import fit_fbg_segment


HERE = Path(__file__).resolve().parent
DENSE_SOURCE = HERE / "dense_single_reflection_135.json"
DEFAULT_OUTPUT = HERE / "stress_points_shape_selected_proposal.json"
DEFAULT_PLOT = HERE / "stress_points_shape_selected_proposal.png"

# Static centers supplied by the user's broadband interrogator screenshot.
# They are priors for choosing a well-observed baseline window only; the live
# stress estimator remains free to move away from them on every frame.
REFERENCE_CENTERS_NM = (
    1527.9107,
    1532.1431,
    1536.3851,
    1540.1685,
    1544.4552,
    1547.8909,
    1551.7849,
    1555.8094,
    1559.9161,
)

# AQ6150B two-repeat replay rejected the original entries at S3/local0 and
# S6/local1 because they reproducibly selected an optical branch about 0.35 nm
# away.  The other entries below replace low-information/low-power neighbours
# at the same time.  Every replacement was independently triggered twice on
# the live meter, was single-main-mode (SMSR >= 25 dB), and stayed below 135 mA.
LIVE_VERIFIED_REPLACEMENTS_NM = {
    0: {5: 1528.02195},
    2: {
        0: 1536.27976,
        1: 1536.39913,
        2: 1536.40656,
        3: 1536.42156,
        4: 1536.43368,
        5: 1536.56020,
        6: 1536.87769,
    },
    5: {1: 1547.63041, 6: 1548.48694},
}


def side_candidates(
    x: np.ndarray,
    y: np.ndarray,
    power: np.ndarray,
    indices: list[int],
    levels: np.ndarray,
    positions: np.ndarray,
    amplitude: float,
    increasing: bool,
    keep: int = 6,
) -> list[tuple[float, tuple[int, int, int]]]:
    ranked: list[tuple[float, tuple[int, int, int]]] = []
    x_scale = max(float(np.ptp(x)), 1e-6)
    for combo in itertools.combinations(indices, 3):
        values = y[list(combo)]
        monotonic = bool(
            values[0] <= values[1] <= values[2]
            if increasing
            else values[0] >= values[1] >= values[2]
        )
        if not monotonic:
            continue
        level_error = float(np.sum(((values - levels) / max(amplitude, 0.03)) ** 2))
        position_error = float(np.sum(((x[list(combo)] - positions) / x_scale) ** 2))
        power_error = float(np.sum(((power[list(combo)] - 1.50) / 0.10) ** 2))
        score = level_error + 0.18 * position_error + 0.015 * power_error
        ranked.append((score, combo))
    ranked.sort(key=lambda item: item[0])
    return ranked[:keep]


def quick_gaussian_fit(x: np.ndarray, y: np.ndarray, prior: float) -> tuple[float, float, float]:
    span = float(x[-1] - x[0])
    centers = np.linspace(max(x[0], prior - 0.16), min(x[-1], prior + 0.16), 61)
    sigmas = np.linspace(max(span / 12.0, 0.025), max(span * 0.65, 0.08), 31)
    centered_y = y - np.mean(y)
    total = max(float(np.dot(centered_y, centered_y)), 1e-12)
    center_grid, sigma_grid = np.meshgrid(centers, sigmas, indexing="ij")
    flat_centers = center_grid.ravel()
    flat_sigmas = sigma_grid.ravel()
    shape = np.exp(
        -0.5 * ((x[None, :] - flat_centers[:, None]) / flat_sigmas[:, None]) ** 2
    )
    shape_centered = shape - np.mean(shape, axis=1, keepdims=True)
    variance = np.sum(shape_centered * shape_centered, axis=1)
    amplitude = (shape_centered @ centered_y) / np.maximum(variance, 1e-12)
    baseline = np.mean(y) - amplitude * np.mean(shape, axis=1)
    residual = y[None, :] - (baseline[:, None] + amplitude[:, None] * shape)
    normalized_sse = np.sum(residual * residual, axis=1) / total
    prior_penalty = 0.035 * ((flat_centers - prior) / 0.020) ** 2
    score = normalized_sse + prior_penalty
    score[amplitude <= 0.0] = math.inf
    best_index = int(np.argmin(score))
    return (
        float(score[best_index]),
        float(flat_centers[best_index]),
        float(flat_sigmas[best_index]),
    )


def choose_segment(rows: list[dict], prior: float) -> tuple[list[dict], dict]:
    x = np.asarray([float(row["candidate_wavelength_nm"]) for row in rows])
    y = np.asarray([float(row["voltage_median_v"][1]) for row in rows])
    power = np.asarray([float(row["candidate_power_mw"]) for row in rows])
    if len(rows) < 7:
        raise RuntimeError("a segment has fewer than seven candidates")

    best = None
    low_level = float(np.quantile(y, 0.08))
    for peak in range(3, len(rows) - 3):
        if abs(x[peak] - prior) > 0.14 or y[peak] < float(np.quantile(y, 0.55)):
            continue
        amplitude = float(y[peak] - low_level)
        if amplitude < 0.07:
            continue
        left_indices = list(range(peak))
        right_indices = list(range(peak + 1, len(rows)))
        left_levels = low_level + amplitude * np.asarray((0.10, 0.48, 0.78))
        right_levels = low_level + amplitude * np.asarray((0.78, 0.48, 0.10))
        left_positions = x[0] + (x[peak] - x[0]) * np.asarray((0.04, 0.55, 0.82))
        right_positions = x[peak] + (x[-1] - x[peak]) * np.asarray((0.18, 0.50, 0.96))
        left = side_candidates(
            x, y, power, left_indices, left_levels, left_positions,
            amplitude, increasing=True,
        )
        right = side_candidates(
            x, y, power, right_indices, right_levels, right_positions,
            amplitude, increasing=False,
        )
        for left_score, left_combo in left:
            for right_score, right_combo in right:
                indices = left_combo + (peak,) + right_combo
                subset_x = x[list(indices)]
                subset_y = y[list(indices)]
                # Two laser states less than 8 pm apart add almost no peak
                # observability but still consume a complete scan slot.
                if np.min(np.diff(subset_x)) < 0.008:
                    continue
                gaussian_score, center, sigma = quick_gaussian_fit(subset_x, subset_y, prior)
                coverage = float((subset_x[-1] - subset_x[0]) / max(x[-1] - x[0], 1e-9))
                coverage_penalty = max(0.72 - coverage, 0.0) ** 2 * 3.0
                score = gaussian_score + 0.04 * (left_score + right_score) + coverage_penalty
                if best is None or score < best[0]:
                    best = (score, indices, center, sigma, coverage)
    if best is None:
        raise RuntimeError(f"no seven-point monotonic design around {prior:.4f} nm")

    _, indices, quick_center, quick_sigma, coverage = best
    selected = [dict(rows[index]) for index in indices]
    selected_x = x[list(indices)]
    selected_y = y[list(indices)]
    official = fit_fbg_segment(
        selected_x, selected_y, min_prominence_v=0.02, allow_edge_peak=True
    )
    diagnostics = {
        "reference_center_nm": prior,
        "quick_center_nm": quick_center,
        "quick_sigma_nm": quick_sigma,
        "official_center_nm": float(official.center_nm),
        "official_center_error_pm": float((official.center_nm - prior) * 1000.0),
        "official_r_squared": float(official.r_squared),
        "official_center_std_pm": float(official.center_std_pm),
        "official_valid": bool(official.valid),
        "wavelength_coverage_percent": coverage * 100.0,
        "selected_monotonic": bool(
            np.all(np.diff(selected_y[:4]) >= 0.0)
            and np.all(np.diff(selected_y[3:]) <= 0.0)
        ),
    }
    return selected, diagnostics


def render_plot(path: Path, source_rows: list[dict], selected_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for segment, axis in enumerate(axes.flat):
        dense = [row for row in source_rows if int(row["segment"]) == segment]
        chosen = [row for row in selected_rows if int(row["segment"]) == segment]
        axis.scatter(
            [row["candidate_wavelength_nm"] for row in dense],
            [row["voltage_median_v"][1] for row in dense],
            s=17, alpha=0.42, label="252-point settled pool",
        )
        axis.plot(
            [row["candidate_wavelength_nm"] for row in chosen],
            [row["voltage_median_v"][1] for row in chosen],
            "o-", linewidth=1.8, label="selected 7",
        )
        axis.axvline(REFERENCE_CENTERS_NM[segment], color="gray", linestyle=":")
        axis.set_title(f"FBG {segment + 1}")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Settled CH1 (V)")
        axis.grid(alpha=0.25)
        if segment == 0:
            axis.legend(fontsize=8)
    figure.suptitle("Shape-constrained 63-point stress scan design")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DENSE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    dense = json.loads(args.source.read_text(encoding="utf-8"))
    selected_rows: list[dict] = []
    segment_diagnostics: list[dict] = []
    for segment, prior in enumerate(REFERENCE_CENTERS_NM):
        rows = [row for row in dense["rows"] if int(row["segment"]) == segment]
        chosen, diagnostics = choose_segment(rows, prior)
        replacements = LIVE_VERIFIED_REPLACEMENTS_NM.get(segment, {})
        for local_index, target_nm in replacements.items():
            replacement = min(
                rows,
                key=lambda row: abs(float(row["candidate_wavelength_nm"]) - target_nm),
            )
            if abs(float(replacement["candidate_wavelength_nm"]) - target_nm) > 0.0001:
                raise RuntimeError(f"missing live replacement {target_nm:.6f} nm")
            chosen[local_index] = dict(replacement)
        if replacements:
            chosen.sort(key=lambda row: float(row["candidate_wavelength_nm"]))
            selected_x = np.asarray([row["candidate_wavelength_nm"] for row in chosen])
            selected_y = np.asarray([row["voltage_median_v"][1] for row in chosen])
            official = fit_fbg_segment(
                selected_x,
                selected_y,
                min_prominence_v=0.02,
                allow_edge_peak=True,
            )
            diagnostics.update({
                "official_center_nm": float(official.center_nm),
                "official_center_error_pm": float((official.center_nm - prior) * 1000.0),
                "official_r_squared": float(official.r_squared),
                "official_center_std_pm": float(official.center_std_pm),
                "official_valid": bool(official.valid),
                "selected_monotonic": bool(
                    np.all(np.diff(selected_y[:4]) >= 0.0)
                    and np.all(np.diff(selected_y[3:]) <= 0.0)
                ),
                "live_verified_replacements": {
                    str(index): wavelength for index, wavelength in replacements.items()
                },
            })
        for row in chosen:
            row["selected_index"] = len(selected_rows)
            selected_rows.append(row)
        diagnostics["segment"] = segment
        diagnostics["selected_wavelengths_nm"] = [
            float(row["candidate_wavelength_nm"]) for row in chosen
        ]
        diagnostics["selected_ch1_v"] = [
            float(row["voltage_median_v"][1]) for row in chosen
        ]
        segment_diagnostics.append(diagnostics)

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "shape-constrained experimental design for fast live strain tracking",
        "source": args.source.name,
        "old_total_table_used": False,
        "live_static_spectrum_predictor_used": False,
        "point_count": len(selected_rows),
        "reference_centers_nm": list(REFERENCE_CENTERS_NM),
        "segments": segment_diagnostics,
        "rows": selected_rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_plot(args.plot, dense["rows"], selected_rows)
    print(json.dumps({
        "point_count": len(selected_rows),
        "segments": segment_diagnostics,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
