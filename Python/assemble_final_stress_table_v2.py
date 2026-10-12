"""Assemble the final 63-point table from post-power-calibration pools."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from fbg_peak_fitting import fit_channel_segments
from select_stress_points_from_dense import (
    REFERENCE_CENTERS_NM,
    choose_segment,
)


HERE = Path(__file__).resolve().parent
FINAL_1 = HERE / "stress_points_shape_selected_final.json"
FINAL_1_SINGLE = HERE / "stress_points_shape_selected_final_single.json"
POOL_MAIN = HERE / "uniform_power_candidate_pool.json"
POOL_EXTRA = HERE / "uniform_power_candidate_pool_extra.json"
OUTPUT = HERE / "stress_shape_power_final_v2.json"
YAML_OUTPUT = HERE / "stress_wave_const_v2.yaml"
PLOT_OUTPUT = HERE / "stress_shape_power_final_v2.png"


def pool_rows(path: Path, segment: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload["rows"]:
        if int(row["segment"]) != segment or not row["accepted"]:
            continue
        converted = dict(row)
        converted["candidate_wavelength_nm"] = float(row["wavelength_nm"])
        converted["candidate_power_mw"] = float(row["power_mw"])
        converted["voltage_median_v"] = [0.0, float(row["ch1_v"]), 0.0, 0.0]
        converted["source_pool"] = path.name
        rows.append(converted)
    rows.sort(key=lambda row: row["candidate_wavelength_nm"])
    return rows


def existing_rows() -> dict[int, list[dict]]:
    calibrated = json.loads(FINAL_1.read_text(encoding="utf-8"))
    measured = json.loads(FINAL_1_SINGLE.read_text(encoding="utf-8"))
    result: dict[int, list[dict]] = {}
    for segment in (0, 1, 5):
        rows = []
        for index in range(segment * 7, (segment + 1) * 7):
            calibration = calibrated["rows"][index]
            single = measured["rows"][index]
            rows.append({
                "candidate_wavelength_nm": float(calibration["final_wavelength_nm"]),
                "candidate_power_mw": float(calibration["final_power_mw"]),
                "voltage_median_v": [0.0, float(single["ch1_v"]), 0.0, 0.0],
                "codes": [int(value) for value in calibration["codes"]],
                "currents_ma": [float(value) for value in calibration["currents_ma"]],
                "smsr_min_db": float(calibration["final_smsr_min_db"]),
                "wavelength_span_pm": float(calibration["final_wavelength_span_pm"]),
                "source_pool": FINAL_1.name,
            })
        result[segment] = rows
    return result


def main() -> int:
    selected_by_segment = existing_rows()
    for segment in (2, 3, 4, 7):
        selected_by_segment[segment], _ = choose_segment(
            pool_rows(POOL_MAIN, segment), REFERENCE_CENTERS_NM[segment]
        )
    for segment in (6, 8):
        selected_by_segment[segment], _ = choose_segment(
            pool_rows(POOL_EXTRA, segment), REFERENCE_CENTERS_NM[segment]
        )

    rows = []
    for segment in range(9):
        selected = sorted(
            selected_by_segment[segment],
            key=lambda row: float(row["candidate_wavelength_nm"]),
        )
        if len(selected) != 7:
            raise RuntimeError(f"segment {segment} has {len(selected)} selected rows")
        for row in selected:
            currents = [float(value) for value in row["currents_ma"]]
            limits = (135.0, 135.0, 10.0, 30.0, 30.0)
            if any(value > limit + 0.002 for value, limit in zip(currents, limits)):
                raise RuntimeError(f"current limit exceeded in segment {segment}")
            rows.append({
                "index": len(rows),
                "segment": segment,
                "wavelength_nm": float(row["candidate_wavelength_nm"]),
                "power_mw": float(row["candidate_power_mw"]),
                "ch1_v": float(row["voltage_median_v"][1]),
                "codes": [int(value) for value in row["codes"]],
                "currents_ma": currents,
                "smsr_min_db": float(row.get("smsr_min_db", 99.0)),
                "wavelength_span_pm": float(row.get("wavelength_span_pm", 0.0)),
                "source_pool": str(row["source_pool"]),
            })

    wavelengths = np.asarray([row["wavelength_nm"] for row in rows])
    ch1 = np.asarray([row["ch1_v"] for row in rows])
    segments = [slice(index * 7, (index + 1) * 7) for index in range(9)]
    fits = fit_channel_segments(
        wavelengths,
        ch1,
        segments,
        min_prominence_v=0.02,
        allow_edge_peak=True,
    )
    fit_rows = []
    for segment, fit in enumerate(fits):
        fit_rows.append({
            "segment": segment,
            "center_nm": float(fit.center_nm),
            "center_error_from_broadband_reference_pm": float(
                (fit.center_nm - REFERENCE_CENTERS_NM[segment]) * 1000.0
            ),
            "amplitude_v": float(fit.amplitude_v),
            "r_squared": float(fit.r_squared),
            "rmse_v": float(fit.rmse_v),
            "center_std_pm": float(fit.center_std_pm),
            "valid": bool(fit.valid),
        })
    if not all(item["valid"] for item in fit_rows):
        raise RuntimeError("one or more assembled segments failed fit validation")
    powers = np.asarray([row["power_mw"] for row in rows])
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "final_post_power_shape_selected_63_point_stress_table",
        "old_total_table_used": False,
        "point_count": len(rows),
        "limits_ma": {
            "gain": 135.0,
            "soa": 135.0,
            "phase": 10.0,
            "wave_a": 30.0,
            "wave_b": 30.0,
        },
        "power_summary_mw": {
            "minimum": float(np.min(powers)),
            "mean": float(np.mean(powers)),
            "maximum": float(np.max(powers)),
            "coefficient_of_variation_percent": float(np.std(powers) / np.mean(powers) * 100.0),
        },
        "fits": fit_rows,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    yaml_lines = ["Wave_DATA:"]
    for row in rows:
        integer = int(math.floor(row["wavelength_nm"]))
        fractional = (row["wavelength_nm"] - integer) * 1000.0
        yaml_lines.append(f"  - [{integer}, {fractional:.3f}]")
    YAML_OUTPUT.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
        for segment, axis in enumerate(axes.flat):
            section = segments[segment]
            axis.plot(wavelengths[section], ch1[section], "o-", linewidth=1.8)
            axis.set_title(
                f"FBG {segment + 1}: R²={fit_rows[segment]['r_squared']:.3f}, "
                f"σc={fit_rows[segment]['center_std_pm']:.1f} pm"
            )
            axis.set_xlabel("Wavelength (nm)")
            axis.set_ylabel("Settled CH1 (V)")
            axis.grid(alpha=0.3)
        figure.suptitle("Final post-power 63-point stress design")
        figure.savefig(PLOT_OUTPUT, dpi=170)
        plt.close(figure)
    except ModuleNotFoundError:
        pass

    print(json.dumps({
        "power_summary_mw": payload["power_summary_mw"],
        "fits": fit_rows,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
