"""Select a fast 63-point table from measured dynamic candidate scans."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from fbg_peak_fitting import fit_fbg_segment
from select_stress_points_from_dense import REFERENCE_CENTERS_NM, choose_segment


HERE = Path(__file__).resolve().parent
DEFAULT_POOL = HERE / "stress_dynamic_training_pool.json"
DEFAULT_CAPTURE = HERE / "stress_dynamic_pool_capture.json"
DEFAULT_OUTPUT = HERE / "stress_shape_power_dynamic_v3.json"
DEFAULT_YAML = HERE / "stress_wave_const_v3.yaml"
DEFAULT_PLOT = HERE / "stress_shape_power_dynamic_v3.png"
ADC_SCALE = 2.5 / 4096.0


def fit_summary(x: np.ndarray, y: np.ndarray, segment: int) -> dict:
    fit = fit_fbg_segment(x, y, min_prominence_v=0.02, allow_edge_peak=True)
    return {
        "center_nm": float(fit.center_nm),
        "center_error_from_broadband_reference_pm": float(
            (fit.center_nm - REFERENCE_CENTERS_NM[segment]) * 1000.0
        ),
        "amplitude_v": float(fit.amplitude_v),
        "r_squared": float(fit.r_squared),
        "center_std_pm": float(fit.center_std_pm),
        "valid": bool(fit.valid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    frames = np.asarray(
        [frame["current_estimate"] for frame in capture["captures"]], dtype=float
    )
    if frames.shape[1] != len(pool["rows"]):
        raise RuntimeError("candidate/capture point count mismatch")
    dynamic_median = np.median(frames, axis=0) * ADC_SCALE
    dynamic_std = np.std(frames, axis=0) * ADC_SCALE

    candidates: list[dict] = []
    for index, source in enumerate(pool["rows"]):
        row = dict(source)
        row["candidate_wavelength_nm"] = float(source["wavelength_nm"])
        row["candidate_power_mw"] = float(source["power_mw"])
        row["voltage_median_v"] = [0.0, float(dynamic_median[index]), 0.0, 0.0]
        row["training_dynamic_ch1_v"] = float(dynamic_median[index])
        row["training_dynamic_std_v"] = float(dynamic_std[index])
        row["training_abs_error_v"] = abs(
            float(dynamic_median[index]) - float(source["ch1_v"])
        )
        row["training_pool_index"] = index
        candidates.append(row)

    selected_rows: list[dict] = []
    segment_reports: list[dict] = []
    for segment in range(9):
        section = [row for row in candidates if int(row["segment"]) == segment]
        section.sort(key=lambda row: row["candidate_wavelength_nm"])
        if len(section) == 7:
            chosen = [dict(row) for row in section]
            selection = {"method": "retained_all_seven_no_alternative_candidates"}
        else:
            chosen, selection = choose_segment(section, REFERENCE_CENTERS_NM[segment])
        chosen.sort(key=lambda row: row["candidate_wavelength_nm"])
        x = np.asarray([row["candidate_wavelength_nm"] for row in chosen])
        settled = np.asarray([row["ch1_v"] for row in chosen])
        dynamic = np.asarray([row["training_dynamic_ch1_v"] for row in chosen])
        settled_fit = fit_summary(x, settled, segment)
        dynamic_fit = fit_summary(x, dynamic, segment)
        segment_reports.append({
            "segment": segment,
            "candidate_count": len(section),
            "selection": selection,
            "settled_fit": settled_fit,
            "training_dynamic_fit": dynamic_fit,
            "dynamic_vs_settled_mae_v": float(np.mean(np.abs(dynamic - settled))),
            "selected_wavelengths_nm": x.tolist(),
        })
        for row in chosen:
            selected_rows.append({
                "index": len(selected_rows),
                "segment": segment,
                "wavelength_nm": float(row["candidate_wavelength_nm"]),
                "power_mw": float(row["candidate_power_mw"]),
                "ch1_v": float(row["ch1_v"]),
                "training_dynamic_ch1_v": float(row["training_dynamic_ch1_v"]),
                "training_dynamic_std_v": float(row["training_dynamic_std_v"]),
                "training_abs_error_v": float(row["training_abs_error_v"]),
                "codes": [int(value) for value in row["codes"]],
                "currents_ma": [float(value) for value in row["currents_ma"]],
                "smsr_min_db": float(row.get("smsr_min_db", 99.0)),
                "wavelength_span_pm": float(row.get("wavelength_span_pm", 0.0)),
                "source_pool": str(row.get("source_pool", args.pool.name)),
                "training_pool_index": int(row["training_pool_index"]),
            })

    if len(selected_rows) != 63:
        raise RuntimeError(f"expected 63 selected rows, got {len(selected_rows)}")
    powers = np.asarray([row["power_mw"] for row in selected_rows])
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "dynamic-measured shape-selected 63-point stress table",
        "point_count": len(selected_rows),
        "limits_ma": {"gain": 135.0, "soa": 135.0, "phase": 10.0,
                      "wave_a": 30.0, "wave_b": 30.0},
        "power_summary_mw": {
            "minimum": float(np.min(powers)),
            "mean": float(np.mean(powers)),
            "maximum": float(np.max(powers)),
            "coefficient_of_variation_percent": float(
                np.std(powers) / np.mean(powers) * 100.0
            ),
        },
        "segments": segment_reports,
        "rows": selected_rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    yaml_lines = ["Wave_DATA:"]
    for row in selected_rows:
        integer = int(math.floor(row["wavelength_nm"]))
        fractional = (row["wavelength_nm"] - integer) * 1000.0
        yaml_lines.append(f"  - [{integer}, {fractional:.3f}]")
    args.yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
        for segment, axis in enumerate(axes.flat):
            section = [row for row in candidates if int(row["segment"]) == segment]
            chosen = [row for row in selected_rows if int(row["segment"]) == segment]
            axis.scatter(
                [row["candidate_wavelength_nm"] for row in section],
                [row["training_dynamic_ch1_v"] for row in section],
                s=12, alpha=0.25, label="dynamic candidate pool",
            )
            axis.plot(
                [row["wavelength_nm"] for row in chosen],
                [row["ch1_v"] for row in chosen],
                "o--", label="settled single",
            )
            axis.plot(
                [row["wavelength_nm"] for row in chosen],
                [row["training_dynamic_ch1_v"] for row in chosen],
                "o-", label="fast scan",
            )
            axis.set_title(f"FBG {segment + 1}")
            axis.grid(alpha=0.25)
            if segment == 0:
                axis.legend(fontsize=8)
        figure.suptitle("Dynamic-trained 63-point stress design")
        figure.savefig(args.plot, dpi=170)
        plt.close(figure)
    except ModuleNotFoundError:
        pass

    print(json.dumps({
        "power_summary_mw": payload["power_summary_mw"],
        "segments": segment_reports,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
