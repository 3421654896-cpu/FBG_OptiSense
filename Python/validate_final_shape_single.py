"""Re-measure the final power-calibrated 63 points as settled CH1 truth."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from fbg_peak_fitting import fit_channel_segments
from validate_rc_against_single import (
    ADC_CODES,
    ADC_VREF,
    collect_single_rt,
    dac_command,
    mode_command,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stress_points_shape_selected_final.json"
DEFAULT_OUTPUT = HERE / "stress_points_shape_selected_final_single.json"
DEFAULT_PLOT = HERE / "stress_points_shape_selected_final_single.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    medians = np.zeros((len(rows), 4), dtype=float)
    spans = np.zeros((len(rows), 4), dtype=float)
    port = serial.Serial(args.port, args.baud, timeout=0.02, write_timeout=1.0)
    try:
        port.dtr = True
        port.write(mode_command(2))
        port.flush()
        time.sleep(0.12)
        port.reset_input_buffer()
        for index, row in enumerate(rows):
            predecessor = rows[(index - 1) % len(rows)]
            port.write(dac_command(predecessor["codes"]))
            port.flush()
            collect_single_rt(port, 0.08)
            port.write(dac_command(row["codes"]))
            port.flush()
            readings = collect_single_rt(port, 0.26)
            if len(readings) < 8:
                raise RuntimeError(f"row {index}: only {len(readings)} ADC frames")
            stable = np.asarray(readings[-min(80, len(readings)):], dtype=float)
            medians[index] = np.median(stable, axis=0)
            spans[index] = np.ptp(stable, axis=0)
            print(
                f"final single {index + 1:02d}/{len(rows)} "
                f"{row['final_wavelength_nm']:.6f} nm "
                f"CH1={medians[index, 1] * ADC_VREF / ADC_CODES:.4f} V",
                flush=True,
            )
    finally:
        port.close()

    wavelengths = np.asarray([float(row["final_wavelength_nm"]) for row in rows])
    ch1_v = medians[:, 1] * ADC_VREF / ADC_CODES
    segments = [slice(index * 7, (index + 1) * 7) for index in range(9)]
    fits = fit_channel_segments(
        wavelengths,
        ch1_v,
        segments,
        min_prominence_v=0.02,
        allow_edge_peak=True,
    )
    fit_rows = []
    for segment, fit in enumerate(fits):
        fit_rows.append({
            "segment": segment,
            "center_nm": float(fit.center_nm),
            "amplitude_v": float(fit.amplitude_v),
            "r_squared": float(fit.r_squared),
            "rmse_v": float(fit.rmse_v),
            "center_std_pm": float(fit.center_std_pm),
            "valid": bool(fit.valid),
        })
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append({
            "index": index,
            "segment": index // 7,
            "wavelength_nm": float(row["final_wavelength_nm"]),
            "power_mw": float(row["final_power_mw"]),
            "codes": [int(value) for value in row["codes"]],
            "currents_ma": [float(value) for value in row["currents_ma"]],
            "ch1_v": float(ch1_v[index]),
            "ch1_span_v": float(spans[index, 1] * ADC_VREF / ADC_CODES),
        })
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "method": "predecessor -> target -> 260 ms settled single-value median",
        "point_count": len(rows),
        "fits": fit_rows,
        "all_fits_valid": all(bool(item["valid"]) for item in fit_rows),
        "rows": output_rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
        for segment, axis in enumerate(axes.flat):
            section = segments[segment]
            axis.plot(wavelengths[section], ch1_v[section], "o-", linewidth=1.8)
            axis.set_title(
                f"FBG {segment + 1}: {fit_rows[segment]['center_nm']:.6f} nm, "
                f"R²={fit_rows[segment]['r_squared']:.3f}"
            )
            axis.set_xlabel("Wavelength (nm)")
            axis.set_ylabel("Settled CH1 (V)")
            axis.grid(alpha=0.3)
        figure.suptitle("Final 63-point power-calibrated settled spectrum")
        figure.savefig(args.plot, dpi=170)
        plt.close(figure)
    except ModuleNotFoundError:
        pass

    print(json.dumps({
        "all_fits_valid": payload["all_fits_valid"],
        "fits": fit_rows,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["all_fits_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
