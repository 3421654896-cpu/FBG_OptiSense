"""Convert the dense live power-drift map into 1.530 mW feedback targets.

The model is deliberately local and auditable.  It interpolates the measured
power-drift ratio every 0.10 nm, preserves each point's original AQ6150B power,
then uses the photodiode/TIA linear current relation around the fixed scan path
to scale both PDT and PDR target codes.  The monitor with the largest resulting
rail margin is selected per point.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ADC_BIAS_CODE = 2048.0


def rail_margin(code: float) -> float:
    if code < 0.0 or code > ADC_BIAS_CODE:
        return -1.0
    return min(code, ADC_BIAS_CODE - code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("fullband_equal_power_operational_2001.json"))
    parser.add_argument("--power-map", type=Path, default=Path("fullband_pdt_pdr_power_map_0p1nm.json"))
    parser.add_argument("--output", type=Path, default=Path("fullband_equal_power_feedback_targets_2001.json"))
    parser.add_argument("--target-power-mw", type=float, default=1.53)
    args = parser.parse_args()

    table = json.loads(args.table.read_text(encoding="utf-8"))
    power_map = json.loads(args.power_map.read_text(encoding="utf-8"))
    table_rows = sorted(table["rows"].values(), key=lambda row: float(row["target_nm"]))
    map_rows = sorted(power_map["rows"], key=lambda row: float(row["target_nm"]))
    if len(table_rows) != 2001 or len(map_rows) != 2001:
        raise ValueError("both inputs must contain the complete 2001-point grid")

    table_by_target = {
        round(float(row["target_nm"]), 6): row for row in table_rows
    }
    valid_spots: list[dict] = []
    for row in map_rows:
        meter = row.get("meter_spot_check")
        if not meter:
            continue
        smsr = meter.get("smsr_db")
        single_mode = smsr is None or float(smsr) >= 20.0
        if not single_mode or abs(float(meter["wavelength_error_pm"])) > 5.0:
            continue
        original = table_by_target[round(float(row["target_nm"]), 6)]
        original_power = float(original["reading"]["power_mw"])
        valid_spots.append({
            "target_nm": float(row["target_nm"]),
            "drift_ratio": float(meter["power_mw"]) / original_power,
            "current_power_mw": float(meter["power_mw"]),
            "original_power_mw": original_power,
            "wavelength_error_pm": float(meter["wavelength_error_pm"]),
            "smsr_db": smsr,
        })
    if len(valid_spots) < 300:
        raise RuntimeError("insufficient valid 0.10 nm power references")

    spot_wavelength = np.asarray([row["target_nm"] for row in valid_spots], dtype=float)
    spot_drift = np.asarray([row["drift_ratio"] for row in valid_spots], dtype=float)
    target_wavelength = np.asarray([float(row["target_nm"]) for row in table_rows], dtype=float)
    interpolated_drift = np.interp(target_wavelength, spot_wavelength, spot_drift)

    results: list[dict] = []
    for index, (original, live, drift_ratio) in enumerate(
        zip(table_rows, map_rows, interpolated_drift)
    ):
        target_nm = float(original["target_nm"])
        if abs(target_nm - float(live["target_nm"])) > 1e-9:
            raise RuntimeError(f"row alignment failure at index {index}")
        estimated_current_power = float(original["reading"]["power_mw"]) * float(drift_ratio)
        power_scale = float(args.target_power_mw) / estimated_current_power

        pdt_signal = ADC_BIAS_CODE - float(live["pdt_code"])
        pdr_signal = ADC_BIAS_CODE - float(live["pdr_code"])
        target_pdt = ADC_BIAS_CODE - pdt_signal * power_scale
        target_pdr = ADC_BIAS_CODE - pdr_signal * power_scale
        candidates = [
            (rail_margin(target_pdt), "PDT", target_pdt),
            (rail_margin(target_pdr), "PDR", target_pdr),
        ]
        margin, channel, reference = max(candidates)
        if margin < 64.0:
            raise RuntimeError(
                f"no feedback channel has safe rail margin at {target_nm:.2f} nm"
            )
        results.append({
            "index": index,
            "target_nm": target_nm,
            "estimated_current_power_mw": estimated_current_power,
            "interpolated_drift_ratio": float(drift_ratio),
            "target_power_mw": float(args.target_power_mw),
            "power_scale": power_scale,
            "captured_pdt_code": int(live["pdt_code"]),
            "captured_pdr_code": int(live["pdr_code"]),
            "target_pdt_code": int(round(target_pdt)),
            "target_pdr_code": int(round(target_pdr)),
            "power_feedback_channel": channel,
            "power_feedback_reference_code": int(round(reference)),
            "power_feedback_rail_margin_code": int(round(margin)),
            "closed_loop_eligible": margin >= 128.0,
        })

    margins = [row["power_feedback_rail_margin_code"] for row in results]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": args.table.name,
        "source_power_map": args.power_map.name,
        "target_power_mw": float(args.target_power_mw),
        "method": (
            "linear interpolation of measured 0.10 nm power-drift ratio; "
            "pointwise preservation of original AQ6150B power; linear PDT/PDR "
            "photocurrent scaling; adaptive rail-margin monitor selection"
        ),
        "valid_power_spot_count": len(valid_spots),
        "valid_power_spots": valid_spots,
        "rows": results,
        "summary": {
            "point_count": len(results),
            "pdt_selected_count": sum(row["power_feedback_channel"] == "PDT" for row in results),
            "pdr_selected_count": sum(row["power_feedback_channel"] == "PDR" for row in results),
            "closed_loop_margin_ge_128_count": sum(row["closed_loop_eligible"] for row in results),
            "minimum_selected_rail_margin_code": min(margins),
            "maximum_selected_rail_margin_code": max(margins),
            "estimated_current_power_min_mw": min(row["estimated_current_power_mw"] for row in results),
            "estimated_current_power_max_mw": max(row["estimated_current_power_mw"] for row in results),
            "power_scale_min": min(row["power_scale"] for row in results),
            "power_scale_max": max(row["power_scale"] for row in results),
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
