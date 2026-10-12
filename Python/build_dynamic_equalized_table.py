"""Build a current-session equal-power table from measured local SOA slopes."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from laser_power_calibration import code_to_current, current_to_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("fullband_equal_power_operational_2001.json"))
    parser.add_argument("--training", type=Path, default=Path("fullband_dynamic_power_training_0p2nm.json"))
    parser.add_argument("--output", type=Path, default=Path("fullband_dynamic_equalized_2001.json"))
    parser.add_argument("--target-power-mw", type=float, default=1.53)
    parser.add_argument("--max-soa-trim-ma", type=float, default=3.0)
    args = parser.parse_args()

    table = json.loads(args.table.read_text(encoding="utf-8"))
    training = json.loads(args.training.read_text(encoding="utf-8"))
    rows = sorted(table["rows"].values(), key=lambda row: float(row["target_nm"]))
    groups: dict[float, list[dict]] = {}
    for sample in training["rows"]:
        groups.setdefault(round(float(sample["target_nm"]), 6), []).append(sample)

    slope_rows: list[dict] = []
    rejected: list[dict] = []
    for target_key in sorted(groups):
        samples = groups[target_key]
        base = next((sample for sample in samples if sample["variant"] == "base"), None)
        valid = [
            sample for sample in samples
            if (
                sample["settled"].get("smsr_db") is None
                or float(sample["settled"]["smsr_db"]) >= 20.0
            )
            and abs(float(sample["settled"]["wavelength_error_pm"])) <= 10.0
        ]
        if base is None or len(valid) < 2:
            rejected.append({"target_nm": target_key, "reason": "fewer than two valid variants"})
            continue
        soa = np.asarray([float(sample["currents_ma"][1]) for sample in valid])
        power = np.asarray([float(sample["settled"]["power_mw"]) for sample in valid])
        wave_pm = np.asarray([float(sample["settled"]["wavelength_error_pm"]) for sample in valid])
        power_slope, power_intercept = np.polyfit(soa, power, 1)
        wave_slope, wave_intercept = np.polyfit(soa, wave_pm, 1)
        if power_slope <= 0.001:
            rejected.append({
                "target_nm": target_key,
                "reason": "non-positive/weak SOA power slope",
                "power_slope_mw_per_ma": float(power_slope),
            })
            continue
        base_soa = float(base["currents_ma"][1])
        base_power = float(base["settled"]["power_mw"])
        raw_trim = (float(args.target_power_mw) - base_power) / float(power_slope)
        trim = max(-float(args.max_soa_trim_ma), min(float(args.max_soa_trim_ma), raw_trim))
        trim = max(-base_soa, min(135.0 - base_soa, trim))
        slope_rows.append({
            "target_nm": target_key,
            "valid_variant_count": len(valid),
            "base_soa_ma": base_soa,
            "base_power_mw": base_power,
            "power_slope_mw_per_ma": float(power_slope),
            "wave_slope_pm_per_ma": float(wave_slope),
            "raw_soa_trim_ma": float(raw_trim),
            "applied_soa_trim_ma": float(trim),
            "predicted_power_mw": float(power_intercept + power_slope * (base_soa + trim)),
            "predicted_wavelength_error_pm": float(wave_intercept + wave_slope * (base_soa + trim)),
        })
    if len(slope_rows) < 150:
        raise RuntimeError("too few valid local SOA slopes")

    slope_x = np.asarray([row["target_nm"] for row in slope_rows], dtype=float)
    slope_trim = np.asarray([row["applied_soa_trim_ma"] for row in slope_rows], dtype=float)
    target_x = np.asarray([float(row["target_nm"]) for row in rows], dtype=float)
    interpolated_trim = np.interp(target_x, slope_x, slope_trim)

    result = deepcopy(table)
    output_rows = result["rows"]
    for source_row, trim in zip(rows, interpolated_trim):
        key = f"{float(source_row['target_nm']):.6f}"
        row = output_rows[key]
        old_soa = code_to_current(int(row["codes"][1]), 1)
        new_soa = max(0.0, min(135.0, old_soa + float(trim)))
        new_code = current_to_code(new_soa, 1)
        row["codes"][1] = int(new_code)
        row["currents_ma"][1] = code_to_current(int(new_code), 1)
        row["dynamic_power_trim"] = {
            "source_base_soa_ma": old_soa,
            "interpolated_trim_ma": float(trim),
            "updated_soa_ma": row["currents_ma"][1],
            "requires_live_validation": True,
        }
    result["dynamic_equalization"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": args.table.name,
        "source_training": args.training.name,
        "target_power_mw": float(args.target_power_mw),
        "method": "local measured SOA power slope every 0.20 nm, linearly interpolated",
        "max_soa_trim_ma": float(args.max_soa_trim_ma),
        "valid_slope_count": len(slope_rows),
        "rejected_slope_count": len(rejected),
        "slope_rows": slope_rows,
        "rejected": rejected,
        "summary": {
            "trim_min_ma": float(np.min(interpolated_trim)),
            "trim_median_ma": float(np.median(interpolated_trim)),
            "trim_max_ma": float(np.max(interpolated_trim)),
            "final_soa_min_ma": min(float(row["currents_ma"][1]) for row in output_rows.values()),
            "final_soa_max_ma": max(float(row["currents_ma"][1]) for row in output_rows.values()),
            "power_slope_min_mw_per_ma": min(row["power_slope_mw_per_ma"] for row in slope_rows),
            "power_slope_median_mw_per_ma": float(np.median([row["power_slope_mw_per_ma"] for row in slope_rows])),
            "power_slope_max_mw_per_ma": max(row["power_slope_mw_per_ma"] for row in slope_rows),
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["dynamic_equalization"]["summary"], indent=2))
    print(f"valid_slopes={len(slope_rows)} rejected={len(rejected)}")


if __name__ == "__main__":
    main()
