"""Assemble the accepted high-power stress rows into one final JSON table."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from laser_power_calibration import save_json


HERE = Path(__file__).resolve().parent
DESIGN = HERE / "high_power_stress_72_design_v3.json"
EQUALIZED = HERE / "high_power_stress_72_equalized.json"
OUTPUT = HERE / "high_power_stress_final_69.json"
EXCLUDED = {
    18: "Repeated longitudinal-mode hop near 1536.276 nm",
    42: "Power equalization changes longitudinal-mode branch near 1547.679 nm",
    48: "Power equalization changes longitudinal-mode branch near 1551.273 nm",
    52: "Independent replay exposed intermittent 334 pm longitudinal-mode hop near 1551.823 nm",
}
ADDED_STABLE_ROW = {
    "original_index": None,
    "segment": 6,
    "target_nm": 1551.860000,
    "measured_nm": 1551.855110,
    "wavelength_error_pm": -4.890000,
    "power_mw": 1.235252174839042,
    "power_dbm": 0.917556272,
    "side_mode_suppression_db": 28.099126372,
    "codes": [45875, 42368, 9156, 24576, 2048],
    "currents_ma": [104.99954223632812, 96.97265625, 2.794189453125, 30.0, 2.5],
    "pdt_code": 647,
    "pdr_code": 842,
    "source": "fresh_replacement_five_repeat_validation",
}


def main() -> int:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    equalized = json.loads(EQUALIZED.read_text(encoding="utf-8"))
    rows = []
    excluded = []
    for original_index, point in enumerate(design["points"]):
        record = equalized["results"].get(str(original_index), {})
        if original_index in EXCLUDED or not record.get("success"):
            excluded.append({
                "original_index": original_index,
                "segment": int(point["segment"]),
                "target_nm": float(point["target_nm"]),
                "reason": EXCLUDED.get(original_index, "Equalization acceptance failed"),
            })
            continue
        final = record["final"]
        rows.append({
            "index": len(rows),
            "original_index": original_index,
            "segment": int(point["segment"]),
            "target_nm": float(point["target_nm"]),
            "measured_nm": float(final["wavelength_nm"]),
            "wavelength_error_pm": float(final["wavelength_error_pm"]),
            "power_mw": float(final["power_mw"]),
            "power_dbm": float(final["power_dbm"]),
            "side_mode_suppression_db": final.get("side_mode_suppression_db"),
            "codes": [int(value) for value in record["codes"]],
            "currents_ma": [float(value) for value in record["currents_ma"]],
            "pdt_code": final.get("pdt_code"),
            "pdr_code": final.get("pdr_code"),
        })
    rows.append(dict(ADDED_STABLE_ROW))
    rows.sort(key=lambda row: (int(row["segment"]), float(row["measured_nm"])))
    for index, row in enumerate(rows):
        row["index"] = index
    powers = [row["power_mw"] for row in rows]
    errors = [abs(row["wavelength_error_pm"]) for row in rows]
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "fresh_high_power_fast_stress_scan",
        "historical_total_workbook_used": False,
        "meter": equalized.get("meter"),
        "common_target_power_mw": float(equalized["target_power_mw"]),
        "limits_ma": equalized["limits_ma"],
        "excluded": excluded,
        "rows": rows,
        "summary": {
            "point_count": len(rows),
            "segment_count": len({row["segment"] for row in rows}),
            "power_min_mw": min(powers),
            "power_max_mw": max(powers),
            "power_mean_mw": statistics.fmean(powers),
            "power_stdev_mw": statistics.pstdev(powers),
            "power_peak_to_peak_percent": (max(powers) - min(powers))
            / statistics.fmean(powers) * 100.0,
            "max_abs_wavelength_error_pm": max(errors),
            "predicted_frame_rate_hz_from_100_point_7_0832hz": 7.0832 * 100.0 / len(rows),
        },
    }
    save_json(OUTPUT, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    for segment in range(9):
        print(f"segment={segment} points={sum(row['segment'] == segment for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
