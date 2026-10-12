"""Seed hybrid calibration with rows verified in the current calibration run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from laser_power_calibration import code_to_current, save_json


HERE = Path(__file__).resolve().parent


def main() -> int:
    selection = json.loads((HERE / "fbg_hybrid_selection.json").read_text(encoding="utf-8"))
    candidate_report = json.loads(
        (HERE / "fbg_candidate_calibration_report.json").read_text(encoding="utf-8")
    )
    wave_rows = yaml.safe_load(
        (HERE / "wave_const_hybrid.yaml").read_text(encoding="utf-8")
    )["Wave_DATA"]
    wavelengths = [float(major) + float(minor) / 1000.0 for major, minor in wave_rows]
    points = [point for segment in selection["segments"] for point in segment["points"]]

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "meter": candidate_report.get("meter"),
        "target_power_mw": 0.5,
        "power_tolerance_mw": 0.006,
        "wavelength_tolerance_pm": 3.0,
        "limits_ma": candidate_report.get("limits_ma", {}),
        "results": {},
    }
    for hybrid_index, (point, target_nm) in enumerate(zip(points, wavelengths)):
        if point["source"] != "new_candidate":
            continue
        original = candidate_report["results"][str(point["source_index"])]
        codes = [int(value) for value in original["calibrated_codes"]]
        final = dict(original["final"])
        final["wavelength_error_pm"] = (
            float(final["wavelength_nm"]) - target_nm
        ) * 1000.0
        payload["results"][str(hybrid_index)] = {
            "index": hybrid_index,
            "target_nm": target_nm,
            "success": True,
            "original_codes": codes,
            "calibrated_codes": codes,
            "gain_ma": code_to_current(codes[0], 0),
            "soa_ma": code_to_current(codes[1], 1),
            "phase_ma": code_to_current(codes[2], 2),
            "final": final,
            "history": [],
            "failure": None,
            "validation": "current_session_candidate_calibration",
        }

    destination = HERE / "fbg_hybrid_calibration_checkpoint.json"
    save_json(destination, payload)
    print(f"seeded={len(payload['results'])} pending={100-len(payload['results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
