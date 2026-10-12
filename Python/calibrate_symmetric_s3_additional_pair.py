"""Calibrate the +/-0.11 nm pair that expands stress FBG 3 to nine points."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibrate_symmetric_stress_segments import (
    CURRENT_TABLE,
    MIN_SMSR_DB,
    REFERENCE_CENTERS_NM,
    calibrate_target,
    load_seed_pool,
)
from laser_power_calibration import AQ6150B, LaserSerial, save_json
from validate_rc_against_single import ADC_CODES, ADC_VREF, collect_single_rt


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "stress_symmetric_fbg3_additional_pair.json"
TARGETS = (
    REFERENCE_CENTERS_NM[2] - 0.11,
    REFERENCE_CENTERS_NM[2] + 0.11,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    installed = json.loads(CURRENT_TABLE.read_text(encoding="utf-8"))
    section = [row for row in installed["rows"] if int(row["segment"]) == 2]
    seeds = load_seed_pool()
    used_codes = {tuple(int(value) for value in row["codes"]) for row in installed["rows"]}
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "additional symmetric +/-0.11 nm FBG-3 pair",
        "installed_table_modified": False,
        "reference_center_nm": REFERENCE_CENTERS_NM[2],
        "targets_nm": TARGETS,
        "minimum_smsr_db": MIN_SMSR_DB,
        "rows": [],
    }
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    park = installed["rows"][0]["codes"]
    try:
        for index, target in enumerate(TARGETS):
            row = calibrate_target(laser, meter, seeds, target, used_codes)
            row["segment"] = 2
            row["index"] = index
            predecessor = min(
                section,
                key=lambda item: abs(float(item["wavelength_nm"]) - target),
            )["codes"]
            laser.set_codes(predecessor)
            collect_single_rt(laser.serial, 0.060)
            laser.set_codes(row["codes"])
            samples = collect_single_rt(laser.serial, 0.32)
            stable = np.asarray(samples[-min(80, len(samples)):], dtype=float)
            row["settled_adc_v"] = (
                np.median(stable, axis=0) * ADC_VREF / ADC_CODES
            ).tolist()
            row["settled_span_v"] = (
                np.ptp(stable, axis=0) * ADC_VREF / ADC_CODES
            ).tolist()
            payload["rows"].append(row)
            save_json(args.output, payload)
            print(
                f"{index + 1}/2 target={target:.6f} "
                f"actual={row['final_wavelength_nm']:.6f} "
                f"power={row['final_power_mw']:.3f}mW "
                f"SMSR>={row['final_smsr_min_db']:.1f}dB "
                f"CH1={row['settled_adc_v'][1]:.4f}V",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()
    payload["passed"] = bool(
        len(payload["rows"]) == 2
        and all(row["passed"] for row in payload["rows"])
        and min(float(row["final_smsr_min_db"]) for row in payload["rows"]) >= 25.0
    )
    save_json(args.output, payload)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
