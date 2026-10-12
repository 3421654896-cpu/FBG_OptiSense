"""Repeat-measure calibrated 72-point rows without changing their codes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import (
    AQ6150B,
    DAC_SOURCE,
    LaserSerial,
    code_to_current,
    current_to_code,
    parse_dac_rows,
    save_json,
    take_reading,
)


HERE = Path(__file__).resolve().parent
CALIBRATION = HERE / "high_power_stress_72_wavelength_calibration_v3.json"
OUTPUT = HERE / "high_power_stress_72_repeat_validation_v3.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,24,53,25,27,31")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    indices = [int(item) for item in args.indices.split(",") if item.strip()]
    source = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "indices": indices,
        "repeats": args.repeats,
        "results": {},
    }
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    rows = parse_dac_rows(DAC_SOURCE)
    park = list(rows[min(44, len(rows) - 1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(120.0, code_to_current(park[1], 1)), 1)
    try:
        for index in indices:
            record = source["results"][str(index)]
            final = record["final"]
            codes = [int(value) for value in final["codes"]]
            target = float(record["target_nm"])
            readings = []
            for repeat in range(args.repeats):
                reading = take_reading(
                    laser,
                    meter,
                    codes,
                    target,
                    settle_s=0.38 if repeat == 0 else 0.20,
                    select_main_peak=True,
                )
                readings.append(asdict(reading))
            powers = [item["power_mw"] for item in readings]
            errors = [item["wavelength_error_pm"] for item in readings]
            payload["results"][str(index)] = {
                "index": index,
                "target_nm": target,
                "codes": codes,
                "readings": readings,
                "power_min_mw": min(powers),
                "power_median_mw": statistics.median(powers),
                "power_max_mw": max(powers),
                "max_abs_wavelength_error_pm": max(abs(value) for value in errors),
            }
            save_json(args.output, payload)
            print(
                f"index={index} target={target:.6f} "
                f"power={min(powers):.4f}/{statistics.median(powers):.4f}/{max(powers):.4f} mW "
                f"max_error={max(abs(value) for value in errors):.2f} pm",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()
    all_minima = [value["power_min_mw"] for value in payload["results"].values()]
    payload["summary"] = {
        "bottleneck_repeat_min_mw": min(all_minima),
        "recommended_common_target_mw": min(all_minima) * 0.97,
    }
    save_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
