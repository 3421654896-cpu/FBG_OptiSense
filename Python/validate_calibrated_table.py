"""Re-measure every row of a calibrated PI11210 table without changing it."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import (
    AQ6150B,
    LaserSerial,
    load_wavelengths,
    parse_dac_rows,
    save_json,
    take_reading,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--dac-source", type=Path, required=True)
    parser.add_argument("--wave-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--power-tolerance", type=float, default=0.015)
    parser.add_argument("--wavelength-tolerance-pm", type=float, default=8.0)
    args = parser.parse_args()

    rows = parse_dac_rows(args.dac_source)
    wavelengths = load_wavelengths(args.wave_yaml)
    count = min(len(rows), len(wavelengths))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "target_power_mw": 0.5,
        "power_tolerance_mw": args.power_tolerance,
        "wavelength_tolerance_pm": args.wavelength_tolerance_pm,
        "results": {},
    }
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for index in range(count):
            try:
                reading = take_reading(
                    laser, meter, rows[index], wavelengths[index], settle_s=0.20
                )
                success = (
                    abs(reading.power_mw - 0.5) <= args.power_tolerance
                    and abs(reading.wavelength_error_pm)
                    <= args.wavelength_tolerance_pm
                )
                result = {
                    "index": index,
                    "target_nm": wavelengths[index],
                    "codes": rows[index],
                    "success": success,
                    "reading": asdict(reading),
                }
            except Exception as error:
                result = {
                    "index": index,
                    "target_nm": wavelengths[index],
                    "codes": rows[index],
                    "success": False,
                    "failure": str(error),
                }
            payload["results"][str(index)] = result
            save_json(args.output, payload)
            reading = result.get("reading", {})
            print(
                f"[{index + 1:03d}/{count:03d}] success={result['success']} "
                f"wave_error={reading.get('wavelength_error_pm', float('nan')):+.3f} pm "
                f"power={reading.get('power_mw', float('nan')):.6f} mW",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(rows[min(44, count - 1)])
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    successes = sum(item["success"] for item in payload["results"].values())
    print(f"Complete: {successes}/{count} rows passed", flush=True)
    return 0 if successes == count else 2


if __name__ == "__main__":
    raise SystemExit(main())
