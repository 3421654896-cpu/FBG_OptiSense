"""Measure the common optical-power ceiling on the currently connected path.

This program deliberately does not use the historical total-table workbook.
It starts from the active, wavelength-meter-calibrated DAC table, clamps GAIN
and SOA to 120 mA, measures every wavelength with the connected AQ6150B, and
checkpoints every result.  The first pass holds GAIN at its active value and
sets SOA to 120 mA; a later optimizer can concentrate extra GAIN/PHASE work on
the measured bottleneck wavelengths.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import (
    AQ6150B,
    DAC_SOURCE,
    LASER_MAX_MA,
    LaserSerial,
    WAVE_YAML,
    code_to_current,
    current_to_code,
    load_wavelengths,
    parse_dac_rows,
    save_json,
    take_reading,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "common_power_ceiling_probe.json"


def safe_probe_codes(row: list[int]) -> list[int]:
    codes = list(row)
    active_gain = min(code_to_current(codes[0], 0), LASER_MAX_MA[0])
    codes[0] = current_to_code(active_gain, 0)
    codes[1] = current_to_code(LASER_MAX_MA[1], 1)
    return codes


def summarize(payload: dict, reserve: float) -> dict:
    valid = []
    for key, result in payload.get("results", {}).items():
        readings = result.get("readings", [])
        powers = [float(item["power_mw"]) for item in readings]
        errors = [abs(float(item["wavelength_error_pm"])) for item in readings]
        if not powers or not all(math.isfinite(value) and value > 0.0 for value in powers):
            continue
        valid.append(
            {
                "index": int(key),
                "target_nm": float(result["target_nm"]),
                "power_median_mw": statistics.median(powers),
                "power_min_mw": min(powers),
                "max_abs_wavelength_error_pm": max(errors),
            }
        )
    valid.sort(key=lambda item: item["power_median_mw"])
    if not valid:
        return {"valid_count": 0, "bottlenecks": []}
    weakest = valid[0]
    return {
        "valid_count": len(valid),
        "weakest_index": weakest["index"],
        "weakest_target_nm": weakest["target_nm"],
        "raw_common_ceiling_mw": weakest["power_median_mw"],
        "reserved_common_target_mw": weakest["power_median_mw"] * reserve,
        "reserve_fraction": reserve,
        "bottlenecks": valid[: min(12, len(valid))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--reserve", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repeats < 1 or not 0.80 <= args.reserve <= 1.0:
        raise SystemExit("Invalid repeats/reserve")

    rows = parse_dac_rows(DAC_SOURCE)
    wavelengths = load_wavelengths(WAVE_YAML)
    count = min(len(rows), len(wavelengths))
    start = max(0, min(args.start, count))
    stop = max(start, min(args.stop if args.stop is not None else count, count))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "active_table_gain_held_soa_120mA",
        "limits_ma": {"gain": 120.0, "soa": 120.0, "phase": 10.0,
                      "wave_a": 30.0, "wave_b": 30.0},
        "meter": None,
        "results": {},
    }
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        payload["results"].update(old.get("results", {}))

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    park = safe_probe_codes(rows[min(44, count - 1)])
    try:
        for index in range(start, stop):
            key = str(index)
            if args.resume and len(payload["results"].get(key, {}).get("readings", [])) >= args.repeats:
                continue
            codes = safe_probe_codes(rows[index])
            readings = []
            for repeat in range(args.repeats):
                reading = take_reading(
                    laser, meter, codes, wavelengths[index],
                    settle_s=0.36 if repeat == 0 else 0.18,
                )
                readings.append(asdict(reading))
            payload["results"][key] = {
                "index": index,
                "target_nm": wavelengths[index],
                "codes": codes,
                "currents_ma": [code_to_current(code, channel)
                                for channel, code in enumerate(codes)],
                "readings": readings,
            }
            payload["summary"] = summarize(payload, args.reserve)
            save_json(args.output, payload)
            med = statistics.median(item["power_mw"] for item in readings)
            err = statistics.median(item["wavelength_error_pm"] for item in readings)
            print(
                f"[{index + 1:03d}/{stop:03d}] {wavelengths[index]:.3f} nm "
                f"power={med:.6f} mW error={err:+.2f} pm",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    payload["summary"] = summarize(payload, args.reserve)
    save_json(args.output, payload)
    summary = payload["summary"]
    print(
        f"valid={summary.get('valid_count', 0)}/{stop-start} "
        f"raw_common_ceiling={summary.get('raw_common_ceiling_mw', float('nan')):.6f} mW "
        f"reserved_target={summary.get('reserved_common_target_mw', float('nan')):.6f} mW",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
