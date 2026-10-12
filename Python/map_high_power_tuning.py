"""Build a fresh high-current tuning map using the connected AQ6150B.

Only the active table's PHASE/WAVE-A/WAVE-B triples are used as safe seed
coordinates.  Every optical wavelength and power value in the output is
measured again on the current optical path; no historical workbook is read.
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
    LaserSerial,
    LASER_MAX_MA,
    WAVE_YAML,
    code_to_current,
    current_to_code,
    load_wavelengths,
    parse_dac_rows,
    save_json,
    take_reading,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "high_power_tuning_map.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--gain-ma", type=float, default=135.0)
    parser.add_argument("--soa-ma", type=float, default=135.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 < args.gain_ma <= LASER_MAX_MA[0] or not 0.0 < args.soa_ma <= LASER_MAX_MA[1]:
        raise SystemExit(f"GAIN/SOA must be in (0, {LASER_MAX_MA[0]:g}] mA")

    rows = parse_dac_rows(DAC_SOURCE)
    nominal = load_wavelengths(WAVE_YAML)
    count = min(len(rows), len(nominal))
    start = max(0, min(args.start, count))
    stop = max(start, min(args.stop if args.stop is not None else count, count))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "fresh_current_path_map_from_active_tuning_seeds",
        "gain_ma": args.gain_ma,
        "soa_ma": args.soa_ma,
        "meter": None,
        "results": {},
    }
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if (math.isclose(float(old.get("gain_ma", -1)), args.gain_ma)
                and math.isclose(float(old.get("soa_ma", -1)), args.soa_ma)):
            payload["results"].update(old.get("results", {}))

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    park = list(rows[min(44, count - 1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(LASER_MAX_MA[1], code_to_current(park[1], 1)), 1)
    try:
        for index in range(start, stop):
            key = str(index)
            if args.resume and payload["results"].get(key, {}).get("reading"):
                continue
            codes = list(rows[index])
            codes[0] = current_to_code(args.gain_ma, 0)
            codes[1] = current_to_code(args.soa_ma, 1)
            reading = take_reading(
                laser,
                meter,
                codes,
                nominal[index],
                settle_s=0.38,
                select_main_peak=True,
            )
            payload["results"][key] = {
                "seed_index": index,
                "nominal_seed_nm": nominal[index],
                "codes": codes,
                "currents_ma": [code_to_current(code, channel)
                                for channel, code in enumerate(codes)],
                "reading": asdict(reading),
            }
            save_json(args.output, payload)
            print(
                f"[{index + 1:03d}/{stop:03d}] seed={nominal[index]:.3f} "
                f"measured={reading.wavelength_nm:.6f} nm "
                f"power={reading.power_mw:.6f} mW "
                f"delta={reading.wavelength_error_pm:+.1f} pm",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    records = [value for value in payload["results"].values() if value.get("reading")]
    powers = [float(value["reading"]["power_mw"]) for value in records]
    payload["summary"] = {
        "measured_count": len(records),
        "power_min_mw": min(powers) if powers else None,
        "power_median_mw": statistics.median(powers) if powers else None,
        "power_max_mw": max(powers) if powers else None,
    }
    save_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
