"""Repeat-measure every accepted final stress-table row on live hardware."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import AQ6150B, DAC_SOURCE, LaserSerial, code_to_current, current_to_code, parse_dac_rows, save_json, take_reading

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "high_power_stress_final_69.json"
OUTPUT = HERE / "high_power_stress_final_69_validation_final.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="COM6")
    p.add_argument("--gpib", default="GPIB0::7::INSTR")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    stop = min(args.stop if args.stop is not None else len(rows), len(rows))
    payload = {"created": datetime.now().isoformat(timespec="seconds"),
               "source": SOURCE.name, "repeats": args.repeats, "results": {}}
    if args.resume and OUTPUT.exists():
        payload["results"].update(json.loads(OUTPUT.read_text(encoding="utf-8")).get("results", {}))
    old_rows = parse_dac_rows(DAC_SOURCE)
    park = list(old_rows[min(44, len(old_rows)-1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(120.0, code_to_current(park[1], 1)), 1)
    laser, meter = LaserSerial(args.port), AQ6150B(args.gpib)
    try:
        for index in range(max(0, args.start), stop):
            key = str(index)
            if args.resume and key in payload["results"]:
                continue
            row = rows[index]
            samples = []
            for repeat in range(args.repeats):
                reading = take_reading(laser, meter, row["codes"], row["target_nm"], 0.30, True)
                samples.append(asdict(reading))
            powers = [float(x["power_mw"]) for x in samples]
            errors = [abs(float(x["wavelength_error_pm"])) for x in samples]
            passed = max(errors) <= 7.0 and min(powers) >= 1.18 and max(powers) <= 1.28
            payload["results"][key] = {"index": index, "original_index": row["original_index"],
                "target_nm": row["target_nm"], "passed": passed, "samples": samples,
                "power_min_mw": min(powers), "power_max_mw": max(powers),
                "max_abs_error_pm": max(errors)}
            save_json(OUTPUT, payload)
            print(f"[{index+1:02d}/{stop:02d}] pass={passed} target={row['target_nm']:.6f} "
                  f"power={min(powers):.4f}..{max(powers):.4f} maxerr={max(errors):.2f} pm", flush=True)
    finally:
        try:
            laser.set_codes(park); time.sleep(0.08)
        finally:
            laser.close(); meter.close()
    values = list(payload["results"].values())
    payload["summary"] = {"tested": len(values), "passed": sum(bool(x["passed"]) for x in values),
        "failed": sum(not bool(x["passed"]) for x in values),
        "power_min_mw": min(float(x["power_min_mw"]) for x in values),
        "power_max_mw": max(float(x["power_max_mw"]) for x in values),
        "power_mean_mw": statistics.fmean(float(x["samples"][0]["power_mw"]) for x in values),
        "max_abs_error_pm": max(float(x["max_abs_error_pm"]) for x in values)}
    save_json(OUTPUT, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
