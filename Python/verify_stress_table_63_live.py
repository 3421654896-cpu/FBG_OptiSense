"""Replay the installed 63-point stress table with exact PI11210 DAC codes."""

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
SOURCE = HERE / "stress_high_power_135_final.json"
OUTPUT = HERE / "stress_high_power_135_live_recheck.json"


def parse_indices(text: str, count: int) -> list[int]:
    if not text.strip():
        return list(range(count))
    values = []
    for part in text.split(","):
        index = int(part.strip())
        if not 0 <= index < count:
            raise ValueError(f"Index {index} is outside 0..{count - 1}")
        if index not in values:
            values.append(index)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--indices", default="")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--settle", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    indices = parse_indices(args.indices, len(rows))
    old_rows = parse_dac_rows(DAC_SOURCE)
    park = list(old_rows[min(44, len(old_rows) - 1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(80.0, code_to_current(park[1], 1)), 1)

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "method": "exact_raw_dac_codes_extra_mode_full_fan",
        "settle_s": args.settle,
        "repeats": args.repeats,
        "indices": indices,
        "results": [],
    }

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for sequence, index in enumerate(indices, start=1):
            row = rows[index]
            reference_nm = float(row["final_wavelength_nm"])
            samples = []
            for _ in range(args.repeats):
                reading = take_reading(
                    laser,
                    meter,
                    [int(value) for value in row["codes"]],
                    reference_nm,
                    settle_s=args.settle,
                    select_main_peak=True,
                )
                samples.append(asdict(reading))
            errors_pm = [
                (float(sample["wavelength_nm"]) - reference_nm) * 1000.0
                for sample in samples
            ]
            waves_nm = [float(sample["wavelength_nm"]) for sample in samples]
            powers_mw = [float(sample["power_mw"]) for sample in samples]
            result = {
                "index": index,
                "segment": int(row["segment"]),
                "reference_nm": reference_nm,
                "codes": [int(value) for value in row["codes"]],
                "samples": samples,
                "errors_pm": errors_pm,
                "mean_error_pm": statistics.fmean(errors_pm),
                "repeat_span_pm": (max(waves_nm) - min(waves_nm)) * 1000.0,
                "power_mean_mw": statistics.fmean(powers_mw),
                "within_2pm": max(abs(value) for value in errors_pm) <= 2.0,
            }
            payload["results"].append(result)
            save_json(args.output, payload)
            print(
                f"[{sequence:02d}/{len(indices):02d}] row={index + 1:02d} "
                f"ref={reference_nm:.6f} nm actual="
                f"{','.join(f'{value:.6f}' for value in waves_nm)} nm "
                f"err={','.join(f'{value:+.2f}' for value in errors_pm)} pm "
                f"power={statistics.fmean(powers_mw):.3f} mW",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.10)
        finally:
            laser.close()
            meter.close()

    all_errors = [value for row in payload["results"] for value in row["errors_pm"]]
    payload["summary"] = {
        "tested_points": len(payload["results"]),
        "points_within_2pm": sum(row["within_2pm"] for row in payload["results"]),
        "mean_signed_error_pm": statistics.fmean(all_errors),
        "maximum_absolute_error_pm": max(abs(value) for value in all_errors),
        "minimum_error_pm": min(all_errors),
        "maximum_error_pm": max(all_errors),
    }
    save_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
