"""Collect early-feedback to settled-power training data on a forward path."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from capture_fullband_feedback_reference import apply_guard, read_monitor_window
from laser_power_calibration import AQ6150B, LaserSerial, code_to_current, current_to_code


def save(path: Path, table_name: str, rows: list[dict], complete: bool) -> None:
    path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": table_name,
        "scan_direction": "forward",
        "complete": complete,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("fullband_equal_power_operational_2001.json"))
    parser.add_argument("--output", type=Path, default=Path("fullband_dynamic_power_training_0p2nm.json"))
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--soa-delta-ma", type=float, default=0.75)
    parser.add_argument("--settle-ms", type=float, default=5.0)
    parser.add_argument("--sample-ms", type=float, default=25.0)
    args = parser.parse_args()

    table = json.loads(args.table.read_text(encoding="utf-8"))
    rows = sorted(table["rows"].values(), key=lambda row: float(row["target_nm"]))
    guards = {
        round(float(item["target_nm"]), 6): item
        for item in table["operational_export"]["transition_guards"]
    }
    samples: list[dict] = []
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    previous_codes = [current_to_code(value, channel) for channel, value in enumerate((60, 90, 0, 10, 10))]
    previous_target_nm = None
    started = time.monotonic()
    try:
        for index, row in enumerate(rows):
            target_nm = float(row["target_nm"])
            apply_guard(laser, guards.get(round(target_nm, 6)))
            base_codes = [int(value) for value in row["codes"]]
            if index % max(1, args.stride) != 0 and index != len(rows) - 1:
                laser.set_codes(base_codes)
                previous_codes = list(base_codes)
                previous_target_nm = target_nm
                time.sleep(0.003)
                continue

            base_soa = code_to_current(base_codes[1], 1)
            variants = [
                ("base", base_soa),
                ("soa_minus", max(0.0, base_soa - args.soa_delta_ma)),
                ("soa_plus", min(135.0, base_soa + args.soa_delta_ma)),
            ]
            for variant, soa_ma in variants:
                codes = list(base_codes)
                codes[1] = current_to_code(soa_ma, 1)
                input_previous_codes = list(previous_codes)
                input_previous_target_nm = previous_target_nm
                laser.set_codes(codes)
                time.sleep(args.settle_ms / 1000.0)
                early_pdt, early_pdr, early_frames = read_monitor_window(
                    laser, args.sample_ms / 1000.0
                )
                reading = meter.measure(target_nm, select_main_peak=True)
                late_pdt, late_pdr, late_frames = read_monitor_window(laser, 0.05)
                sample = {
                    "index": index,
                    "target_nm": target_nm,
                    "variant": variant,
                    "codes": codes,
                    "currents_ma": [code_to_current(code, channel) for channel, code in enumerate(codes)],
                    "previous_target_nm": input_previous_target_nm,
                    "previous_codes": input_previous_codes,
                    "previous_currents_ma": [
                        code_to_current(code, channel)
                        for channel, code in enumerate(input_previous_codes)
                    ],
                    "early_pdt_code": early_pdt,
                    "early_pdr_code": early_pdr,
                    "early_frame_count": early_frames,
                    "late_pdt_code": late_pdt,
                    "late_pdr_code": late_pdr,
                    "late_frame_count": late_frames,
                    "settled": {
                        "wavelength_nm": reading.wavelength_nm,
                        "wavelength_error_pm": reading.wavelength_error_pm,
                        "power_mw": reading.power_mw,
                        "power_dbm": reading.power_dbm,
                        "peak_count": reading.peak_count,
                        "smsr_db": reading.side_mode_suppression_db,
                    },
                }
                samples.append(sample)
                previous_codes = list(codes)
                previous_target_nm = target_nm
            save(args.output, args.table.name, samples, complete=False)
            print(
                f"training target {target_nm:.2f}nm samples={len(samples)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    finally:
        try:
            park = [current_to_code(value, channel) for channel, value in enumerate((60, 90, 0, 10, 10))]
            laser.set_codes(park)
            time.sleep(0.1)
        finally:
            laser.close()
            meter.close()

    save(args.output, args.table.name, samples, complete=True)
    powers = [sample["settled"]["power_mw"] for sample in samples]
    smsrs = [
        sample["settled"]["smsr_db"]
        for sample in samples
        if sample["settled"]["smsr_db"] is not None
    ]
    print(json.dumps({
        "samples": len(samples),
        "targets": len({sample["target_nm"] for sample in samples}),
        "power_min_mw": min(powers),
        "power_max_mw": max(powers),
        "minimum_smsr_db": min(smsrs),
        "elapsed_s": time.monotonic() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
