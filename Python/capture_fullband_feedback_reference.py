"""Capture forward-scan PDT/PDR references for the 2001-point calibration.

The wavelength-meter calibration and the internal feedback capture are kept as
separate, auditable measurements.  AQ6150B spot checks are recorded every
``--meter-stride`` points while every point receives a settled PDT/PDR sample.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from laser_dac_safety import FULLBAND_2001_CODE_LIMITS
from laser_power_calibration import AQ6150B, LaserSerial, current_to_code


MONITOR_MARKER = b"\xFF\xFF\x02\x02"


def read_monitor_window(laser: LaserSerial, duration_s: float) -> tuple[int, int, int]:
    for attempt in range(4):
        laser.serial.reset_input_buffer()
        laser._rx.clear()
        deadline = time.monotonic() + duration_s * (1.0 + 0.5 * attempt)
        raw = bytearray()
        while time.monotonic() < deadline:
            waiting = laser.serial.in_waiting
            if waiting:
                raw.extend(laser.serial.read(waiting))
            time.sleep(0.005)
        waiting = laser.serial.in_waiting
        if waiting:
            raw.extend(laser.serial.read(waiting))

        pdt_values: list[int] = []
        pdr_values: list[int] = []
        start = 0
        while True:
            index = raw.find(MONITOR_MARKER, start)
            if index < 0:
                break
            if index + 8 <= len(raw):
                pdt_values.append((raw[index + 4] << 8) | raw[index + 5])
                pdr_values.append((raw[index + 6] << 8) | raw[index + 7])
            start = index + 4
        if pdt_values:
            return (
                int(round(statistics.median(pdt_values))),
                int(round(statistics.median(pdr_values))),
                len(pdt_values),
            )
        time.sleep(0.02)
    raise RuntimeError("no complete PDT/PDR monitor frame received after four windows")


def save_checkpoint(path: Path, table_name: str, rows: list[dict], complete: bool) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": table_name,
        "complete": complete,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_guard(laser: LaserSerial, guard: dict | None) -> None:
    if not guard:
        return
    hold_s = float(guard["hold_ms_per_stage"]) / 1000.0
    for codes in guard["precondition_codes"]:
        laser.set_codes([int(value) for value in codes])
        time.sleep(hold_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("fullband_equal_power_operational_2001.json"))
    parser.add_argument("--output", type=Path, default=Path("fullband_pdt_pdr_reference_2001.json"))
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--settle-ms", type=float, default=80.0)
    parser.add_argument("--sample-ms", type=float, default=70.0)
    parser.add_argument("--meter-stride", type=int, default=50)
    args = parser.parse_args()

    source = json.loads(args.table.read_text(encoding="utf-8"))
    rows = sorted(source["rows"].values(), key=lambda row: float(row["target_nm"]))
    export = source["operational_export"]
    guards = {
        round(float(item["target_nm"]), 6): item
        for item in export["transition_guards"]
    }

    laser = LaserSerial(args.port, code_limits=FULLBAND_2001_CODE_LIMITS)
    meter = AQ6150B(args.meter) if args.meter else None
    results: list[dict] = []
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            previous.get("source_table") == args.table.name
            and not previous.get("complete", False)
            and isinstance(previous.get("rows"), list)
        ):
            results = previous["rows"]
            print(f"resuming from checkpoint row {len(results)}", flush=True)
    started = time.monotonic()
    try:
        # Re-establish the exact forward branch before a checkpoint resume.
        for row in rows[: len(results)]:
            target_nm = float(row["target_nm"])
            apply_guard(laser, guards.get(round(target_nm, 6)))
            laser.set_codes([int(value) for value in row["codes"]])
            time.sleep(0.005)
        for index, row in enumerate(rows):
            if index < len(results):
                continue
            target_nm = float(row["target_nm"])
            guard = guards.get(round(target_nm, 6))
            apply_guard(laser, guard)
            laser.set_codes([int(value) for value in row["codes"]])
            time.sleep(args.settle_ms / 1000.0)
            pdt_code, pdr_code, frame_count = read_monitor_window(
                laser, args.sample_ms / 1000.0
            )
            record = {
                "index": index,
                "target_nm": target_nm,
                "codes": row["codes"],
                "pdt_code": pdt_code,
                "pdr_code": pdr_code,
                "pdt_signal_code": 2048 - pdt_code,
                "pdr_signal_code": 2048 - pdr_code,
                "pdr_pdt_signal_ratio": (
                    (2048 - pdr_code) / (2048 - pdt_code)
                    if (2048 - pdt_code) != 0 else None
                ),
                "monitor_frame_count": frame_count,
                "guard_applied": bool(guard),
            }
            if meter is not None and (
                index % max(1, args.meter_stride) == 0 or index == len(rows) - 1
            ):
                reading = meter.measure(target_nm, select_main_peak=True)
                record["meter_spot_check"] = {
                    "wavelength_nm": reading.wavelength_nm,
                    "wavelength_error_pm": reading.wavelength_error_pm,
                    "power_mw": reading.power_mw,
                    "power_dbm": reading.power_dbm,
                    "peak_count": reading.peak_count,
                    "smsr_db": reading.side_mode_suppression_db,
                }
            results.append(record)
            if (index + 1) % 50 == 0:
                save_checkpoint(args.output, args.table.name, results, complete=False)
            if index == 0 or (index + 1) % 100 == 0 or index == len(rows) - 1:
                elapsed = time.monotonic() - started
                print(
                    f"feedback {index + 1}/{len(rows)} "
                    f"target={target_nm:.2f}nm PDT={pdt_code} PDR={pdr_code} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        try:
            park = [current_to_code(value, channel) for channel, value in enumerate((60, 90, 0, 10, 10))]
            laser.set_codes(park)
            time.sleep(0.1)
        finally:
            laser.close()
            if meter is not None:
                meter.close()

    meter_rows = [row for row in results if "meter_spot_check" in row]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": args.table.name,
        "scan_direction": "forward",
        "point_count": len(results),
        "settle_ms": args.settle_ms,
        "sample_window_ms": args.sample_ms,
        "meter_resource": args.meter,
        "meter_spot_stride": args.meter_stride,
        "rows": results,
        "summary": {
            "pdt_code_min": min(row["pdt_code"] for row in results),
            "pdt_code_max": max(row["pdt_code"] for row in results),
            "pdr_code_min": min(row["pdr_code"] for row in results),
            "pdr_code_max": max(row["pdr_code"] for row in results),
            "meter_spot_checks": len(meter_rows),
            "elapsed_s": time.monotonic() - started,
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
