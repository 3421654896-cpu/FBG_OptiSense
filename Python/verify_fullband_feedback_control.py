"""Verify bounded PDT/PDR-referenced SOA control against the AQ6150B.

The feedback loop never changes GAIN, PHASE, WAVE-A, or WAVE-B.  SOA changes
are secant-estimated from a small live probe and are limited to 1 mA per
iteration and 3 mA total, always below the configured 135 mA ceiling.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from capture_fullband_feedback_reference import apply_guard, read_monitor_window
from laser_power_calibration import AQ6150B, LaserSerial, code_to_current, current_to_code


def selected_code(channel: str, pdt: int, pdr: int) -> int:
    return pdt if channel == "PDT" else pdr


def settle_feedback(laser: LaserSerial, settle_ms: float = 5.0, sample_ms: float = 25.0):
    time.sleep(settle_ms / 1000.0)
    return read_monitor_window(laser, sample_ms / 1000.0)


def close_power(
    laser: LaserSerial,
    base_codes: list[int],
    channel: str,
    reference_code: int,
    max_iterations: int = 3,
) -> tuple[list[int], list[dict]]:
    working = list(base_codes)
    base_soa_ma = code_to_current(working[1], 1)
    history: list[dict] = []
    laser.set_codes(working)
    pdt, pdr, frames = settle_feedback(laser)

    for iteration in range(max_iterations):
        current_code = selected_code(channel, pdt, pdr)
        error_code = current_code - reference_code
        history.append({
            "iteration": iteration,
            "stage": "base",
            "soa_ma": code_to_current(working[1], 1),
            "pdt_code": pdt,
            "pdr_code": pdr,
            "selected_code": current_code,
            "error_code": error_code,
            "monitor_frames": frames,
        })
        if abs(error_code) <= 3:
            break

        current_soa_ma = code_to_current(working[1], 1)
        probe_delta_ma = 0.25 if current_soa_ma <= 134.70 else -0.25
        probe_soa_ma = current_soa_ma + probe_delta_ma
        probe_codes = list(working)
        probe_codes[1] = current_to_code(probe_soa_ma, 1)
        laser.set_codes(probe_codes)
        probe_pdt, probe_pdr, probe_frames = settle_feedback(laser)
        probe_selected = selected_code(channel, probe_pdt, probe_pdr)
        actual_probe_delta = code_to_current(probe_codes[1], 1) - current_soa_ma
        slope_code_per_ma = (probe_selected - current_code) / actual_probe_delta
        history.append({
            "iteration": iteration,
            "stage": "probe",
            "soa_ma": code_to_current(probe_codes[1], 1),
            "pdt_code": probe_pdt,
            "pdr_code": probe_pdr,
            "selected_code": probe_selected,
            "slope_code_per_ma": slope_code_per_ma,
            "monitor_frames": probe_frames,
        })
        if slope_code_per_ma >= -1.0:
            laser.set_codes(working)
            history.append({
                "iteration": iteration,
                "stage": "abort",
                "reason": "feedback slope is not safely negative",
            })
            break

        correction_ma = (reference_code - current_code) / slope_code_per_ma
        correction_ma = max(-1.0, min(1.0, correction_ma))
        requested_soa = current_soa_ma + correction_ma
        requested_soa = max(base_soa_ma - 3.0, min(base_soa_ma + 3.0, requested_soa))
        requested_soa = max(0.0, min(135.0, requested_soa))
        working[1] = current_to_code(requested_soa, 1)
        laser.set_codes(working)
        pdt, pdr, frames = settle_feedback(laser)
    return working, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("fullband_equal_power_operational_2001.json"))
    parser.add_argument("--targets", type=Path, default=Path("fullband_equal_power_feedback_targets_2001.json"))
    parser.add_argument("--output", type=Path, default=Path("fullband_feedback_control_validation.json"))
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    table = json.loads(args.table.read_text(encoding="utf-8"))
    target_payload = json.loads(args.targets.read_text(encoding="utf-8"))
    rows = sorted(table["rows"].values(), key=lambda row: float(row["target_nm"]))
    target_rows = target_payload["rows"]
    guards = {
        round(float(item["target_nm"]), 6): item
        for item in table["operational_export"]["transition_guards"]
    }
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    results: list[dict] = []
    started = time.monotonic()
    try:
        for index, row in enumerate(rows):
            target_nm = float(row["target_nm"])
            apply_guard(laser, guards.get(round(target_nm, 6)))
            if index % max(1, args.stride) != 0 and index != len(rows) - 1:
                laser.set_codes([int(value) for value in row["codes"]])
                time.sleep(0.003)
                continue
            control_target = target_rows[index]
            adjusted_codes, history = close_power(
                laser,
                [int(value) for value in row["codes"]],
                str(control_target["power_feedback_channel"]),
                int(control_target["power_feedback_reference_code"]),
                args.max_iterations,
            )
            reading = meter.measure(target_nm, select_main_peak=True)
            pdt, pdr, frames = read_monitor_window(laser, 0.05)
            selected = selected_code(
                str(control_target["power_feedback_channel"]), pdt, pdr
            )
            record = {
                "index": index,
                "target_nm": target_nm,
                "base_codes": row["codes"],
                "adjusted_codes": adjusted_codes,
                "base_soa_ma": code_to_current(int(row["codes"][1]), 1),
                "adjusted_soa_ma": code_to_current(int(adjusted_codes[1]), 1),
                "feedback_channel": control_target["power_feedback_channel"],
                "feedback_reference_code": control_target["power_feedback_reference_code"],
                "final_feedback_code": selected,
                "final_feedback_error_code": selected - int(control_target["power_feedback_reference_code"]),
                "pdt_code": pdt,
                "pdr_code": pdr,
                "monitor_frames": frames,
                "reading": {
                    "wavelength_nm": reading.wavelength_nm,
                    "wavelength_error_pm": reading.wavelength_error_pm,
                    "power_mw": reading.power_mw,
                    "power_error_percent": (reading.power_mw / float(target_payload["target_power_mw"]) - 1.0) * 100.0,
                    "peak_count": reading.peak_count,
                    "smsr_db": reading.side_mode_suppression_db,
                },
                "history": history,
            }
            results.append(record)
            print(
                f"control {len(results)} target={target_nm:.2f}nm "
                f"power={reading.power_mw:.6f}mW wave_err={reading.wavelength_error_pm:+.2f}pm "
                f"feedback_err={record['final_feedback_error_code']:+d}code",
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

    powers = [row["reading"]["power_mw"] for row in results]
    power_errors = [abs(row["reading"]["power_error_percent"]) for row in results]
    wavelength_errors = [abs(row["reading"]["wavelength_error_pm"]) for row in results]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": args.table.name,
        "source_targets": args.targets.name,
        "stride": args.stride,
        "rows": results,
        "summary": {
            "tested": len(results),
            "power_min_mw": min(powers),
            "power_max_mw": max(powers),
            "power_abs_error_median_percent": sorted(power_errors)[len(power_errors) // 2],
            "power_abs_error_max_percent": max(power_errors),
            "wavelength_abs_error_median_pm": sorted(wavelength_errors)[len(wavelength_errors) // 2],
            "wavelength_abs_error_max_pm": max(wavelength_errors),
            "power_within_0p5_percent": sum(value <= 0.5 for value in power_errors),
            "wavelength_within_2pm": sum(value <= 2.0 for value in wavelength_errors),
            "single_mode_pass": sum(
                row["reading"]["smsr_db"] is None or row["reading"]["smsr_db"] >= 20.0
                for row in results
            ),
            "elapsed_s": time.monotonic() - started,
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
