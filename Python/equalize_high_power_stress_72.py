"""Equalize the fresh 72-point stress table to one high common power."""

from __future__ import annotations

import argparse
import json
import math
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
SOURCE = HERE / "high_power_stress_72_wavelength_calibration_v3.json"
DESIGN = HERE / "high_power_stress_72_design_v3.json"
OUTPUT = HERE / "high_power_stress_72_equalized.json"
TARGET_POWER_MW = 1.2311414157220184
POWER_TOLERANCE_MW = 0.015
WAVELENGTH_TOLERANCE_PM = 5.0
MIN_SMSR_DB = 15.0
DEFAULT_PHASE_SLOPE_PM_PER_MA = -75.0


def usable(reading, segment: dict) -> bool:
    smsr = reading.side_mode_suppression_db
    return bool(
        math.isfinite(reading.wavelength_nm)
        and float(segment["start_nm"]) - 0.35 <= reading.wavelength_nm
        <= float(segment["stop_nm"]) + 0.35
        and reading.power_mw >= 0.70
        and (smsr is None or smsr >= MIN_SMSR_DB)
    )


def score(reading) -> float:
    return (
        abs(reading.power_mw - TARGET_POWER_MW) / POWER_TOLERANCE_MW
        + abs(reading.wavelength_error_pm) / WAVELENGTH_TOLERANCE_PM
    )


def tune_point(laser, meter, point: dict, segment: dict, source: dict) -> dict:
    target_nm = float(point["target_nm"])
    codes = [int(value) for value in source["codes"]]
    # Re-quantize both high-current channels through the strict <=120 mA cap.
    codes[0] = current_to_code(code_to_current(codes[0], 0), 0)
    codes[1] = current_to_code(code_to_current(codes[1], 1), 1)
    phase_ma = code_to_current(codes[2], 2)
    soa_ma = code_to_current(codes[1], 1)
    history = []
    best = None
    previous_phase_sample = None
    phase_slope = DEFAULT_PHASE_SLOPE_PM_PER_MA

    for iteration in range(16):
        reading = take_reading(
            laser,
            meter,
            codes,
            target_nm,
            settle_s=0.34 if iteration == 0 else 0.22,
            select_main_peak=True,
        )
        sample = {
            "iteration": iteration,
            "codes": list(codes),
            "currents_ma": [
                code_to_current(code, channel) for channel, code in enumerate(codes)
            ],
            "reading": asdict(reading),
        }
        history.append(sample)
        if usable(reading, segment) and (best is None or score(reading) < score_obj(best)):
            best = sample

        power_ok = abs(reading.power_mw - TARGET_POWER_MW) <= POWER_TOLERANCE_MW
        wavelength_ok = abs(reading.wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
        if usable(reading, segment) and power_ok and wavelength_ok:
            return make_result(True, target_nm, sample, history)
        if not usable(reading, segment):
            break

        # First move SOA toward the common optical power.  GAIN remains high
        # and unchanged, preserving the user's high-power operating choice.
        if not power_ok:
            ratio = TARGET_POWER_MW / max(reading.power_mw, 0.05)
            requested = soa_ma * ratio
            requested = min(max(requested, soa_ma - 12.0), soa_ma + 8.0)
            requested = min(max(requested, 20.0), 120.0)
            if abs(requested - soa_ma) < 0.015:
                break
            soa_ma = requested
            codes[1] = current_to_code(soa_ma, 1)
            # Keep the latest PHASE probe across the SOA correction.  The
            # wavelength perturbation from this small SOA move is far smaller
            # than the PHASE response, and retaining the paired probe lets us
            # interpolate instead of bouncing between two 0.10 mA endpoints.
            continue

        # Once power is in tolerance, recover wavelength with small PHASE
        # steps.  The bound avoids crossing a longitudinal-mode boundary.
        if previous_phase_sample is not None:
            old_phase, old_error = previous_phase_sample
            delta = phase_ma - old_phase
            if abs(delta) >= 0.008:
                measured_slope = (reading.wavelength_error_pm - old_error) / delta
                if -500.0 <= measured_slope <= -8.0:
                    phase_slope = measured_slope
        previous_phase_sample = (phase_ma, reading.wavelength_error_pm)
        requested = phase_ma - reading.wavelength_error_pm / phase_slope
        requested = min(max(requested, phase_ma - 0.10), phase_ma + 0.10)
        requested = min(max(requested, 0.0), 10.0)
        if abs(requested - phase_ma) < 0.003:
            break
        phase_ma = requested
        codes[2] = current_to_code(phase_ma, 2)

    if best is not None:
        final_reading = best["reading"]
        success = (
            abs(float(final_reading["power_mw"]) - TARGET_POWER_MW)
            <= POWER_TOLERANCE_MW
            and abs(float(final_reading["wavelength_error_pm"]))
            <= WAVELENGTH_TOLERANCE_PM
        )
        return make_result(success, target_nm, best, history)
    return {
        "success": False,
        "target_nm": target_nm,
        "failure": "no_stable_single_mode_sample",
        "history": history,
    }


def score_obj(sample: dict) -> float:
    reading = sample["reading"]
    return (
        abs(float(reading["power_mw"]) - TARGET_POWER_MW) / POWER_TOLERANCE_MW
        + abs(float(reading["wavelength_error_pm"])) / WAVELENGTH_TOLERANCE_PM
    )


def make_result(success: bool, target_nm: float, sample: dict, history: list) -> dict:
    return {
        "success": bool(success),
        "target_nm": target_nm,
        "codes": list(sample["codes"]),
        "currents_ma": list(sample["currents_ma"]),
        "final": dict(sample["reading"]),
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    points = design["points"]
    stop = min(args.stop if args.stop is not None else len(points), len(points))
    start = max(0, min(args.start, stop))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "fresh_live_common_power_soa_then_phase_closed_loop",
        "source": SOURCE.name,
        "target_power_mw": TARGET_POWER_MW,
        "power_tolerance_mw": POWER_TOLERANCE_MW,
        "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
        "limits_ma": {"gain": 120.0, "soa": 120.0, "phase": 10.0,
                      "wave_a": 30.0, "wave_b": 30.0},
        "meter": None,
        "results": {},
    }
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        payload["created"] = old.get("created", payload["created"])
        payload["results"].update(old.get("results", {}))

    rows = parse_dac_rows(DAC_SOURCE)
    park = list(rows[min(44, len(rows) - 1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(120.0, code_to_current(park[1], 1)), 1)
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for index in range(start, stop):
            key = str(index)
            if args.resume and payload["results"].get(key, {}).get("success"):
                continue
            point = points[index]
            source_record = source["results"][key]
            if not source_record.get("success"):
                raise RuntimeError(f"Source point {index} is not calibrated")
            result = tune_point(
                laser,
                meter,
                point,
                design["segments"][int(point["segment"])],
                source_record["final"],
            )
            payload["results"][key] = {
                "point_index": index,
                "segment": int(point["segment"]),
                "target_nm": float(point["target_nm"]),
                **result,
            }
            save_json(args.output, payload)
            final = result.get("final", {})
            print(
                f"[{index + 1:02d}/{stop:02d}] success={result['success']} "
                f"wave={final.get('wavelength_nm', float('nan')):.6f} "
                f"error={final.get('wavelength_error_pm', float('nan')):+.2f} pm "
                f"power={final.get('power_mw', float('nan')):.4f} mW "
                f"SOA={result.get('currents_ma', [0,0])[1]:.3f} mA",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    values = list(payload["results"].values())
    success_count = sum(bool(item.get("success")) for item in values)
    powers = [float(item["final"]["power_mw"]) for item in values if item.get("success")]
    payload["summary"] = {
        "success_count": success_count,
        "attempted_count": len(values),
        "power_min_mw": min(powers) if powers else None,
        "power_max_mw": max(powers) if powers else None,
    }
    save_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
