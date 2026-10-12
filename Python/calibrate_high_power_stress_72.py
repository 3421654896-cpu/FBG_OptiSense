"""Calibrate the designed 72-point stress table on the live optical path."""

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
DESIGN = HERE / "high_power_stress_72_design_v3.json"
MAPS = (
    HERE / "high_power_tuning_map_mainpeak_v2.json",
    HERE / "high_power_tuning_map_gain105.json",
    HERE / "high_power_tuning_map_gain90.json",
)
OUTPUT = HERE / "high_power_stress_72_wavelength_calibration_v3.json"
WAVELENGTH_TOLERANCE_PM = 5.0
MIN_POWER_MW = 0.75
MIN_SMSR_DB = 15.0
DEFAULT_PHASE_SLOPE_PM_PER_MA = -75.0


def reading_is_usable(reading, start_nm: float, stop_nm: float) -> bool:
    smsr = reading.side_mode_suppression_db
    return bool(
        math.isfinite(reading.wavelength_nm)
        and start_nm - 0.40 <= reading.wavelength_nm <= stop_nm + 0.40
        and reading.power_mw >= MIN_POWER_MW
        and (smsr is None or smsr >= MIN_SMSR_DB)
    )


def candidate_pool(segment: dict) -> list[dict]:
    rows = []
    for path in MAPS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        gain = float(payload["gain_ma"])
        for item in payload["results"].values():
            reading = item["reading"]
            smsr = reading.get("side_mode_suppression_db")
            if smsr is None:
                smsr = 99.0
            measured = float(reading["wavelength_nm"])
            power = float(reading["power_mw"])
            if not (segment["start_nm"] - 0.40 <= measured
                    <= segment["stop_nm"] + 0.40):
                continue
            if power < MIN_POWER_MW or float(smsr) < MIN_SMSR_DB:
                continue
            rows.append(
                {
                    "map": path.name,
                    "seed_index": int(item["seed_index"]),
                    "gain_ma": gain,
                    "measured_nm": measured,
                    "power_mw": power,
                    "smsr_db": float(smsr),
                    "codes": [int(value) for value in item["codes"]],
                }
            )
    return rows


def ranked_candidates(segment: dict, target_nm: float) -> list[dict]:
    pool = candidate_pool(segment)
    selected = []
    # Try the nearest three seeds at the highest gain first.  Only if that
    # branch cannot reach the target do we fall back to the next GAIN layer.
    for gain in (120.0, 105.0, 90.0):
        layer = [item for item in pool if abs(item["gain_ma"] - gain) < 0.1]
        layer.sort(
            key=lambda item: (
                abs(item["measured_nm"] - target_nm),
                -item["power_mw"],
            )
        )
        selected.extend(layer[:3])
    return selected


def tune_seed(laser, meter, segment, target_nm: float, seed: dict) -> dict:
    codes = list(seed["codes"])
    # Re-quantize the two high-current channels with the strict <=120 mA cap.
    codes[0] = current_to_code(seed["gain_ma"], 0)
    codes[1] = current_to_code(120.0, 1)
    phase_ma = code_to_current(codes[2], 2)
    history = []
    best = None
    previous = None
    slope = DEFAULT_PHASE_SLOPE_PM_PER_MA

    for iteration in range(15):
        reading = take_reading(
            laser,
            meter,
            codes,
            target_nm,
            settle_s=0.36 if iteration == 0 else 0.22,
            select_main_peak=True,
        )
        sample = {
            "iteration": iteration,
            "phase_ma": phase_ma,
            "codes": list(codes),
            "reading": asdict(reading),
        }
        history.append(sample)
        usable = reading_is_usable(
            reading, float(segment["start_nm"]), float(segment["stop_nm"])
        )
        if usable and (best is None or abs(reading.wavelength_error_pm)
                       < abs(best["reading"]["wavelength_error_pm"])):
            best = sample
        if usable and abs(reading.wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM:
            return {
                "success": True,
                "target_nm": target_nm,
                "seed": seed,
                "codes": list(codes),
                "currents_ma": [code_to_current(code, channel)
                                for channel, code in enumerate(codes)],
                "final": asdict(reading),
                "history": history,
            }

        if not usable:
            break
        if previous is not None:
            delta_phase = phase_ma - float(previous["phase_ma"])
            delta_error = reading.wavelength_error_pm - float(
                previous["reading"]["wavelength_error_pm"]
            )
            if abs(delta_phase) >= 0.02:
                measured_slope = delta_error / delta_phase
                if -500.0 <= measured_slope <= -8.0:
                    slope = measured_slope
        previous = sample
        requested = phase_ma - reading.wavelength_error_pm / slope
        # Large phase jumps can cross a longitudinal-mode boundary.  Walk the
        # branch in bounded increments and re-estimate the local slope.
        requested = min(max(requested, phase_ma - 0.60), phase_ma + 0.60)
        requested = min(max(requested, 0.0), 10.0)
        if abs(requested - phase_ma) < 0.004:
            break
        phase_ma = requested
        codes[2] = current_to_code(phase_ma, 2)

    return {
        "success": False,
        "target_nm": target_nm,
        "seed": seed,
        "best": best,
        "history": history,
        "failure": "phase_branch_did_not_reach_target",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-report",
        type=Path,
        default=None,
        help="reuse successful points only when point index and target match exactly",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    points = design["points"]
    stop = min(args.stop if args.stop is not None else len(points), len(points))
    start = max(0, min(args.start, stop))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "gain_priority_live_phase_closed_loop",
        "limits_ma": {"gain": 120.0, "soa": 120.0, "phase": 10.0,
                      "wave_a": 30.0, "wave_b": 30.0},
        "meter": None,
        "results": {},
    }
    if args.resume and args.output.exists():
        payload["results"].update(
            json.loads(args.output.read_text(encoding="utf-8")).get("results", {})
        )
    if args.reuse_report is not None and args.reuse_report.exists():
        old_results = json.loads(
            args.reuse_report.read_text(encoding="utf-8")
        ).get("results", {})
        for point_index, point in enumerate(points):
            old = old_results.get(str(point_index))
            if (
                old
                and old.get("success")
                and abs(float(old.get("target_nm", float("nan")))
                        - float(point["target_nm"])) < 1e-9
            ):
                payload["results"][str(point_index)] = old

    original_rows = parse_dac_rows(DAC_SOURCE)
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    park = list(original_rows[min(44, len(original_rows) - 1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(120.0, code_to_current(park[1], 1)), 1)
    try:
        for point_index in range(start, stop):
            key = str(point_index)
            if args.resume and payload["results"].get(key, {}).get("success"):
                continue
            point = points[point_index]
            segment = design["segments"][int(point["segment"])]
            attempts = []
            final = None
            for seed in ranked_candidates(segment, float(point["target_nm"])):
                result = tune_seed(
                    laser, meter, segment, float(point["target_nm"]), seed
                )
                attempts.append(result)
                if result["success"]:
                    final = result
                    break
            record = {
                "point_index": point_index,
                "segment": int(point["segment"]),
                "target_nm": float(point["target_nm"]),
                "success": final is not None,
                "final": final,
                "attempts": attempts,
            }
            payload["results"][key] = record
            save_json(args.output, payload)
            if final:
                reading = final["final"]
                print(
                    f"[{point_index + 1:02d}/{stop:02d}] "
                    f"target={point['target_nm']:.6f} actual={reading['wavelength_nm']:.6f} "
                    f"error={reading['wavelength_error_pm']:+.2f} pm "
                    f"power={reading['power_mw']:.4f} mW "
                    f"GAIN={final['currents_ma'][0]:.3f} PHASE={final['currents_ma'][2]:.3f}",
                    flush=True,
                )
            else:
                print(
                    f"[{point_index + 1:02d}/{stop:02d}] "
                    f"target={point['target_nm']:.6f} FAILED attempts={len(attempts)}",
                    flush=True,
                )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    success = sum(bool(item.get("success")) for item in payload["results"].values())
    payload["summary"] = {"success_count": success, "attempted_count": len(payload["results"])}
    save_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
