"""Calibrate the shape-selected table to the highest common safe power."""

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
    LaserSerial,
    code_to_current,
    current_to_code,
    take_reading,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stress_points_shape_selected_proposal.json"
DEFAULT_OUTPUT = HERE / "stress_points_shape_selected_calibrated.json"
TARGET_POWER_MW = 1.50
POWER_TOLERANCE_MW = 0.055
WAVELENGTH_TOLERANCE_PM = 3.0
MIN_SMSR_DB = 20.0
PHASE_SLOPE_PM_PER_MA = -75.0


def smsr(reading) -> float:
    if reading.peak_count == 1 and reading.side_mode_suppression_db is None:
        return 99.0
    return float(reading.side_mode_suppression_db or -999.0)


def calibrate_row(laser: LaserSerial, meter: AQ6150B, row: dict) -> dict:
    target_nm = float(row["candidate_wavelength_nm"])
    codes = [int(value) for value in row["codes"]]
    history: list[dict] = []
    best = None
    for iteration in range(7):
        reading = take_reading(
            laser,
            meter,
            codes,
            target_nm,
            settle_s=0.30 if iteration == 0 else 0.23,
            select_main_peak=True,
        )
        item = {"codes": list(codes), "reading": asdict(reading)}
        history.append(item)
        branch_ok = abs(reading.wavelength_nm - target_nm) <= 0.020
        clean = smsr(reading) >= MIN_SMSR_DB
        if branch_ok and clean:
            score = (
                abs(reading.power_mw - TARGET_POWER_MW) / POWER_TOLERANCE_MW
                + abs(reading.wavelength_error_pm) / WAVELENGTH_TOLERANCE_PM
            )
            if best is None or score < best[0]:
                best = (score, item)
        if (
            branch_ok
            and clean
            and abs(reading.power_mw - TARGET_POWER_MW) <= 0.015
            and abs(reading.wavelength_error_pm) <= 1.5
        ):
            break
        if not branch_ok or not clean:
            break

        # SOA controls optical power.  Limit each iteration so a clean branch
        # cannot be skipped by a large correction.
        soa_ma = code_to_current(codes[1], 1)
        requested_soa = soa_ma * TARGET_POWER_MW / max(reading.power_mw, 0.20)
        requested_soa = min(135.0, max(30.0, requested_soa))
        requested_soa = min(max(requested_soa, soa_ma - 18.0), soa_ma + 18.0)

        # SOA adjustment slightly moves wavelength.  PHASE correction is
        # deliberately small and remains below the user-authorised 10 mA.
        phase_ma = code_to_current(codes[2], 2)
        phase_delta = -reading.wavelength_error_pm / PHASE_SLOPE_PM_PER_MA
        phase_delta = min(max(phase_delta, -0.35), 0.35)
        requested_phase = min(10.0, max(0.0, phase_ma + phase_delta))
        next_codes = list(codes)
        next_codes[1] = current_to_code(requested_soa, 1)
        next_codes[2] = current_to_code(requested_phase, 2)
        if next_codes == codes:
            break
        codes = next_codes

    if best is None:
        raise RuntimeError(f"no clean branch near {target_nm:.6f} nm")
    final_codes = list(best[1]["codes"])
    confirmations = []
    for repeat in range(2):
        confirmations.append(asdict(take_reading(
            laser,
            meter,
            final_codes,
            target_nm,
            settle_s=0.27 if repeat == 0 else 0.22,
            select_main_peak=True,
        )))
    waves = [float(value["wavelength_nm"]) for value in confirmations]
    powers = [float(value["power_mw"]) for value in confirmations]
    smsrs = [
        99.0
        if int(value.get("peak_count", 0)) == 1
        and value.get("side_mode_suppression_db") is None
        else float(value.get("side_mode_suppression_db") or -999.0)
        for value in confirmations
    ]
    final_wave = statistics.median(waves)
    final_power = statistics.median(powers)
    passed = bool(
        max(abs(value - target_nm) for value in waves) * 1000.0 <= WAVELENGTH_TOLERANCE_PM
        and max(waves) - min(waves) <= 0.002
        and min(smsrs) >= MIN_SMSR_DB
        and min(powers) >= TARGET_POWER_MW - POWER_TOLERANCE_MW
        and max(powers) <= TARGET_POWER_MW + POWER_TOLERANCE_MW
    )
    return {
        **row,
        "codes": final_codes,
        "currents_ma": [
            code_to_current(code, channel) for channel, code in enumerate(final_codes)
        ],
        "history": history,
        "confirmations": confirmations,
        "final_wavelength_nm": final_wave,
        "final_wavelength_error_pm": (final_wave - target_nm) * 1000.0,
        "final_wavelength_span_pm": (max(waves) - min(waves)) * 1000.0,
        "final_power_mw": final_power,
        "final_power_min_mw": min(powers),
        "final_power_max_mw": max(powers),
        "final_smsr_min_db": min(smsrs),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "target_power_mw": TARGET_POWER_MW,
        "power_tolerance_mw": POWER_TOLERANCE_MW,
        "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
        "minimum_smsr_db": MIN_SMSR_DB,
        "limits_ma": {
            "gain": 135.0,
            "soa": 135.0,
            "phase": 10.0,
            "wave_a": 30.0,
            "wave_b": 30.0,
        },
        "rows": [],
    }
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        payload["rows"] = old.get("rows", [])

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for index in range(len(payload["rows"]), len(source["rows"])):
            calibrated = calibrate_row(laser, meter, source["rows"][index])
            calibrated["index"] = index
            calibrated["segment"] = index // 7
            payload["rows"].append(calibrated)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"power {index + 1:02d}/{len(source['rows'])} "
                f"{calibrated['final_wavelength_nm']:.6f} nm "
                f"{calibrated['final_power_mw']:.3f} mW "
                f"SMSR>={calibrated['final_smsr_min_db']:.1f} "
                f"pass={calibrated['passed']}",
                flush=True,
            )
    finally:
        try:
            if payload["rows"]:
                laser.set_codes(payload["rows"][0]["codes"])
                time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    good = sum(bool(row["passed"]) for row in payload["rows"])
    payload["summary"] = {
        "tested": len(payload["rows"]),
        "passed": good,
        "failed": len(payload["rows"]) - good,
        "power_min_mw": min(row["final_power_min_mw"] for row in payload["rows"]),
        "power_mean_mw": statistics.mean(row["final_power_mw"] for row in payload["rows"]),
        "power_max_mw": max(row["final_power_max_mw"] for row in payload["rows"]),
        "power_cv_percent": statistics.pstdev(
            row["final_power_mw"] for row in payload["rows"]
        ) / statistics.mean(row["final_power_mw"] for row in payload["rows"]) * 100.0,
        "minimum_smsr_db": min(row["final_smsr_min_db"] for row in payload["rows"]),
        "maximum_wavelength_error_pm": max(
            abs(row["final_wavelength_error_pm"]) for row in payload["rows"]
        ),
        "maximum_wavelength_span_pm": max(
            row["final_wavelength_span_pm"] for row in payload["rows"]
        ),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if good == len(payload["rows"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
