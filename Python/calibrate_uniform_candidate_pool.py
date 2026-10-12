"""Build a settled CH1 candidate pool after, not before, power equalisation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from laser_power_calibration import (
    AQ6150B,
    LaserSerial,
    code_to_current,
    current_to_code,
    take_reading,
)
from validate_rc_against_single import ADC_CODES, ADC_VREF, collect_single_rt


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dense_single_reflection_135.json"
DEFAULT_OUTPUT = HERE / "uniform_power_candidate_pool.json"
TARGET_POWER_MW = 1.50
TARGET_SEGMENTS = {2, 3, 4, 7}  # zero-based segments that failed final shape QA


def smsr(reading: dict) -> float:
    if int(reading.get("peak_count", 0)) == 1 and reading.get("side_mode_suppression_db") is None:
        return 99.0
    return float(reading.get("side_mode_suppression_db") or -999.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--segments",
        default=",".join(str(value) for value in sorted(TARGET_SEGMENTS)),
        help="comma-separated zero-based wavelength segments",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    target_segments = {
        int(value.strip()) for value in str(args.segments).split(",") if value.strip()
    }
    if not target_segments or any(value < 0 or value > 8 for value in target_segments):
        raise ValueError("segments must be zero-based values from 0 through 8")
    candidates = [
        row for row in source["rows"] if int(row["segment"]) in target_segments
    ]
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE.name,
        "old_total_table_used": False,
        "target_power_mw": TARGET_POWER_MW,
        "method": (
            "hold GAIN/PHASE/WAVE A/WAVE B; adjust only SOA to 1.50 mW; "
            "accept the newly measured wavelength; then acquire settled CH1"
        ),
        "target_segments": sorted(target_segments),
        "rows": [],
    }
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        payload["rows"] = old.get("rows", [])

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for candidate_index in range(len(payload["rows"]), len(candidates)):
            candidate = candidates[candidate_index]
            cached_nm = float(candidate["candidate_wavelength_nm"])
            codes = [int(value) for value in candidate["codes"]]
            attempts: list[dict] = []
            reading = take_reading(
                laser, meter, codes, cached_nm, settle_s=0.26, select_main_peak=True
            )
            attempts.append({"codes": list(codes), "reading": asdict(reading)})
            # One proportional SOA move plus one small residual correction is
            # enough for this candidate-screening pool.  Wavelength is measured
            # after the move instead of forcing PHASE back to the cached value.
            for _ in range(2):
                if abs(reading.power_mw - TARGET_POWER_MW) <= 0.018:
                    break
                soa_ma = code_to_current(codes[1], 1)
                requested = soa_ma * TARGET_POWER_MW / max(reading.power_mw, 0.2)
                requested = min(135.0, max(30.0, requested))
                requested = min(max(requested, soa_ma - 20.0), soa_ma + 20.0)
                next_code = current_to_code(requested, 1)
                if next_code == codes[1]:
                    break
                codes[1] = next_code
                reading = take_reading(
                    laser, meter, codes, cached_nm, settle_s=0.23, select_main_peak=True
                )
                attempts.append({"codes": list(codes), "reading": asdict(reading)})

            confirmation = take_reading(
                laser,
                meter,
                codes,
                reading.wavelength_nm,
                settle_s=0.22,
                select_main_peak=True,
            )
            adc = collect_single_rt(laser.serial, 0.13)
            stable = np.asarray(adc[-min(80, len(adc)):], dtype=float) if adc else np.empty((0, 4))
            same_branch = abs(reading.wavelength_nm - cached_nm) <= 0.18
            repeat_wave_pm = abs(confirmation.wavelength_nm - reading.wavelength_nm) * 1000.0
            clean = min(smsr(asdict(reading)), smsr(asdict(confirmation))) >= 20.0
            power_ok = bool(
                min(reading.power_mw, confirmation.power_mw) >= 1.445
                and max(reading.power_mw, confirmation.power_mw) <= 1.555
            )
            accepted = bool(
                same_branch and repeat_wave_pm <= 2.0 and clean and power_ok and len(stable) >= 8
            )
            row = {
                "candidate_index": candidate_index,
                "segment": int(candidate["segment"]),
                "cached_wavelength_nm": cached_nm,
                "codes": list(codes),
                "currents_ma": [
                    code_to_current(code, channel) for channel, code in enumerate(codes)
                ],
                "attempts": attempts,
                "confirmation": asdict(confirmation),
                "wavelength_nm": statistics.median(
                    [reading.wavelength_nm, confirmation.wavelength_nm]
                ),
                "wavelength_span_pm": repeat_wave_pm,
                "power_mw": statistics.median([reading.power_mw, confirmation.power_mw]),
                "power_min_mw": min(reading.power_mw, confirmation.power_mw),
                "power_max_mw": max(reading.power_mw, confirmation.power_mw),
                "smsr_min_db": min(smsr(asdict(reading)), smsr(asdict(confirmation))),
                "ch1_v": float(np.median(stable[:, 1]) * ADC_VREF / ADC_CODES)
                if len(stable) else None,
                "ch1_span_v": float(np.ptp(stable[:, 1]) * ADC_VREF / ADC_CODES)
                if len(stable) else None,
                "accepted": accepted,
            }
            payload["rows"].append(row)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"pool {candidate_index + 1:03d}/{len(candidates)} "
                f"S{row['segment'] + 1} {row['wavelength_nm']:.6f} nm "
                f"{row['power_mw']:.3f} mW CH1={row['ch1_v']} accepted={accepted}",
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

    accepted_rows = [row for row in payload["rows"] if row["accepted"]]
    payload["summary"] = {
        "tested": len(payload["rows"]),
        "accepted": len(accepted_rows),
        "rejected": len(payload["rows"]) - len(accepted_rows),
        "accepted_per_segment": {
            str(segment): sum(row["segment"] == segment for row in accepted_rows)
            for segment in sorted(target_segments)
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
