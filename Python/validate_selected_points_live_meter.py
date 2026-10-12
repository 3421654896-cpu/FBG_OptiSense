"""Replay proposed stress points and trigger AQ6150B for every live reading."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from laser_power_calibration import AQ6150B, LaserSerial, take_reading
from validate_rc_against_single import ADC_CODES, ADC_VREF, collect_single_rt


HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "stress_points_shape_selected_proposal.json"
DEFAULT_OUTPUT = HERE / "stress_points_shape_selected_live_validation.json"


def smsr_value(reading: dict) -> float:
    if int(reading.get("peak_count", 0)) == 1 and reading.get("side_mode_suppression_db") is None:
        return 99.0
    return float(reading.get("side_mode_suppression_db") or -999.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    rows = proposal["rows"]
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": PROPOSAL.name,
        "method": (
            "Each point is applied in EXTRA mode; AQ6150B is explicitly INIT/TRG via "
            "GPIB for every repeat; CH1 is then collected at the same settled state."
        ),
        "limits": {
            "maximum_wavelength_drift_from_cached_pm": 20.0,
            "minimum_smsr_db": 20.0,
            "minimum_live_power_mw": 1.40,
        },
        "rows": [],
    }
    if args.start and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        payload["rows"] = old.get("rows", [])[: args.start]

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for index in range(len(payload["rows"]), len(rows)):
            row = rows[index]
            samples = []
            for repeat in range(args.repeats):
                reading = take_reading(
                    laser,
                    meter,
                    row["codes"],
                    float(row["candidate_wavelength_nm"]),
                    settle_s=0.28 if repeat == 0 else 0.22,
                    select_main_peak=True,
                )
                samples.append(asdict(reading))
            adc = collect_single_rt(laser.serial, 0.12)
            if len(adc) < 8:
                raise RuntimeError(f"row {index}: only {len(adc)} ADC frames")
            stable = np.asarray(adc[-min(80, len(adc)):], dtype=float)
            waves = [float(sample["wavelength_nm"]) for sample in samples]
            powers = [float(sample["power_mw"]) for sample in samples]
            smsrs = [smsr_value(sample) for sample in samples]
            cached = float(row["candidate_wavelength_nm"])
            maximum_error_pm = max(abs(wave - cached) for wave in waves) * 1000.0
            passed = bool(
                maximum_error_pm <= payload["limits"]["maximum_wavelength_drift_from_cached_pm"]
                and min(smsrs) >= payload["limits"]["minimum_smsr_db"]
                and min(powers) >= payload["limits"]["minimum_live_power_mw"]
            )
            result = {
                "index": index,
                "segment": int(row["segment"]),
                "codes": [int(value) for value in row["codes"]],
                "currents_ma": [float(value) for value in row["currents_ma"]],
                "cached_wavelength_nm": cached,
                "samples": samples,
                "live_wavelength_nm": statistics.median(waves),
                "live_wavelength_span_pm": (max(waves) - min(waves)) * 1000.0,
                "maximum_cached_error_pm": maximum_error_pm,
                "live_power_mw": statistics.median(powers),
                "live_power_min_mw": min(powers),
                "live_power_max_mw": max(powers),
                "live_smsr_min_db": min(smsrs),
                "live_ch1_v": float(np.median(stable[:, 1]) * ADC_VREF / ADC_CODES),
                "live_ch1_span_v": float(np.ptp(stable[:, 1]) * ADC_VREF / ADC_CODES),
                "passed": passed,
            }
            payload["rows"].append(result)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"meter {index + 1:02d}/{len(rows)} S{result['segment'] + 1} "
                f"{result['live_wavelength_nm']:.6f} nm "
                f"{result['live_power_mw']:.3f} mW SMSR>={result['live_smsr_min_db']:.1f} "
                f"pass={passed}",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(rows[0]["codes"])
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    passed_count = sum(bool(row["passed"]) for row in payload["rows"])
    payload["summary"] = {
        "tested": len(payload["rows"]),
        "passed": passed_count,
        "failed": len(payload["rows"]) - passed_count,
        "live_power_min_mw": min(row["live_power_min_mw"] for row in payload["rows"]),
        "live_power_median_mw": statistics.median(
            row["live_power_mw"] for row in payload["rows"]
        ),
        "live_power_max_mw": max(row["live_power_max_mw"] for row in payload["rows"]),
        "minimum_smsr_db": min(row["live_smsr_min_db"] for row in payload["rows"]),
        "maximum_wavelength_span_pm": max(
            row["live_wavelength_span_pm"] for row in payload["rows"]
        ),
        "maximum_cached_error_pm": max(
            row["maximum_cached_error_pm"] for row in payload["rows"]
        ),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if passed_count == len(payload["rows"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
