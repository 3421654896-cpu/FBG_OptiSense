"""Measure a dense settled FBG reflection spectrum from our 135 mA candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from build_and_validate_stress_135 import CENTERS, collect
from laser_power_calibration import LaserSerial, code_to_current
from validate_rc_against_single import ADC_CODES, ADC_VREF, collect_single_rt


HERE = Path(__file__).resolve().parent
CANDIDATE_SOURCE = HERE / "single_mode_candidates_135.json"
DEFAULT_OUTPUT = HERE / "dense_single_reflection_135.json"


def usable_rows(source: dict) -> list[dict]:
    rows: list[dict] = []
    for segment in range(len(CENTERS)):
        for candidate in collect(source, segment):
            reading = candidate["reading"]
            smsr = reading.get("side_mode_suppression_db")
            peak_count = int(reading.get("peak_count", 0))
            clean = peak_count == 1 or (smsr is not None and float(smsr) >= 20.0)
            if not clean:
                continue
            rows.append({
                "segment": segment,
                "fbg_center_nm": float(CENTERS[segment]),
                "candidate_wavelength_nm": float(reading["wavelength_nm"]),
                "candidate_power_mw": float(reading["power_mw"]),
                "candidate_smsr_db": 99.0 if peak_count == 1 else float(smsr),
                "candidate_peak_count": peak_count,
                "codes": [int(value) for value in candidate["codes"]],
            })
    unique: dict[tuple[int, ...], dict] = {}
    for row in rows:
        key = tuple(row["codes"])
        previous = unique.get(key)
        if previous is None or row["candidate_smsr_db"] > previous["candidate_smsr_db"]:
            unique[key] = row
    return sorted(unique.values(), key=lambda row: (row["segment"], row["candidate_wavelength_nm"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--precondition", type=float, default=0.055)
    parser.add_argument("--target-settle", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = json.loads(CANDIDATE_SOURCE.read_text(encoding="utf-8"))
    candidates = usable_rows(source)
    if not candidates:
        raise RuntimeError("no usable locally calibrated 135 mA candidates")
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "candidate_source": CANDIDATE_SOURCE.name,
            "old_total_table_used": False,
            "mode": "EXTRA/single-value; 40 kOhm; fan forced to full by firmware",
            "precondition_s": args.precondition,
            "target_settle_s": args.target_settle,
            "value": "median of the final up to 80 ADC telemetry samples",
        },
        "rows": [],
    }

    laser = LaserSerial(args.port)
    try:
        predecessor = candidates[-1]
        for index, row in enumerate(candidates):
            laser.set_codes(predecessor["codes"])
            collect_single_rt(laser.serial, args.precondition)
            laser.set_codes(row["codes"])
            readings = collect_single_rt(laser.serial, args.target_settle)
            if len(readings) < 8:
                raise RuntimeError(f"candidate {index}: only {len(readings)} ADC frames")
            stable = np.asarray(readings[-min(80, len(readings)):], dtype=float)
            medians = np.median(stable, axis=0)
            spans = np.ptp(stable, axis=0)
            measured = dict(row)
            measured.update({
                "index": index,
                "currents_ma": [
                    code_to_current(code, channel)
                    for channel, code in enumerate(row["codes"])
                ],
                "adc_median_codes": [float(value) for value in medians],
                "adc_span_codes": [float(value) for value in spans],
                "voltage_median_v": [
                    float(value * ADC_VREF / ADC_CODES) for value in medians
                ],
                "voltage_span_v": [
                    float(value * ADC_VREF / ADC_CODES) for value in spans
                ],
                "telemetry_frames": len(readings),
            })
            payload["rows"].append(measured)
            predecessor = row
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"dense {index + 1:03d}/{len(candidates)} "
                f"S{row['segment'] + 1} {row['candidate_wavelength_nm']:.6f} nm "
                f"CH1={measured['voltage_median_v'][1]:.4f} V",
                flush=True,
            )
    finally:
        # Leave the laser at a known, already validated high-power table state.
        try:
            laser.set_codes(candidates[0]["codes"])
            time.sleep(0.08)
        finally:
            laser.close()

    segment_summary = []
    for segment in range(len(CENTERS)):
        subset = [row for row in payload["rows"] if row["segment"] == segment]
        segment_summary.append({
            "segment": segment,
            "count": len(subset),
            "wavelength_min_nm": min(row["candidate_wavelength_nm"] for row in subset),
            "wavelength_max_nm": max(row["candidate_wavelength_nm"] for row in subset),
            "ch1_min_v": min(row["voltage_median_v"][1] for row in subset),
            "ch1_max_v": max(row["voltage_median_v"][1] for row in subset),
            "median_repeat_span_v": statistics.median(
                row["voltage_span_v"][1] for row in subset
            ),
        })
    payload["summary"] = {
        "point_count": len(payload["rows"]),
        "segments": segment_summary,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
