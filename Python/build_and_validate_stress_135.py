"""Build a uniform high-power 135 mA stress table and replay it twice."""

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
    AQ6150B, DAC_SOURCE, LaserSerial, code_to_current, current_to_code,
    parse_dac_rows, save_json, take_reading,
)

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "single_mode_candidates_135.json"
OUTPUT = HERE / "stress_high_power_135_final.json"
VALIDATION = HERE / "stress_high_power_135_validation.json"

CENTERS = (1527.941, 1532.238, 1536.326, 1540.220, 1544.456,
           1547.902, 1551.702, 1555.827, 1559.884)
TARGET_POWER_MW = 1.50
POWER_TOLERANCE_MW = 0.055
MIN_SMSR_DB = 20.0
SEGMENT_HALF_WIDTH_NM = (0.60, 0.60, 0.60, 0.60, 0.60, 0.70, 0.70, 0.60, 0.70)


def clean(row: dict, segment: int) -> bool:
    reading = row["reading"]
    return bool(
        abs(float(reading["wavelength_nm"]) - CENTERS[segment]) <= SEGMENT_HALF_WIDTH_NM[segment]
        and float(reading["power_mw"]) >= TARGET_POWER_MW
        and reading.get("side_mode_suppression_db") is not None
        and float(reading["side_mode_suppression_db"]) >= MIN_SMSR_DB
    )


def collect(payload: dict, segment: int) -> list[dict]:
    rows = []
    for row in payload["base_candidates"]:
        if clean(row, segment):
            rows.append(row)
    for row in payload["phase_scans"].values():
        if int(row["segment"]) == segment and row.get("accepted") and clean(row, segment):
            rows.append(row)
    if segment == 2:
        gain_layer_path = HERE / "segment2_gain_layers_soa135.json"
        if gain_layer_path.exists():
            gain_layers = json.loads(gain_layer_path.read_text(encoding="utf-8"))
            for row in gain_layers.get("results", {}).values():
                reading = row["reading"]
                smsr = reading.get("side_mode_suppression_db")
                if (abs(float(reading["wavelength_nm"]) - CENTERS[segment]) <= SEGMENT_HALF_WIDTH_NM[segment]
                        and float(reading["power_mw"]) >= 1.45
                        and (smsr is None and int(reading.get("peak_count", 0)) == 1
                             or smsr is not None and float(smsr) >= MIN_SMSR_DB)):
                    rows.append(row)
    if segment in (5, 8):
        gap_layer_path = HERE / "gap_gain_layers_soa135.json"
        if gap_layer_path.exists():
            gap_layers = json.loads(gap_layer_path.read_text(encoding="utf-8"))
            for row in gap_layers.get("results", {}).values():
                if int(row["segment"]) != segment:
                    continue
                reading = row["reading"]
                smsr = reading.get("side_mode_suppression_db")
                if (abs(float(reading["wavelength_nm"]) - CENTERS[segment]) <= SEGMENT_HALF_WIDTH_NM[segment]
                        and float(reading["power_mw"]) >= 1.45
                        and (smsr is None and int(reading.get("peak_count", 0)) == 1
                             or smsr is not None and float(smsr) >= MIN_SMSR_DB)):
                    rows.append(row)
    unique = {}
    for row in rows:
        key = tuple(int(value) for value in row["codes"])
        unique[key] = row
    return list(unique.values())


def ranked_selections(payload: dict) -> list[list[dict]]:
    result = []
    for segment, center in enumerate(CENTERS):
        pool = collect(payload, segment)
        if len(pool) < 7:
            raise RuntimeError(f"segment {segment} has only {len(pool)} usable rows")
        pool_waves = sorted(float(row["reading"]["wavelength_nm"]) for row in pool)
        stable_start = pool_waves[0]
        stable_stop = pool_waves[-1]
        targets = [stable_start + index * (stable_stop - stable_start) / 6.0
                   for index in range(7)]
        used = set()
        for target in targets:
            available = [row for row in pool if tuple(row["codes"]) not in used]
            ranked = sorted(available, key=lambda item: (
                abs(float(item["reading"]["wavelength_nm"]) - target),
                -float(item["currents_ma"][0]),
                abs(float(item["reading"]["power_mw"]) - TARGET_POWER_MW),
                -float(item["reading"]["side_mode_suppression_db"]
                       if item["reading"].get("side_mode_suppression_db") is not None
                       else 99.0),
            ))
            row = ranked[0]
            used.add(tuple(row["codes"]))
            choices = []
            for candidate in ranked[:8]:
                choices.append({
                    "index": len(result), "segment": segment,
                    "fbg_center_nm": center, "desired_wavelength_nm": target,
                    "selection_wavelength_nm": float(candidate["reading"]["wavelength_nm"]),
                    "selection_power_mw": float(candidate["reading"]["power_mw"]),
                    "selection_smsr_db": (
                        float(candidate["reading"]["side_mode_suppression_db"])
                        if candidate["reading"].get("side_mode_suppression_db") is not None
                        else 99.0
                    ),
                    "codes": [int(value) for value in candidate["codes"]],
                })
            result.append(choices)
    return result


def tune_power(laser, meter, row: dict) -> dict:
    """Reduce SOA only as needed; keep GAIN at the 135 mA high-power rail."""
    codes = list(row["codes"])
    history = []
    best = None
    for iteration in range(6):
        reading = take_reading(laser, meter, codes, row["selection_wavelength_nm"],
                               settle_s=0.30 if iteration == 0 else 0.22,
                               select_main_peak=True)
        sample = {"codes": list(codes), "reading": asdict(reading)}
        history.append(sample)
        same_branch = abs(reading.wavelength_nm - row["selection_wavelength_nm"]) <= 0.10
        clean_mode = (reading.side_mode_suppression_db is not None
                      and reading.side_mode_suppression_db >= MIN_SMSR_DB)
        if same_branch and clean_mode and (best is None or
                abs(reading.power_mw - TARGET_POWER_MW) <
                abs(best["reading"]["power_mw"] - TARGET_POWER_MW)):
            best = sample
        if (same_branch and clean_mode
                and abs(reading.power_mw - TARGET_POWER_MW) <= POWER_TOLERANCE_MW):
            break
        if not same_branch or not clean_mode or reading.power_mw < 0.8:
            break
        soa_ma = code_to_current(codes[1], 1)
        requested = soa_ma * TARGET_POWER_MW / max(reading.power_mw, 0.2)
        requested = min(135.0, max(70.0, requested))
        if abs(requested - soa_ma) < 0.05:
            break
        codes[1] = current_to_code(requested, 1)
    if best is None:
        raise RuntimeError(f"No clean power-tuning state at row {row['index']}")
    return {
        **row,
        "codes": list(best["codes"]),
        "currents_ma": [code_to_current(code, channel)
                        for channel, code in enumerate(best["codes"])],
        "calibration": dict(best["reading"]),
        "history": history,
    }


def tune_from_choices(laser, meter, choices: list[dict]) -> dict:
    """Try nearby optical branches until one is independently reproducible."""
    failures = []
    for choice_number, row in enumerate(choices):
        try:
            calibrated = tune_power(laser, meter, row)
            # A second immediate re-application must reproduce the same branch;
            # this rejects discovery-only and transition-path-dependent modes.
            check = take_reading(
                laser, meter, calibrated["codes"],
                calibrated["calibration"]["wavelength_nm"],
                settle_s=0.26, select_main_peak=True,
            )
            same = abs(check.wavelength_nm - calibrated["calibration"]["wavelength_nm"]) <= 0.010
            clean_mode = (check.side_mode_suppression_db is not None
                          and check.side_mode_suppression_db >= MIN_SMSR_DB)
            power_ok = abs(check.power_mw - TARGET_POWER_MW) <= POWER_TOLERANCE_MW
            if same and clean_mode and power_ok:
                calibrated["choice_number"] = choice_number
                calibrated["confirmation"] = asdict(check)
                return calibrated
            failures.append({"choice": choice_number, "reason": "confirmation_failed",
                             "confirmation": asdict(check)})
        except RuntimeError as exc:
            failures.append({"choice": choice_number, "reason": str(exc)})
    raise RuntimeError(
        f"No reproducible branch at row {choices[0]['index']}: {failures}"
    )


def replay(laser, meter, row: dict, repeats: int) -> dict:
    samples = []
    for repeat in range(repeats):
        samples.append(asdict(take_reading(
            laser, meter, row["codes"], row["calibration"]["wavelength_nm"],
            settle_s=0.30 if repeat == 0 else 0.22, select_main_peak=True,
        )))
    waves = [float(x["wavelength_nm"]) for x in samples]
    powers = [float(x["power_mw"]) for x in samples]
    # AQ6150B reports no secondary peak (and therefore no SMSR value) when
    # exactly one peak is detected.  That is a cleaner result than any finite
    # SMSR and must not be treated as a double-peak failure.
    smsrs = [
        (99.0 if int(x.get("peak_count", 0)) == 1
         and x.get("side_mode_suppression_db") is None
         else float(x["side_mode_suppression_db"] or -999.0))
        for x in samples
    ]
    reference = float(row["calibration"]["wavelength_nm"])
    passed = bool(
        max(abs(x - reference) for x in waves) <= 0.010
        and max(waves) - min(waves) <= 0.006
        and min(smsrs) >= MIN_SMSR_DB
        and min(powers) >= TARGET_POWER_MW - POWER_TOLERANCE_MW
        and max(powers) <= TARGET_POWER_MW + POWER_TOLERANCE_MW
    )
    return {
        "index": row["index"], "segment": row["segment"], "passed": passed,
        "samples": samples, "wavelength_span_pm": (max(waves)-min(waves))*1000.0,
        "max_abs_from_calibration_pm": max(abs(x-reference) for x in waves)*1000.0,
        "power_min_mw": min(powers), "power_max_mw": max(powers),
        "smsr_min_db": min(smsrs),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="COM6")
    p.add_argument("--gpib", default="GPIB0::7::INSTR")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--resume-calibration", action="store_true")
    p.add_argument("--start", type=int, default=0)
    args = p.parse_args()
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    selected = ranked_selections(candidates)
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "fresh_135mA_single_mode_high_power_stress_scan",
        "old_69_wavelengths_used": False,
        "target_power_mw": TARGET_POWER_MW,
        "power_tolerance_mw": POWER_TOLERANCE_MW,
        "minimum_smsr_db": MIN_SMSR_DB,
        "limits_ma": {"gain": 135.0, "soa": 135.0, "phase": 10.0,
                      "wave_a": 30.0, "wave_b": 30.0},
        "meter": candidates.get("meter"), "rows": [],
    }
    if args.resume_calibration and OUTPUT.exists():
        old = json.loads(OUTPUT.read_text(encoding="utf-8"))
        payload["rows"] = old.get("rows", [])
    old_rows = parse_dac_rows(DAC_SOURCE)
    park = list(old_rows[44])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(80.0, 1)
    laser, meter = LaserSerial(args.port), AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        if args.start > 0:
            payload["rows"] = payload["rows"][:args.start]
        for index in range(len(payload["rows"]), len(selected)):
            calibrated = tune_from_choices(laser, meter, selected[index])
            payload["rows"].append(calibrated)
            save_json(OUTPUT, payload)
            reading = calibrated["calibration"]
            print(f"cal {index+1:02d}/{len(selected)} wave={reading['wavelength_nm']:.6f} "
                  f"power={reading['power_mw']:.4f} SMSR={reading['side_mode_suppression_db']:.2f} "
                  f"SOA={calibrated['currents_ma'][1]:.2f}", flush=True)
        validation = {"created": datetime.now().isoformat(timespec="seconds"),
                      "source": OUTPUT.name, "repeats": args.repeats, "results": []}
        for index, row in enumerate(payload["rows"]):
            result = replay(laser, meter, row, args.repeats)
            validation["results"].append(result)
            save_json(VALIDATION, validation)
            print(f"val {index+1:02d}/{len(payload['rows'])} pass={result['passed']} "
                  f"span={result['wavelength_span_pm']:.2f}pm "
                  f"power={result['power_min_mw']:.3f}..{result['power_max_mw']:.3f} "
                  f"SMSR={result['smsr_min_db']:.1f}", flush=True)
    finally:
        try:
            laser.set_codes(park); time.sleep(0.08)
        finally:
            laser.close(); meter.close()
    good = sum(bool(x["passed"]) for x in validation["results"])
    validation["summary"] = {
        "tested": len(validation["results"]), "passed": good,
        "failed": len(validation["results"])-good,
        "power_min_mw": min(x["power_min_mw"] for x in validation["results"]),
        "power_max_mw": max(x["power_max_mw"] for x in validation["results"]),
        "minimum_smsr_db": min(x["smsr_min_db"] for x in validation["results"]),
        "max_wavelength_span_pm": max(x["wavelength_span_pm"] for x in validation["results"]),
    }
    save_json(VALIDATION, validation)
    print(json.dumps(validation["summary"], ensure_ascii=False), flush=True)
    return 0 if good == len(validation["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
