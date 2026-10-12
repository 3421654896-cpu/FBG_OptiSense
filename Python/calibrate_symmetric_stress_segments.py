"""Live-calibrate symmetric seven-point designs for stress FBG 3 and FBG 7.

The output is deliberately a proposal.  It never edits the installed stress
table or firmware.  Each laser state is tuned with the AQ6150B, equalised to
1.50 mW, confirmed twice, and then replayed in EXTRA mode to capture settled
CH1 reflection voltage before it can be considered for installation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from calibrate_shape_selected_uniform_power import calibrate_row as equalize_row
from fbg_peak_fitting import fit_fbg_segment
from laser_power_calibration import (
    AQ6150B,
    LaserSerial,
    code_to_current,
    current_to_code,
    save_json,
    take_reading,
)
from validate_rc_against_single import ADC_CODES, ADC_VREF, collect_single_rt


HERE = Path(__file__).resolve().parent
HIGH_MAPS = (
    HERE / "high_power_tuning_map_gain135_soa135.json",
    HERE / "high_power_tuning_map_mainpeak_v2.json",
    HERE / "high_power_tuning_map_gain105.json",
    HERE / "high_power_tuning_map_gain90.json",
)
PHASE_MAP = HERE / "single_mode_candidates_135.json"
CURRENT_TABLE = HERE / "stress_shape_power_dynamic_v3.json"
DEFAULT_OUTPUT = HERE / "stress_symmetric_segments_3_7_proposal.json"

# Static centers from the user's broadband interrogator.  The current sparse
# scan is intentionally not used as the center prior because its asymmetry is
# the defect being corrected.
REFERENCE_CENTERS_NM = {2: 1536.3851, 6: 1551.7849}
# Use the widest exactly symmetric positions that remain inside the clean
# high-power longitudinal branch measured at GAIN 90/105/120/135 mA.
SYMMETRIC_OFFSETS_NM = {
    # FBG 3 is about 1.03 nm FWHM.  The former +/-0.20 nm design sampled only
    # its flat crown and its rightmost state landed on a measured laser-state
    # response bump.  +/-0.25 nm reaches the real descending shoulder while
    # remaining inside the clean branch pool.
    2: (-0.25, -0.13, -0.065, 0.0, 0.065, 0.13, 0.25),
    6: (-0.39, -0.26, -0.13, 0.0, 0.13, 0.26, 0.39),
}

TARGET_POWER_MW = 1.50
MIN_SMSR_DB = 25.0
TUNE_TOLERANCE_PM = 3.0
DEFAULT_PHASE_SLOPE_PM_PER_MA = -75.0
MAX_PHASE_STEP_MA = 0.60


def _smsr(reading: dict) -> float:
    if int(reading.get("peak_count", 0)) == 1 and reading.get(
        "side_mode_suppression_db"
    ) is None:
        return 99.0
    return float(reading.get("side_mode_suppression_db") or -999.0)


def load_seed_pool() -> list[dict]:
    seeds: list[dict] = []
    for source in HIGH_MAPS:
        high = json.loads(source.read_text(encoding="utf-8"))
        for item in high["results"].values():
            reading = dict(item["reading"])
            seeds.append({
                "source": source.name,
                "wavelength_nm": float(reading["wavelength_nm"]),
                "power_mw": float(reading["power_mw"]),
                "smsr_db": _smsr(reading),
                "codes": [int(value) for value in item["codes"]],
            })

    phase = json.loads(PHASE_MAP.read_text(encoding="utf-8"))
    phase_rows = list(phase.get("base_candidates", []))
    phase_rows.extend(
        row for row in phase.get("phase_scans", {}).values()
        if bool(row.get("accepted"))
    )
    for item in phase_rows:
        reading = dict(item["reading"])
        seeds.append({
            "source": PHASE_MAP.name,
            "wavelength_nm": float(reading["wavelength_nm"]),
            "power_mw": float(reading["power_mw"]),
            "smsr_db": _smsr(reading),
            "codes": [int(value) for value in item["codes"]],
        })

    current = json.loads(CURRENT_TABLE.read_text(encoding="utf-8"))
    for item in current["rows"]:
        seeds.append({
            "source": CURRENT_TABLE.name,
            "wavelength_nm": float(item["wavelength_nm"]),
            "power_mw": float(item["power_mw"]),
            "smsr_db": float(item.get("smsr_min_db", 99.0)),
            "codes": [int(value) for value in item["codes"]],
        })

    unique: dict[tuple[int, ...], dict] = {}
    for seed in seeds:
        if seed["power_mw"] < 0.70 or seed["smsr_db"] < 15.0:
            continue
        key = tuple(seed["codes"])
        previous = unique.get(key)
        if previous is None or seed["smsr_db"] > previous["smsr_db"]:
            unique[key] = seed
    return list(unique.values())


def tune_seed_to_wavelength(laser, meter, seed: dict, target_nm: float) -> dict:
    codes = list(seed["codes"])
    # Preserve the seed's measured GAIN branch.  Lower-GAIN maps are needed at
    # the extreme left edges where the 135 mA branch does not remain clean;
    # SOA equalisation below may still use the full authorised 135 mA range.
    phase_ma = code_to_current(codes[2], 2)
    slope = DEFAULT_PHASE_SLOPE_PM_PER_MA
    previous = None
    history = []
    best = None

    for iteration in range(15):
        reading = take_reading(
            laser, meter, codes, target_nm,
            settle_s=0.32 if iteration == 0 else 0.23,
            select_main_peak=True,
        )
        sample = {
            "iteration": iteration,
            "phase_ma": phase_ma,
            "codes": list(codes),
            "reading": asdict(reading),
        }
        history.append(sample)
        clean = (
            math.isfinite(reading.wavelength_nm)
            and abs(reading.wavelength_nm - target_nm) <= 1.20
            and reading.power_mw >= 0.70
            and (
                reading.side_mode_suppression_db is None
                or reading.side_mode_suppression_db >= 15.0
            )
        )
        if not clean:
            break
        if best is None or abs(reading.wavelength_error_pm) < abs(
            best["reading"]["wavelength_error_pm"]
        ):
            best = sample
        if abs(reading.wavelength_error_pm) <= TUNE_TOLERANCE_PM:
            return {"success": True, "codes": list(codes), "history": history}

        if previous is not None:
            delta_phase = phase_ma - float(previous["phase_ma"])
            delta_error = reading.wavelength_error_pm - float(
                previous["reading"]["wavelength_error_pm"]
            )
            if abs(delta_phase) >= 0.015:
                measured_slope = delta_error / delta_phase
                if 8.0 <= abs(measured_slope) <= 600.0:
                    slope = measured_slope
        previous = sample
        requested = phase_ma - reading.wavelength_error_pm / slope
        requested = min(
            max(requested, phase_ma - MAX_PHASE_STEP_MA),
            phase_ma + MAX_PHASE_STEP_MA,
        )
        requested = min(max(requested, 0.0), 10.0)
        if abs(requested - phase_ma) < 0.003:
            break
        phase_ma = requested
        codes[2] = current_to_code(phase_ma, 2)

    return {"success": False, "best": best, "history": history}


def calibrate_target(laser, meter, seeds, target_nm: float, used_codes: set) -> dict:
    ranked = sorted(
        seeds,
        key=lambda seed: (
            abs(float(seed["wavelength_nm"]) - target_nm),
            -float(seed["smsr_db"]),
            -float(seed["power_mw"]),
        ),
    )
    attempts = []
    for seed in ranked[:14]:
        tuned = tune_seed_to_wavelength(laser, meter, seed, target_nm)
        attempts.append({"seed": seed, "tuning": tuned})
        if not tuned["success"]:
            continue
        proposal = {
            "candidate_wavelength_nm": target_nm,
            "codes": list(tuned["codes"]),
        }
        try:
            calibrated = equalize_row(laser, meter, proposal)
        except RuntimeError as exc:
            attempts[-1]["equalization_error"] = str(exc)
            continue
        code_key = tuple(int(value) for value in calibrated["codes"])
        clean = bool(
            calibrated["passed"]
            and float(calibrated["final_smsr_min_db"]) >= MIN_SMSR_DB
            and code_key not in used_codes
        )
        attempts[-1]["equalized_summary"] = {
            "passed": bool(calibrated["passed"]),
            "final_wavelength_nm": float(calibrated["final_wavelength_nm"]),
            "final_wavelength_error_pm": float(
                calibrated["final_wavelength_error_pm"]
            ),
            "final_power_mw": float(calibrated["final_power_mw"]),
            "final_smsr_min_db": float(calibrated["final_smsr_min_db"]),
            "codes": [int(value) for value in calibrated["codes"]],
        }
        if not clean:
            continue
        used_codes.add(code_key)
        calibrated["target_nm"] = target_nm
        calibrated["attempts"] = attempts
        return calibrated
    raise RuntimeError(f"no confirmed clean 1.50 mW state for {target_nm:.6f} nm")


def capture_settled_reflection(laser: LaserSerial, rows: list[dict]) -> None:
    for segment in sorted({int(row["segment"]) for row in rows}):
        section = [row for row in rows if int(row["segment"]) == segment]
        predecessor = section[-1]
        for row in section:
            laser.set_codes(predecessor["codes"])
            collect_single_rt(laser.serial, 0.055)
            laser.set_codes(row["codes"])
            samples = collect_single_rt(laser.serial, 0.30)
            if len(samples) < 8:
                raise RuntimeError(
                    f"segment {segment + 1} {row['target_nm']:.6f}: "
                    f"only {len(samples)} EXTRA monitor frames"
                )
            stable = np.asarray(samples[-min(80, len(samples)):], dtype=float)
            row["settled_adc_v"] = (
                np.median(stable, axis=0) * ADC_VREF / ADC_CODES
            ).tolist()
            row["settled_span_v"] = (
                np.ptp(stable, axis=0) * ADC_VREF / ADC_CODES
            ).tolist()
            predecessor = row


def segment_diagnostics(rows: list[dict]) -> list[dict]:
    diagnostics = []
    for segment, center in REFERENCE_CENTERS_NM.items():
        section = [row for row in rows if int(row["segment"]) == segment]
        if not section:
            continue
        if len(section) != 7:
            raise RuntimeError(
                f"segment {segment + 1}: expected seven rows, got {len(section)}"
            )
        wavelengths = np.asarray(
            [float(row["final_wavelength_nm"]) for row in section], dtype=float
        )
        ch1 = np.asarray(
            [float(row["settled_adc_v"][1]) for row in section], dtype=float
        )
        fit = fit_fbg_segment(
            wavelengths, ch1, min_prominence_v=0.02, allow_edge_peak=True
        )
        pair_distance_error_pm = [
            abs(
                (wavelengths[3] - wavelengths[left])
                - (wavelengths[right] - wavelengths[3])
            ) * 1000.0
            for left, right in ((0, 6), (1, 5), (2, 4))
        ]
        diagnostics.append({
            "segment": segment,
            "reference_center_nm": center,
            "settled_fit_center_nm": float(fit.center_nm),
            "settled_fit_center_error_pm": float((fit.center_nm - center) * 1000.0),
            "settled_fit_r_squared": float(fit.r_squared),
            "settled_fit_center_std_pm": float(fit.center_std_pm),
            "settled_fit_valid": bool(fit.valid),
            "maximum_pair_distance_error_pm": max(pair_distance_error_pm),
            "ch1_voltage_v": ch1.tolist(),
        })
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--segments",
        default="3,7",
        help="comma-separated one-based FBG numbers to calibrate (default: 3,7)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_segments = {
        int(value.strip()) - 1
        for value in args.segments.split(",")
        if value.strip()
    }
    invalid_segments = selected_segments.difference(REFERENCE_CENTERS_NM)
    if not selected_segments or invalid_segments:
        raise SystemExit(f"unsupported --segments value: {args.segments}")
    targets = [
        {"segment": segment, "target_nm": center + offset}
        for segment, center in REFERENCE_CENTERS_NM.items()
        if segment in selected_segments
        for offset in SYMMETRIC_OFFSETS_NM[segment]
    ]
    seeds = load_seed_pool()
    if args.dry_run:
        for target in targets:
            nearest = min(seeds, key=lambda seed: abs(
                float(seed["wavelength_nm"]) - target["target_nm"]
            ))
            print(
                f"S{target['segment'] + 1} target={target['target_nm']:.6f} "
                f"seed={nearest['wavelength_nm']:.6f} "
                f"delta={(nearest['wavelength_nm'] - target['target_nm']) * 1000:+.1f} pm"
            )
        return 0

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "symmetric replacement proposal for stress FBG 3 and FBG 7",
        "installed_table_modified": False,
        "reference_centers_nm": REFERENCE_CENTERS_NM,
        "offsets_nm": {
            str(segment): list(offsets)
            for segment, offsets in SYMMETRIC_OFFSETS_NM.items()
        },
        "target_power_mw": TARGET_POWER_MW,
        "minimum_smsr_db": MIN_SMSR_DB,
        "limits_ma": {
            "gain": 135.0, "soa": 135.0, "phase": 10.0,
            "wave_a": 30.0, "wave_b": 30.0,
        },
        "meter": None,
        "rows": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        payload["rows"] = previous.get("rows", [])

    used_codes = {tuple(int(value) for value in row["codes"])
                  for row in payload["rows"]}
    current = json.loads(CURRENT_TABLE.read_text(encoding="utf-8"))
    park = [int(value) for value in current["rows"][0]["codes"]]
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for index in range(len(payload["rows"]), len(targets)):
            target = targets[index]
            calibrated = calibrate_target(
                laser, meter, seeds, float(target["target_nm"]), used_codes
            )
            calibrated["index"] = index
            calibrated["segment"] = int(target["segment"])
            payload["rows"].append(calibrated)
            save_json(args.output, payload)
            print(
                f"[{index + 1:02d}/{len(targets):02d}] "
                f"S{target['segment'] + 1} "
                f"target={target['target_nm']:.6f} "
                f"actual={calibrated['final_wavelength_nm']:.6f} "
                f"power={calibrated['final_power_mw']:.3f}mW "
                f"SMSR>={calibrated['final_smsr_min_db']:.1f}dB",
                flush=True,
            )
        capture_settled_reflection(laser, payload["rows"])
        payload["segments"] = segment_diagnostics(payload["rows"])
        payload["summary"] = {
            "passed_points": sum(bool(row["passed"]) for row in payload["rows"]),
            "point_count": len(payload["rows"]),
            "power_min_mw": min(float(row["final_power_min_mw"])
                                for row in payload["rows"]),
            "power_max_mw": max(float(row["final_power_max_mw"])
                                for row in payload["rows"]),
            "minimum_smsr_db": min(float(row["final_smsr_min_db"])
                                   for row in payload["rows"]),
            "maximum_wavelength_error_pm": max(
                abs(float(row["final_wavelength_error_pm"]))
                for row in payload["rows"]
            ),
        }
        save_json(args.output, payload)
        print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
        for item in payload["segments"]:
            print(json.dumps(item, ensure_ascii=False), flush=True)
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
