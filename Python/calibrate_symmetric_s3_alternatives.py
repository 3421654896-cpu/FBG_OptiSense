"""Explore multiple laser branches for a symmetric stress FBG-3 design.

This is intentionally a proposal-only tool.  It does not edit the installed
Python/YAML tables or firmware.  Every alternative is wavelength/power/SMSR
qualified with the AQ6150B and is then measured on CH1 in EXTRA mode.  A joint
search chooses one state per wavelength using the measured reflection shape,
not merely wavelength proximity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from calibrate_shape_selected_uniform_power import calibrate_row as equalize_row
from calibrate_symmetric_stress_segments import (
    CURRENT_TABLE,
    MIN_SMSR_DB,
    REFERENCE_CENTERS_NM,
    SYMMETRIC_OFFSETS_NM,
    load_seed_pool,
    tune_seed_to_wavelength,
)
from fbg_peak_fitting import fit_fbg_segment
from laser_power_calibration import AQ6150B, LaserSerial, code_to_current, save_json
from validate_rc_against_single import ADC_CODES, ADC_VREF, collect_single_rt


HERE = Path(__file__).resolve().parent
SOURCE_PROPOSAL = HERE / "stress_symmetric_segments_3_7_proposal.json"
DEFAULT_OUTPUT = HERE / "stress_symmetric_fbg3_branch_search.json"
SEGMENT = 2
TARGETS_NM = [
    REFERENCE_CENTERS_NM[SEGMENT] + offset
    for offset in SYMMETRIC_OFFSETS_NM[SEGMENT]
]


def branch_signature(codes: list[int]) -> str:
    """Identify gain, PHASE cycle and both WAVE controls.

    PHASE is periodic: two states with the same GAIN/WAVE controls can reach
    the same wavelength on different phase cycles and can produce measurably
    different FBG reflection.  They must therefore remain separate choices.
    """
    return "/".join(
        f"{code_to_current(int(codes[index]), index):.2f}"
        for index in (0, 2, 3, 4)
    )


def measure_reflection(
    laser: LaserSerial, predecessor_codes: list[int], target_codes: list[int]
) -> dict:
    # EXTRA mode forces the board fan to its physical maximum.  The identical
    # predecessor for all alternatives at a target makes their comparison fair.
    laser.set_codes(predecessor_codes)
    collect_single_rt(laser.serial, 0.060)
    laser.set_codes(target_codes)
    samples = collect_single_rt(laser.serial, 0.32)
    if len(samples) < 8:
        raise RuntimeError(f"only {len(samples)} EXTRA monitor frames")
    stable = np.asarray(samples[-min(80, len(samples)):], dtype=float)
    return {
        "adc_voltage_v": (
            np.median(stable, axis=0) * ADC_VREF / ADC_CODES
        ).tolist(),
        "adc_span_v": (np.ptp(stable, axis=0) * ADC_VREF / ADC_CODES).tolist(),
        "telemetry_frames": int(len(samples)),
    }


def compact_calibrated(row: dict, signature: str, reflection: dict) -> dict:
    return {
        "target_nm": float(row["target_nm"]),
        "final_wavelength_nm": float(row["final_wavelength_nm"]),
        "final_wavelength_error_pm": float(row["final_wavelength_error_pm"]),
        "final_power_mw": float(row["final_power_mw"]),
        "final_power_min_mw": float(row["final_power_min_mw"]),
        "final_power_max_mw": float(row["final_power_max_mw"]),
        "final_smsr_min_db": float(row["final_smsr_min_db"]),
        "codes": [int(value) for value in row["codes"]],
        "currents_ma": [
            code_to_current(int(code), index)
            for index, code in enumerate(row["codes"])
        ],
        "branch_signature": signature,
        "reflection": reflection,
    }


def candidate_shape_score(candidates: tuple[dict, ...]) -> tuple[float, dict]:
    wavelengths = np.asarray(
        [float(item["final_wavelength_nm"]) for item in candidates], dtype=float
    )
    signal = np.asarray(
        [float(item["reflection"]["adc_voltage_v"][1]) for item in candidates],
        dtype=float,
    )
    peak_index = int(np.argmax(signal))
    left_violation = sum(
        max(0.0, float(signal[index] - signal[index + 1]))
        for index in range(peak_index)
    )
    right_violation = sum(
        max(0.0, float(signal[index + 1] - signal[index]))
        for index in range(peak_index, len(signal) - 1)
    )
    monotonic_violation = left_violation + right_violation
    symmetry_error = sum(
        abs(float(signal[left] - signal[right]))
        for left, right in ((0, 6), (1, 5), (2, 4))
    )
    amplitude = float(np.max(signal) - np.min(signal))
    fit = fit_fbg_segment(
        wavelengths, signal, min_prominence_v=0.02, allow_edge_peak=True
    )
    center_error_pm = abs(
        (float(fit.center_nm) - REFERENCE_CENTERS_NM[SEGMENT]) * 1000.0
    )
    r_squared = float(fit.r_squared) if math.isfinite(fit.r_squared) else -1.0
    center_std_pm = (
        float(fit.center_std_pm) if math.isfinite(fit.center_std_pm) else 999.0
    )
    branch_changes = sum(
        candidates[index]["branch_signature"]
        != candidates[index - 1]["branch_signature"]
        for index in range(1, len(candidates))
    )

    # Pair symmetry and monotonicity are direct requirements.  Fit terms keep
    # the chosen peak physically centred and reject a merely pretty polyline.
    amplitude_penalty = (0.05 - amplitude) / 0.002 if amplitude < 0.05 else 0.0
    score = (
        monotonic_violation / 0.004
        + symmetry_error / 0.012
        + center_error_pm / 8.0
        + max(0.0, 0.96 - r_squared) * 35.0
        + center_std_pm / 8.0
        + abs(peak_index - 3) * 3.0
        + branch_changes * 0.12
        + amplitude_penalty
    )
    return score, {
        "score": float(score),
        "ch1_voltage_v": signal.tolist(),
        "peak_index": peak_index,
        "monotonic_violation_v": float(monotonic_violation),
        "pair_symmetry_error_v": float(symmetry_error),
        "amplitude_v": amplitude,
        "fit_center_nm": float(fit.center_nm),
        "fit_center_error_pm": float(
            (fit.center_nm - REFERENCE_CENTERS_NM[SEGMENT]) * 1000.0
        ),
        "fit_r_squared": r_squared,
        "fit_center_std_pm": center_std_pm,
        "fit_valid": bool(fit.valid),
        "branch_changes": int(branch_changes),
    }


def choose_joint(candidates_by_target: list[list[dict]]) -> tuple[list[dict], dict]:
    best = None
    for combination in itertools.product(*candidates_by_target):
        score, diagnostics = candidate_shape_score(combination)
        if best is None or score < best[0]:
            best = (score, list(combination), diagnostics)
    if best is None:
        raise RuntimeError("no complete candidate combination")
    return best[1], best[2]


def replay_selected(laser: LaserSerial, rows: list[dict], repeats: int = 4) -> dict:
    traces = []
    fits = []
    for _ in range(repeats):
        predecessor = rows[-1]["codes"]
        trace = []
        for row in rows:
            reflection = measure_reflection(laser, predecessor, row["codes"])
            trace.append(float(reflection["adc_voltage_v"][1]))
            predecessor = row["codes"]
        wavelengths = np.asarray(
            [float(row["final_wavelength_nm"]) for row in rows], dtype=float
        )
        fit = fit_fbg_segment(
            wavelengths, np.asarray(trace), min_prominence_v=0.02,
            allow_edge_peak=True,
        )
        traces.append(trace)
        fits.append({
            "center_nm": float(fit.center_nm),
            "center_error_pm": float(
                (fit.center_nm - REFERENCE_CENTERS_NM[SEGMENT]) * 1000.0
            ),
            "r_squared": float(fit.r_squared),
            "center_std_pm": float(fit.center_std_pm),
            "valid": bool(fit.valid),
        })
    centers = np.asarray([item["center_nm"] for item in fits], dtype=float)
    r2 = np.asarray([item["r_squared"] for item in fits], dtype=float)
    return {
        "repeats": repeats,
        "ch1_voltage_traces_v": traces,
        "fits": fits,
        "center_repeat_std_pm": float(np.std(centers) * 1000.0),
        "minimum_r_squared": float(np.min(r2)),
        "all_valid": bool(all(item["valid"] for item in fits)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alternatives", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.alternatives < 2:
        raise SystemExit("--alternatives must be at least 2")

    source = json.loads(SOURCE_PROPOSAL.read_text(encoding="utf-8"))
    installed = json.loads(CURRENT_TABLE.read_text(encoding="utf-8"))
    baseline_rows = [
        row for row in source["rows"] if int(row["segment"]) == SEGMENT
    ]
    if len(baseline_rows) != 7:
        raise RuntimeError("source proposal does not contain seven FBG-3 rows")
    seeds = load_seed_pool()
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "FBG-3 symmetric multi-branch reflection search",
        "installed_table_modified": False,
        "source": SOURCE_PROPOSAL.name,
        "reference_center_nm": REFERENCE_CENTERS_NM[SEGMENT],
        "targets_nm": TARGETS_NM,
        "alternatives_requested": args.alternatives,
        "minimum_smsr_db": MIN_SMSR_DB,
        "targets": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        payload["targets"] = previous.get("targets", [])

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    park = [int(value) for value in installed["rows"][0]["codes"]]
    try:
        for index, target_nm in enumerate(TARGETS_NM):
            if index < len(payload["targets"]):
                entry = payload["targets"][index]
            else:
                entry = {"index": index, "target_nm": target_nm, "candidates": []}
                payload["targets"].append(entry)

            predecessor = baseline_rows[index - 1 if index else -1]["codes"]
            signatures = {
                item["branch_signature"] for item in entry["candidates"]
            }
            code_keys = {
                tuple(int(value) for value in item["codes"])
                for item in entry["candidates"]
            }
            if not entry["candidates"]:
                base = baseline_rows[index]
                reflection = measure_reflection(
                    laser, predecessor, [int(value) for value in base["codes"]]
                )
                entry["candidates"].append(compact_calibrated(
                    base, branch_signature(base["codes"]), reflection
                ))
                signatures.add(branch_signature(base["codes"]))
                code_keys.add(tuple(int(value) for value in base["codes"]))
                save_json(args.output, payload)

            ranked = sorted(
                seeds,
                key=lambda seed: (
                    abs(float(seed["wavelength_nm"]) - target_nm),
                    -float(seed["smsr_db"]),
                    -float(seed["power_mw"]),
                ),
            )
            tried_signatures = set(signatures)
            for seed in ranked[:32]:
                if len(entry["candidates"]) >= args.alternatives:
                    break
                signature = branch_signature(seed["codes"])
                if signature in tried_signatures:
                    continue
                tried_signatures.add(signature)
                print(
                    f"S3 point {index + 1}/7 {target_nm:.6f}: "
                    f"testing branch {signature}", flush=True,
                )
                tuned = tune_seed_to_wavelength(laser, meter, seed, target_nm)
                if not tuned["success"]:
                    continue
                try:
                    row = equalize_row(laser, meter, {
                        "candidate_wavelength_nm": target_nm,
                        "codes": list(tuned["codes"]),
                    })
                except RuntimeError:
                    continue
                code_key = tuple(int(value) for value in row["codes"])
                if (
                    not bool(row["passed"])
                    or float(row["final_smsr_min_db"]) < MIN_SMSR_DB
                    or code_key in code_keys
                ):
                    continue
                row["target_nm"] = target_nm
                reflection = measure_reflection(laser, predecessor, row["codes"])
                item = compact_calibrated(row, signature, reflection)
                entry["candidates"].append(item)
                signatures.add(signature)
                code_keys.add(code_key)
                save_json(args.output, payload)
                print(
                    f"  accepted wave={item['final_wavelength_nm']:.6f} "
                    f"power={item['final_power_mw']:.3f}mW "
                    f"SMSR>={item['final_smsr_min_db']:.1f}dB "
                    f"CH1={item['reflection']['adc_voltage_v'][1]:.4f}V",
                    flush=True,
                )
            if len(entry["candidates"]) < args.alternatives:
                raise RuntimeError(
                    f"target {target_nm:.6f}: only {len(entry['candidates'])} "
                    "qualified distinct branches"
                )

        selected, diagnostics = choose_joint(
            [entry["candidates"] for entry in payload["targets"]]
        )
        payload["selection"] = {
            "rows": selected,
            "candidate_shape": diagnostics,
        }
        payload["sequential_replay"] = replay_selected(laser, selected)
        replay = payload["sequential_replay"]
        payload["accepted_for_merge"] = bool(
            diagnostics["monotonic_violation_v"] <= 0.006
            and diagnostics["pair_symmetry_error_v"] <= 0.055
            and abs(diagnostics["fit_center_error_pm"]) <= 25.0
            and diagnostics["fit_r_squared"] >= 0.90
            and replay["all_valid"]
            and replay["minimum_r_squared"] >= 0.88
            and replay["center_repeat_std_pm"] <= 8.0
        )
        save_json(args.output, payload)
        print(json.dumps({
            "candidate_shape": diagnostics,
            "sequential_replay": replay,
            "accepted_for_merge": payload["accepted_for_merge"],
        }, ensure_ascii=False, indent=2), flush=True)
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()
    return 0 if payload.get("accepted_for_merge") else 2


if __name__ == "__main__":
    raise SystemExit(main())
