"""Track the 1539.120-nm longitudinal mode while reducing SOA power.

This is a deliberately narrow recovery tool.  It never edits the operational
2001-point table.  Every exit path sends the PI11210 SOA shutter command and
records whether the acknowledgement was received.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import app_JDSU as app
import fullband_equal_power_model as model
from audit_fullband_2001_lan import wait_connected
from diagnose_adaptive_stress_peaks_live import _shutter
from fbg_lan_serial import FbgLanSerial
from laser_power_calibration import LaserSerial, Reading, save_json
from optimize_fullband_anomalies_lan import (
    LanLaserAdapter,
    RetryingMeter,
    find_point_index,
    reading_passes,
    smsr_db,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs" / "fullband_1539_soa_continuation_20260904.json"
TARGET_NM = 1539.120
TARGET_POWER_MW = 1.530
HIGH_SOA_SEED = (58981, 58981, 14737, 3651, 2458)

# These two slopes are the measured trajectory that originally connected the
# 135-mA clean state to the equal-power state.  They are feed-forward only;
# every step is checked by the wavelength meter before it is accepted.
PHASE_MA_PER_SOA_MA = 0.01187
WAVE_A_MA_PER_SOA_MA = -0.00111
INITIAL_PHASE_WAVELENGTH_SLOPE_NM_PER_MA = -0.055


def measure_after(laser, meter, codes, target_nm: float, settle_s: float) -> Reading:
    laser.set_codes(codes)
    time.sleep(settle_s)
    return meter.measure(target_nm, select_main_peak=True)


def set_predecessors(laser, points, index: int, hold_s: float) -> None:
    for predecessor_index in range(max(0, index - 3), index):
        laser.set_codes(points[predecessor_index].codes)
        time.sleep(hold_s)


def stable_branch(reading: Reading, previous: Reading | None = None) -> bool:
    if smsr_db(reading) < 20.0:
        return False
    if abs(reading.wavelength_nm - TARGET_NM) > 0.040:
        return False
    if previous is not None and abs(reading.wavelength_nm - previous.wavelength_nm) > 0.040:
        return False
    return True


def correct_phase(
    laser,
    meter,
    codes: list[int],
    reading: Reading,
    settle_s: float,
    phase_slope: float,
) -> tuple[list[int], Reading, float, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    for correction_index in range(4):
        error_nm = reading.wavelength_nm - TARGET_NM
        if abs(error_nm) <= 0.00065:
            break
        requested_delta = -error_nm / phase_slope
        requested_delta = min(max(requested_delta, -0.018), 0.018)
        old_phase = model.code_to_current(codes[2], 2)
        requested_phase = min(max(old_phase + requested_delta, 0.0), 10.0)
        trial_codes = list(codes)
        trial_codes[2] = model.current_to_code(requested_phase, 2)
        if trial_codes == codes:
            break
        trial = measure_after(laser, meter, trial_codes, TARGET_NM, settle_s)
        delta_phase = model.code_to_current(trial_codes[2], 2) - old_phase
        measured_slope = (
            (trial.wavelength_nm - reading.wavelength_nm) / delta_phase
            if abs(delta_phase) >= 0.001 else float("nan")
        )
        improved = stable_branch(trial, reading) and (
            abs(trial.wavelength_nm - TARGET_NM)
            < abs(reading.wavelength_nm - TARGET_NM)
        )
        history.append({
            "action": f"phase_feedback_{correction_index}",
            "accepted": improved,
            "codes": list(trial_codes),
            "reading": asdict(trial),
            "measured_phase_slope_nm_per_ma": measured_slope,
        })
        if not improved:
            laser.set_codes(codes)
            time.sleep(settle_s)
            break
        codes, reading = trial_codes, trial
        if math.isfinite(measured_slope) and -0.20 <= measured_slope <= -0.010:
            phase_slope = 0.65 * measured_slope + 0.35 * phase_slope
    return codes, reading, phase_slope, history


def follow_soa_branch(laser, meter, points, index: int, args) -> dict[str, Any]:
    set_predecessors(laser, points, index, args.predecessor_hold_s)
    tuned = model.tune_local_three_controls(
        laser,
        meter,
        list(HIGH_SOA_SEED),
        TARGET_NM,
        args.settle_s,
        max_measurements=12,
    )
    codes = [int(value) for value in tuned["codes"]]
    reading = Reading(**tuned["reading"])
    history: list[dict[str, Any]] = [{
        "action": "high_soa_wavelength_close",
        "codes": list(codes),
        "reading": asdict(reading),
        "tuning": tuned,
    }]
    if not stable_branch(reading):
        return {"success": False, "reason": "high_soa_seed_not_stable", "history": history}

    phase_slope = INITIAL_PHASE_WAVELENGTH_SLOPE_NM_PER_MA
    accepted_states: list[dict[str, Any]] = [{
        "codes": list(codes), "reading": asdict(reading),
    }]
    observations: list[tuple[float, float]] = [
        (model.code_to_current(codes[1], 1), reading.power_mw)
    ]
    step_limit_ma = args.soa_step_ma

    for step_index in range(args.max_steps):
        wavelength_error_pm = (reading.wavelength_nm - TARGET_NM) * 1000.0
        relative_power_error = reading.power_mw / args.target_power_mw - 1.0
        if (
            abs(wavelength_error_pm) <= 2.0
            and abs(relative_power_error) <= args.power_relative_tolerance
            and smsr_db(reading) >= 20.0
        ):
            return {
                "success": True,
                "codes": list(codes),
                "currents_ma": [
                    model.code_to_current(value, control_index)
                    for control_index, value in enumerate(codes)
                ],
                "reading": asdict(reading),
                "wavelength_error_pm": wavelength_error_pm,
                "power_relative_error": relative_power_error,
                "history": history,
                "accepted_states": accepted_states,
            }

        current_soa = model.code_to_current(codes[1], 1)
        desired_delta = -step_limit_ma if reading.power_mw > args.target_power_mw else step_limit_ma
        if len(observations) >= 2:
            old_soa, old_power = observations[-2]
            new_soa, new_power = observations[-1]
            delta_soa = new_soa - old_soa
            slope = (new_power - old_power) / delta_soa if abs(delta_soa) >= 0.20 else float("nan")
            if math.isfinite(slope) and 0.002 <= slope <= 0.030:
                desired_delta = (args.target_power_mw - reading.power_mw) / slope
                desired_delta = min(max(desired_delta, -step_limit_ma), step_limit_ma)
        if abs(desired_delta) < 0.06:
            desired_delta = math.copysign(0.06, desired_delta)

        requested_soa = min(max(current_soa + desired_delta, 85.0), 135.0)
        proposed = list(codes)
        proposed[1] = model.current_to_code(requested_soa, 1)
        actual_delta_soa = model.code_to_current(proposed[1], 1) - current_soa
        if abs(actual_delta_soa) < 0.01:
            break
        current_phase = model.code_to_current(codes[2], 2)
        current_wave_a = model.code_to_current(codes[3], 3)
        proposed[2] = model.current_to_code(
            min(max(current_phase + PHASE_MA_PER_SOA_MA * actual_delta_soa, 0.0), 10.0), 2
        )
        proposed[3] = model.current_to_code(
            min(max(current_wave_a + WAVE_A_MA_PER_SOA_MA * actual_delta_soa, 0.0), 30.0), 3
        )
        trial = measure_after(laser, meter, proposed, TARGET_NM, args.settle_s)
        accepted = stable_branch(trial, reading)
        history.append({
            "action": f"soa_continuation_{step_index}",
            "accepted": accepted,
            "step_limit_ma": step_limit_ma,
            "actual_delta_soa_ma": actual_delta_soa,
            "codes": list(proposed),
            "reading": asdict(trial),
        })
        if not accepted:
            laser.set_codes(codes)
            time.sleep(args.settle_s)
            step_limit_ma *= 0.5
            print(
                f"  continuation reject step={step_index + 1} "
                f"SOA={model.code_to_current(proposed[1], 1):.3f} "
                f"wave={trial.wavelength_nm:.6f} SMSR={smsr_db(trial):.2f}; "
                f"retry step={step_limit_ma:.3f}mA",
                flush=True,
            )
            if step_limit_ma < 0.055:
                break
            continue

        codes, reading = proposed, trial
        codes, reading, phase_slope, correction_history = correct_phase(
            laser, meter, codes, reading, args.settle_s, phase_slope
        )
        history.extend(correction_history)
        if not stable_branch(reading):
            break
        observations.append((model.code_to_current(codes[1], 1), reading.power_mw))
        accepted_states.append({"codes": list(codes), "reading": asdict(reading)})
        print(
            f"  continuation {step_index + 1}: "
            f"SOA={model.code_to_current(codes[1], 1):.3f}mA "
            f"wave={reading.wavelength_nm:.8f} "
            f"err={(reading.wavelength_nm - TARGET_NM) * 1000.0:+.2f}pm "
            f"power={reading.power_mw:.4f}mW SMSR={smsr_db(reading):.2f}",
            flush=True,
        )

    return {
        "success": False,
        "reason": "continuation_did_not_converge",
        "codes": list(codes),
        "reading": asdict(reading),
        "history": history,
        "accepted_states": accepted_states,
    }


def compressed_stages(states: list[dict[str, Any]], count: int) -> list[list[int]]:
    if count <= 0 or len(states) <= 2:
        return []
    last_source_index = len(states) - 2
    indices = sorted({
        max(0, min(last_source_index, round(i * last_source_index / count)))
        for i in range(count)
    })
    return [list(states[index]["codes"]) for index in indices]


def confirm_path(laser, meter, points, index: int, final_codes, stages, args) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for repeat in range(args.confirmations):
        set_predecessors(laser, points, index, args.predecessor_hold_s)
        for stage in stages:
            laser.set_codes(stage)
            time.sleep(args.guard_hold_s)
        reading = measure_after(laser, meter, final_codes, TARGET_NM, args.settle_s)
        passed = reading_passes(
            reading, TARGET_NM, args.target_power_mw,
            args.power_relative_tolerance, 20.0,
        )
        row = {
            "repeat": repeat + 1,
            "passed": passed,
            "reading": asdict(reading),
            "wavelength_error_pm": (reading.wavelength_nm - TARGET_NM) * 1000.0,
            "power_relative_error_percent":
                (reading.power_mw / args.target_power_mw - 1.0) * 100.0,
            "smsr_db": smsr_db(reading),
        }
        rows.append(row)
        print(
            f"  confirm stages={len(stages)} repeat={repeat + 1} "
            f"err={row['wavelength_error_pm']:+.2f}pm "
            f"power={reading.power_mw:.4f}mW SMSR={row['smsr_db']:.2f} "
            f"ok={int(passed)}",
            flush=True,
        )
    return {"success": all(row["passed"] for row in rows), "stages": stages, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-power-mw", type=float, default=TARGET_POWER_MW)
    parser.add_argument("--power-relative-tolerance", type=float, default=0.005)
    parser.add_argument("--settle-s", type=float, default=0.35)
    parser.add_argument("--predecessor-hold-s", type=float, default=0.25)
    parser.add_argument("--guard-hold-s", type=float, default=0.08)
    parser.add_argument("--soa-step-ma", type=float, default=0.50)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--confirmations", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.10 <= args.soa_step_ma <= 1.0:
        raise SystemExit("SOA步长必须位于0.10–1.0mA")
    LaserSerial.validate_codes(list(HIGH_SOA_SEED))
    points = app.load_fullband_accuracy_table(HERE)
    index = find_point_index(points, TARGET_NM)
    if args.dry_run:
        print(f"dry_run=PASS index={index} points={len(points)}")
        return 0

    payload: dict[str, Any] = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "LAN_SOA_homotopy_with_PHASE_WAVE_feedforward_and_live_meter_feedback",
        "target_nm": TARGET_NM,
        "target_power_mw": args.target_power_mw,
        "limits_ma": [135.0, 135.0, 10.0, 30.0, 30.0],
    }
    device = FbgLanSerial(board_host=args.host, timeout=0.10, write_timeout=3.0)
    worker = app.UnlimitedAccuracyWorker(device, points, {})
    laser = LanLaserAdapter(worker)
    meter: RetryingMeter | None = None
    shutter_confirmed = False
    try:
        wait_connected(device)
        device.write(app.build_work_mode_command(2))
        time.sleep(0.12)
        device.reset_input_buffer()
        worker._rx_buffer.clear()
        meter = RetryingMeter(args.meter)
        payload["meter"] = meter.identity
        result = follow_soa_branch(laser, meter, points, index, args)
        payload["continuation"] = result
        save_json(args.output, payload)
        if not result.get("success"):
            print("continuation_result=FAILED", flush=True)
            return 2

        final_codes = result["codes"]
        states = result["accepted_states"]
        confirmations: list[dict[str, Any]] = []
        for stage_count in (0, 1, 3, 6, 12, 24):
            stages = compressed_stages(states, stage_count)
            confirmation = confirm_path(
                laser, meter, points, index, final_codes, stages, args
            )
            confirmation["requested_stage_count"] = stage_count
            confirmations.append(confirmation)
            payload["confirmations"] = confirmations
            save_json(args.output, payload)
            if confirmation["success"]:
                payload["success"] = True
                payload["final_codes"] = final_codes
                payload["final_currents_ma"] = result["currents_ma"]
                payload["transition_precondition_codes"] = stages
                payload["transition_hold_ms_per_stage"] = round(args.guard_hold_s * 1000.0)
                save_json(args.output, payload)
                print(f"continuation_result=PASS stages={len(stages)}", flush=True)
                return 0
        payload["success"] = False
        payload["reason"] = "no_compressed_real_path_passed_5_of_5"
        save_json(args.output, payload)
        print("continuation_result=FAILED_CONFIRMATION", flush=True)
        return 3
    finally:
        try:
            if device.is_open and device.connected:
                shutter_confirmed = bool(_shutter(device))
        finally:
            if meter is not None:
                meter.close()
            device.close()
            payload["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
            payload["soa_shutter_confirmed"] = shutter_confirmed
            save_json(args.output, payload)
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
