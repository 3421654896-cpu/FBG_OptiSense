"""Capture pre-RC pairs without CNC motion, table edits or firmware changes.

Diagnostic requests force MAP45 and perturb sparse cadence. This is a paired
ADC/RC stability diagnostic, NOT a sparse bandwidth or stable-truth validation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter, wait_single
from train_rc_dynamic_model import capture_frames, metrics, cross_validate_by_segment
from capture_ch1_teacher_spectrum import acquire_settled_ch1
from laser_dac_safety import validate_pi11210_codes

HERE = Path(__file__).resolve().parent


def load_installed_rows():
    rows = json.loads((HERE / "mode_tables_from_fullband_2001.json").read_text("utf-8"))["modes"]["stress"]["rows"]
    source = (HERE.parent / "JDSU/Core/Src/stress_table.c").read_text("utf-8")
    table = source.split("const uint16_t Stress_Wave_DAC", 1)[1].split("};", 1)[0]
    compiled_rows = [tuple(map(int, group.split(","))) for group in re.findall(r"\{([0-9,]+)\}", table)]
    if len(rows) != 45 or [tuple(row["codes"]) for row in rows] != compiled_rows:
        raise RuntimeError("installed desktop and firmware 45-point DAC tables differ")
    for row in rows:
        validate_pi11210_codes(row["codes"])
    hidden_text = source.split("const uint16_t Stress_Path_Precondition_DAC", 1)[1].split("};", 1)[0]
    hidden = tuple(map(int, re.search(r"\{([0-9,\s]+)", hidden_text)[1].replace("\n", "").split(",")))
    validate_pi11210_codes(hidden)
    return rows, hidden


def capture_teacher(device, rows, hidden, selector):
    _safe_disarm(device)
    points = [app.FullbandAccuracyPoint(index, row["target_wavelength_nm"],
              row["measured_wavelength_nm"], tuple(row["codes"]))
              for index, row in enumerate(rows)]
    worker = app.EqualIntervalWorker(device, points, {}, settle_s=.3,
                                    feedback_selectors=(0, selector))
    if not worker._feedback_command_and_ack():
        raise RuntimeError("settled teacher feedback ACK missing")
    # Establish the actual wrap predecessor before the first point.
    worker._command_and_ack(points[-1].codes)
    worker._interruptible_wait(.3)
    result = []
    for point in points:
        if point.index == 2:
            worker._command_and_ack(hidden)
            worker._interruptible_wait(.005)
        row = acquire_settled_ch1(worker, point, .3)
        result.append(row)
        if (point.index + 1) % 5 == 0:
            print(f"teacher={point.index + 1}/45 stable={sum(r['stable'] for r in result)}", flush=True)
    return result


def summarize(captures):
    result = []
    for index in range(captures.shape[1]):
        first, second, _, estimate, recheck = captures[:, index, :].T
        result.append({
            "point_index": index, "grating": 1 + index // 5,
            "first_mean": float(np.mean(first)),
            "second_mean": float(np.mean(second)),
            "rc_mean": float(np.mean(estimate)),
            "first_std": float(np.std(first)),
            "second_std": float(np.std(second)),
            "rc_std": float(np.std(estimate)),
            "rc_std_over_second_std": float(np.std(estimate) / max(np.std(second), 0.01)),
            "rc_clipped_count": int(np.sum((estimate == 0) | (estimate == 4095))),
            "recheck_count": int(np.sum(recheck)),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--selector", type=int, choices=range(4), default=1)
    parser.add_argument("--captures", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-teacher", action="store_true",
                        help="capture settled DAC-ACK teacher before and after the fast pairs")
    args = parser.parse_args()
    if not 3 <= args.captures <= 100:
        raise ValueError("bounded diagnostic requires 3..100 captures")
    rows, hidden = load_installed_rows()
    payload = {
        "schema": "ch1_pre_rc_pair_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "MAP45 forced diagnostic; not sparse timing or accuracy truth",
        "feedback_selector": args.selector,
        "columns": ["first_direct_adc", "second_direct_adc", "slow_direct_adc", "rc_estimate", "recheck"],
        "complete": False,
        "dac_rows": rows,
    }
    with serial.Serial(args.port, 2_000_000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            if args.stable_teacher:
                payload["teacher_before"] = capture_teacher(device, rows, hidden, args.selector)
            _safe_disarm(device)
            device.write(app.build_stress_feedback_command(args.selector, args.selector))
            if wait_single(device, app.decode_stress_feedback_status) != ((args.selector, args.selector), app.ACK_VALUE):
                raise RuntimeError("exact analogue feedback ACK missing")
            # No multirate arm: all 45 rows genuinely converted, never cached.
            device.write(app.build_work_mode_command(0))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                device.read(4096)
            captures = np.asarray(capture_frames(device, args.captures, 45))
            payload["captures"] = captures.astype(int).tolist()
            payload["points"] = summarize(captures)
            if args.stable_teacher:
                payload["teacher_after"] = capture_teacher(device, rows, hidden, args.selector)
                before = np.asarray([r["ch1_adc_code"] for r in payload["teacher_before"]])
                after = np.asarray([r["ch1_adc_code"] for r in payload["teacher_after"]])
                # Train only one shared two-read gain, with entire gratings held
                # out. No row identity or wavelengths enter the predictor.
                cv, folds = cross_validate_by_segment(captures, before, np.arange(45) // 5)
                truth = np.broadcast_to(before, captures.shape[:2])
                payload["teacher_comparison"] = {
                    "scope": "unloaded static paired-path comparison, not dynamic pressure accuracy",
                    "stable_before": sum(r["stable"] for r in payload["teacher_before"]),
                    "stable_after": sum(r["stable"] for r in payload["teacher_after"]),
                    "teacher_after_minus_before": metrics(before, after),
                    "first_adc_vs_teacher": metrics(truth, captures[..., 0]),
                    "second_adc_vs_teacher": metrics(truth, captures[..., 1]),
                    "firmware_rc_vs_teacher": metrics(truth, captures[..., 3]),
                    "held_out_grating_shared_gain_vs_teacher": metrics(truth, cv),
                    "gain_folds": folds,
                }
            payload["complete"] = True
        finally:
            errors = []
            disarmed = shuttered = False
            try:
                disarmed = _safe_disarm(device)
            except Exception as exc:
                errors.append(f"disarm: {exc}")
            try:
                shuttered = _close_shutter(device)
            except Exception as exc:
                errors.append(f"shutter: {exc}")
            payload["safety_cleanup"] = {
                "exact_disarm_ack": disarmed, "exact_soa_shutter_ack": shuttered,
                "errors": errors,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}", flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError("diagnostic safe shutdown not confirmed")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
