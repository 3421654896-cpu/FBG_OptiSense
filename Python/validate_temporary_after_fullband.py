"""Acquire and validate a fresh CH1 9x5 temporary route after 2001-point release.

This script deliberately refuses to run before the 145 mA full-band workflow
has published a complete forward-only 2001-point table.  It then performs the
same sequence as the desktop page, while preserving every raw acquisition:

1. acquire the complete 2001-point CH1 spectrum in the deployed forward path;
2. select nine equal-spaced five-point windows with point three at each apex;
3. replay only those 45 points slowly and require all nine shape-margin gates;
4. run a conservative finite T45 capture and fit every post-warmup raw frame.

No runtime/default file is replaced automatically.  A new route can be made
the desktop default only after the generated summary has been reviewed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import serial

from capture_ch1_teacher_spectrum import save_checkpoint
from temporary_peak_fit import fit_temporary_frame
from temporary_reference import acquire_reference
from temporary_test_mode import (
    assess_five_point_shape,
    build_selected_plan,
    execute_device,
    select_points,
)


HERE = Path(__file__).resolve().parent
DEFAULT_STATE = HERE / "outputs" / "fullband_145_auto" / "auto_calibration_state.json"
DEFAULT_RUNTIME_TABLE = HERE / "fullband_equal_power_operational_2001.json"
DEFAULT_OUTPUT_ROOT = HERE / "outputs" / "temporary_test"
EXPECTED_POINT_COUNT = 2001
EXPECTED_TARGET_POWER_MW = 2.056
EXPECTED_POWER_TOLERANCE_PERCENT = 3.0


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层不是对象：{path}")
    return value


def validate_fullband_release(state_path: Path, runtime_table_path: Path) -> dict:
    """Prove the new table was published before opening either instrument."""

    state = _read_json(state_path)
    result = state.get("result", {})
    if result.get("status") != "complete":
        raise RuntimeError("145 mA的2001点校准尚未完成发布，禁止开始九峰采集")
    if state.get("certification_direction") != "forward":
        raise RuntimeError("2001点结果不是固定升序认证")
    if not math.isclose(
        float(state.get("single_point_power_relative_tolerance", float("nan"))),
        0.03,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("2001点结果不是±3%单点功率放行配置")

    runtime = _read_json(runtime_table_path)
    rows = runtime.get("rows")
    summary = runtime.get("final_summary", {})
    certification = runtime.get("independent_repeatability_certification", {})
    forward = certification.get("forward", {})
    if not isinstance(rows, dict) or len(rows) != EXPECTED_POINT_COUNT:
        raise RuntimeError("正式运行点表不是完整2001点")
    if int(summary.get("tested", -1)) != EXPECTED_POINT_COUNT or int(
        summary.get("passed", -1)
    ) != EXPECTED_POINT_COUNT:
        raise RuntimeError("正式运行点表汇总不是2001/2001")
    if not math.isclose(
        float(summary.get("power_acceptance_tolerance_percent", float("nan"))),
        EXPECTED_POWER_TOLERANCE_PERCENT,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("正式运行点表未记录±3%功率门槛")
    if certification.get("mode") != "forward-only" or int(
        forward.get("passed", -1)
    ) != EXPECTED_POINT_COUNT:
        raise RuntimeError("正式运行点表缺少固定升序2001/2001独立复测")
    target_power_mw = float(summary.get("target_power_mw", float("nan")))
    if not math.isclose(
        target_power_mw, EXPECTED_TARGET_POWER_MW, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("正式运行点表的共同目标功率不是2.056 mW")
    return {
        "state": str(Path(state_path).resolve()),
        "runtime_table": str(Path(runtime_table_path).resolve()),
        "point_count": len(rows),
        "target_power_mw": target_power_mw,
        "power_tolerance_percent": float(
            summary["power_acceptance_tolerance_percent"]
        ),
        "forward_certified": int(forward["passed"]),
    }


def validate_runtime_table_identity(runtime_table_path: Path, table) -> dict:
    """Prove the audited JSON and the hardware-facing CSV are one table."""

    runtime = _read_json(runtime_table_path)
    rows = runtime.get("rows", {})
    points = tuple(table)
    if not isinstance(rows, dict) or len(rows) != EXPECTED_POINT_COUNT:
        raise RuntimeError("正式运行JSON不是完整2001点")
    if len(points) != EXPECTED_POINT_COUNT:
        raise RuntimeError("硬件运行CSV不是完整2001点")
    ordered = sorted(rows.items(), key=lambda item: float(item[0]))
    for expected_index, ((key, row), point) in enumerate(zip(ordered, points)):
        try:
            target_nm = float(row["target_nm"])
            measured_nm = float(row["reading"]["wavelength_nm"])
            codes = tuple(int(value) for value in row["codes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"正式运行JSON第{expected_index + 1}点字段无效") from exc
        if (
            int(point.index) != expected_index
            or not math.isclose(float(key), target_nm, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(float(point.target_nm), target_nm, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(
                float(point.measured_nm), measured_nm, rel_tol=0.0, abs_tol=5e-9
            )
            or tuple(int(value) for value in point.codes) != codes
        ):
            raise RuntimeError(
                f"正式运行JSON与硬件CSV第{expected_index + 1}点不一致"
            )
    return {
        "point_count": len(points),
        "json": str(Path(runtime_table_path).resolve()),
        "csv_directory": str(Path(runtime_table_path).resolve().parent),
    }


def validate_sparse_reference(plan: dict, report: dict) -> tuple[list[float], list[dict]]:
    """Apply the desktop page's exact 45-point identity and shape gates."""

    rows = report.get("rows", [])
    selected_rows = plan.get("rows", [])
    if (
        not report.get("complete")
        or not report.get("disarm_ack")
        or not report.get("shutter_ack")
        or report.get("ch1_feedback_selector") != 2
        or len(rows) != 45
        or len(selected_rows) != 45
    ):
        raise RuntimeError("45点慢速参考未完整完成、停光未确认或模拟档位错误")

    values: list[float] = []
    for acquired, selected in zip(rows, selected_rows):
        value = float(acquired.get("ch1_adc_code", float("nan")))
        if (
            acquired.get("index") != selected.get("index")
            or acquired.get("dac_codes") != selected.get("codes")
            or not acquired.get("stable")
            or acquired.get("ch1_saturated")
            or not math.isfinite(value)
            or not 0.0 <= value < 4080.0
        ):
            raise RuntimeError("45点慢速参考存在错点、不稳定、饱和或无效值")
        values.append(value)

    checks = []
    for group in range(9):
        start = group * 5
        selected = selected_rows[start : start + 5]
        check = assess_five_point_shape(
            [row["measured_nm"] for row in selected], values[start : start + 5]
        )
        check["group"] = group + 1
        checks.append(check)
    failed = [check for check in checks if not check.get("passed")]
    if failed:
        detail = "、".join(
            f"峰{check['group']}({check.get('reason', '未知')})" for check in failed
        )
        raise RuntimeError(f"当前45点稀疏路径未保持九峰形状：{detail}")
    return values, checks


def analyze_live_capture(plan: dict, capture: dict) -> tuple[dict, list[dict]]:
    """Fit every post-warmup raw frame and require 9/9 on every one."""

    if (
        not capture.get("complete")
        or not capture.get("disarm_ack")
        or not capture.get("shutter_ack")
    ):
        raise RuntimeError("临时45点实时采集未完整结束或停光未确认")
    frames = [frame for frame in capture.get("frames", []) if frame.get("sequence", -1) >= 16]
    if not frames:
        raise RuntimeError("临时45点采集没有预热后的有效帧")

    fitted_frames = []
    for frame in frames:
        records = frame.get("records", [])
        if len(records) != 45:
            raise RuntimeError("临时45点帧记录数不完整")
        values = [record["second_code"] for record in records]
        fits = fit_temporary_frame(plan["rows"], values, warmed_up=True)
        fitted_frames.append({"sequence": frame["sequence"], "fits": fits})

    peak_summaries = []
    for peak in range(9):
        results = [frame["fits"][peak] for frame in fitted_frames]
        valid = [result for result in results if result.get("valid")]
        centers = [float(result["center_nm"]) for result in valid]
        reasons = Counter(str(result.get("reason", "未知")) for result in results)
        peak_summaries.append(
            {
                "peak": peak + 1,
                "valid_frames": len(valid),
                "tested_frames": len(results),
                "valid_fraction": len(valid) / len(results),
                "center_median_nm": statistics.median(centers) if centers else None,
                "center_std_pm": (
                    statistics.pstdev(centers) * 1000.0 if len(centers) > 1 else 0.0
                ) if centers else None,
                "reasons": dict(reasons),
            }
        )
    all_nine = sum(
        all(result.get("valid") for result in frame["fits"])
        for frame in fitted_frames
    )
    passed = all_nine == len(fitted_frames) and all(
        peak["valid_frames"] == peak["tested_frames"] for peak in peak_summaries
    )
    summary = {
        "passed": passed,
        "post_warmup_frames": len(fitted_frames),
        "all_nine_valid_frames": all_nine,
        "peak_summaries": peak_summaries,
        "raw_adc_only": True,
        "fit_model": "five_measured_points_gaussian_plus_constant_baseline",
        "optical_accuracy_verified": False,
    }
    return summary, fitted_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--runtime-table", type=Path, default=DEFAULT_RUNTIME_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--point-wait-us", type=int, default=5000)
    args = parser.parse_args(argv)
    if not 32 <= args.cycles <= 512:
        raise ValueError("验收帧数必须在32到512之间")
    if not 50 <= args.point_wait_us <= 15000 or args.point_wait_us % 25:
        raise ValueError("逐点等待必须为50～15000 µs的25 µs整数倍")

    release = validate_fullband_release(args.state, args.runtime_table)
    folder = args.output_root / f"nine_peak_after_fullband_{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=False)
    summary_path = folder / "summary.json"
    summary = {
        "schema": "temporary_nine_peak_after_fullband_v1",
        "complete": False,
        "release": release,
        "output_dir": str(folder.resolve()),
        "started_s": time.time(),
    }
    save_checkpoint(summary_path, summary)

    import app_JDSU as app

    try:
        runtime_dir = args.runtime_table.resolve().parent
        table = app.load_fullband_accuracy_table(runtime_dir)
        release["runtime_table_identity"] = validate_runtime_table_identity(
            args.runtime_table, table,
        )
        save_checkpoint(summary_path, summary)
        dense_path = folder / "dense_reference_2001.json"
        with serial.Serial(args.port, 2_000_000, timeout=0.1, write_timeout=3) as device:
            dense = acquire_reference(
                device,
                dense_path,
                feedback_selector=2,
                require_stable=True,
                table_base_dir=runtime_dir,
                on_progress=lambda done, total: print(
                    f"dense {done}/{total}", flush=True
                ) if done == 1 or done % 25 == 0 or done == total else None,
            )
        wavelengths = [float(row["measured_wavelength_nm"]) for row in dense["rows"]]
        values = [float(row["ch1_adc_code"]) for row in dense["rows"]]
        dense_plan = select_points(
            table,
            wavelengths,
            values,
            spacing_nm=0,
            baseline_fraction=0.1,
            soft_baseline=True,
        )
        dense_plan.update(
            reference_source=str(dense_path.resolve()),
            reference_created_s=dense.get("started_s"),
            reference_wavelengths_nm=wavelengths,
            reference_values=values,
        )
        dense_plan_path = folder / "dense_selected_plan.json"
        save_checkpoint(dense_plan_path, dense_plan)

        slow_path = folder / "selected_reference_45.json"
        with serial.Serial(args.port, 2_000_000, timeout=0.1, write_timeout=3) as device:
            slow = acquire_reference(
                device,
                slow_path,
                selected_rows=dense_plan["rows"],
                feedback_selector=2,
                require_stable=True,
                table_base_dir=runtime_dir,
                on_progress=lambda done, total: print(
                    f"selected {done}/{total}", flush=True
                ) if done == 1 or done % 5 == 0 or done == total else None,
            )
        reference_values, reference_checks = validate_sparse_reference(dense_plan, slow)
        groups = [
            [row["index"] for row in dense_plan["rows"][group * 5 : group * 5 + 5]]
            for group in range(9)
        ]
        validated_plan = build_selected_plan(
            table,
            groups,
            reference_values,
            source=str(slow_path.resolve()),
        )
        validated_plan["dense_reference_source"] = str(dense_path.resolve())
        validated_plan["dense_selection_plan"] = str(dense_plan_path.resolve())
        validated_plan["live_reference_fit_checks"] = reference_checks
        validated_plan_path = folder / "validated_selected_plan.json"
        save_checkpoint(validated_plan_path, validated_plan)

        capture_path = folder / "live_capture.json"
        fitted_frames: list[dict] = []
        with serial.Serial(args.port, 2_000_000, timeout=0.03, write_timeout=3) as device:
            capture = execute_device(
                device,
                validated_plan,
                capture_path,
                cycles=args.cycles,
                feedback_selector=2,
                first_delay_us=650,
                boundary_extra_us=0,
                point_delays_us=[args.point_wait_us] * 45,
                on_frame=lambda frame: print(
                    f"live {frame['sequence'] + 1}/{args.cycles}", flush=True
                ) if (frame["sequence"] + 1) % 16 == 0 else None,
            )
        fit_summary, fitted_frames = analyze_live_capture(validated_plan, capture)
        save_checkpoint(folder / "fits.json", fitted_frames)
        summary.update(
            complete=bool(fit_summary["passed"]),
            dense_reference=str(dense_path.resolve()),
            dense_selected_plan=str(dense_plan_path.resolve()),
            selected_reference=str(slow_path.resolve()),
            validated_selected_plan=str(validated_plan_path.resolve()),
            live_capture=str(capture_path.resolve()),
            point_wait_us=args.point_wait_us,
            capture_mean_hz=capture.get("mean_hz"),
            fit=fit_summary,
            completed_s=time.time(),
            shutdown_confirmed=bool(capture.get("disarm_ack") and capture.get("shutter_ack")),
        )
        save_checkpoint(summary_path, summary)
        if not fit_summary["passed"]:
            raise RuntimeError("预热后的实时帧未全部达到9/9峰可拟合")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        summary["complete"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["failed_s"] = time.time()
        save_checkpoint(summary_path, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
