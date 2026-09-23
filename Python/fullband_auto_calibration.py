"""One-command 145 mA, 2001-point wavelength/power calibration.

This is the non-Codex operator workflow used by the desktop calibration page.
It owns the AQ6150B and STM32 USB port for the duration of the run, resumes
each measurement file in place, derives the highest robust common power from
all 2001 live measurements, certifies the configured scan path, captures fresh
PDT/PDR references, and promotes a table only after every gate passes.  The
default is the deployed fixed forward path; bidirectional hysteresis auditing
is available only when explicitly requested.

The ordinary 18/45-point and single-value routes keep their 135 mA limits.
Only the dedicated full-band commands use the authorized 145 mA profile.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import serial

from laser_dac_safety import FULLBAND_2001_CODE_LIMITS
from laser_power_calibration import AQ6150B, pd_code_to_current_ma


HERE = Path(__file__).resolve().parent
MODEL_SCRIPT = HERE / "fullband_equal_power_model.py"
EXPORT_SCRIPT = HERE / "export_fullband_calibration.py"
FEEDBACK_SCRIPT = HERE / "capture_fullband_feedback_reference.py"
DEFAULT_OUTPUT_DIR = HERE / "outputs" / "fullband_145_auto"
DEFAULT_WARM_START_TABLE = HERE / "fullband_dense_high_power_2001.json"
STATE_NAME = "auto_calibration_state.json"
LOG_NAME = "auto_calibration.log"
PROFILE = "fullband_2001_145mA_v1"
CURRENT_LIMIT_MA = 145.0
TARGET_COUNT = 2001
MAX_CERTIFICATION_ROUNDS = 10
PRECONDITION_ROUNDS = 4
STAGE_COUNT = 17
DEFAULT_SINGLE_POINT_POWER_REL_TOLERANCE = 0.03
# Re-centre rows before they approach the public +/-2 pm release boundary.
# Machine-2 forward-sweep evidence showed that the old 1.5 pm selector missed
# the entire next set of failures; repaired rows themselves had no repeats.
CERTIFICATION_REPAIR_WAVELENGTH_GUARD_PM = 0.5
CERTIFICATION_REPAIR_POWER_GUARD_RELATIVE = 0.02
SCRATCH_MAP_WAVE_STEP_MA = 1.0
SCRATCH_PHASE_STEP_MA = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit_event(kind: str, **fields: Any) -> None:
    print(
        "@@AUTO@@ "
        + json.dumps({"kind": kind, **fields}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "created_utc": utc_now(),
            "profile": PROFILE,
            "current_limit_ma": CURRENT_LIMIT_MA,
            "stages": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile") != PROFILE:
        raise RuntimeError(
            f"输出目录属于其他校准配置：{payload.get('profile')!r}，当前要求{PROFILE}"
        )
    return payload


def save_state(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_utc"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # On Windows a read-only progress window can briefly hold the destination
    # without FILE_SHARE_DELETE.  Retry the atomic swap instead of aborting a
    # hardware run on that transient reader race.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def stage_complete(
    state: dict[str, Any], name: str, outputs: list[Path] | tuple[Path, ...]
) -> bool:
    record = state.get("stages", {}).get(name, {})
    return record.get("status") == "complete" and all(path.exists() for path in outputs)


def run_stage(
    state_path: Path,
    state: dict[str, Any],
    name: str,
    index: int,
    total: int,
    action: Callable[[], Any],
    outputs: list[Path] | tuple[Path, ...] = (),
) -> Any:
    if stage_complete(state, name, outputs):
        emit_event("stage_skipped", name=name, index=index, total=total)
        return None
    emit_event("stage_started", name=name, index=index, total=total)
    state.setdefault("stages", {})[name] = {
        "status": "running",
        "started_utc": utc_now(),
    }
    save_state(state_path, state)
    try:
        result = action()
    except BaseException as exc:
        state["stages"][name] = {
            "status": "failed",
            "failed_utc": utc_now(),
            "error": str(exc),
        }
        save_state(state_path, state)
        emit_event("stage_failed", name=name, index=index, total=total, error=str(exc))
        raise
    state["stages"][name] = {
        "status": "complete",
        "completed_utc": utc_now(),
        "outputs": [str(path) for path in outputs],
    }
    save_state(state_path, state)
    emit_event("stage_completed", name=name, index=index, total=total)
    return result


def run_command(
    command: list[str],
    log_path: Path,
    *,
    dry_run: bool = False,
    accepted_exit_codes: tuple[int, ...] = (0,),
) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n> {printable}", flush=True)
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] > {printable}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            # The UI creates this orchestrator in a Windows process group.  Its
            # CTRL_BREAK also reaches the active child so the model's finally
            # block can park the laser before returning here.
            try:
                process.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5.0)
            raise
        code = process.wait()
        if code not in accepted_exit_codes:
            raise RuntimeError(f"校准子任务退出码为{code}：{printable}")


def _read_single_value_ack(device: serial.Serial, timeout_s: float = 3.0) -> bytes:
    buffer = bytearray()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = int(device.in_waiting or 0)
        chunk = device.read(min(max(waiting, 20), 4096))
        if chunk:
            buffer.extend(chunk)
        start = 0
        while True:
            offset = buffer.find(b"\xff\xff\x02\x00", start)
            if offset < 0:
                break
            frame = bytes(buffer[offset : offset + 20])
            if len(frame) == 20:
                return frame
            start = offset + 1
        time.sleep(0.002)
    raise TimeoutError("3秒内未收到板卡单值DAC回读")


def _send_shuttered_codes_and_verify(
    device: serial.Serial, codes: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int, int]:
    import app_JDSU as app

    device.write(app.build_work_mode_command(2))
    device.flush()
    time.sleep(0.12)
    device.reset_input_buffer()
    command = app.build_single_value_dac_command(
        codes,
        soa_mode=app.PI11210_SOA_SHUTTER_MODE,
        fullband_2001=True,
    )
    device.write(command)
    device.flush()
    frame = _read_single_value_ack(device)
    returned = tuple(
        int.from_bytes(frame[4 + channel * 2 : 6 + channel * 2], "big")
        for channel in range(5)
    )
    if int(frame[14]) != app.PI11210_SOA_SHUTTER_MODE or not (int(frame[15]) & 0x80):
        raise RuntimeError("板卡未确认SOA硬件关光，禁止开始145 mA校准")
    if int(frame[15]) & 0x03 != 0x03 or not (int(frame[15]) & 0x04):
        raise RuntimeError("PI11210状态或写入回读异常，禁止开始145 mA校准")
    return returned  # type: ignore[return-value]


def hardware_preflight(port: str, meter_resource: str) -> dict[str, Any]:
    """Prove meter identity and the paired 145 mA release while shuttered.

    Shutter mode intentionally forces the physical SOA command and its ACK to
    zero, so the GAIN ceiling is the safe on-wire capability marker.  Official
    full-band releases change the paired GAIN/SOA ceilings together.  The
    first source-mode command still requires an exact five-code ACK and will
    fail immediately if SOA is not accepted at 145 mA.
    """

    meter = AQ6150B(meter_resource)
    try:
        identity = meter.identity
    finally:
        meter.close()

    requested = (
        FULLBAND_2001_CODE_LIMITS[0],
        0,
        0,
        0,
        0,
    )
    returned: tuple[int, int, int, int, int] | None = None
    with serial.Serial(port, 2_000_000, timeout=0.04, write_timeout=1.0) as device:
        device.dtr = True
        time.sleep(0.12)
        try:
            returned = _send_shuttered_codes_and_verify(device, requested)
            if returned != requested:
                raise RuntimeError(
                    "板卡仍在使用较低电流上限："
                    f"关光请求GAIN={requested[0]}，回读GAIN={returned[0]}；"
                    "请先安装支持145 mA专用配置的固件"
                )
        finally:
            # Leave both codes at zero and retain the hardware SOA gate even
            # when the capability check reports an older firmware clamp.
            try:
                _send_shuttered_codes_and_verify(device, (0, 0, 0, 0, 0))
            except Exception:
                pass
    return {
        "meter": identity,
        "port": port,
        "requested_codes": list(requested),
        "returned_codes": list(returned or ()),
        "gain_145_profile_marker_confirmed": returned == requested,
        "soa_shutter_forced_code": 0,
        "soa_shutter_confirmed": returned is not None,
    }


def best_effort_final_shutter(port: str) -> None:
    try:
        with serial.Serial(port, 2_000_000, timeout=0.04, write_timeout=1.0) as device:
            device.dtr = True
            time.sleep(0.12)
            returned = _send_shuttered_codes_and_verify(device, (0, 0, 0, 0, 0))
            if returned != (0, 0, 0, 0, 0):
                raise RuntimeError(f"最终关光回读不是全零：{returned}")
        emit_event("safety", shutter_confirmed=True)
    except Exception as exc:
        emit_event("safety", shutter_confirmed=False, error=str(exc))
        raise


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scratch_power_tuning_tolerance(release_tolerance: float) -> float:
    """Keep useful release margin without restoring the old 0.5% time sink."""

    return min(0.025, max(0.005, float(release_tolerance) * 0.5))


def write_scratch_initialization(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "created_utc": utc_now(),
        "method": "machine_specific_live_discovery_without_legacy_seeds",
        "from_zero": True,
        "port": args.port,
        "meter_resource": args.meter,
        "gain_limit_ma": CURRENT_LIMIT_MA,
        "soa_limit_ma": CURRENT_LIMIT_MA,
        "phase_limit_ma": 10.0,
        "wave_a_limit_ma": 30.0,
        "wave_b_limit_ma": 30.0,
        "release_power_relative_tolerance": args.power_relative_tolerance,
        "internal_power_relative_tolerance": scratch_power_tuning_tolerance(
            args.power_relative_tolerance
        ),
        "legacy_tables_used": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _finite_correlation(left: list[float], right: list[float]) -> float | None:
    pairs = [
        (float(a), float(b)) for a, b in zip(left, right)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if max(xs) == min(xs) or max(ys) == min(ys):
        return None
    return statistics.correlation(xs, ys)


def analyze_feedback_landscape(
    mode_maps: list[Path], output: Path,
) -> dict[str, Any]:
    """Summarize live PDT/PDR evidence without treating it as a release gate.

    Both monitor channels are biased transimpedance measurements.  Their
    absolute values depend on the laser package and front end, while abrupt
    changes along a continuous DAC walk are useful evidence of an etalon or
    longitudinal-mode boundary.  AQ6150B wavelength/power remains authoritative.
    """

    observations: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for path in mode_maps:
        payload = read_json(path)
        previous: dict[str, Any] | None = None
        for row in payload.get("rows", {}).values():
            reading = row.get("reading", {})
            if reading.get("pdt_code") is None or reading.get("pdr_code") is None:
                previous = None
                continue
            pdt_code = int(reading["pdt_code"])
            pdr_code = int(reading["pdr_code"])
            pdt_ma = pd_code_to_current_ma(pdt_code)
            pdr_ma = pd_code_to_current_ma(pdr_code)
            item = {
                "source": path.name,
                "wavelength_nm": float(reading["wavelength_nm"]),
                "power_mw": float(reading["power_mw"]),
                "pdt_code": pdt_code,
                "pdr_code": pdr_code,
                "pdt_monitor_ma": pdt_ma,
                "pdr_monitor_ma": pdr_ma,
                "pdr_pdt_ratio": pdr_ma / pdt_ma if pdt_ma > 1e-9 else None,
                "single_mode": bool(row.get("single_mode")),
            }
            observations.append(item)
            if previous is not None:
                wave_jump = abs(item["wavelength_nm"] - previous["wavelength_nm"])
                transitions.append({
                    "source": path.name,
                    "wavelength_jump_nm": wave_jump,
                    "pdt_code_jump": abs(item["pdt_code"] - previous["pdt_code"]),
                    "pdr_code_jump": abs(item["pdr_code"] - previous["pdr_code"]),
                    "mode_boundary": (
                        wave_jump > 0.35
                        or not item["single_mode"]
                        or not previous["single_mode"]
                    ),
                })
            previous = item

    powers = [row["power_mw"] for row in observations]
    pdt_currents = [row["pdt_monitor_ma"] for row in observations]
    pdr_currents = [row["pdr_monitor_ma"] for row in observations]
    ratio_rows = [row for row in observations if row["pdr_pdt_ratio"] is not None]
    boundaries = [row for row in transitions if row["mode_boundary"]]
    continuous = [row for row in transitions if not row["mode_boundary"]]

    def median_jump(rows: list[dict[str, Any]], field: str) -> float | None:
        values = [float(row[field]) for row in rows]
        return statistics.median(values) if values else None

    result = {
        "created_utc": utc_now(),
        "method": "live_pdt_pdr_landscape_evidence_v1",
        "source_mode_maps": [str(path) for path in mode_maps],
        "release_authority": "AQ6150B wavelength, power and SMSR",
        "pdt_pdr_role": "branch continuity evidence only; not a hard acceptance gate",
        "circuit_model": {
            "adc_reference_v": 2.5,
            "photodiode_bias_v": 1.25,
            "transimpedance_ohm": 2000.0,
            "formula": "max(0, (1.25 - adc_code*2.5/4096)/2000)",
        },
        "summary": {
            "observations": len(observations),
            "transitions": len(transitions),
            "mode_boundary_transitions": len(boundaries),
            "single_mode_observations": sum(row["single_mode"] for row in observations),
            "power_correlation_with_pdt": _finite_correlation(powers, pdt_currents),
            "power_correlation_with_pdr": _finite_correlation(powers, pdr_currents),
            "power_correlation_with_pdr_pdt_ratio": _finite_correlation(
                [row["power_mw"] for row in ratio_rows],
                [row["pdr_pdt_ratio"] for row in ratio_rows],
            ),
            "median_pdt_jump_at_mode_boundary_codes": median_jump(
                boundaries, "pdt_code_jump"
            ),
            "median_pdr_jump_at_mode_boundary_codes": median_jump(
                boundaries, "pdr_code_jump"
            ),
            "median_pdt_jump_inside_continuous_branch_codes": median_jump(
                continuous, "pdt_code_jump"
            ),
            "median_pdr_jump_inside_continuous_branch_codes": median_jump(
                continuous, "pdr_code_jump"
            ),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def require_complete_rows(path: Path, *, label: str) -> dict[str, Any]:
    payload = read_json(path)
    limits = payload.get("limits_ma", {})
    if (
        float(limits.get("gain_ma", -1.0)) != CURRENT_LIMIT_MA
        or float(limits.get("soa_ma", -1.0)) != CURRENT_LIMIT_MA
    ):
        raise RuntimeError(f"{label}没有声明145 mA专用配置")
    rows = payload.get("rows")
    if not isinstance(rows, dict) or len(rows) != TARGET_COUNT:
        raise RuntimeError(f"{label}不是完整2001点：{0 if not isinstance(rows, dict) else len(rows)}")
    failed = [key for key, row in rows.items() if not row.get("success")]
    if failed:
        raise RuntimeError(f"{label}仍有{len(failed)}个失败点，首点{failed[0]}")
    return payload


def validate_warm_start_table(path: Path) -> dict[str, Any]:
    """Validate a complete older live table before using it only as a seed.

    GAIN/SOA values from this table are never sent unchanged: the live dense
    tuner replaces them with the dedicated 145 mA discovery values.  The old
    PHASE/WAVE controls merely identify already-proven full-band branches.
    """

    if not path.exists():
        raise RuntimeError(f"暖启动的2001点实测表不存在：{path}")
    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, dict) or len(rows) != TARGET_COUNT:
        raise RuntimeError(
            f"暖启动表不是完整2001点："
            f"{0 if not isinstance(rows, dict) else len(rows)}"
        )
    powers: list[float] = []
    maximum_error_pm = 0.0
    minimum_smsr_db = math.inf
    for index in range(TARGET_COUNT):
        target = 1525.0 + index * 0.02
        key = f"{target:.6f}"
        row = rows.get(key)
        if not isinstance(row, dict) or not row.get("success"):
            raise RuntimeError(f"暖启动表{key} nm未通过旧的实测校准")
        if not math.isclose(float(row.get("target_nm", float("nan"))), target,
                            rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError(f"暖启动表{key} nm的目标索引不一致")
        codes = row.get("codes", ())
        if len(codes) != 5:
            raise RuntimeError(f"暖启动表{key} nm不是五通道DAC")
        for channel, (raw, maximum) in enumerate(
            zip(codes, FULLBAND_2001_CODE_LIMITS)
        ):
            value = int(raw)
            historical_boundary = channel >= 2 and value == maximum + 1
            if value < 0 or (value > maximum and not historical_boundary):
                raise RuntimeError(f"暖启动表{key} nm超出专用DAC上限")
        reading = row.get("reading", {})
        measured_wave = float(reading.get("wavelength_nm", float("nan")))
        power = float(reading.get("power_mw", float("nan")))
        if not math.isfinite(measured_wave) or not math.isfinite(power) or power <= 0.0:
            raise RuntimeError(f"暖启动表{key} nm的实测数据无效")
        error_pm = abs(measured_wave - target) * 1000.0
        if error_pm > 2.0 + 1e-9:
            raise RuntimeError(f"暖启动表{key} nm的旧波长误差超过±2 pm")
        smsr_value = reading.get("side_mode_suppression_db")
        if smsr_value is None and int(reading.get("peak_count", 0)) == 1:
            smsr = 99.0
        else:
            smsr = float(smsr_value if smsr_value is not None else -math.inf)
        if not math.isfinite(smsr) or smsr < 20.0:
            raise RuntimeError(f"暖启动表{key} nm的旧SMSR未达20 dB")
        powers.append(power)
        maximum_error_pm = max(maximum_error_pm, error_pm)
        minimum_smsr_db = min(minimum_smsr_db, smsr)
    return {
        "path": str(path),
        "rows": len(rows),
        "source_gain_ma": payload.get("limits_ma", {}).get("gain_ma"),
        "source_soa_ma": payload.get("limits_ma", {}).get("soa_ma"),
        "power_min_mw": min(powers),
        "power_max_mw": max(powers),
        "maximum_abs_wavelength_error_pm": maximum_error_pm,
        "minimum_smsr_db": minimum_smsr_db,
    }


def validation_failures(path: Path) -> set[str]:
    payload = read_json(path)
    rows = payload.get("rows", {})
    return {str(key) for key, row in rows.items() if not row.get("success")}


def validation_rework_targets(
    path: Path, release_power_relative_tolerance: float,
) -> set[str]:
    """Include release failures and successful rows too near a drift boundary.

    A 2001-point certification sweep lasts long enough that a row measured at
    1.9 pm or 2.9% can cross the public 2 pm / 3% gate on the next sweep even
    though its DAC branch is valid.  Proactively re-centering the guard-band
    rows lets the following full sweep test useful margin instead of repeatedly
    discovering a different set of edge points.  This only broadens the repair
    set; it never changes the final release thresholds.
    """

    payload = read_json(path)
    rows = payload.get("rows", {})
    power_guard = min(
        float(release_power_relative_tolerance),
        CERTIFICATION_REPAIR_POWER_GUARD_RELATIVE,
    )
    targets: set[str] = set()
    for key, row in rows.items():
        if not row.get("success"):
            targets.add(str(key))
            continue
        try:
            wavelength_error_pm = abs(float(row["wavelength_error_pm"]))
            power_relative_error = abs(float(row["power_relative_error"]))
        except (KeyError, TypeError, ValueError):
            targets.add(str(key))
            continue
        if (
            wavelength_error_pm > CERTIFICATION_REPAIR_WAVELENGTH_GUARD_PM
            or power_relative_error > power_guard
        ):
            targets.add(str(key))
    return targets


def write_rework_seed(
    source: Path, output: Path, failed: set[str], certification_round: int
) -> None:
    payload = read_json(source)
    # Guard-band expansion belongs to one specific validation round.  A
    # repaired table is the source for the following round, so carrying this
    # metadata forward would make the next repair subprocess reuse the old
    # target list and skip newly failed points.
    payload.pop("certification_guard_band_expansion", None)
    payload.pop("repair_guard_band_targets", None)
    for key in failed:
        row = payload["rows"].get(key)
        if row is not None:
            row["success"] = False
            row.setdefault("certification_rework", []).append(
                {"round": certification_round, "created_utc": utc_now()}
            )
    payload["certification_rework_created_utc"] = utc_now()
    payload["certification_rework_round"] = certification_round
    payload["certification_failed_targets"] = sorted(failed, key=float)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_recentered_rework_seed(
    source: Path,
    output: Path,
    failed: set[str],
    certification_round: int,
    validations: list[Path],
    high_power: Path,
    soa_pilot: Path,
    *,
    require_live_verification: bool = True,
    correction_gain: float = 1.0,
    maximum_step_ma: float = 4.0,
    defer_unusable_evidence: bool = False,
) -> dict[str, int]:
    """Use robust validation power evidence to pre-center SOA offline."""
    write_rework_seed(source, output, failed, certification_round)
    payload = read_json(output)
    validation_payloads = [read_json(path) for path in validations if path.exists()]
    high_rows = read_json(high_power)["rows"]
    ratios = read_json(soa_pilot)["summary"]["normalized_power_ratio_median_by_soa"]
    curve = sorted((float(current), float(ratio)) for current, ratio in ratios.items())

    def local_slope(baseline_power_mw: float, soa_ma: float) -> float:
        pair = None
        for left, right in zip(curve, curve[1:]):
            if left[0] <= soa_ma <= right[0]:
                pair = left, right
                break
        if pair is None:
            pair = (curve[0], curve[1]) if soa_ma < curve[0][0] else (curve[-2], curve[-1])
        (left_ma, left_ratio), (right_ma, right_ratio) = pair
        return baseline_power_mw * (right_ratio - left_ratio) / (right_ma - left_ma)

    predicted = 0
    live_required = 0
    for key in failed:
        row = payload["rows"].get(key)
        if row is None:
            continue
        usable_powers: list[float] = []
        evidence: list[dict[str, Any]] = []
        for validation in validation_payloads:
            checked = validation.get("rows", {}).get(key)
            if checked is None:
                continue
            reading = checked.get("reading", {})
            smsr_value = reading.get("side_mode_suppression_db")
            smsr = 99.0 if smsr_value is None and int(reading.get("peak_count", 0)) == 1 \
                else float(smsr_value if smsr_value is not None else -math.inf)
            sample_count = int(checked.get("sample_count", 1))
            mode_pass_count = int(
                checked.get("mode_pass_count", 1 if smsr >= 20.0 else 0)
            )
            mode_ok = mode_pass_count >= sample_count // 2 + 1 and smsr >= 20.0
            wave_ok = abs(float(checked.get("wavelength_error_pm", math.inf))) <= 2.0
            power_mw = float(reading.get("power_mw", float("nan")))
            evidence.append({
                "direction": validation.get("direction"),
                "power_mw": power_mw,
                "wavelength_error_pm": checked.get("wavelength_error_pm"),
                "mode_ok": mode_ok,
            })
            if mode_ok and wave_ok and math.isfinite(power_mw) and power_mw > 0.0:
                usable_powers.append(power_mw)
        if not usable_powers:
            if defer_unusable_evidence:
                row["success"] = True
                row["certification_power_prediction_unverified"] = True
            else:
                live_required += 1
            row.setdefault("certification_rework", []).append({
                "strategy": (
                    "deferred_to_strict_live_branch_reclosure"
                    if defer_unusable_evidence
                    else "live_branch_reclosure_required"
                ),
                "validation_evidence": evidence,
            })
            continue

        measured_power = float(statistics.median(usable_powers))
        target_power = float(payload["target_power_mw"])
        codes = [int(value) for value in row["codes"]]
        current_soa = codes[1] * 150.0 / 65536.0
        baseline_power = float(high_rows[key]["reading"]["power_mw"])
        slope = local_slope(baseline_power, current_soa)
        if not math.isfinite(slope) or slope <= 0.0005:
            live_required += 1
            continue
        requested_soa = (
            current_soa
            + float(correction_gain) * (target_power - measured_power) / slope
        )
        requested_soa = min(
            max(requested_soa, current_soa - float(maximum_step_ma)),
            current_soa + float(maximum_step_ma),
        )
        requested_soa = min(max(requested_soa, 85.0), CURRENT_LIMIT_MA)
        new_code = min(
            max(int(round(requested_soa / 150.0 * 65536.0)), 0),
            int(FULLBAND_2001_CODE_LIMITS[1]),
        )
        codes[1] = new_code
        row["codes"] = codes
        currents = [float(value) for value in row["currents_ma"]]
        currents[1] = new_code * 150.0 / 65536.0
        row["currents_ma"] = currents
        # A preconditioning pass may defer this predicted power-only change to
        # the next full sweep.  Strict certification rework keeps it invalid
        # until the AQ6150B closes and confirms the candidate live.
        row["success"] = not require_live_verification
        row["certification_power_prediction_unverified"] = True
        row.setdefault("certification_rework", []).append({
            "strategy": "robust_validation_median_soa_recentering_v1",
            "validation_evidence": evidence,
            "measured_power_mw": measured_power,
            "target_power_mw": target_power,
            "estimated_slope_mw_per_ma": slope,
            "old_soa_ma": current_soa,
            "new_soa_ma": currents[1],
            "correction_gain": float(correction_gain),
            "maximum_step_ma": float(maximum_step_ma),
            "live_verification_required": bool(require_live_verification),
        })
        predicted += 1
    live_required = sum(
        1 for key in failed
        if key in payload["rows"] and not payload["rows"][key].get("success")
    )
    payload["certification_predicted_power_recentering"] = {
        "predicted_rows": predicted,
        "live_reclosure_rows": live_required,
        "validation_sources": [path.name for path in validations],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    return {"predicted": predicted, "live_required": live_required}


def atomic_promote(source: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".new")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def promote_runtime_table(
    output_dir: Path,
    prefix: Path,
    feedback: Path,
    *,
    runtime_dir: Path = HERE,
) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    canonical_prefix = runtime_dir / "fullband_equal_power_operational_2001"
    pairs = [
        (prefix.with_suffix(".json"), canonical_prefix.with_suffix(".json")),
        (prefix.with_suffix(".csv"), canonical_prefix.with_suffix(".csv")),
        (prefix.with_suffix(".h"), canonical_prefix.with_suffix(".h")),
        (
            prefix.with_name(prefix.name + "_transition_guards").with_suffix(".json"),
            canonical_prefix.with_name(
                canonical_prefix.name + "_transition_guards"
            ).with_suffix(".json"),
        ),
        (feedback, runtime_dir / "fullband_pdt_pdr_reference_2001.json"),
    ]
    missing = [str(source) for source, _ in pairs if not source.exists()]
    if missing:
        raise RuntimeError("发布文件缺失：" + ", ".join(missing))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = output_dir / f"previous_runtime_table_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for _, target in pairs:
        if target.exists():
            shutil.copy2(target, backup / target.name)
    for source, target in pairs:
        atomic_promote(source, target)
    return {
        "backup_dir": str(backup),
        "promoted": [str(target) for _, target in pairs],
    }


def model_command(*arguments: str) -> list[str]:
    return [sys.executable, str(MODEL_SCRIPT), *map(str, arguments)]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / STATE_NAME
    log_path = output / LOG_NAME
    state = load_state(state_path)
    discovery_mode = "from_zero" if args.from_zero else "legacy_seeded"
    previous_mode = state.get("discovery_mode")
    if previous_mode is not None and previous_mode != discovery_mode:
        raise RuntimeError(
            f"输出目录已经属于{previous_mode}模式，不能按{discovery_mode}继续"
        )
    state.update(
        {
            "port": args.port,
            "meter_resource": args.meter,
            "runtime_dir": str(args.runtime_dir.resolve()),
            "certification_direction": args.certification_direction,
            "single_point_power_relative_tolerance": (
                args.power_relative_tolerance
            ),
            "internal_power_relative_tolerance": scratch_power_tuning_tolerance(
                args.power_relative_tolerance
            ),
            "discovery_mode": discovery_mode,
            "legacy_tables_used": not args.from_zero,
        }
    )
    state["result"] = {"status": "running", "started_utc": utc_now()}
    save_state(state_path, state)

    mode_map = output / "mode_map_145.json"
    mode_map_offset = output / "mode_map_offset_145.json"
    scratch_marker = output / "scratch_initialization.json"
    feedback_landscape = output / "pdt_pdr_landscape.json"
    branches = output / "phase_branches_145.json"
    phase_base = output / "phase_map_base_145.json"
    phase_gap = output / "phase_map_gap_145.json"
    gap_branches = output / "phase_gap_branches_145.json"
    phase_alternate = output / "phase_map_alternate_145.json"
    alternate_branches = output / "phase_alternate_branches_145.json"
    phase_candidates = output / "phase_candidates_145.json"
    dense_raw = output / "dense_raw_145_2001.json"
    dense_repaired = output / "dense_repaired_145_2001.json"
    high_power = output / "maximum_power_145_2001.json"
    soa_pilot = output / "soa_transfer_145.json"
    equalized = output / "equal_power_145_2001.json"
    repaired = output / "equal_power_repaired_145_2001.json"
    preliminary_prefix = output / "operational_145_2001_no_feedback"
    final_prefix = output / "operational_145_2001"
    feedback = output / "pdt_pdr_reference_145_2001.json"

    warm_start: Path | None = None
    if args.from_zero:
        state.pop("warm_start", None)
        emit_event("scratch_discovery_selected", legacy_tables_used=False)
    else:
        warm_start = args.warm_start_table.resolve()
        warm_start_summary = validate_warm_start_table(warm_start)
        state["warm_start"] = warm_start_summary
        emit_event("warm_start_validated", **warm_start_summary)
    save_state(state_path, state)

    total = STAGE_COUNT
    stage = 1
    if not args.dry_run:
        preflight = hardware_preflight(args.port, args.meter)
        state["last_preflight"] = {"at_utc": utc_now(), **preflight}
        save_state(state_path, state)
        emit_event("preflight", **preflight)
    else:
        emit_event("preflight", dry_run=True)

    if args.from_zero:
        run_stage(
            state_path, state, "scratch_initialization", stage, total,
            lambda: write_scratch_initialization(scratch_marker, args),
            [scratch_marker],
        )
    else:
        run_stage(
            state_path, state, "inventory_and_model", stage, total,
            lambda: (
                run_command(model_command("inventory", "--output", str(output / "inventory_145.json")), log_path, dry_run=args.dry_run),
                run_command(model_command("train", "--output", str(output / "surrogate_145.joblib")), log_path, dry_run=args.dry_run),
            ),
            [output / "inventory_145.json", output / "surrogate_145.joblib"],
        )
    stage += 1
    fixed_mode_maps = [mode_map]
    if args.from_zero:
        fixed_mode_maps.append(mode_map_offset)
    run_stage(
        state_path, state, "mode_map", stage, total,
        lambda: (
            run_command(
                model_command(
                    "map", "--port", args.port, "--gpib", args.meter,
                    "--output", str(mode_map), "--gain-ma", "145", "--soa-ma", "145",
                    "--phase-ma", "5", "--wave-step-ma",
                    str(SCRATCH_MAP_WAVE_STEP_MA if args.from_zero else 1.5),
                    "--resume",
                ), log_path, dry_run=args.dry_run,
            ),
            run_command(
                model_command(
                    "map", "--port", args.port, "--gpib", args.meter,
                    "--output", str(mode_map_offset), "--gain-ma", "145",
                    "--soa-ma", "145", "--phase-ma", "5", "--wave-step-ma", "1",
                    "--wave-a-start-ma", "0.5", "--wave-a-stop-ma", "29.5",
                    "--wave-b-start-ma", "0.5", "--wave-b-stop-ma", "29.5",
                    "--resume",
                ), log_path, dry_run=args.dry_run,
            ) if args.from_zero else None,
            analyze_feedback_landscape(fixed_mode_maps, feedback_landscape)
            if not args.dry_run else None,
        ), [*fixed_mode_maps, *([feedback_landscape] if not args.dry_run else [])],
    )
    stage += 1
    run_stage(
        state_path, state, "select_phase_branches", stage, total,
        lambda: run_command(
            model_command(
                "select-phase-branches", "--mode-maps",
                *[str(path) for path in fixed_mode_maps],
                "--output", str(branches), "--target-step-nm",
                "0.25" if args.from_zero else "0.20",
                "--radius-nm", "0.50" if args.from_zero else "0.30",
                "--branches-per-target", "2",
            ), log_path, dry_run=args.dry_run,
        ), [branches],
    )
    stage += 1
    run_stage(
        state_path, state, "phase_map_base", stage, total,
        lambda: run_command(
            model_command(
                "phase-map", "--port", args.port, "--gpib", args.meter,
                "--branches", str(branches), "--output", str(phase_base),
                "--gain-ma", "145", "--soa-ma", "145", "--phase-start-ma", "0",
                "--phase-stop-ma", "10", "--phase-step-ma",
                str(SCRATCH_PHASE_STEP_MA if args.from_zero else 1.0), "--resume",
            ), log_path, dry_run=args.dry_run,
        ), [phase_base],
    )
    stage += 1

    phase_maps = [phase_base]
    candidate_command = lambda maps: model_command(
        "build-phase-candidates", "--phase-maps", *[str(path) for path in maps],
        "--output", str(phase_candidates), "--candidates-per-target",
        "8" if args.from_zero else "5",
    )
    run_stage(
        state_path, state, "phase_candidates_base", stage, total,
        # Exit 2 here means that measured branches still leave wavelength
        # gaps.  That is the expected trigger for stages 6/7, not a fatal
        # calibration error.  The standalone command remains fail-closed.
        lambda: run_command(
            candidate_command(phase_maps), log_path,
            dry_run=args.dry_run, accepted_exit_codes=(0, 2),
        ),
        [phase_candidates],
    )
    stage += 1

    uncovered = 0 if args.dry_run else int(
        read_json(phase_candidates).get("summary", {}).get("uncovered_targets", TARGET_COUNT)
    )
    if uncovered:
        run_stage(
            state_path, state, "phase_gap_refinement", stage, total,
            lambda: (
                run_command(
                    model_command(
                        "select-gap-phase-branches", "--candidates", str(phase_candidates),
                        "--phase-map", str(phase_base), "--output", str(gap_branches),
                        "--radius-nm", "0.80" if args.from_zero else "0.60",
                        "--branches-per-target", "3" if args.from_zero else "2",
                    ), log_path, dry_run=args.dry_run,
                ),
                run_command(
                    model_command(
                        "phase-map", "--port", args.port, "--gpib", args.meter,
                        "--branches", str(gap_branches), "--output", str(phase_gap),
                        "--gain-ma", "145", "--soa-ma", "145",
                        "--phase-start-ma", "0.25" if args.from_zero else "0.5",
                        "--phase-stop-ma", "9.75" if args.from_zero else "9.5",
                        "--phase-step-ma", "0.5" if args.from_zero else "1",
                        "--resume",
                    ), log_path, dry_run=args.dry_run,
                ),
            ), [gap_branches, phase_gap],
        )
        phase_maps.append(phase_gap)
        run_command(
            candidate_command(phase_maps), log_path,
            dry_run=args.dry_run, accepted_exit_codes=(0, 2),
        )
        uncovered = 0 if args.dry_run else int(
            read_json(phase_candidates).get("summary", {}).get("uncovered_targets", TARGET_COUNT)
        )
    stage += 1
    if uncovered:
        run_stage(
            state_path, state, "phase_alternate_refinement", stage, total,
            lambda: (
                run_command(
                    model_command(
                        "select-alternate-mode-branches", "--candidates", str(phase_candidates),
                        "--exclude-phase-maps", *[str(path) for path in phase_maps],
                        "--mode-maps", *[str(path) for path in fixed_mode_maps],
                        "--output", str(alternate_branches),
                        "--radius-nm", "1.25" if args.from_zero else "1.0",
                        "--branches-per-target", "5" if args.from_zero else "4",
                    ), log_path, dry_run=args.dry_run,
                ),
                run_command(
                    model_command(
                        "phase-map", "--port", args.port, "--gpib", args.meter,
                        "--branches", str(alternate_branches), "--output", str(phase_alternate),
                        "--gain-ma", "145", "--soa-ma", "145",
                        "--phase-start-ma", "0", "--phase-stop-ma", "10",
                        "--phase-step-ma", "0.5", "--resume",
                    ), log_path, dry_run=args.dry_run,
                ),
            ), [alternate_branches, phase_alternate],
        )
        phase_maps.append(phase_alternate)
        run_command(
            candidate_command(phase_maps), log_path,
            dry_run=args.dry_run, accepted_exit_codes=(0, 2),
        )
        uncovered = 0 if args.dry_run else int(
            read_json(phase_candidates).get("summary", {}).get("uncovered_targets", TARGET_COUNT)
        )
    stage += 1
    # Gaps mean only that no interpolation lies wholly inside two measured
    # same-mode endpoints; they do not prove that a target is unreachable.
    # Scratch mode intentionally permits only measurements made during this
    # machine-specific run.  Legacy-seeded mode retains the older workflow.
    phase_seed_sources = list(phase_maps)
    if warm_start is not None:
        phase_seed_sources.append(warm_start)
    if uncovered:
        event = {
            "uncovered_targets": uncovered,
            "legacy_tables_used": warm_start is not None,
        }
        if warm_start is not None:
            event["warm_start_table"] = str(warm_start)
        emit_event("candidate_gaps_deferred_to_live_dense_closure", **event)

    internal_power_tolerance = (
        scratch_power_tuning_tolerance(args.power_relative_tolerance)
        if args.from_zero else 0.005
    )
    internal_power_tolerance_text = f"{internal_power_tolerance:.9g}"
    power_target_arguments = (
        ["--power-headroom-relative", "0", "--power-lift-relative", "0.015"]
        if args.from_zero else
        ["--power-headroom-relative", "0.005", "--power-lift-relative", "0"]
    )
    repair_target_arguments = (
        ["--repair-power-target-relative", internal_power_tolerance_text]
        if args.from_zero else []
    )
    phase_candidate_repair_arguments = (
        ["--phase-candidates", str(phase_candidates)] if args.from_zero else []
    )
    inventory_arguments = (
        [] if args.from_zero
        else ["--inventory", str(output / "inventory_145.json")]
    )

    run_stage(
        state_path, state, "dense_2001", stage, total,
        lambda: run_command(
            model_command(
                "dense-phase", "--port", args.port, "--gpib", args.meter,
                "--candidates", str(phase_candidates), "--phase-maps",
                *[str(path) for path in phase_seed_sources], "--output", str(dense_raw),
                "--ranked-seeds", "2" if args.from_zero else "5",
                "--nearest-seeds", "2" if args.from_zero else "6",
                "--refine-seeds", "2" if args.from_zero else "4",
                "--max-measurements", "24" if args.from_zero else "18", "--resume",
            ), log_path, dry_run=args.dry_run, accepted_exit_codes=(0, 2),
        ), [dense_raw],
    )
    stage += 1
    run_stage(
        state_path, state, "repair_dense", stage, total,
        lambda: run_command(
            model_command(
                "repair-dense", "--port", args.port, "--gpib", args.meter,
                "--input", str(dense_raw), "--output", str(dense_repaired),
                "--passes", "4", "--max-measurements", "40", "--resume",
            ), log_path, dry_run=args.dry_run,
        ), [dense_repaired],
    )
    if not args.dry_run:
        require_complete_rows(dense_repaired, label="密集波长闭环")
    stage += 1
    run_stage(
        state_path, state, "maximize_power", stage, total,
        lambda: run_command(
            model_command(
                "upgrade-power", "--port", args.port, "--gpib", args.meter,
                "--input", str(dense_repaired), "--output", str(high_power),
                "--candidates", str(phase_candidates), "--phase-maps",
                *[str(path) for path in [*phase_seed_sources, dense_repaired]],
                "--adaptive-bottleneck",
                "--minimum-gain-mw", "0.002", "--screen-seeds", "10",
                "--refine-seeds", "4", "--max-measurements", "20", "--resume",
            ), log_path, dry_run=args.dry_run, accepted_exit_codes=(0, 2),
        ), [high_power],
    )
    if not args.dry_run:
        require_complete_rows(high_power, label="逐点最大功率源表")
    stage += 1
    run_stage(
        state_path, state, "soa_transfer", stage, total,
        lambda: run_command(
            model_command(
                "soa-pilot", "--port", args.port, "--gpib", args.meter,
                "--input", str(high_power), "--output", str(soa_pilot),
                "--soa-max-ma", "145", "--soa-min-ma", "85", "--soa-step-ma", "5",
                "--wavelength-samples", "9", "--low-power-samples", "3",
                "--high-power-samples", "3",
            ), log_path, dry_run=args.dry_run,
        ), [soa_pilot],
    )
    stage += 1
    run_stage(
        state_path, state, "equalize_common_power", stage, total,
        lambda: run_command(
            model_command(
                "equalize-power", "--port", args.port, "--gpib", args.meter,
                "--input", str(high_power), "--soa-pilot", str(soa_pilot),
                "--output", str(equalized), "--soa-max-ma", "145",
                *power_target_arguments, "--power-resolution-mw", "0.001",
                "--power-relative-tolerance", internal_power_tolerance_text,
                "--max-rounds", "8", "--max-tune-measurements", "18", "--resume",
            ), log_path, dry_run=args.dry_run, accepted_exit_codes=(0, 2),
        ), [equalized],
    )
    stage += 1
    target_power = 1.0 if args.dry_run else float(read_json(equalized)["target_power_mw"])
    run_stage(
        state_path, state, "repair_equal_power", stage, total,
        lambda: run_command(
            model_command(
                "repair-equal-power", "--port", args.port, "--gpib", args.meter,
                "--input", str(equalized), "--output", str(repaired),
                "--source-table", str(high_power), "--soa-pilot", str(soa_pilot),
                *inventory_arguments,
                *phase_candidate_repair_arguments,
                "--target-power-mw", f"{target_power:.9f}", "--passes", "4",
                "--power-relative-tolerance", internal_power_tolerance_text,
                *repair_target_arguments,
                "--max-rounds", "6", "--max-tune-measurements", "20", "--resume",
            ), log_path, dry_run=args.dry_run,
        ), [repaired],
    )
    if not args.dry_run:
        require_complete_rows(repaired, label="等功率闭环")
    stage += 1

    current_table = repaired
    certification_power_tolerance = f"{args.power_relative_tolerance:.9g}"
    final_forward: Path | None = None
    final_reverse: Path | None = None
    if args.dry_run:
        final_forward = output / "validation_forward_round1.json"
        run_command(
            model_command(
                "validate-equal-power", "--port", args.port, "--gpib", args.meter,
                "--input", str(current_table), "--output", str(final_forward),
                "--direction", "forward", "--power-relative-tolerance",
                certification_power_tolerance,
                "--resume",
            ), log_path, dry_run=True,
        )
        if args.certification_direction == "bidirectional":
            final_reverse = output / "validation_reverse_round1.json"
            run_command(
                model_command(
                    "validate-equal-power", "--port", args.port, "--gpib", args.meter,
                    "--input", str(current_table), "--output", str(final_reverse),
                    "--direction", "reverse", "--power-relative-tolerance",
                    certification_power_tolerance,
                    "--resume",
                ), log_path, dry_run=True,
            )
    else:
        # Fast sequential-scan learning brings the old table into the current
        # thermal state before the expensive nine-sweep certification.  Each
        # reading is already a three-sweep AQ6150B average.  Power-only updates
        # remain provisional until the next full sweep; wavelength/mode faults
        # are still repaired live.  Nothing from this loop is publishable.
        for precondition_round in range(1, PRECONDITION_ROUNDS + 1):
            precheck = output / (
                f"precondition_forward_round{precondition_round}_avg3s1.json"
            )
            run_command(
                model_command(
                    "validate-equal-power", "--port", args.port, "--gpib", args.meter,
                    "--input", str(current_table), "--output", str(precheck),
                    "--direction", "forward", "--power-relative-tolerance",
                    certification_power_tolerance,
                    "--samples-per-point", "1", "--quiet-failures", "--resume",
                ), log_path, accepted_exit_codes=(0, 2),
            )
            precondition_failed = validation_failures(precheck)
            emit_event(
                "preconditioning_scan_completed",
                round=precondition_round,
                passed=TARGET_COUNT - len(precondition_failed),
                failed=len(precondition_failed),
            )
            if not precondition_failed:
                break
            precondition_seed = output / (
                f"precondition_rework_seed_round{precondition_round}_avg3s1.json"
            )
            precondition_output = output / (
                f"precondition_repaired_round{precondition_round}_avg3s1.json"
            )
            if not precondition_seed.exists():
                recentering = write_recentered_rework_seed(
                    current_table, precondition_seed, precondition_failed,
                    precondition_round, [precheck], high_power, soa_pilot,
                    require_live_verification=False,
                    correction_gain=0.5,
                    maximum_step_ma=1.5,
                    defer_unusable_evidence=True,
                )
                emit_event(
                    "preconditioning_power_recentering",
                    round=precondition_round,
                    **recentering,
                )
            run_command(
                model_command(
                    "repair-equal-power", "--port", args.port, "--gpib", args.meter,
                    "--input", str(precondition_seed),
                    "--output", str(precondition_output),
                    "--source-table", str(high_power), "--soa-pilot", str(soa_pilot),
                    *inventory_arguments,
                    *phase_candidate_repair_arguments,
                    "--target-power-mw", f"{target_power:.9f}", "--passes", "4",
                    "--power-relative-tolerance", internal_power_tolerance_text,
                    *repair_target_arguments,
                    "--max-rounds", "7", "--max-tune-measurements", "24", "--resume",
                ), log_path,
            )
            require_complete_rows(
                precondition_output, label="快速顺扫预调理后的完整点表",
            )
            current_table = precondition_output

        for certification_round in range(1, MAX_CERTIFICATION_ROUNDS + 1):
            forward = output / (
                f"validation_forward_round{certification_round}_avg3_postcondition.json"
            )
            reverse = output / (
                f"validation_reverse_round{certification_round}_avg3_postcondition.json"
            )
            run_command(
                model_command(
                    "validate-equal-power", "--port", args.port, "--gpib", args.meter,
                    "--input", str(current_table), "--output", str(forward),
                    "--direction", "forward", "--power-relative-tolerance",
                    certification_power_tolerance,
                    "--samples-per-point", "3", "--repeat-settle-s", "0.02",
                    "--quiet-failures", "--resume",
                ), log_path, accepted_exit_codes=(0, 2),
            )
            forward_failed = validation_failures(forward)
            if forward_failed:
                if certification_round == MAX_CERTIFICATION_ROUNDS:
                    raise RuntimeError(
                        f"{MAX_CERTIFICATION_ROUNDS}轮自动修复后正向仍有"
                        f"{len(forward_failed)}个复测失败点，未发布正式点表"
                    )
                rework_seed = output / (
                    f"certification_rework_seed_forward_round{certification_round}_avg3_postcondition.json"
                )
                rework_output = output / (
                    f"certification_repaired_forward_round{certification_round}_avg3_postcondition.json"
                )
                rework_targets = validation_rework_targets(
                    forward, args.power_relative_tolerance,
                )
                emit_event(
                    "certification_guard_band_rework",
                    round=certification_round,
                    release_failed=len(forward_failed),
                    rework_targets=len(rework_targets),
                    wavelength_guard_pm=CERTIFICATION_REPAIR_WAVELENGTH_GUARD_PM,
                    power_guard_relative=min(
                        args.power_relative_tolerance,
                        CERTIFICATION_REPAIR_POWER_GUARD_RELATIVE,
                    ),
                )
                if not rework_seed.exists():
                    recentering = write_recentered_rework_seed(
                        current_table, rework_seed, rework_targets, certification_round,
                        [forward], high_power, soa_pilot,
                    )
                    emit_event(
                        "certification_power_recentering",
                        round=certification_round,
                        direction="forward",
                        **recentering,
                    )
                run_command(
                    model_command(
                        "repair-equal-power", "--port", args.port, "--gpib", args.meter,
                        "--input", str(rework_seed), "--output", str(rework_output),
                        "--source-table", str(high_power), "--soa-pilot", str(soa_pilot),
                        *inventory_arguments,
                        *phase_candidate_repair_arguments,
                        "--target-power-mw", f"{target_power:.9f}", "--passes", "4",
                        "--power-relative-tolerance", certification_power_tolerance,
                        *repair_target_arguments,
                        "--max-rounds", "7", "--max-tune-measurements", "24", "--resume",
                    ), log_path,
                )
                require_complete_rows(
                    rework_output, label="正向独立复测失败点同点自动修复",
                )
                current_table = rework_output
                continue
            if args.certification_direction == "forward":
                require_complete_rows(forward, label="固定升序独立复测")
                final_forward = forward
                final_reverse = None
                break
            run_command(
                model_command(
                    "validate-equal-power", "--port", args.port, "--gpib", args.meter,
                    "--input", str(current_table), "--output", str(reverse),
                    "--direction", "reverse", "--power-relative-tolerance",
                    certification_power_tolerance,
                    "--samples-per-point", "3", "--repeat-settle-s", "0.02",
                    "--quiet-failures", "--resume",
                ), log_path, accepted_exit_codes=(0, 2),
            )
            failed = validation_failures(forward) | validation_failures(reverse)
            if not failed:
                require_complete_rows(forward, label="正向独立复测")
                require_complete_rows(reverse, label="反向独立复测")
                final_forward, final_reverse = forward, reverse
                break
            if certification_round == MAX_CERTIFICATION_ROUNDS:
                raise RuntimeError(
                    f"{MAX_CERTIFICATION_ROUNDS}轮自动修复后仍有"
                    f"{len(failed)}个复测失败点，未发布正式点表"
                )
            rework_seed = output / (
                f"certification_rework_seed_round{certification_round}_avg3_postcondition.json"
            )
            rework_output = output / (
                f"certification_repaired_round{certification_round}_avg3_postcondition.json"
            )
            if not rework_seed.exists():
                recentering = write_recentered_rework_seed(
                    current_table, rework_seed, failed, certification_round,
                    [forward, reverse], high_power, soa_pilot,
                )
                emit_event(
                    "certification_power_recentering",
                    round=certification_round,
                    direction="both",
                    **recentering,
                )
            run_command(
                model_command(
                    "repair-equal-power", "--port", args.port, "--gpib", args.meter,
                    "--input", str(rework_seed), "--output", str(rework_output),
                    "--source-table", str(high_power), "--soa-pilot", str(soa_pilot),
                    *inventory_arguments,
                    *phase_candidate_repair_arguments,
                    "--target-power-mw", f"{target_power:.9f}", "--passes", "4",
                    "--power-relative-tolerance", certification_power_tolerance,
                    *repair_target_arguments,
                    "--max-rounds", "7", "--max-tune-measurements", "24", "--resume",
                ), log_path,
            )
            require_complete_rows(rework_output, label="独立复测失败点自动修复")
            current_table = rework_output
    if final_forward is None or (
        args.certification_direction == "bidirectional" and final_reverse is None
    ):
        raise RuntimeError("未生成所选运行路径的独立复测报告")
    state["current_certified_table"] = str(current_table)
    state["forward_validation"] = str(final_forward)
    state["reverse_validation"] = (
        str(final_reverse) if final_reverse is not None else None
    )
    save_state(state_path, state)
    emit_event(
        "stage_completed", name="bidirectional_certification", index=stage, total=total,
        certification_direction=args.certification_direction,
    )
    stage += 1

    def export_command(prefix: Path, feedback_path: Path) -> list[str]:
        command = [
            sys.executable, str(EXPORT_SCRIPT), "--source", str(current_table),
            "--prefix", str(prefix), "--feedback", str(feedback_path),
            "--forward-validation", str(final_forward),
            "--certification-mode",
            "forward-only"
            if args.certification_direction == "forward"
            else "bidirectional",
            "--power-relative-tolerance",
            certification_power_tolerance,
        ]
        if final_reverse is not None:
            command.extend(["--reverse-validation", str(final_reverse)])
        return command

    run_stage(
        state_path, state, "preliminary_export", stage, total,
        lambda: run_command(
            export_command(preliminary_prefix, output / "absent.json"),
            log_path, dry_run=args.dry_run,
        ),
        [preliminary_prefix.with_suffix(".json"), preliminary_prefix.with_suffix(".csv")],
    )
    stage += 1
    run_stage(
        state_path, state, "feedback_reference", stage, total,
        lambda: run_command(
            [
                sys.executable, str(FEEDBACK_SCRIPT), "--table",
                str(preliminary_prefix.with_suffix(".json")), "--output", str(feedback),
                "--port", args.port, "--meter", args.meter, "--meter-stride", "50",
            ], log_path, dry_run=args.dry_run,
        ), [feedback],
    )
    stage += 1
    run_stage(
        state_path, state, "final_export", stage, total,
        lambda: run_command(
            export_command(final_prefix, feedback), log_path, dry_run=args.dry_run,
        ),
        [
            final_prefix.with_suffix(".json"), final_prefix.with_suffix(".csv"),
            final_prefix.with_suffix(".h"),
            final_prefix.with_name(final_prefix.name + "_transition_guards").with_suffix(".json"),
        ],
    )
    stage += 1
    if not args.dry_run:
        feedback_payload = read_json(feedback)
        if len(feedback_payload.get("rows", [])) != TARGET_COUNT:
            raise RuntimeError("PDT/PDR参考不是完整2001点，未发布")
        promotion = promote_runtime_table(
            output,
            final_prefix,
            feedback,
            runtime_dir=args.runtime_dir,
        )
        final_payload = read_json(final_prefix.with_suffix(".json"))
        final_summary = final_payload.get("final_summary", {})
        state["result"] = {
            "status": "complete",
            "completed_utc": utc_now(),
            "target_power_mw": target_power,
            "final_summary": final_summary,
            **promotion,
        }
        save_state(state_path, state)
        emit_event(
            "complete",
            target_power_mw=target_power,
            maximum_abs_power_error_percent=final_summary.get(
                "maximum_abs_power_error_percent"
            ),
            power_peak_to_peak_percent=final_summary.get(
                "power_peak_to_peak_percent_of_target"
            ),
            output=str(final_prefix.with_suffix(".json")),
            backup_dir=promotion["backup_dir"],
        )
    else:
        emit_event("complete", dry_run=True, output=str(final_prefix.with_suffix(".json")))
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=HERE,
        help=(
            "directory that receives the completed runtime table; use a "
            "machine-specific directory to avoid replacing another unit's calibration"
        ),
    )
    parser.add_argument(
        "--warm-start-table", type=Path, default=DEFAULT_WARM_START_TABLE,
        help=(
            "complete older live 2001-point table used only for PHASE/WAVE branch seeds; "
            "every point is remeasured at the dedicated 145 mA profile"
        ),
    )
    parser.add_argument(
        "--from-zero", action="store_true",
        help=(
            "discover this laser only from new live WAVE/PHASE maps; do not read "
            "a legacy 2001-point table or the shared historical inventory"
        ),
    )
    parser.add_argument(
        "--certification-direction",
        choices=("forward", "bidirectional"),
        default="forward",
        help=(
            "forward certifies the fixed increasing-wavelength path used by "
            "2001/45-point runtime scans; bidirectional additionally audits hysteresis"
        ),
    )
    parser.add_argument(
        "--power-relative-tolerance",
        type=float,
        default=DEFAULT_SINGLE_POINT_POWER_REL_TOLERANCE,
        help=(
            "single-point power release tolerance as a fraction; "
            "0.03 means ±3% around the common target"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the complete plan without opening hardware or writing calibration results",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="only verify AQ6150B and shuttered 145 mA firmware support",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.power_relative_tolerance) or not (
        0.0 < args.power_relative_tolerance <= 0.10
    ):
        raise ValueError("单点功率容差必须在0到10%之间")
    emit_event(
        "started", profile=PROFILE, current_limit_ma=CURRENT_LIMIT_MA,
        output_dir=str(args.output_dir.resolve()),
        single_point_power_relative_tolerance=args.power_relative_tolerance,
        discovery_mode="from_zero" if args.from_zero else "legacy_seeded",
    )
    try:
        if args.preflight_only:
            if args.dry_run:
                emit_event("preflight", dry_run=True)
            else:
                emit_event("preflight", **hardware_preflight(args.port, args.meter))
            emit_event("complete", preflight_only=True)
            return 0
        run_pipeline(args)
        return 0
    except KeyboardInterrupt:
        emit_event("cancelled", message="操作员已取消；保留检查点，可从原目录继续")
        return 130
    except Exception as exc:
        try:
            state_path = args.output_dir.resolve() / STATE_NAME
            if state_path.exists():
                failed_state = load_state(state_path)
                failed_state["result"] = {
                    "status": "failed",
                    "failed_utc": utc_now(),
                    "error": str(exc),
                }
                save_state(state_path, failed_state)
        except Exception:
            pass
        emit_event("failed", error=str(exc))
        print(f"自动校准失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if not args.dry_run:
            try:
                best_effort_final_shutter(args.port)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
