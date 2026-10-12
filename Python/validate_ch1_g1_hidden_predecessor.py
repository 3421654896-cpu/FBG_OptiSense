"""Qualify the one hidden predecessor needed by the CH1 G1 sparse path.

This tool is deliberately separate from candidate selection and calibration.  It
never edits the 45-point candidate, the AQ6150 calibration report, desktop
tables, firmware sources, or the controller.  It only replays the fixed local
runtime chain

    order0 -> order1 -> fullband index140 -> order2 -> order3

and writes an auditable JSON result.  The index140 state is a control-only
predecessor: it is not sampled and is not added to the 45-point spectrum.

Two batches of five trials are run for each requested hidden hold time.  Hold
times are tried in ascending order and the first value with 10/10 fully passing
trials is selected.  Every DAC write uses the production exact-ACK path.  Every
hardware exit, including exceptions and Ctrl-C, requests and confirms the
exact SOA shutter before success can be reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import app_JDSU as app
from calibrate_selected_ch1_table import (
    currents_ma,
    load_candidates,
    normalized_smsr_db,
    now_text,
    reading_dict,
    sha256_file,
    validate_absolute_currents,
    validate_and_hash_fullband_points,
    validate_meter_reading,
)
from capture_ch1_teacher_spectrum import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_HOST,
    command_with_link_retry,
    safe_shutter_and_confirm,
    save_checkpoint,
    wait_connected,
)
from fbg_lan_serial import FbgLanSerial
from laser_power_calibration import AQ6150B, Reading


HERE = Path(__file__).resolve().parent
SCHEMA = "ch1_g1_hidden_predecessor_path_validation/v1"
DEFAULT_CANDIDATE = HERE / "artifacts" / "ch1_selected_candidate_45_v2_20260905.json"
DEFAULT_CALIBRATION_REPORT = (
    HERE / "artifacts" / "ch1_45_aq6150_calibration_v2_20260905.json"
)

CHAIN_ORDERS = (0, 1, 2, 3)
CHAIN_FULLBAND_INDICES = (118, 122, 141, 170)
HIDDEN_FULLBAND_INDEX = 140
HIDDEN_TARGET_NM = 1527.800000
HIDDEN_CODES = (58981, 52531, 66, 4687, 1201)
EXPECTED_CALIBRATED_CODES = {
    0: (58981, 52431, 5288, 5120, 1638),
    1: (58981, 53086, 4571, 4710, 1434),
    2: (58981, 52431, 15877, 4710, 1229),
    3: (58981, 54396, 2655, 3072, 16179),
}
EXPECTED_TARGET_POWER_MW = 1.667248938547319
POWER_TOLERANCE_MW = 0.03334497877094638
WAVELENGTH_TOLERANCE_PM = 2.0
MINIMUM_SMSR_DB = 20.0
ALLOWED_HOLD_MS = (5, 10, 20, 40, 80)
BATCH_COUNT = 2
TRIALS_PER_BATCH = 5
CHAIN_STATE_HOLD_S = 0.08
TARGET_SETTLE_S = 0.30


def _load_calibrated_chain(
    candidate_path: Path,
    calibration_path: Path,
    fullband_points: Sequence[Any],
) -> tuple[tuple[Any, ...], dict[int, tuple[int, ...]], dict[str, Any]]:
    """Load and pin the exact evidence used by this one-purpose test."""

    _, candidates = load_candidates(candidate_path)
    if len(candidates) != 45:
        raise ValueError("G1局部路径验证只允许当前45点CH1候选表")
    for order, expected_index in zip(CHAIN_ORDERS, CHAIN_FULLBAND_INDICES):
        point = candidates[order]
        if point.order != order or point.fullband_index != expected_index:
            raise ValueError(
                f"固定链order{order}必须是fullband index{expected_index}"
            )

    report = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("标定报告根结点必须是对象")
    candidate_sha256 = sha256_file(candidate_path)
    candidate_source = report.get("candidate_source")
    report_candidate_sha256 = (
        candidate_source.get("sha256")
        if isinstance(candidate_source, Mapping)
        else report.get("candidate_sha256")
    )
    if report_candidate_sha256 != candidate_sha256:
        raise ValueError("标定报告与45点候选表SHA256不一致")
    if not bool(report.get("shutter_confirmed")):
        raise ValueError("标定报告的最终SOA关光未获确认")

    fullband_sha256 = validate_and_hash_fullband_points(fullband_points)
    if report.get("fullband_predecessor_sha256") != fullband_sha256:
        raise ValueError("标定报告与当前2001点前驱表不一致")
    if len(fullband_points) != app.FULLBAND_ACCURACY_POINT_COUNT:
        raise ValueError("必须使用完整2001点审计表")
    hidden_point = fullband_points[HIDDEN_FULLBAND_INDEX]
    hidden_target = float(getattr(hidden_point, "target_nm", HIDDEN_TARGET_NM))
    if abs(hidden_target - HIDDEN_TARGET_NM) > 1e-9:
        raise ValueError("fullband index140不是1527.800000 nm")
    hidden_codes = validate_absolute_currents(hidden_point.codes)
    if hidden_codes != HIDDEN_CODES:
        raise ValueError(
            "fullband index140规范化码不是"
            f"{list(HIDDEN_CODES)}，拒绝硬件验证"
        )

    calibration = report.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("标定报告缺calibration")
    summary = calibration.get("summary")
    if not isinstance(summary, Mapping) or not bool(summary.get("all_points_passed")):
        raise ValueError("45点共同功率标定尚未全部通过")
    target_power = float(calibration.get("target_power_mw", float("nan")))
    if (
        not math.isfinite(target_power)
        or abs(target_power - EXPECTED_TARGET_POWER_MW) > 1e-9
    ):
        raise ValueError(
            f"共同功率必须是{EXPECTED_TARGET_POWER_MW:.15f} mW"
        )
    raw_rows = calibration.get("rows")
    if not isinstance(raw_rows, Mapping):
        raise ValueError("标定报告缺calibration.rows")

    chain_codes: dict[int, tuple[int, ...]] = {}
    for order in CHAIN_ORDERS:
        row = raw_rows.get(str(order))
        if not isinstance(row, Mapping) or not bool(row.get("passed")):
            raise ValueError(f"order{order}没有通过的标定码")
        codes = validate_absolute_currents(row.get("codes", ()))
        if codes != EXPECTED_CALIBRATED_CODES[order]:
            raise ValueError(
                f"order{order}标定码已变更：{list(codes)}，"
                "请生成新的局部验证工具版本"
            )
        chain_codes[order] = codes
    return candidates, chain_codes, dict(report)


def _validate_hold_schedule(values: Sequence[object]) -> tuple[int, ...]:
    if not values:
        raise ValueError("至少需要一个hidden hold时延")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("hidden hold必须是整数毫秒")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("hidden hold必须是整数毫秒") from exc
        if not math.isfinite(numeric) or numeric != math.trunc(numeric):
            raise ValueError("hidden hold必须是整数毫秒")
        parsed.append(int(numeric))
    if any(value not in ALLOWED_HOLD_MS for value in parsed):
        raise ValueError(f"hidden hold只允许{list(ALLOWED_HOLD_MS)} ms")
    if parsed != sorted(set(parsed)):
        raise ValueError("hidden hold必须严格递增且不重复")
    return tuple(parsed)


def _wait(worker: Any, seconds: float, label: str) -> None:
    if not worker._interruptible_wait(float(seconds)):
        raise RuntimeError(f"{label}等待被中止")


def _command_exact(
    worker: Any,
    codes: Sequence[int],
    *,
    label: str,
    hold_s: float,
) -> dict[str, Any]:
    exact = validate_absolute_currents(codes)
    acknowledgement = command_with_link_retry(worker, exact)
    if acknowledgement is None:
        raise RuntimeError(f"{label}未获精确DAC回读确认")
    last_codes = validate_absolute_currents(worker._last_codes)
    if last_codes != exact:
        raise RuntimeError(
            f"{label}回读码{list(last_codes)}与命令码{list(exact)}不一致"
        )
    _wait(worker, hold_s, label)
    latency: float | None = None
    try:
        candidate = float(acknowledgement)
        if math.isfinite(candidate):
            latency = candidate
    except (TypeError, ValueError):
        pass
    return {
        "label": label,
        "codes": list(exact),
        "currents_ma": currents_ma(exact),
        "exact_ack": True,
        "ack_latency_s": latency,
        "hold_ms": float(hold_s) * 1000.0,
    }


def _measure_and_gate(
    meter: Any,
    *,
    target_nm: float,
    target_power_mw: float,
) -> tuple[dict[str, Any], dict[str, bool]]:
    reading: Reading = meter.measure(target_nm, select_main_peak=True)
    validate_meter_reading(reading)
    payload = reading_dict(reading, target_nm)
    gates = {
        "wavelength": abs(float(payload["wavelength_error_pm"]))
        <= WAVELENGTH_TOLERANCE_PM,
        "common_power": abs(float(payload["power_mw"]) - target_power_mw)
        <= POWER_TOLERANCE_MW,
        "single_mode_smsr": float(payload["normalized_smsr_db"])
        >= MINIMUM_SMSR_DB,
    }
    return payload, gates


def _run_trial(
    worker: Any,
    meter: Any,
    candidates: Sequence[Any],
    chain_codes: Mapping[int, Sequence[int]],
    *,
    hold_ms: int,
    target_power_mw: float,
    batch_index: int,
    trial_index: int,
    trial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if trial is None:
        trial = {
            "batch": batch_index,
            "trial": trial_index,
            "hidden_hold_ms": hold_ms,
            "commands": [],
            "passed": False,
            "status": "running",
        }
    commands = trial.setdefault("commands", [])
    try:
        commands.append(
            _command_exact(
                worker,
                chain_codes[0],
                label="order0_idx118",
                hold_s=CHAIN_STATE_HOLD_S,
            )
        )
        commands.append(
            _command_exact(
                worker,
                chain_codes[1],
                label="order1_idx122",
                hold_s=CHAIN_STATE_HOLD_S,
            )
        )
        commands.append(
            _command_exact(
                worker,
                HIDDEN_CODES,
                label="hidden_idx140",
                hold_s=float(hold_ms) / 1000.0,
            )
        )
        commands.append(
            _command_exact(
                worker,
                chain_codes[2],
                label="order2_idx141",
                hold_s=TARGET_SETTLE_S,
            )
        )
        order2_reading, order2_gates = _measure_and_gate(
            meter,
            target_nm=float(candidates[2].target_nm),
            target_power_mw=target_power_mw,
        )
        trial["order2"] = {"reading": order2_reading, "gates": order2_gates}

        commands.append(
            _command_exact(
                worker,
                chain_codes[3],
                label="order3_idx170",
                hold_s=TARGET_SETTLE_S,
            )
        )
        order3_reading, order3_gates = _measure_and_gate(
            meter,
            target_nm=float(candidates[3].target_nm),
            target_power_mw=target_power_mw,
        )
        trial["order3"] = {"reading": order3_reading, "gates": order3_gates}
        trial["passed"] = bool(
            all(order2_gates.values()) and all(order3_gates.values())
        )
        trial["status"] = "complete"
    except BaseException as exc:
        trial["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        trial["error"] = f"{type(exc).__name__}: {exc}"
        raise
    return trial


def _new_audit(
    *,
    candidate_path: Path,
    calibration_path: Path,
    fullband_sha256: str,
    chain_codes: Mapping[int, Sequence[int]],
    hold_schedule: Sequence[int],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created": now_text(),
        "updated": now_text(),
        "status": "starting",
        "candidate_source": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "calibration_source": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "fullband_predecessor_sha256": fullband_sha256,
        "fixed_chain": [
            {
                "order": order,
                "fullband_index": index,
                "codes": list(chain_codes[order]),
            }
            for order, index in zip(CHAIN_ORDERS, CHAIN_FULLBAND_INDICES)
        ],
        "hidden_predecessor": {
            "fullband_index": HIDDEN_FULLBAND_INDEX,
            "target_wavelength_nm": HIDDEN_TARGET_NM,
            "codes": list(HIDDEN_CODES),
            "currents_ma": currents_ma(HIDDEN_CODES),
            "control_only": True,
            "adc_sampled": False,
            "included_in_45_point_spectrum": False,
        },
        "acceptance": {
            "batches": BATCH_COUNT,
            "trials_per_batch": TRIALS_PER_BATCH,
            "required_passes_per_hold": BATCH_COUNT * TRIALS_PER_BATCH,
            "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
            "target_power_mw": EXPECTED_TARGET_POWER_MW,
            "power_tolerance_mw": POWER_TOLERANCE_MW,
            "minimum_smsr_db": MINIMUM_SMSR_DB,
            "order3_must_also_pass": True,
        },
        "hold_schedule_ms": list(hold_schedule),
        "hold_results": {},
        "accepted_hidden_hold_ms": None,
        "session_id": uuid.uuid4().hex,
        "meter_identity": None,
        "board_host": None,
        "shutter_confirmed": False,
        "writes_candidate": False,
        "writes_calibration_report": False,
        "writes_runtime_table": False,
        "writes_firmware": False,
    }


def execute(
    args: argparse.Namespace,
    *,
    device_factory: Callable[..., Any] = FbgLanSerial,
    meter_factory: Callable[[str], Any] = AQ6150B,
    fullband_points: Sequence[Any] | None = None,
    worker_factory: Callable[..., Any] = app.UnlimitedAccuracyWorker,
    shutter_callback: Callable[[Any, Any], bool] = safe_shutter_and_confirm,
) -> int:
    hold_schedule = _validate_hold_schedule(args.hold_ms)
    connect_timeout_s = float(args.connect_timeout_s)
    if not math.isfinite(connect_timeout_s) or connect_timeout_s <= 0.0:
        raise ValueError("connect-timeout-s必须是正有限数")

    candidate_path = args.candidate.expanduser().resolve()
    calibration_path = args.calibration_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"为避免混合硬件证据，输出已存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    points = (
        tuple(fullband_points)
        if fullband_points is not None
        else app.load_fullband_accuracy_table(HERE)
    )
    candidates, chain_codes, _ = _load_calibrated_chain(
        candidate_path, calibration_path, points
    )
    fullband_sha256 = validate_and_hash_fullband_points(points)
    audit = _new_audit(
        candidate_path=candidate_path,
        calibration_path=calibration_path,
        fullband_sha256=fullband_sha256,
        chain_codes=chain_codes,
        hold_schedule=hold_schedule,
    )
    save_checkpoint(output, audit)

    device = None
    worker = None
    meter = None
    operation_error: BaseException | None = None
    shutter_error: BaseException | None = None
    shutter_confirmed = False
    accepted_hold: int | None = None
    try:
        device = device_factory(board_host=args.host, timeout=0.10, write_timeout=3.0)
        worker = worker_factory(device, (), {}, False)
        wait_connected(device, connect_timeout_s)
        device.write(app.build_work_mode_command(2))
        _wait(worker, 0.12, "切换EXTRA模式")
        device.reset_input_buffer()
        worker._rx_buffer.clear()
        meter = meter_factory(args.gpib)
        audit["meter_identity"] = str(getattr(meter, "identity", "unknown"))
        audit["board_host"] = str(getattr(device, "board_ip", args.host))
        audit["status"] = "running"
        audit["updated"] = now_text()
        save_checkpoint(output, audit)

        for hold_ms in hold_schedule:
            hold_result: dict[str, Any] = {
                "hidden_hold_ms": hold_ms,
                "batches": [],
                "passed_trials": 0,
                "completed_trials": 0,
                "accepted": False,
            }
            audit["hold_results"][str(hold_ms)] = hold_result
            for batch_index in range(BATCH_COUNT):
                batch: dict[str, Any] = {
                    "batch": batch_index,
                    "trials": [],
                    "reset_shutter_confirmed": False,
                }
                hold_result["batches"].append(batch)
                for trial_index in range(TRIALS_PER_BATCH):
                    trial: dict[str, Any] = {
                        "batch": batch_index,
                        "trial": trial_index,
                        "hidden_hold_ms": hold_ms,
                        "commands": [],
                        "passed": False,
                        "status": "running",
                    }
                    # Append before commanding hardware so an exception or
                    # Ctrl-C still leaves every acknowledged partial step in
                    # the final audit checkpoint.
                    batch["trials"].append(trial)
                    _run_trial(
                        worker,
                        meter,
                        candidates,
                        chain_codes,
                        hold_ms=hold_ms,
                        target_power_mw=EXPECTED_TARGET_POWER_MW,
                        batch_index=batch_index,
                        trial_index=trial_index,
                        trial=trial,
                    )
                    hold_result["completed_trials"] += 1
                    if trial["passed"]:
                        hold_result["passed_trials"] += 1
                    audit["updated"] = now_text()
                    save_checkpoint(output, audit)
                batch_shutter = bool(shutter_callback(device, worker))
                batch["reset_shutter_confirmed"] = batch_shutter
                if not batch_shutter:
                    raise RuntimeError(
                        f"{hold_ms}ms第{batch_index + 1}批后SOA关光未获精确回读确认"
                    )
                audit["updated"] = now_text()
                save_checkpoint(output, audit)

            hold_result["accepted"] = bool(
                hold_result["completed_trials"]
                == BATCH_COUNT * TRIALS_PER_BATCH
                and hold_result["passed_trials"]
                == BATCH_COUNT * TRIALS_PER_BATCH
                and all(
                    bool(batch["reset_shutter_confirmed"])
                    for batch in hold_result["batches"]
                )
            )
            audit["updated"] = now_text()
            save_checkpoint(output, audit)
            print(
                f"hidden_hold_ms={hold_ms} "
                f"passes={hold_result['passed_trials']}/"
                f"{hold_result['completed_trials']} "
                f"accepted={int(bool(hold_result['accepted']))}",
                flush=True,
            )
            if hold_result["accepted"]:
                accepted_hold = hold_ms
                audit["accepted_hidden_hold_ms"] = hold_ms
                break
        audit["status"] = "stage_complete"
    except BaseException as exc:
        operation_error = exc
        audit["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        audit["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if device is not None and worker is not None:
            try:
                shutter_confirmed = bool(shutter_callback(device, worker))
            except BaseException as exc:
                shutter_error = exc
                audit["shutter_error"] = f"{type(exc).__name__}: {exc}"
        if meter is not None:
            try:
                meter.close()
            except Exception:
                pass
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        audit["finished"] = now_text()
        audit["shutter_confirmed"] = shutter_confirmed
        audit["updated"] = now_text()
        if operation_error is None and not shutter_confirmed:
            audit["status"] = "failed"
        elif operation_error is None:
            audit["status"] = "complete" if accepted_hold is not None else "rejected"
        save_checkpoint(output, audit)
        print(f"soa_shutter_confirmed={int(bool(shutter_confirmed))}", flush=True)

    if operation_error is not None:
        raise operation_error
    if not shutter_confirmed:
        detail = f"：{shutter_error}" if shutter_error is not None else ""
        raise RuntimeError(f"SOA安全关光未获精确回读确认{detail}")
    return 0 if accepted_hold is not None else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT
    )
    parser.add_argument("--output", type=Path, required=True, help="新的局部路径审计JSON")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument(
        "--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S
    )
    parser.add_argument(
        "--hold-ms",
        type=int,
        nargs="+",
        default=list(ALLOWED_HOLD_MS),
        help="严格递增子集，只允许5/10/20/40/80，首个10/10通过即停",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except KeyboardInterrupt:
        print("g1_hidden_path_interrupted=1", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"g1_hidden_path_error={exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
