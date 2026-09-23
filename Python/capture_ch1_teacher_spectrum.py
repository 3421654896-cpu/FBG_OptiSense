"""Safely capture a settled CH1 teacher spectrum over the LAN raw channel.

The script replays the wavelength-meter-audited 2001-point operational table
through the same exact DAC acknowledgement, transition-guard and robust
stability code used by the desktop single-value/unlimited-accuracy modes.  It
exports CH1 only; the other optical ADC channels are never written to the
teacher-spectrum file.  This module deliberately has no Mach3 dependency and
never issues machine-motion commands.

Indices are zero based and ``--end-index`` is inclusive.  Reusing the same
output path resumes a previously interrupted, forward-only capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import app_JDSU as app
from fbg_lan_serial import FbgLanSerial
from laser_power_calibration import save_json


HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE / "fullband_equal_power_operational_2001.csv"
GUARD_PATH = HERE / "fullband_equal_power_operational_2001_transition_guards.json"
SCHEMA = "fbg_ch1_teacher_spectrum_v1"
CH1_MONITOR_COLUMN = 3  # Monitor frame columns: PDT, PDR, CH0, CH1, CH2, CH3.
CH0_SAFETY_SELECTOR = 0  # CH0 is not a target; keep it at the lowest 2 kOhm range.
DEFAULT_HOST = "192.168.3.46"
DEFAULT_CONNECT_TIMEOUT_S = 15.0
CHECKPOINT_RETRY_TIMEOUT_S = 3.0


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist a checkpoint despite short Windows reader/indexer locks.

    ``save_json`` already writes to a temporary file and atomically replaces
    the destination.  On Windows, an antivirus/indexer or a live diagnostic
    reader can briefly deny that final replace.  Retrying the same immutable
    payload preserves atomicity and prevents an otherwise healthy optical
    scan from being abandoned.
    """

    deadline = time.monotonic() + CHECKPOINT_RETRY_TIMEOUT_S
    delay_s = 0.01
    while True:
        try:
            save_json(path, payload)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 1.7, 0.20)


def wait_connected(device: FbgLanSerial, timeout_s: float) -> None:
    device.open()
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if device.connected:
            return
        time.sleep(0.05)
    raise RuntimeError(f"{timeout_s:.1f}秒内未建立局域网原始数据通道")


def command_with_link_retry(
    worker: app.UnlimitedAccuracyWorker,
    codes: Sequence[int],
    *,
    soa_mode: int = app.PI11210_SOA_SOURCE_MODE,
):
    """Retry transport loss without ever accepting a relaxed DAC ACK."""

    last_error: OSError | None = None
    for attempt in range(3):
        try:
            return worker._command_and_ack(codes, soa_mode=soa_mode)
        except OSError as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"局域网命令连续3次未确认：{last_error}")


def summarize_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows, key=int)]
    stable = [row for row in ordered if bool(row.get("stable"))]
    voltages = [float(row["ch1_voltage_v"]) for row in ordered]
    return {
        "captured_points": len(ordered),
        "stable_points": len(stable),
        "review_points": len(ordered) - len(stable),
        "first_index": int(ordered[0]["index"]) if ordered else None,
        "last_index": int(ordered[-1]["index"]) if ordered else None,
        "ch1_voltage_min_v": min(voltages, default=None),
        "ch1_voltage_max_v": max(voltages, default=None),
    }


def validate_existing_payload(
    payload: dict[str, Any],
    *,
    table_sha256: str,
    guard_sha256: str,
    settle_s: float,
    selector: int,
) -> None:
    if payload.get("schema") != SCHEMA:
        raise ValueError("输出文件不是CH1教师光谱检查点")
    source = payload.get("source", {})
    if source.get("table_sha256") != table_sha256:
        raise ValueError("输出文件对应的2001点标定表已改变，请改用新的输出文件")
    if source.get("transition_guards_sha256") != guard_sha256:
        raise ValueError("输出文件对应的过渡保护表已改变，请改用新的输出文件")
    settings = payload.get("settings", {})
    if not math.isclose(
        float(settings.get("settle_s", float("nan"))),
        float(settle_s),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("续跑的settle与已有检查点不一致")
    if int(settings.get("ch1_feedback_selector", -1)) != int(selector):
        raise ValueError("续跑的CH1模拟跨阻与已有检查点不一致")
    if int(payload.get("channel", -1)) != 1:
        raise ValueError("输出文件不是CH1数据")
    rows = payload.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("输出文件rows字段损坏")
    for key, row in rows.items():
        index = int(key)
        if index < 0 or index >= app.FULLBAND_ACCURACY_POINT_COUNT:
            raise ValueError(f"输出文件包含越界索引：{index}")
        if not isinstance(row, dict) or int(row.get("index", -1)) != index:
            raise ValueError(f"输出文件第{index}点结构损坏")


def load_or_create_payload(
    output: Path,
    *,
    host: str,
    settle_s: float,
    selector: int,
    table_sha256: str,
    guard_sha256: str,
) -> dict[str, Any]:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        validate_existing_payload(
            payload,
            table_sha256=table_sha256,
            guard_sha256=guard_sha256,
            settle_s=settle_s,
            selector=selector,
        )
        return payload

    return {
        "schema": SCHEMA,
        "created": now_text(),
        "updated": now_text(),
        "channel": 1,
        "spectrum_axis": "wavelength_meter_audited_measured_wavelength_nm",
        "scan_direction": "forward",
        "source": {
            "table": TABLE_PATH.name,
            "table_sha256": table_sha256,
            "transition_guards": GUARD_PATH.name,
            "transition_guards_sha256": guard_sha256,
            "point_count": app.FULLBAND_ACCURACY_POINT_COUNT,
        },
        "settings": {
            "board_host": str(host),
            "settle_s": float(settle_s),
            "ch1_feedback_selector": int(selector),
            "ch1_feedback_kohm": app.PD_FEEDBACK_KOHM_BY_SELECTOR[selector],
            "ch0_safety_selector": CH0_SAFETY_SELECTOR,
            "ch0_safety_kohm": app.PD_FEEDBACK_KOHM_BY_SELECTOR[
                CH0_SAFETY_SELECTOR
            ],
            "stability_algorithm": "app_JDSU.fullband_accuracy_statistics",
            "exported_adc_channels": [1],
        },
        "sessions": [],
        "rows": {},
        "summary": summarize_rows({}),
        "soa_shutter_confirmed": False,
    }


def first_pending_index(
    rows: dict[str, dict[str, Any]], start_index: int, end_index: int
) -> int | None:
    """Return the next missing index and reject non-forward holes."""

    missing_seen = False
    first_missing: int | None = None
    for index in range(start_index, end_index + 1):
        present = str(index) in rows
        if not present and not missing_seen:
            first_missing = index
            missing_seen = True
        elif present and missing_seen:
            raise ValueError(
                f"检查点在第{first_missing}点之后又包含第{index}点；"
                "为保护正向光路状态，请使用新的输出文件"
            )
    return first_missing


def establish_forward_state(
    worker: app.EqualIntervalWorker,
    points: Sequence[app.FullbandAccuracyPoint],
    next_index: int,
) -> None:
    predecessor_index = max(0, int(next_index) - 1)
    predecessor = points[predecessor_index]
    if command_with_link_retry(worker, predecessor.codes) is None:
        raise RuntimeError("无法建立续跑点的前一波长状态")
    if not worker._interruptible_wait(app.FULLBAND_ACCURACY_FIRST_POINT_SETTLE_S):
        raise RuntimeError("建立正向扫描初始状态时被中止")


def acquire_settled_ch1(
    worker: app.EqualIntervalWorker,
    point: app.FullbandAccuracyPoint,
    settle_s: float,
    *,
    retain_monitor_samples: bool = False,
    apply_codes: bool = True,
    max_samples: int | None = None,
    point_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Acquire a robust window, exporting only the CH1 result."""

    sample_limit = (
        app.FULLBAND_ACCURACY_MAX_SAMPLES
        if max_samples is None else int(max_samples)
    )
    timeout_s = (
        app.FULLBAND_ACCURACY_POINT_TIMEOUT_S
        if point_timeout_s is None else float(point_timeout_s)
    )
    if sample_limit < app.FULLBAND_ACCURACY_MIN_SAMPLES:
        raise ValueError("max_samples不能小于稳定判定的最小帧数")
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("point_timeout_s必须是正的有限秒数")

    started = time.monotonic()
    if apply_codes and command_with_link_retry(worker, point.codes) is None:
        raise RuntimeError("扫描已中止")
    if not worker._interruptible_wait(settle_s):
        raise RuntimeError("稳定等待期间扫描被中止")

    # Discard every monitor frame produced before the requested stable window.
    worker.port.reset_input_buffer()
    worker._rx_buffer.clear()
    samples: list[tuple[int, ...]] = []
    median = sigma = drift = np.full(6, np.nan, dtype=float)
    settled = False
    deadline = time.monotonic() + timeout_s
    while worker.running and time.monotonic() < deadline:
        for frame in worker._read_available_frames():
            values = app.decode_single_value_monitor_frame(frame)
            if values is None or any(value < 0 or value > 4095 for value in values):
                continue
            samples.append(tuple(int(value) for value in values))
            median, sigma, drift, settled = app.fullband_accuracy_statistics(samples)
            if settled or len(samples) >= sample_limit:
                break
        if settled or len(samples) >= sample_limit:
            break
        time.sleep(0.001)

    if not samples:
        raise RuntimeError(
            f"{point.target_nm:.2f} nm等待{settle_s:.3f}秒后未收到ADC监测数据"
        )
    median, sigma, drift, settled = app.fullband_accuracy_statistics(samples)
    ch1_code = float(median[CH1_MONITOR_COLUMN])
    ch1_saturated = ch1_code >= 4080.0
    stable = bool(settled and not ch1_saturated)
    code_to_volt = app.PD_ADC_REFERENCE_V / app.PD_ADC_CODE_COUNT
    result = {
        "index": int(point.index),
        "target_wavelength_nm": float(point.target_nm),
        "measured_wavelength_nm": float(point.measured_nm),
        "dac_codes": [int(value) for value in point.codes],
        "ch1_adc_code": ch1_code,
        "ch1_voltage_v": ch1_code * code_to_volt,
        "ch1_sigma_codes": float(sigma[CH1_MONITOR_COLUMN]),
        "ch1_sigma_v": float(sigma[CH1_MONITOR_COLUMN] * code_to_volt),
        "ch1_drift_codes": float(drift[CH1_MONITOR_COLUMN]),
        "ch1_drift_v": float(drift[CH1_MONITOR_COLUMN] * code_to_volt),
        "ch1_saturated": bool(ch1_saturated),
        "stable": stable,
        "window_stable": stable,
        "repeatability_verified": False,
        "stability_scope": "within_window_only_not_cross_scan_repeatability",
        "all_monitor_channels_stable": bool(settled),
        "sample_count": len(samples),
        "sample_limit": sample_limit,
        "point_timeout_s": timeout_s,
        "settle_s": float(settle_s),
        "elapsed_s": time.monotonic() - started,
        "captured": now_text(),
        "adc_acquisition_method": "checkRT/ADC_Write_Read_Stable",
        "adc_samples_are_uniform_raw_stream": False,
        "firmware_window_selected_samples": True,
    }
    if retain_monitor_samples:
        result["monitor_channels"] = ["PDT", "PDR", "CH0", "CH1", "CH2", "CH3"]
        result["monitor_samples"] = [list(row) for row in samples]
        result["adc_value_kind"] = "window_selected_adc"
    return result


def safe_shutter_and_confirm(
    device: FbgLanSerial, worker: app.UnlimitedAccuracyWorker
) -> bool:
    """Enter EXTRA mode, assert SOA CLR and require its exact status ACK."""

    if not device.is_open:
        return False
    if not device.connected:
        wait_connected(device, min(5.0, DEFAULT_CONNECT_TIMEOUT_S))
    device.write(app.build_work_mode_command(2))
    time.sleep(0.12)
    device.reset_input_buffer()
    worker._rx_buffer.clear()
    return bool(
        command_with_link_retry(
            worker,
            worker._last_codes,
            soa_mode=app.PI11210_SOA_SHUTTER_MODE,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="局域网板卡IP")
    parser.add_argument("--output", type=Path, required=True, help="增量JSON文件")
    parser.add_argument(
        "--settle",
        "--settle-s",
        dest="settle_s",
        type=float,
        default=app.FULLBAND_ACCURACY_MIN_SETTLE_S,
        help="每次DAC回读确认后的激光稳定时间（秒，至少0.20）",
    )
    parser.add_argument(
        "--selector",
        type=int,
        choices=range(len(app.PD_FEEDBACK_KOHM_BY_SELECTOR)),
        default=0,
        help="CH1模拟跨阻选择码：0=2k, 1=40k, 2=5k, 3=20k",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="起始索引（0基，包含）"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=app.FULLBAND_ACCURACY_POINT_COUNT - 1,
        help="结束索引（0基，包含）",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_S,
        help="等待板卡局域网原始通道的最长秒数",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.settle_s) or args.settle_s < app.FULLBAND_ACCURACY_MIN_SETTLE_S:
        raise ValueError(
            f"settle必须至少为{app.FULLBAND_ACCURACY_MIN_SETTLE_S:.2f}秒，"
            "教师光谱不接受瞬态采样"
        )
    if not math.isfinite(args.connect_timeout_s) or args.connect_timeout_s <= 0.0:
        raise ValueError("connect-timeout-s必须为正数")
    maximum = app.FULLBAND_ACCURACY_POINT_COUNT - 1
    if not 0 <= args.start_index <= maximum:
        raise ValueError(f"start-index必须位于0～{maximum}")
    if not args.start_index <= args.end_index <= maximum:
        raise ValueError(f"end-index必须位于start-index～{maximum}")


def run_capture(args: argparse.Namespace) -> int:
    validate_arguments(args)
    points = app.load_fullband_accuracy_table(HERE)
    guards = app.load_fullband_transition_guards(points, HERE)
    table_sha256 = sha256_file(TABLE_PATH)
    guard_sha256 = sha256_file(GUARD_PATH)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = load_or_create_payload(
        args.output,
        host=args.host,
        settle_s=args.settle_s,
        selector=args.selector,
        table_sha256=table_sha256,
        guard_sha256=guard_sha256,
    )
    rows = payload["rows"]
    next_index = first_pending_index(rows, args.start_index, args.end_index)
    if next_index is None:
        print(
            f"requested_range_already_complete={args.start_index}:{args.end_index}",
            flush=True,
        )
        return 0

    session: dict[str, Any] = {
        "started": now_text(),
        "requested_start_index": int(args.start_index),
        "requested_end_index": int(args.end_index),
        "actual_start_index": int(next_index),
        "status": "starting",
    }
    payload["sessions"].append(session)
    payload["updated"] = now_text()
    payload["summary"] = summarize_rows(rows)
    save_checkpoint(args.output, payload)

    device = FbgLanSerial(
        board_host=args.host,
        timeout=0.10,
        write_timeout=3.0,
    )
    worker = app.EqualIntervalWorker(
        device,
        points,
        guards,
        settle_s=args.settle_s,
        feedback_selectors=(CH0_SAFETY_SELECTOR, args.selector),
    )
    scan_error: BaseException | None = None
    shutter_error: Exception | None = None
    shutter_confirmed = False
    try:
        wait_connected(device, args.connect_timeout_s)
        device.write(app.build_work_mode_command(2))
        if not worker._interruptible_wait(0.12):
            raise RuntimeError("切换单值模式时被中止")
        device.reset_input_buffer()
        worker._rx_buffer.clear()
        if not worker._feedback_command_and_ack():
            raise RuntimeError("未能确认CH1模拟跨阻")
        if not worker._interruptible_wait(0.005):
            raise RuntimeError("模拟跨阻确认后被中止")
        device.reset_input_buffer()
        worker._rx_buffer.clear()

        establish_forward_state(worker, points, next_index)
        session["status"] = "capturing"
        for sequence_index, index in enumerate(
            range(next_index, args.end_index + 1)
        ):
            if (
                sequence_index > 0
                and sequence_index % app.EQUAL_INTERVAL_FEEDBACK_VERIFY_EVERY_POINTS == 0
            ):
                if not worker._feedback_command_and_ack():
                    raise RuntimeError("周期性复核CH1模拟跨阻失败")
            guard = guards.get(index)
            if guard is not None and not worker._apply_guard(guard):
                raise RuntimeError(f"第{index}点的跨模过渡保护被中止")
            row = acquire_settled_ch1(worker, points[index], args.settle_s)
            rows[str(index)] = row
            payload["updated"] = now_text()
            payload["summary"] = summarize_rows(rows)
            session["last_saved_index"] = int(index)
            save_checkpoint(args.output, payload)
            print(
                f"point={index}/{args.end_index} "
                f"target={row['target_wavelength_nm']:.2f}nm "
                f"measured={row['measured_wavelength_nm']:.8f}nm "
                f"ch1={row['ch1_voltage_v']:.6f}V "
                f"stable={int(row['stable'])} samples={row['sample_count']}",
                flush=True,
            )
        session["status"] = "capture_complete"
    except BaseException as exc:
        scan_error = exc
        session["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        session["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            shutter_confirmed = safe_shutter_and_confirm(device, worker)
        except Exception as exc:
            shutter_error = exc
            print(f"shutter_error={exc}", file=sys.stderr, flush=True)
        finally:
            device.close()
            payload["soa_shutter_confirmed"] = bool(shutter_confirmed)
            payload["updated"] = now_text()
            payload["summary"] = summarize_rows(rows)
            session["finished"] = now_text()
            session["soa_shutter_confirmed"] = bool(shutter_confirmed)
            if shutter_error is not None:
                session["shutter_error"] = str(shutter_error)
            if scan_error is None and not shutter_confirmed:
                session["status"] = "failed"
            elif scan_error is None:
                session["status"] = "complete"
            try:
                save_checkpoint(args.output, payload)
            except Exception as exc:
                if scan_error is None:
                    scan_error = exc
                else:
                    print(f"checkpoint_error={exc}", file=sys.stderr, flush=True)
            print(
                f"soa_shutter_confirmed={int(shutter_confirmed)}",
                flush=True,
            )

    if scan_error is not None:
        raise scan_error
    if not shutter_confirmed:
        detail = f"：{shutter_error}" if shutter_error is not None else ""
        raise RuntimeError(f"SOA安全关光未获回读确认{detail}")
    print("summary=" + json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_capture(args)
    except KeyboardInterrupt:
        print("capture_interrupted=1", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"capture_error={exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
