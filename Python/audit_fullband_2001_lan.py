"""Audit the operational 2001-point laser table through the LAN raw channel.

The scan replays the exact forward order and transition guards used by the
desktop unlimited-accuracy mode.  AQ6150B measurements are compared with both
the nominal target and the original live calibration.  The operational table
is never modified by this tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import app_JDSU as app
from fbg_lan_serial import FbgLanSerial
from laser_power_calibration import AQ6150B, save_json

try:
    import serial
except ImportError:  # pragma: no cover - only relevant outside the workstation venv
    serial = None


HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE / "fullband_equal_power_operational_2001.csv"
DEFAULT_WAVELENGTH_TOLERANCE_PM = 2.0
DEFAULT_POWER_RELATIVE_TOLERANCE = 0.005
DEFAULT_SMSR_DB = 20.0


def wait_connected(device: FbgLanSerial, timeout_s: float = 15.0) -> None:
    device.open()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if device.connected:
            return
        time.sleep(0.05)
    raise RuntimeError("15秒内未建立局域网原始数据通道")


def load_reference_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row["index"])
            raw_smsr = str(row.get("smsr_db", "")).strip()
            rows[index] = {
                "measured_wavelength_nm": float(row["measured_wavelength_nm"]),
                "measured_power_mw": float(row["measured_power_mw"]),
                "target_power_mw": float(row["target_power_mw"]),
                # AQ6150B has no secondary-mode value to subtract when only
                # one peak is reported; the export intentionally leaves SMSR
                # blank for those valid single-peak rows.
                "smsr_db": float(raw_smsr) if raw_smsr else None,
            }
    if len(rows) != app.FULLBAND_ACCURACY_POINT_COUNT:
        raise RuntimeError(f"正式参考表只有{len(rows)}点，不是2001点")
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(rows: dict[str, dict]) -> dict:
    measured = [row for row in rows.values() if row.get("reading")]
    wave_errors = [float(row["wavelength_error_pm"]) for row in measured]
    wave_drifts = [float(row["wavelength_drift_from_calibration_pm"]) for row in measured]
    powers = [float(row["reading"]["power_mw"]) for row in measured]
    power_drifts = [float(row["power_drift_from_calibration_percent"]) for row in measured]
    smsr_values = [
        99.0 if row["reading"].get("side_mode_suppression_db") is None
        else float(row["reading"]["side_mode_suppression_db"])
        for row in measured
    ]
    failed = [row for row in rows.values() if not row.get("success")]
    return {
        "tested": len(rows),
        "measured": len(measured),
        "passed_current_spec": len(measured) - len(failed),
        "failed_current_spec": len(failed),
        "wavelength_error_pm": {
            "mean": statistics.fmean(wave_errors) if wave_errors else None,
            "median": statistics.median(wave_errors) if wave_errors else None,
            "p95_abs": percentile([abs(value) for value in wave_errors], 0.95),
            "maximum_abs": max((abs(value) for value in wave_errors), default=None),
        },
        "wavelength_drift_from_original_pm": {
            "mean": statistics.fmean(wave_drifts) if wave_drifts else None,
            "median": statistics.median(wave_drifts) if wave_drifts else None,
            "p95_abs": percentile([abs(value) for value in wave_drifts], 0.95),
            "maximum_abs": max((abs(value) for value in wave_drifts), default=None),
        },
        "power_mw": {
            "minimum": min(powers, default=None),
            "median": statistics.median(powers) if powers else None,
            "maximum": max(powers, default=None),
        },
        "power_drift_from_original_percent": {
            "mean": statistics.fmean(power_drifts) if power_drifts else None,
            "median": statistics.median(power_drifts) if power_drifts else None,
            "p95_abs": percentile([abs(value) for value in power_drifts], 0.95),
            "maximum_abs": max((abs(value) for value in power_drifts), default=None),
        },
        "minimum_smsr_db": min(smsr_values, default=None),
        "failed_targets_nm": [float(row["target_nm"]) for row in failed],
    }


def command_with_link_retry(
    worker: app.UnlimitedAccuracyWorker,
    codes,
    soa_mode=app.PI11210_SOA_SOURCE_MODE,
):
    """Retry a lost LAN transport acknowledgement without relaxing DAC ACKs."""

    last_error = None
    for attempt in range(3):
        try:
            return worker._command_and_ack(codes, soa_mode=soa_mode)
        except OSError as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"局域网命令连续3次未确认：{last_error}")


def meter_measure_with_retry(
    meter: AQ6150B,
    target_nm: float,
    *,
    attempts: int = 3,
):
    """Repeat only transient AQ6150B array/empty-read failures.

    A trigger can occasionally be queried while the instrument is swapping its
    wavelength and power arrays.  Retrying the complete trigger is safe; other
    meter errors still fail immediately so a real instrument fault is visible.
    """

    transient_messages = (
        "returned an empty measurement",
        "wavelength/power arrays are inconsistent",
    )
    for attempt in range(1, attempts + 1):
        try:
            return meter.measure(target_nm, select_main_peak=True), attempt
        except RuntimeError as exc:
            if not any(message in str(exc) for message in transient_messages):
                raise
            if attempt >= attempts:
                raise
            time.sleep(0.25 * attempt)
    raise AssertionError("unreachable")


def measure_point(
    worker: app.UnlimitedAccuracyWorker,
    meter: AQ6150B,
    point,
    reference: dict,
    settle_s: float,
) -> dict:
    if command_with_link_retry(worker, point.codes) is None:
        raise RuntimeError("扫描已中止")
    time.sleep(settle_s)
    reading, meter_attempts = meter_measure_with_retry(meter, point.target_nm)
    wavelength_error_pm = (reading.wavelength_nm - point.target_nm) * 1000.0
    wavelength_drift_pm = (
        reading.wavelength_nm - reference["measured_wavelength_nm"]
    ) * 1000.0
    target_power_mw = float(reference["target_power_mw"])
    power_relative_error = (
        reading.power_mw - target_power_mw
    ) / target_power_mw
    power_drift_percent = (
        (reading.power_mw - reference["measured_power_mw"])
        / reference["measured_power_mw"]
        * 100.0
    )
    smsr_db = (
        99.0 if reading.side_mode_suppression_db is None
        else float(reading.side_mode_suppression_db)
    )
    success = (
        abs(wavelength_error_pm) <= DEFAULT_WAVELENGTH_TOLERANCE_PM
        and abs(power_relative_error) <= DEFAULT_POWER_RELATIVE_TOLERANCE
        and smsr_db >= DEFAULT_SMSR_DB
    )
    return {
        "index": int(point.index),
        "target_nm": float(point.target_nm),
        "codes": list(point.codes),
        "success": bool(success),
        "reading": asdict(reading),
        "wavelength_error_pm": wavelength_error_pm,
        "wavelength_drift_from_calibration_pm": wavelength_drift_pm,
        "power_relative_error_percent": power_relative_error * 100.0,
        "power_drift_from_calibration_percent": power_drift_percent,
        "meter_measure_attempts": meter_attempts,
        "calibration_reference": reference,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument(
        "--port",
        default="",
        help="use an exact-ACK USB CDC port (for example COM6) instead of LAN",
    )
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-s", type=float, default=0.24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--indices",
        default="",
        help="comma-separated zero-based indices for a quick transport smoke check",
    )
    args = parser.parse_args()
    if not 0.18 <= args.settle_s <= 2.0:
        raise SystemExit("settle-s必须位于0.18～2.0秒")

    points = app.load_fullband_accuracy_table(HERE)
    guards = app.load_fullband_transition_guards(points, HERE)
    references = load_reference_rows(DEFAULT_TABLE)
    target_powers = {round(float(row["target_power_mw"]), 9) for row in references.values()}
    if len(target_powers) != 1:
        raise RuntimeError("2001点参考表中的目标功率不一致")
    target_power_mw = target_powers.pop()
    selected_indices = (
        [int(value.strip()) for value in args.indices.split(",") if value.strip()]
        if args.indices else list(range(len(points)))
    )
    if any(index < 0 or index >= len(points) for index in selected_indices):
        raise SystemExit("indices超出2001点范围")

    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "method": (
                "USB_exact_forward_path_AQ6150B_independent_audit"
                if args.port
                else "LAN_exact_forward_path_AQ6150B_independent_audit"
            ),
            "source_table": DEFAULT_TABLE.name,
            "board_host": args.host,
            "board_port": args.port or None,
            "scan_direction": "forward",
            "settle_s": args.settle_s,
            "limits_ma": {
                "gain": 140.0,
                "soa": 140.0,
                "phase": 10.0,
                "wave_a": 30.0,
                "wave_b": 30.0,
            },
            "gates": {
                "target_power_mw": target_power_mw,
                "power_relative_tolerance": DEFAULT_POWER_RELATIVE_TOLERANCE,
                "wavelength_tolerance_pm": DEFAULT_WAVELENGTH_TOLERANCE_PM,
                "minimum_smsr_db": DEFAULT_SMSR_DB,
            },
            "rows": {},
        }

    if args.port:
        if serial is None:
            raise RuntimeError("当前Python环境缺少pyserial，无法使用USB审计")
        device = serial.Serial(
            port=args.port,
            baudrate=115200,
            timeout=0.10,
            write_timeout=3.0,
        )
    else:
        device = FbgLanSerial(
            board_host=args.host, timeout=0.10, write_timeout=3.0
        )
    meter = None
    worker = app.UnlimitedAccuracyWorker(device, points, guards)
    shutter_confirmed = False
    try:
        if not args.port:
            wait_connected(device)
        device.write(app.build_work_mode_command(2))
        time.sleep(0.12)
        device.reset_input_buffer()
        worker._rx_buffer.clear()
        meter = AQ6150B(args.meter)
        payload["meter"] = meter.identity

        if args.indices:
            scan_sequence = selected_indices
            for sequence_index, index in enumerate(scan_sequence, start=1):
                predecessor = points[max(0, index - 1)]
                if command_with_link_retry(worker, predecessor.codes) is None:
                    raise RuntimeError("无法建立抽测点的前置状态")
                time.sleep(0.35 if index else 0.50)
                guard = guards.get(index)
                if guard is not None and not worker._apply_guard(guard):
                    raise RuntimeError("跨模保护执行失败")
                result = measure_point(
                    worker, meter, points[index], references[index], args.settle_s
                )
                payload["rows"][str(index)] = result
                print(
                    f"smoke {sequence_index}/{len(scan_sequence)} "
                    f"target={result['target_nm']:.2f}nm "
                    f"wave={result['reading']['wavelength_nm']:.8f}nm "
                    f"drift={result['wavelength_drift_from_calibration_pm']:+.2f}pm "
                    f"power={result['reading']['power_mw']:.4f}mW "
                    f"smsr={result['reading']['side_mode_suppression_db']}",
                    flush=True,
                )
        else:
            completed_indices = sorted(int(key) for key in payload["rows"])
            first_missing = 0
            while first_missing in completed_indices:
                first_missing += 1
            if first_missing:
                # A resumed scan starts from the immediately preceding proven
                # point.  This is the same one-step state used by normal table
                # traversal and avoids flooding the LAN link with thousands of
                # already completed commands.
                predecessor_index = max(0, first_missing - 1)
                if command_with_link_retry(
                    worker, points[predecessor_index].codes
                ) is None:
                    raise RuntimeError("无法恢复中断点的前一波长状态")
                time.sleep(app.FULLBAND_ACCURACY_FIRST_POINT_SETTLE_S)
            else:
                if command_with_link_retry(worker, points[0].codes) is None:
                    raise RuntimeError("无法建立1525 nm正向扫描初始状态")
                time.sleep(app.FULLBAND_ACCURACY_FIRST_POINT_SETTLE_S)

            for index in range(first_missing, len(points)):
                point = points[index]
                guard = guards.get(index)
                if guard is not None and not worker._apply_guard(guard):
                    raise RuntimeError("跨模保护执行失败")
                result = measure_point(
                    worker, meter, point, references[index], args.settle_s
                )
                payload["rows"][str(index)] = result
                if (index + 1) % 10 == 0 or index == 0:
                    print(
                        f"audit {index + 1}/2001 target={point.target_nm:.2f}nm "
                        f"drift={result['wavelength_drift_from_calibration_pm']:+.2f}pm "
                        f"power={result['reading']['power_mw']:.4f}mW "
                        f"ok={int(result['success'])}",
                        flush=True,
                    )
                if (index + 1) % 10 == 0 or not result["success"]:
                    payload["summary"] = summarize(payload["rows"])
                    save_json(args.output, payload)

        payload["summary"] = summarize(payload["rows"])
        save_json(args.output, payload)
        print("summary=" + json.dumps(payload["summary"], ensure_ascii=False), flush=True)
        return 0
    finally:
        try:
            if device.is_open:
                shutter_confirmed = bool(
                    command_with_link_retry(
                        worker,
                        (0, 0, 0, 0, 0),
                        soa_mode=app.PI11210_SOA_SHUTTER_MODE,
                    )
                )
        except Exception as exc:
            print(f"shutter_error={exc}", flush=True)
        finally:
            if meter is not None:
                meter.close()
            device.close()
            payload["soa_shutter_confirmed"] = shutter_confirmed
            payload["finished"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            try:
                payload["summary"] = summarize(payload.get("rows", {}))
                save_json(args.output, payload)
            except Exception:
                pass
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
