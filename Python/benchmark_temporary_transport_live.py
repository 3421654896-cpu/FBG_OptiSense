"""Finite, machine-guarded USB/LAN temporary-route throughput capture."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import serial
from serial.tools import list_ports

from fbg_lan_serial import FbgLanSerial
from machine_profile import ensure_machine_compatible, set_runtime_machine_id
from temporary_test_mode import execute_device, validate_plan_payload


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("usb", "lan"), required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=128)
    parser.add_argument("--usb-port", default="COM18")
    parser.add_argument("--usb-serial", default="207933893232")
    parser.add_argument("--host", default="192.168.4.62")
    parser.add_argument("--expected-firmware", default="1.0.96")
    parser.add_argument("--usb-baseline", type=Path)
    parser.add_argument("--min-usb-ratio", type=float, default=0.95)
    parser.add_argument("--require-compact", action="store_true")
    parser.add_argument(
        "--first-peaks", type=int,
        help="Bench only the first N five-point peaks of the saved route; do not alter it",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not 32 <= args.cycles <= 512:
        parser.error("cycles must be 32..512")

    saved = json.loads(args.record.read_text(encoding="utf-8"))
    set_runtime_machine_id("machine_3")
    ensure_machine_compatible(saved)
    if saved.get("machine_id") != "machine_3":
        raise RuntimeError("this benchmark is guarded to machine_3")
    selected_plan = saved["plan"]
    if args.first_peaks is not None:
        if not 1 <= args.first_peaks <= 9:
            parser.error("--first-peaks must be 1..9")
        point_limit = args.first_peaks * 5
        if point_limit > len(selected_plan["rows"]):
            parser.error("--first-peaks exceeds the saved route")
        selected_plan = dict(selected_plan)
        selected_plan["rows"] = selected_plan["rows"][:point_limit]
        selected_plan["peaks_nm"] = selected_plan["peaks_nm"][:args.first_peaks]
        selected_plan["schema"] = "temporary_test_variable_plan_v2"
        selected_plan["point_count"] = point_limit
        selected_plan["peak_count"] = args.first_peaks
        if "selection_fit_checks" in selected_plan:
            selected_plan["selection_fit_checks"] = selected_plan["selection_fit_checks"][:args.first_peaks]
        for key, limit in (("endpoint_checks", args.first_peaks),
                           ("validated_reference_values", point_limit)):
            if isinstance(selected_plan.get(key), list):
                selected_plan[key] = selected_plan[key][:limit]
    plan = validate_plan_payload(selected_plan)
    selected_point_delays = (
        list(saved["point_delays_us"][:len(plan["rows"])])
        if saved.get("use_point_waits") else None
    )
    if len(plan["rows"]) not in (15, 45):
        raise RuntimeError("expected a saved 15- or 45-point route")
    if args.require_compact and args.transport != "lan":
        parser.error("--require-compact applies only to LAN")
    if args.usb_baseline and args.transport != "lan":
        parser.error("--usb-baseline applies only to LAN")
    if not 0 < args.min_usb_ratio <= 1:
        parser.error("--min-usb-ratio must be in (0, 1]")
    usb_baseline = None
    if args.usb_baseline:
        usb_baseline = json.loads(args.usb_baseline.read_text(encoding="utf-8"))
        if not (usb_baseline.get("complete") and usb_baseline.get("disarm_ack")
                and usb_baseline.get("shutter_ack")
                and usb_baseline.get("point_count") == len(plan["rows"])
                and usb_baseline.get("mean_hz", 0) > 0
                and usb_baseline.get("plan") == plan
                and usb_baseline.get("received_frames") == args.cycles
                and usb_baseline.get("first_delay_us") == int(saved["first_delay_us"])
                and usb_baseline.get("boundary_extra_us") == int(saved["boundary_extra_us"])
                and usb_baseline.get("feedback_selector") == int(saved["feedback_selector"])
                and usb_baseline.get("signal_channel") == int(saved["signal_channel"])
                and usb_baseline.get("point_delays_us") == selected_point_delays):
            raise RuntimeError("USB baseline is incomplete or uses a different route/cycle count")

    options = dict(
        first_delay_us=int(saved["first_delay_us"]),
        boundary_extra_us=int(saved["boundary_extra_us"]),
        feedback_selector=int(saved["feedback_selector"]),
        adc_channel=int(saved["signal_channel"]),
    )
    if selected_point_delays is not None:
        options["point_delays_us"] = selected_point_delays
    device = None
    if args.transport == "usb":
        matches = [
            port for port in list_ports.comports()
            if port.device.upper() == args.usb_port.upper()
            and (port.serial_number or "").upper() == args.usb_serial.upper()
        ]
        if len(matches) != 1:
            raise RuntimeError("USB serial/port does not identify machine_3")
        device = serial.Serial(
            args.usb_port, 2_000_000, timeout=0.05, write_timeout=5.0
        )
    else:
        device = FbgLanSerial(
            board_host=args.host, timeout=0.05, write_timeout=5.0,
            connect_timeout=0.5, discovery_interval=0.5,
        )
        device.open()
        deadline = time.monotonic() + 25.0
        while not device.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        if not device.connected or device.board_ip != args.host:
            device.close()
            raise TimeoutError("machine_3 LAN RAW READY was not received")

    try:
        report = execute_device(
            device, plan, args.output, cycles=args.cycles, **options
        )
        versions = {int(frame["firmware"]) for frame in report["frames"]}
        version = tuple(int(part) for part in args.expected_firmware.split("."))
        expected = version[0] << 16 | version[1] << 8 | version[2]
        board_gaps = [
            (current["cycle_start_us"] - previous["cycle_start_us"]) / 1000.0
            for previous, current in zip(report["frames"][16:-1], report["frames"][17:])
        ]
        host_gaps = [
            (current["host_received_s"] - previous["host_received_s"]) * 1000.0
            for previous, current in zip(report["frames"][16:-1], report["frames"][17:])
        ]
        warm_frames = report["frames"][16:]
        socket_frames = [frame for frame in warm_frames if "socket_arrival_s" in frame]
        socket_gaps = [
            (current["socket_arrival_s"] - previous["socket_arrival_s"]) * 1000.0
            for previous, current in zip(socket_frames[:-1], socket_frames[1:])
        ]
        batch_sizes = {}
        for frame in socket_frames:
            batch = frame["socket_receive_batch"]
            batch_sizes[batch] = batch_sizes.get(batch, 0) + 1
        parse_lags = [
            (frame["host_received_s"] - frame["socket_arrival_s"]) * 1000.0
            for frame in socket_frames
        ]
        result = {
            "transport": args.transport,
            "point_count": len(plan["rows"]),
            "requested_frames": args.cycles,
            "received_frames": report["received_frames"],
            "complete": bool(report["complete"]),
            "disarm_ack": bool(report["disarm_ack"]),
            "shutter_ack": bool(report["shutter_ack"]),
            "firmware_values": [f"0x{value:08X}" for value in sorted(versions)],
            "mean_hz": report["mean_hz"],
            "board_period_p99_ms": report["p99_period_ms"],
            "board_period_mean_ms": statistics.mean(board_gaps) if board_gaps else None,
            "acquisition_mean_ms": statistics.mean(
                frame["elapsed_us"] / 1000.0 for frame in report["frames"][16:]
            ),
            "host_arrival_p99_ms": _percentile(host_gaps, 99),
            "host_arrival_median_ms": _percentile(host_gaps, 50),
            "host_arrival_max_ms": max(host_gaps) if host_gaps else None,
            "socket_arrival_p99_ms": _percentile(socket_gaps, 99),
            "socket_arrival_median_ms": _percentile(socket_gaps, 50),
            "socket_batch_max_frames": max(batch_sizes.values()) if batch_sizes else None,
            "socket_batch_mean_frames": (
                statistics.mean(batch_sizes.values()) if batch_sizes else None
            ),
            "host_parse_lag_p99_ms": _percentile(parse_lags, 99),
            "transport_stats": (
                report.get("transport_stats_before_shutdown")
                if args.transport == "lan" else None
            ),
            "usb_baseline_hz": usb_baseline.get("mean_hz") if usb_baseline else None,
            "lan_to_usb_hz_ratio": (
                report["mean_hz"] / usb_baseline["mean_hz"]
                if usb_baseline and usb_baseline.get("mean_hz", 0) > 0 else None
            ),
            "report": str(args.output.resolve()),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if not (
            report["complete"] and report["received_frames"] == args.cycles
            and report["disarm_ack"] and report["shutter_ack"]
            and versions == {expected}
        ):
            raise RuntimeError("temporary transport benchmark failed acceptance")
        if args.require_compact and (
            result["transport_stats"]["compact_temporary_frames"] != args.cycles
        ):
            raise RuntimeError("LAN did not use TC4 compact temporary batches for every frame")
        if (usb_baseline and result["lan_to_usb_hz_ratio"] is not None
                and result["lan_to_usb_hz_ratio"] < args.min_usb_ratio):
            raise RuntimeError("LAN throughput fell below the USB comparison target")
    finally:
        device.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
