"""Run one finite temporary-mode acceptance capture over same-LAN TCP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fbg_lan_serial import FbgLanSerial
from machine_profile import ensure_machine_compatible
from temporary_test_mode import execute_device, validate_plan_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument(
        "--expected-firmware",
        help="required runtime version such as 1.0.85",
    )
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 32 <= args.cycles <= 512:
        parser.error("cycles must be 32..512")
    expected_firmware = None
    if args.expected_firmware:
        try:
            major, minor, patch = (
                int(part) for part in args.expected_firmware.split(".")
            )
        except (TypeError, ValueError):
            parser.error("expected firmware must use major.minor.patch")
        if not all(0 <= value <= 255 for value in (major, minor, patch)):
            parser.error("expected firmware components must be 0..255")
        expected_firmware = f"0x{(major << 16 | minor << 8 | patch):08X}"

    payload = json.loads(args.record.read_text(encoding="utf-8"))
    ensure_machine_compatible(payload)
    plan = validate_plan_payload(payload.get("plan", payload))
    point_count = len(plan["rows"])
    if point_count not in (15, 45):
        raise ValueError("live LAN acceptance requires a saved 15- or 45-point route")
    selector = int(payload.get("feedback_selector", 2))
    options = {
        "first_delay_us": int(payload.get("first_delay_us", 350)),
        "boundary_extra_us": int(payload.get("boundary_extra_us", 1800)),
        "feedback_selector": selector,
    }
    waits = payload.get("point_delays_us")
    if payload.get("use_point_waits") and isinstance(waits, list):
        if len(waits) != point_count:
            raise ValueError("saved point-wait count does not match the route")
        options.update(
            first_delay_us=650,
            boundary_extra_us=0,
            point_delays_us=[int(value) for value in waits],
        )

    device = FbgLanSerial(
        board_host=args.host,
        timeout=0.05,
        write_timeout=5.0,
        connect_timeout=0.5,
        discovery_interval=0.5,
    )
    try:
        device.open()
        deadline = time.monotonic() + args.connect_timeout
        while not device.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        if not device.connected:
            raise TimeoutError("LAN RAW READY was not received")
        report = execute_device(
            device,
            plan,
            args.output,
            cycles=args.cycles,
            **options,
        )
        summary = {
            "complete": bool(report.get("complete")),
            "point_count": point_count,
            "peak_count": point_count // 5,
            "received_frames": int(report.get("received_frames", 0)),
            "firmware": (
                f"0x{int(report['frames'][-1]['firmware']):08X}"
                if report.get("frames") else None
            ),
            "mean_hz": report.get("mean_hz"),
            "p99_period_ms": report.get("p99_period_ms"),
            "disarm_ack": bool(report.get("disarm_ack")),
            "shutter_ack": bool(report.get("shutter_ack")),
            "transport_stats": device.transport_stats,
            "output": str(args.output.resolve()),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not (
            summary["complete"]
            and summary["received_frames"] == args.cycles
            and (
                expected_firmware is None
                or summary["firmware"] == expected_firmware
            )
            and summary["disarm_ack"]
            and summary["shutter_ack"]
        ):
            raise RuntimeError(f"LAN temporary acceptance failed: {summary}")
    finally:
        device.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
