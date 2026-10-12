"""Passively log one STM32 USB CDC device's Windows presence during OTA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from serial.tools import list_ports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--duration", type=float, default=260.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.duration <= 600:
        raise ValueError("duration must be 1..600 seconds")

    started = time.monotonic()
    changes: list[dict] = []
    previous: bool | None = None
    try:
        while time.monotonic() - started < args.duration:
            present = any(
                item.device.upper() == args.port.upper()
                and item.vid == 0x0483 and item.pid == 0x5740
                and item.serial_number == args.device_serial
                for item in list_ports.comports()
            )
            if present != previous:
                event = {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "present": present,
                }
                changes.append(event)
                print(json.dumps(event), flush=True)
                previous = present
            time.sleep(0.5)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"port": args.port, "device_serial": args.device_serial,
                        "events": changes}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
