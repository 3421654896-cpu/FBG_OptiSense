"""Hold bounded machine-3 GAIN/SOA currents for a manual electrical check."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
from serial.tools import list_ports

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fullband_equal_power_model import current_to_code
from guarded_application_upgrade import zero_all_dacs_and_verify


EXPECTED_SERIAL = "207933893232"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM18")
    parser.add_argument("--device-serial", default=EXPECTED_SERIAL,
                        help="Exact STM32 USB CDC serial number; required to target another board")
    parser.add_argument("--gain-ma", type=float, default=0.0)
    parser.add_argument("--soa-ma", type=float, default=5.0)
    parser.add_argument("--hold-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.gain_ma <= 20.0:
        parser.error("GAIN diagnostic current must be 0..20 mA")
    if not 0.0 <= args.soa_ma <= 20.0:
        parser.error("SOA diagnostic current must be 0..20 mA")
    if not 10.0 <= args.hold_seconds <= 1800.0:
        parser.error("hold time must be 10..1800 seconds")
    if args.output.exists():
        parser.error("output already exists")
    matches = [port for port in list_ports.comports()
               if port.device.upper() == args.port.upper()
               and port.vid == 0x0483 and port.pid == 0x5740
               and port.serial_number == args.device_serial]
    if len(matches) != 1:
        raise RuntimeError("STM32 USB identity mismatch; no current command sent")

    gain_code = current_to_code(args.gain_ma, 0)
    soa_code = current_to_code(args.soa_ma, 1)
    soa_mode = (app.PI11210_SOA_SHUTTER_MODE if args.soa_ma == 0.0
                else app.PI11210_SOA_SOURCE_MODE)
    report = {
        "schema": "identified_bounded_soa_current_hold_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device_serial": args.device_serial,
        "port": args.port,
        "requested_currents_ma": {"gain": args.gain_ma, "soa": args.soa_ma,
                                  "phase": 0.0, "wave_a": 0.0, "wave_b": 0.0},
        "requested_codes": [gain_code, soa_code, 0, 0, 0],
        "soa_mode": soa_mode,
        "hold_seconds_requested": args.hold_seconds,
        "acknowledged": False,
        "note": "DAC acknowledgement is not an electrical-current measurement",
    }
    device = None
    started = None
    try:
        device = serial.Serial(args.port, 2_000_000, timeout=0.03, write_timeout=3)
        device.dtr = True
        _safe_disarm(device)
        zero_all_dacs_and_verify(device)
        _close_shutter(device)
        worker = app.EqualIntervalWorker(device, (), {}, settle_s=0.0,
                                         feedback_selectors=(0, 2))
        if worker._command_and_ack(
            (gain_code, soa_code, 0, 0, 0), soa_mode=soa_mode
        ) is None:
            raise RuntimeError("GAIN/SOA command had no matching DAC ACK")
        report["acknowledged"] = True
        started = time.monotonic()
        print(json.dumps({"event": "HOLD_READY", "gain_ma": args.gain_ma,
                          "soa_ma": args.soa_ma, "gain_code": gain_code,
                          "soa_code": soa_code, "soa_mode": soa_mode,
                          "other_channels_ma": 0,
                          "auto_off_seconds": args.hold_seconds}), flush=True)
        while time.monotonic() - started < args.hold_seconds:
            time.sleep(0.1)
        report["hold_complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["hold_seconds_actual"] = (
            None if started is None else round(time.monotonic() - started, 3))
        cleanup = {}
        if device is not None:
            for label, action in (("disarm_ack", _safe_disarm),
                                  ("all_zero_ack", zero_all_dacs_and_verify),
                                  ("shutter_ack", _close_shutter)):
                try:
                    action(device)
                    cleanup[label] = True
                except Exception as exc:
                    cleanup[label] = f"{type(exc).__name__}: {exc}"
            device.close()
        report["safety_cleanup"] = cleanup
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(json.dumps({"event": "HOLD_ENDED", "safety_cleanup": cleanup}),
              flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
