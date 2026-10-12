"""Guarded STM32 software reset for Wi-Fi listener lifecycle diagnosis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import serial
from serial.tools import list_ports
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_ota_protocol import load_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--installed-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    matches = [
        item for item in list_ports.comports()
        if item.device.upper() == args.port.upper()
        and item.vid == 0x0483 and item.pid == 0x5740
        and item.serial_number == args.device_serial
    ]
    if len(matches) != 1:
        raise RuntimeError("Native USB identity mismatch; refusing reset")
    with serial.Serial(args.port, 2_000_000, timeout=0.03, write_timeout=2.0) as device:
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("Could not verify all-zero source and closed shutter")

    package = load_package(args.installed_package)
    options = {
        "target_override": "stm32f205re", "connect_mode": "attach",
        "auto_unlock": False, "resume_on_disconnect": False,
        "no_config": True, "frequency": 1_000_000,
        "cache.enable_memory": False, "cache.enable_register": False,
    }
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=args.probe, options=options
    )
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("Exact debug probe unavailable; refusing reset")
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("CPU not running; refusing reset")
        for offset in range(0, package.image_size, 1024):
            expected = package.image[offset : offset + 1024]
            actual = bytes(target.read_memory_block8(
                package.link_address + offset, len(expected)
            ))
            if actual != expected:
                raise RuntimeError("Installed application differs from expected package")
        target.reset(Target.ResetType.SYSRESETREQ)
        state = target.get_state()
        if state == Target.State.HALTED:
            pc = int(target.read_core_register("pc"))
            if not 0x08000000 <= pc < 0x08040000:
                raise RuntimeError("Reset halted outside verified boot/application range")
            target.resume()
            state = target.get_state()
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "probe": args.probe,
            "device_serial": args.device_serial,
            "installed_package_version": package.version_text,
            "installed_image_verified": True,
            "source_disarmed_and_shutter_closed": True,
            "reset_type": "SYSRESETREQ",
            "state_after_reset": str(state),
            "flash_written": False,
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
