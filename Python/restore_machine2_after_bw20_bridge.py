"""Restore the exact backed-up machine-2 application after UART bridge diagnosis.

Only application sectors (0x08008000..0x0803ffff) may be programmed.  The
bootloader and configuration sectors are checked before and after but never
written.  The installed bridge image and the original backup must match
exactly before any target write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from pyocd.flash.loader import MemoryLoader

from fbg_ota_protocol import load_package
from guarded_application_upgrade import prepare_flash_clock


FLASH_START = 0x08000000
APP_START = 0x08008000
APP_END = 0x08040000


def read_block(target, address: int, length: int) -> bytes:
    result = bytearray()
    for offset in range(0, length, 1024):
        result.extend(target.read_memory_block8(address + offset, min(1024, length - offset)))
    return bytes(result)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.report.exists():
        parser.error("Report already exists")
    bridge = load_package(args.bridge)
    original = load_package(args.original)
    if bridge.link_address != APP_START or original.link_address != APP_START:
        raise RuntimeError("Unexpected package link address")
    backup_protected = (args.backup_dir / "protected_flash.bin").read_bytes()
    backup_app = (args.backup_dir / "application_flash.bin").read_bytes()
    if len(backup_protected) != APP_START - FLASH_START or len(backup_app) != APP_END - APP_START:
        raise RuntimeError("Unexpected backup dimensions")
    if backup_app[: len(original.image)] != original.image:
        raise RuntimeError("Original package differs from backed-up application")

    report = {
        "probe": args.probe,
        "apply_requested": args.apply,
        "bridge_image_sha256": digest(bridge.image),
        "original_application_sha256": digest(backup_app),
        "protected_sha256": digest(backup_protected),
        "program_range": [f"0x{APP_START:08X}", f"0x{APP_END:08X}"],
        "verified": False,
    }

    def save() -> None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    options = {
        "target_override": "stm32f205re",
        "connect_mode": "attach",
        "auto_unlock": False,
        "resume_on_disconnect": False,
        "no_config": True,
        "frequency": 1000000,
        "cache.enable_memory": False,
        "cache.enable_register": False,
    }
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=args.probe, options=options
    )
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("Expected probe unavailable")
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("Bridge target is not running")
        protected = read_block(target, FLASH_START, APP_START - FLASH_START)
        current_app = read_block(target, APP_START, APP_END - APP_START)
        if protected != backup_protected:
            raise RuntimeError("Protected boot/config region changed")
        if current_app[: len(bridge.image)] != bridge.image:
            raise RuntimeError("Installed application is not the exact diagnostic bridge")
        report["preflight_verified"] = True
        save()
        if not args.apply:
            print(json.dumps(report))
            return 0

        try:
            target.reset_and_halt(Target.ResetType.SYSRESETREQ)
            report["flash_clock"] = prepare_flash_clock(target)
            if read_block(target, APP_START, len(bridge.image)) != bridge.image:
                raise RuntimeError("Bridge image changed after halt")
            loader = MemoryLoader(
                session, chip_erase="sector", smart_flash=False,
                trust_crc=False, keep_unwritten=True,
            )
            loader.add_data(APP_START, backup_app)
            loader.commit()
            target.halt()
            if read_block(target, APP_START, len(backup_app)) != backup_app:
                raise RuntimeError("Restored application readback mismatch")
            if read_block(target, FLASH_START, len(backup_protected)) != backup_protected:
                raise RuntimeError("Protected boot/config readback changed")
            report["verified"] = True
            save()
            target.reset()
            if target.get_state() == Target.State.HALTED:
                pc = int(target.read_core_register("pc"))
                if not FLASH_START <= pc < APP_END:
                    raise RuntimeError("Unexpected reset PC; refusing resume")
                target.resume()
            report["state_after_reset"] = str(target.get_state())
        except BaseException:
            target.halt()
            raise
        finally:
            save()
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
