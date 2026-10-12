"""Read-only flash backup before a temporary BW20 UART bridge install.

The protected bootloader/configuration area is saved locally but never printed.
No target flash or RAM writes, reset, or halt are issued by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import load_package


FLASH_START = 0x08000000
APP_START = 0x08008000
APP_END = 0x08040000
WORK_STATE = 0x200001CA
OTA_ACTIVE = 0x200002BC


def read_block(target, address: int, length: int) -> bytes:
    result = bytearray()
    for offset in range(0, length, 1024):
        result.extend(target.read_memory_block8(address + offset, min(1024, length - offset)))
    return bytes(result)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error("Backup output directory already exists")
    package = load_package(args.current)
    if package.link_address != APP_START or not 0 < len(package.image) <= APP_END - APP_START:
        raise RuntimeError("Unexpected application image bounds")

    options = {
        "target_override": "cortex_m",
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
            raise RuntimeError("Target CPU is not running")
        work_state = target.read8(WORK_STATE)
        ota_active = target.read8(OTA_ACTIVE)
        if work_state != 2 or ota_active != 0:
            raise RuntimeError(f"Unsafe preflight: workState={work_state}, OTA active={ota_active}")
        protected = read_block(target, FLASH_START, APP_START - FLASH_START)
        application = read_block(target, APP_START, APP_END - APP_START)
        if application[: len(package.image)] != package.image:
            raise RuntimeError("Installed application differs from expected package")
        report = {
            "probe": args.probe,
            "work_state": work_state,
            "ota_active": ota_active,
            "installed_version": package.version_text,
            "protected_sha256": sha256(protected),
            "application_sha256": sha256(application),
            "expected_image_sha256": sha256(package.image),
            "protected_bytes": len(protected),
            "application_bytes": len(application),
            "target_state_after": str(target.get_state()),
        }

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protected_flash.bin").write_bytes(protected)
    (args.output_dir / "application_flash.bin").write_bytes(application)
    (args.output_dir / "backup_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
