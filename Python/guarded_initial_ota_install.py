"""Guarded one-time OTA bootloader + relocated application installation.

The tool never mass-erases.  It writes flash sector 0 and application sectors
2..5 only, proves sector 1 (Wi-Fi credentials and OTA metadata) is byte-for-
byte unchanged, and leaves the CPU halted after any failed write or readback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import serial
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from pyocd.flash.loader import MemoryLoader

from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_ota_protocol import OTA_RAM_END, OTA_RAM_START, load_package
from guarded_application_upgrade import END, START, digest, prepare_flash_clock, read


BOOT_START = 0x08000000
BOOT_END = 0x08004000
CONFIG_START = 0x08004000
CONFIG_END = START


def validate_bootloader_image(image: bytes) -> tuple[int, int]:
    if len(image) < 8 or len(image) > BOOT_END - BOOT_START:
        raise ValueError("Bootloader does not fit flash sector 0")
    initial_sp = int.from_bytes(image[0:4], "little")
    reset_vector = int.from_bytes(image[4:8], "little")
    reset_address = reset_vector & ~1
    if not OTA_RAM_START <= initial_sp <= OTA_RAM_END or initial_sp & 3:
        raise ValueError("Bootloader initial stack pointer is invalid")
    if not reset_vector & 1:
        raise ValueError("Bootloader reset vector is not Thumb code")
    if not BOOT_START <= reset_address < BOOT_START + len(image):
        raise ValueError("Bootloader reset vector is outside its image")
    return initial_sp, reset_vector


def _write_backup(
    directory: Path, boot_sector: bytes, config_sector: bytes, application: bytes
) -> dict[str, str]:
    if directory.exists():
        raise FileExistsError(f"Backup directory already exists: {directory}")
    directory.mkdir(parents=True)
    files = {
        "boot_sector_08000000.bin": boot_sector,
        "config_sector_08004000_PRIVATE.bin": config_sector,
        "application_08008000.bin": application,
    }
    hashes: dict[str, str] = {}
    for name, data in files.items():
        path = directory / name
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "contains_private_wifi_configuration": True,
                "sha256": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootloader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output report already exists")
    if args.apply and args.backup_dir.exists():
        parser.error("Backup directory already exists")

    old = load_package(args.current)
    new = load_package(args.candidate)
    bootloader = args.bootloader.read_bytes()
    initial_sp, reset_vector = validate_bootloader_image(bootloader)
    if old.link_address != START or new.link_address != START:
        raise ValueError("Application package link address is not 0x08008000")

    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "apply_requested": args.apply,
        "candidate_version": new.version_text,
        "candidate_package_sha256": digest(args.candidate.read_bytes()),
        "candidate_image_sha256": digest(new.image),
        "bootloader_sha256": digest(bootloader),
        "bootloader_initial_sp": f"0x{initial_sp:08X}",
        "bootloader_reset_vector": f"0x{reset_vector:08X}",
        "no_mass_erase": True,
        "no_cnc_commands": True,
        "verified": False,
        "reset_sent": False,
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    session = None
    target = None
    try:
        with serial.Serial("COM6", 2_000_000, timeout=0.03, write_timeout=3) as device:
            device.dtr = True
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError("Pre-install disarm/shutter confirmation missing")
            report["preinstall_disarm_and_shutter_confirmed"] = True

        session = ConnectHelper.session_with_chosen_probe(
            blocking=False,
            unique_id=args.probe,
            options={
                "target_override": "stm32f205re",
                "connect_mode": "attach",
                "auto_unlock": False,
                "resume_on_disconnect": False,
                "no_config": True,
                "frequency": 1_000_000,
                "cache.enable_memory": False,
                "cache.enable_register": False,
            },
        )
        if session is None or session.probe.unique_id != args.probe:
            raise RuntimeError("Wrong or missing debug probe")

        with session:
            target = session.target
            if target.get_state() != Target.State.RUNNING:
                raise RuntimeError("Target is not running; refusing automatic install")
            boot_before = read(target, BOOT_START, BOOT_END - BOOT_START)
            config_before = read(target, CONFIG_START, CONFIG_END - CONFIG_START)
            app_before = read(target, START, END - START)
            if app_before[: len(old.image)] != old.image:
                raise RuntimeError("Current application differs from the approved package")
            report.update(
                current_image_verified=True,
                current_image_sha256=digest(old.image),
                boot_sector_sha256_before=digest(boot_before),
                config_sector_sha256_before=digest(config_before),
                application_region_sha256_before=digest(app_before),
                flash_sectors_written=[0, 2, 3, 4, 5],
                flash_sector_1_written=False,
            )
            save()
            if not args.apply:
                return 0

            report["backup_sha256"] = _write_backup(
                args.backup_dir, boot_before, config_before, app_before
            )
            report["backup_directory"] = str(args.backup_dir.resolve())
            save()

            try:
                target.reset_and_halt(Target.ResetType.SYSRESETREQ)
                report["flash_clock"] = prepare_flash_clock(target)
                if read(target, START, len(old.image)) != old.image:
                    raise RuntimeError("Application changed after flash-clock preparation")
                if read(target, CONFIG_START, len(config_before)) != config_before:
                    raise RuntimeError("Configuration changed before programming")
                save()

                loader = MemoryLoader(
                    session,
                    chip_erase="sector",
                    smart_flash=False,
                    trust_crc=False,
                    keep_unwritten=True,
                )
                loader.add_data(BOOT_START, bootloader)
                loader.add_data(START, new.image)
                loader.commit()
                target.halt()

                boot_after = read(target, BOOT_START, len(boot_before))
                config_after = read(target, CONFIG_START, len(config_before))
                app_after = read(target, START, len(app_before))
                if boot_after[: len(bootloader)] != bootloader:
                    raise RuntimeError("Bootloader readback mismatch")
                if boot_after[len(bootloader) :] != boot_before[len(bootloader) :]:
                    raise RuntimeError("Unwritten boot-sector tail changed")
                if config_after != config_before:
                    raise RuntimeError("Wi-Fi configuration sector changed")
                if app_after[: len(new.image)] != new.image:
                    raise RuntimeError("Application readback mismatch")
                if app_after[len(new.image) :] != app_before[len(new.image) :]:
                    raise RuntimeError("Unwritten application tail changed")

                report.update(
                    verified=True,
                    bootloader_sha256_after=digest(boot_after[: len(bootloader)]),
                    boot_sector_sha256_after=digest(boot_after),
                    config_sector_sha256_after=digest(config_after),
                    application_image_sha256_after=digest(app_after[: len(new.image)]),
                    application_region_sha256_after=digest(app_after),
                    config_sector_preserved=True,
                    unwritten_boot_tail_preserved=True,
                    unwritten_application_tail_preserved=True,
                )
                save()
                target.reset()
                report["reset_sent"] = True
                if target.get_state() == Target.State.HALTED:
                    pc = int(target.read_core_register("pc"))
                    if not BOOT_START <= pc < END:
                        raise RuntimeError("Unexpected reset PC; refusing resume")
                    target.resume()
                report["state_after_reset"] = str(target.get_state())
            except BaseException:
                target.halt()
                raise
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
