"""Install the OTA bootloader and application on a provably blank STM32F205RE.

This tool is intentionally limited to factory-blank boards. It refuses to
program if any byte of internal flash is not erased, never performs a mass
erase, writes only boot sector 0 and application sectors 2..5, and compares
the complete 512 KiB flash image before allowing the MCU to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from pyocd.flash.loader import MemoryLoader

from fbg_ota_protocol import load_package
from guarded_application_upgrade import END as APPLICATION_END
from guarded_application_upgrade import START as APPLICATION_START
from guarded_application_upgrade import prepare_flash_clock, read
from guarded_initial_ota_install import (
    BOOT_END,
    BOOT_START,
    CONFIG_END,
    CONFIG_START,
    validate_bootloader_image,
)


FLASH_END = 0x08080000
ERASED_BYTE = 0xFF


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_expected_flash(
    erased_flash: bytes, bootloader: bytes, application: bytes
) -> bytes:
    if len(erased_flash) != FLASH_END - BOOT_START:
        raise ValueError("Unexpected STM32F205RE flash size")
    if any(value != ERASED_BYTE for value in erased_flash):
        raise ValueError("Target flash is not completely erased")
    if len(bootloader) > BOOT_END - BOOT_START:
        raise ValueError("Bootloader crosses sector 0")
    if not 0 < len(application) <= APPLICATION_END - APPLICATION_START:
        raise ValueError("Application crosses the OTA application partition")

    expected = bytearray(erased_flash)
    expected[: len(bootloader)] = bootloader
    application_offset = APPLICATION_START - BOOT_START
    expected[application_offset : application_offset + len(application)] = application
    return bytes(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootloader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output report already exists")

    candidate = load_package(args.candidate)
    if candidate.link_address != APPLICATION_START:
        raise ValueError("Candidate application is not linked at 0x08008000")
    bootloader = args.bootloader.read_bytes()
    initial_sp, reset_vector = validate_bootloader_image(bootloader)

    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "apply_requested": args.apply,
        "probe": args.probe,
        "candidate_version": candidate.version_text,
        "candidate_package_sha256": digest(args.candidate.read_bytes()),
        "candidate_image_sha256": digest(candidate.image),
        "bootloader_sha256": digest(bootloader),
        "bootloader_initial_sp": f"0x{initial_sp:08X}",
        "bootloader_reset_vector": f"0x{reset_vector:08X}",
        "no_mass_erase": True,
        "flash_sectors_written": [0, 2, 3, 4, 5],
        "flash_sector_1_written": False,
        "verified": False,
        "reset_sent": False,
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    target = None
    programming_started = False
    try:
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
            report["cpu_state_before"] = str(target.get_state())
            before = read(target, BOOT_START, FLASH_END - BOOT_START)
            expected = build_expected_flash(before, bootloader, candidate.image)
            report.update(
                full_flash_blank_before=True,
                full_flash_sha256_before=digest(before),
                protected_config_sha256_before=digest(
                    before[CONFIG_START - BOOT_START : CONFIG_END - BOOT_START]
                ),
            )
            save()
            if not args.apply:
                return 0

            target.reset_and_halt(Target.ResetType.SYSRESETREQ)
            if target.get_state() != Target.State.HALTED:
                raise RuntimeError("Blank target did not enter a halted reset state")
            report["flash_clock"] = prepare_flash_clock(target)
            if read(target, BOOT_START, FLASH_END - BOOT_START) != before:
                raise RuntimeError("Blank flash changed before programming")
            save()

            programming_started = True
            loader = MemoryLoader(
                session,
                chip_erase="sector",
                smart_flash=False,
                trust_crc=False,
                keep_unwritten=True,
            )
            loader.add_data(BOOT_START, bootloader)
            loader.add_data(APPLICATION_START, candidate.image)
            loader.commit()
            target.halt()

            after = read(target, BOOT_START, FLASH_END - BOOT_START)
            if after != expected:
                mismatch = next(
                    index for index, values in enumerate(zip(after, expected))
                    if values[0] != values[1]
                )
                raise RuntimeError(
                    f"Full-flash readback mismatch at 0x{BOOT_START + mismatch:08X}"
                )

            config = after[
                CONFIG_START - BOOT_START : CONFIG_END - BOOT_START
            ]
            report.update(
                verified=True,
                full_flash_sha256_after=digest(after),
                bootloader_sha256_after=digest(after[: len(bootloader)]),
                application_sha256_after=digest(
                    after[
                        APPLICATION_START - BOOT_START :
                        APPLICATION_START - BOOT_START + len(candidate.image)
                    ]
                ),
                protected_config_sha256_after=digest(config),
                protected_config_preserved=(
                    config
                    == before[CONFIG_START - BOOT_START : CONFIG_END - BOOT_START]
                ),
                complete_flash_readback_matches=True,
            )
            save()

            target.reset()
            report["reset_sent"] = True
            state = target.get_state()
            if state == Target.State.HALTED:
                pc = int(target.read_core_register("pc"))
                if not BOOT_START <= pc < APPLICATION_END:
                    raise RuntimeError("Unexpected reset PC; refusing resume")
                target.resume()
                state = target.get_state()
            if state == Target.State.LOCKUP:
                raise RuntimeError("Installed target entered lockup after reset")
            report["state_after_reset"] = str(state)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if target is not None and programming_started:
            try:
                target.halt()
            except Exception:
                pass
        raise
    finally:
        save()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
