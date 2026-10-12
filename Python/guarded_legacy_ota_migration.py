"""One-time guarded migration from a legacy image to the OTA flash layout.

This tool is deliberately narrower than a general programmer.  It requires an
exact debug-probe ID, proves that the target still has a legacy vector table,
commands all laser DACs to zero, confirms the dark optical path with the
AQ6150B, backs up the occupied flash range, and then writes only sector 0 and
the relocated application sectors.  Sector 1 is never erased or programmed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import serial
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from pyocd.flash.loader import MemoryLoader

import app_JDSU as app
from fbg_ota_protocol import load_package
from guarded_application_upgrade import END, START, digest, prepare_flash_clock, read
from guarded_initial_ota_install import (
    BOOT_END,
    BOOT_START,
    CONFIG_END,
    CONFIG_START,
    _write_backup,
    validate_bootloader_image,
)
from laser_power_calibration import AQ6150B


FLASH_END = END
ZERO_CODES = (0, 0, 0, 0, 0)


def vector_is_valid(image: bytes, base: int, end: int) -> bool:
    """Return whether *image* begins with a plausible Cortex-M vector table."""

    if len(image) < 8:
        return False
    initial_sp = int.from_bytes(image[0:4], "little")
    reset_vector = int.from_bytes(image[4:8], "little")
    reset_address = reset_vector & ~1
    return (
        0x20000000 <= initial_sp <= 0x20020000
        and initial_sp % 4 == 0
        and bool(reset_vector & 1)
        and base <= reset_address < end
    )


def command_safe_zero(port: str, timeout_s: float = 3.0) -> dict[str, object]:
    """Command zero current and accept only an exact legacy or modern ACK."""

    with serial.Serial(
        port, 2_000_000, timeout=0.04, write_timeout=1.0
    ) as device:
        device.dtr = True
        time.sleep(0.12)
        device.write(app.build_work_mode_command(2))
        device.flush()
        time.sleep(0.12)
        device.reset_input_buffer()
        device.write(
            app.build_single_value_dac_command(
                ZERO_CODES,
                soa_mode=app.PI11210_SOA_SHUTTER_MODE,
            )
        )
        device.flush()

        buffer = bytearray()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            waiting = int(device.in_waiting or 0)
            chunk = device.read(min(max(waiting, 20), 4096))
            if chunk:
                buffer.extend(chunk)
            for frame in app.extract_single_value_frames(buffer):
                if frame[:4] != b"\xff\xff\x02\x00":
                    continue
                returned = tuple(
                    int.from_bytes(frame[4 + channel * 2 : 6 + channel * 2], "big")
                    for channel in range(5)
                )
                if returned != ZERO_CODES:
                    continue
                flags = int(frame[15])
                modern = (
                    int(frame[14]) == app.PI11210_SOA_SHUTTER_MODE
                    and bool(flags & 0x80)
                    and flags & 0x03 == 0x03
                    and bool(flags & 0x04)
                )
                legacy = frame[14:20] == b"\x00" * 6
                if not modern and not legacy:
                    raise RuntimeError(
                        "零电流回读存在，但既不是旧版严格零ACK，也不是新版硬件关光ACK"
                    )
                return {
                    "returned_codes": list(returned),
                    "ack_kind": "modern_hardware_shutter" if modern else "legacy_zero_current",
                    "frame_hex": frame.hex(),
                    "modern_hardware_shutter": modern,
                }
            time.sleep(0.002)
    raise TimeoutError("3秒内未收到全零DAC回读")


def confirm_dark_path(resource: str, maximum_dbm: float) -> dict[str, object]:
    """Require no AQ6150B peak, or a peak below the explicit dark ceiling."""

    meter = AQ6150B(resource, average_count=1)
    try:
        result: dict[str, object] = {"identity": meter.identity}
        try:
            reading = meter.measure(1535.0, select_main_peak=True)
        except RuntimeError as exc:
            if "empty measurement" not in str(exc):
                raise
            result.update(no_valid_peak=True, dark_confirmed=True)
            return result
        result.update(
            no_valid_peak=False,
            wavelength_nm=reading.wavelength_nm,
            power_dbm=reading.power_dbm,
            peak_count=reading.peak_count,
            dark_confirmed=reading.power_dbm <= float(maximum_dbm),
        )
        if not result["dark_confirmed"]:
            raise RuntimeError(
                f"AQ6150B仍检测到{reading.power_dbm:.3f} dBm，超过迁移安全阈值"
            )
        return result
    finally:
        meter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootloader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--maximum-dark-dbm", type=float, default=-20.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error("Output report already exists")
    if args.apply and args.backup_dir.exists():
        parser.error("Backup directory already exists")

    candidate = load_package(args.candidate)
    if candidate.link_address != START or not 0 < len(candidate.image) <= END - START:
        raise ValueError("Candidate is not a bounded relocated OTA application")
    bootloader = args.bootloader.read_bytes()
    initial_sp, reset_vector = validate_bootloader_image(bootloader)

    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "apply_requested": args.apply,
        "probe": args.probe,
        "port": args.port,
        "meter": args.meter,
        "candidate_version": candidate.version_text,
        "candidate_package_sha256": digest(args.candidate.read_bytes()),
        "candidate_image_sha256": digest(candidate.image),
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
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    session = None
    target = None
    try:
        report["preinstall_zero_ack"] = command_safe_zero(args.port)
        report["preinstall_dark_path"] = confirm_dark_path(
            args.meter, args.maximum_dark_dbm
        )
        save()

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
                raise RuntimeError("Target is not running; refusing automatic migration")
            boot_before = read(target, BOOT_START, BOOT_END - BOOT_START)
            config_before = read(target, CONFIG_START, CONFIG_END - CONFIG_START)
            app_before = read(target, START, FLASH_END - START)
            if not vector_is_valid(boot_before, BOOT_START, FLASH_END):
                raise RuntimeError("Legacy vector table is invalid; refusing migration")
            if vector_is_valid(app_before, START, FLASH_END):
                raise RuntimeError(
                    "A relocated application vector already exists; use the normal guarded upgrader"
                )

            report.update(
                cpu_state_before=str(target.get_state()),
                legacy_vector_verified=True,
                relocated_vector_absent=True,
                boot_sector_sha256_before=digest(boot_before),
                protected_sector_1_sha256_before=digest(config_before),
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
                if read(target, BOOT_START, len(boot_before)) != boot_before:
                    raise RuntimeError("Legacy flash changed before programming")
                save()

                loader = MemoryLoader(
                    session,
                    chip_erase="sector",
                    smart_flash=False,
                    trust_crc=False,
                    keep_unwritten=True,
                )
                loader.add_data(BOOT_START, bootloader)
                loader.add_data(START, candidate.image)
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
                    raise RuntimeError("Protected sector 1 changed")
                if app_after[: len(candidate.image)] != candidate.image:
                    raise RuntimeError("Relocated application readback mismatch")
                if app_after[len(candidate.image) :] != app_before[len(candidate.image) :]:
                    raise RuntimeError("Unwritten application tail changed")

                report.update(
                    verified=True,
                    bootloader_sha256_after=digest(boot_after[: len(bootloader)]),
                    protected_sector_1_sha256_after=digest(config_after),
                    application_image_sha256_after=digest(
                        app_after[: len(candidate.image)]
                    ),
                    protected_sector_1_preserved=True,
                    unwritten_boot_tail_preserved=True,
                    unwritten_application_tail_preserved=True,
                )
                save()
                target.reset()
                report["reset_sent"] = True
                if target.get_state() == Target.State.HALTED:
                    pc = int(target.read_core_register("pc"))
                    if not BOOT_START <= pc < FLASH_END:
                        raise RuntimeError("Unexpected reset PC; refusing resume")
                    target.resume()
                report["state_after_reset"] = str(target.get_state())
            except BaseException:
                target.halt()
                raise

        time.sleep(1.0)
        post_ack = command_safe_zero(args.port)
        if not post_ack.get("modern_hardware_shutter"):
            raise RuntimeError("Migrated firmware did not confirm the modern hardware shutter")
        report["postinstall_shutter_ack"] = post_ack
        report["postinstall_dark_path"] = confirm_dark_path(
            args.meter, args.maximum_dark_dbm
        )
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
