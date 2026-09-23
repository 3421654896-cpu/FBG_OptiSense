"""Guarded recovery for an OTA reset intercepted by an attached debugger.

The helper is deliberately optional and imports pyOCD only when it is called.
It never resets the target and never writes target memory or flash.  A halted
core is resumed only when the boot entry, OTA metadata, staged image and flash
controller state all match the package that the desktop has just uploaded.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
from typing import Optional

from fbg_ota_protocol import FirmwarePackage


OTA_METADATA_ADDRESS = 0x08006000
OTA_METADATA_SIZE = 64
OTA_METADATA_MAGIC_PENDING = 0x4F544131
OTA_METADATA_FORMAT_VERSION = 1
OTA_TARGET_ID = 0xF2050052
OTA_STAGING_ADDRESS = 0x08040000
OTA_BOOT_ENTRY = 0x08000230
FLASH_SR = 0x40023C0C
FLASH_CR = 0x40023C10
FLASH_CR_LOCK = 0x80000000


@dataclass(frozen=True)
class VerifiedOtaState:
    phase: str
    boot_token: int


def verify_halted_ota_images(
    package: FirmwarePackage,
    metadata_raw: bytes,
    staging: bytes,
    application: bytes,
    *,
    flash_status: int,
    flash_control: int,
) -> VerifiedOtaState:
    """Validate all bytes needed before allowing a debugger-held resume."""

    if len(metadata_raw) != OTA_METADATA_SIZE:
        raise RuntimeError("OTA 元数据长度不正确")
    words = struct.unpack("<16I", metadata_raw)
    normalized = bytearray(metadata_raw)
    # The bootloader programs the one-way retry word before copying.  Metadata
    # CRC intentionally treats that word as erased in both pending states.
    normalized[56:60] = b"\xff\xff\xff\xff"
    calculated_crc = binascii.crc32(normalized[4:60]) & 0xFFFFFFFF
    common = (
        words[1] == OTA_METADATA_FORMAT_VERSION,
        words[2] == OTA_TARGET_ID,
        words[3] == package.firmware_version,
        words[4] == package.image_size,
        words[5] == package.image_crc32,
        words[6] == package.link_address,
        words[7] == OTA_STAGING_ADDRESS,
        words[15] == calculated_crc,
    )
    pending = (
        words[0] == OTA_METADATA_MAGIC_PENDING
        and words[14] == 0xFFFFFFFF
        and all(common)
    )
    installed = (
        words[0] == 0x00000000
        and words[14] == 0x00000000
        and all(common)
    )
    if not (pending or installed):
        raise RuntimeError("OTA 元数据与本次升级包不完全匹配")
    if staging != package.image:
        raise RuntimeError("OTA 暂存区与本次升级包不完全匹配")
    # Before the first resume a different-version package has not yet been
    # installed, so the application is required to match only after metadata
    # has been cleared by the bootloader.
    if installed and application != package.image:
        raise RuntimeError("OTA 已安装标记存在，但应用区与升级包不匹配")
    if int(flash_status) != 0 or not (int(flash_control) & FLASH_CR_LOCK):
        raise RuntimeError("Flash 控制器不是空闲且锁定状态")
    return VerifiedOtaState(
        phase="pending" if pending else "installed",
        boot_token=words[8],
    )


def resume_halted_verified_ota(
    package: FirmwarePackage,
    *,
    probe_id: Optional[str] = None,
) -> Optional[str]:
    """Resume one verified debugger-held boot, or return ``None`` if inapplicable.

    If no probe id is configured, automatic recovery is allowed only when
    exactly one debug probe is connected.  Any ambiguity or failed safety
    check leaves the core untouched and raises a diagnostic exception.
    """

    try:
        from pyocd.core.helpers import ConnectHelper
        from pyocd.core.target import Target
    except ImportError:
        # The packaged GUI historically ran with a slim runtime without
        # pyOCD.  Keep recovery available by delegating to the project's
        # isolated environment, where pyOCD is installed.  The helper itself
        # performs the same exact read-only checks before calling resume().
        project_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
        helper = Path(__file__).resolve().parent / "resume_verified_ota_bootloader.py"
        if not project_python.is_file() or not helper.is_file():
            return None
        output_path = Path(os.environ.get("TEMP", Path.cwd())) / (
            f"fbg_ota_debug_recovery_{os.getpid()}.json"
        )
        if output_path.exists():
            output_path.unlink()
        command = [
            str(project_python),
            str(helper),
            "--package",
            str(package.path),
            "--output",
            str(output_path),
        ]
        if probe_id:
            command.extend(("--probe", str(probe_id)))
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=25.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            if not output_path.is_file():
                return None
            result = json.loads(output_path.read_text(encoding="utf-8"))
            if result.get("resumed"):
                return str(result.get("message") or "已核验 OTA 状态并继续运行调试器截停的引导")
            if completed.returncode not in (0,):
                detail = str(result.get("error") or completed.stderr.strip())
                if detail:
                    raise RuntimeError(detail)
            return None
        finally:
            try:
                output_path.unlink()
            except OSError:
                pass

    selected_probe = str(probe_id or "").strip()
    if not selected_probe:
        probes = ConnectHelper.get_all_connected_probes(blocking=False)
        if not probes:
            return None
        if len(probes) != 1:
            raise RuntimeError("检测到多个调试器，无法安全确定 OTA 目标")
        selected_probe = str(probes[0].unique_id)

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
        blocking=False,
        unique_id=selected_probe,
        options=options,
    )
    if session is None or str(session.probe.unique_id) != selected_probe:
        return None

    with session:
        target = session.target
        if target.get_state() != Target.State.HALTED:
            return None
        pc = int(target.read_core_register("pc"))
        if pc != OTA_BOOT_ENTRY:
            raise RuntimeError(
                f"CPU 虽已暂停，但地址 0x{pc:08X} 不是已验证启动入口"
            )
        metadata = bytes(
            target.read_memory_block8(OTA_METADATA_ADDRESS, OTA_METADATA_SIZE)
        )
        staging = bytes(
            target.read_memory_block8(OTA_STAGING_ADDRESS, package.image_size)
        )
        application = bytes(
            target.read_memory_block8(package.link_address, package.image_size)
        )
        verified = verify_halted_ota_images(
            package,
            metadata,
            staging,
            application,
            flash_status=target.read32(FLASH_SR),
            flash_control=target.read32(FLASH_CR),
        )
        target.resume()
        if verified.phase == "pending":
            return "已核验 OTA 暂存包，继续运行被调试器截停的安装引导"
        return "已核验新应用，继续运行被调试器二次截停的启动引导"


__all__ = [
    "VerifiedOtaState",
    "resume_halted_verified_ota",
    "verify_halted_ota_images",
]
