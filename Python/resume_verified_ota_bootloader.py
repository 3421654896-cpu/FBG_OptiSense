"""Resume a halted OTA bootloader only after exact package/metadata checks."""

from __future__ import annotations

import argparse
import binascii
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import load_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    package = load_package(args.package)
    report = {
        "schema": "resume_verified_ota_bootloader_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "no_reset": True,
        "no_host_flash_writes": True,
        "resumed": False,
    }
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
    probe_id = str(args.probe or "").strip()
    if not probe_id:
        probes = ConnectHelper.get_all_connected_probes(blocking=False)
        if len(probes) != 1:
            raise RuntimeError("未指定调试器且当前不是唯一一个已连接的调试器")
        probe_id = str(probes[0].unique_id)
    try:
        session = ConnectHelper.session_with_chosen_probe(
            blocking=False, unique_id=probe_id, options=options
        )
        if session is None or session.probe.unique_id != probe_id:
            raise RuntimeError("exact requested probe unavailable")
        with session:
            target = session.target
            state = target.get_state()
            pc = int(target.read_core_register("pc"))
            report["state_before"] = str(state)
            report["pc_before"] = hex(pc)
            if state != Target.State.HALTED or pc != 0x08000230:
                raise RuntimeError("CPU is not halted at the verified boot entry")

            raw = bytes(target.read_memory_block8(0x08006000, 64))
            words = struct.unpack("<16I", raw)
            normalized = bytearray(raw)
            normalized[56:60] = b"\xff\xff\xff\xff"
            calculated = binascii.crc32(normalized[4:60]) & 0xFFFFFFFF
            common_fields = (
                words[1] == 1,
                words[2] == 0xF2050052,
                words[3] == package.firmware_version,
                words[4] == package.image_size,
                words[5] == package.image_crc32,
                words[6] == package.link_address,
                words[7] == 0x08040000,
                words[15] == calculated,
            )
            pending = (
                words[0] == 0x4F544131
                and words[14] == 0xFFFFFFFF
                and all(common_fields)
            )
            installed = (
                words[0] == 0x00000000
                and words[14] == 0x00000000
                and all(common_fields)
            )
            if not (pending or installed):
                raise RuntimeError("OTA metadata does not exactly match package state")
            staging = bytes(
                target.read_memory_block8(0x08040000, package.image_size)
            )
            if staging != package.image:
                raise RuntimeError("staging image does not exactly match package")
            application = bytes(
                target.read_memory_block8(package.link_address, package.image_size)
            )
            if application != package.image:
                raise RuntimeError("installed application does not exactly match package")
            flash_sr = target.read32(0x40023C0C)
            flash_cr = target.read32(0x40023C10)
            report["flash_status_register"] = hex(flash_sr)
            report["flash_control_register"] = hex(flash_cr)
            if flash_sr != 0 or not (flash_cr & 0x80000000):
                raise RuntimeError("flash controller is not idle and locked")
            report["metadata_crc32"] = f"{words[15]:08X}"
            report["staging_crc32"] = f"{binascii.crc32(staging) & 0xFFFFFFFF:08X}"
            report["install_boot_token"] = f"{words[8]:08X}"
            report["verified_ota_state"] = "pending" if pending else "installed"
            target.resume()
            report["resumed"] = True
            report["message"] = (
                "已核验 OTA 暂存包，继续运行被调试器截停的安装引导"
                if pending
                else "已核验新应用，继续运行被调试器二次截停的启动引导"
            )
            report["state_after"] = str(target.get_state())
    finally:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
