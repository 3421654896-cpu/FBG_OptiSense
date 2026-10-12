"""Read non-secret OTA metadata and CPU state through SWD without writes."""

from __future__ import annotations

import argparse
import binascii
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct

from pyocd.core.helpers import ConnectHelper

from fbg_ota_protocol import load_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    package = load_package(args.package)
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
        raise RuntimeError("exact requested probe unavailable")
    with session:
        target = session.target
        state = str(target.get_state())
        registers = {}
        if "HALTED" in state:
            for name in ("pc", "msp", "psp", "xpsr", "primask"):
                registers[name] = hex(int(target.read_core_register(name)))
        metadata_raw = bytes(target.read_memory_block8(0x08006000, 64))
        words = struct.unpack("<16I", metadata_raw)
        normalized = bytearray(metadata_raw)
        normalized[56:60] = b"\xff\xff\xff\xff"
        metadata_crc_calculated = binascii.crc32(normalized[4:60]) & 0xFFFFFFFF
        staging = bytes(
            target.read_memory_block8(0x08040000, package.image_size)
        )
        application = bytes(
            target.read_memory_block8(package.link_address, package.image_size)
        )
        report = {
            "schema": "ota_metadata_readonly_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
            "cpu_state": state,
            "registers_if_halted": registers,
            "metadata": {
                "magic": f"{words[0]:08X}",
                "format_version": words[1],
                "target_id": f"{words[2]:08X}",
                "image_version": f"{words[3]:08X}",
                "image_size": words[4],
                "image_crc32": f"{words[5]:08X}",
                "link_address": f"{words[6]:08X}",
                "staging_address": f"{words[7]:08X}",
                "install_boot_token": f"{words[8]:08X}",
                "retry_marker": f"{words[14]:08X}",
                "metadata_crc32": f"{words[15]:08X}",
                "calculated_crc32": f"{metadata_crc_calculated:08X}",
                "crc_valid_after_retry_normalization": (
                    words[15] == metadata_crc_calculated
                ),
            },
            "package_version": package.version_text,
            "package_image_crc32": f"{package.image_crc32:08X}",
            "staging_crc32": f"{binascii.crc32(staging) & 0xFFFFFFFF:08X}",
            "staging_matches_package": staging == package.image,
            "application_sha256": hashlib.sha256(application).hexdigest(),
            "application_matches_package": application == package.image,
            "flash_status_register": hex(target.read32(0x40023C0C)),
            "flash_control_register": hex(target.read32(0x40023C10)),
            "cpu_state_after": str(target.get_state()),
        }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
