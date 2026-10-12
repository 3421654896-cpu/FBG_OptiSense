"""Read an interrupted OTA session without resetting or writing the target.

Only an exact running legacy application and exact target/package session are
accepted.  The private per-session nonce is returned to the caller but must
never be logged; it permits resuming a partial staging image without erase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import FirmwarePackage, load_package


SYMBOL = re.compile(r"^\s*(\w+)\s+(0x[0-9a-fA-F]+)\s+Data\s+(\d+)\s+", re.M)


@dataclass(frozen=True)
class VerifiedResume:
    nonce: str
    offset: int
    session_id: int
    boot_id: int


def _addresses(path: Path) -> dict[str, int]:
    found = {
        name: int(address, 16)
        for name, address, _size in SYMBOL.findall(path.read_text(encoding="utf-8", errors="replace"))
        if name in {"otaSession", "otaRuntimeBootId", "bootId", "workState"}
    }
    if set(found) != {"otaSession", "otaRuntimeBootId", "bootId", "workState"}:
        raise RuntimeError("OTA state symbols missing from map")
    return found


def read_verified_resume(
    *, probe_id: str, package: FirmwarePackage,
    installed_package: Path, map_path: Path,
) -> VerifiedResume:
    installed = load_package(installed_package)
    if installed.link_address != package.link_address:
        raise RuntimeError("Installed/package link addresses differ")
    addresses = _addresses(map_path)
    options = {
        "target_override": "cortex_m", "connect_mode": "attach",
        "auto_unlock": False, "resume_on_disconnect": False,
        "no_config": True, "frequency": 1_000_000,
        "cache.enable_memory": False, "cache.enable_register": False,
    }
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=probe_id, options=options
    )
    if session is None or str(session.probe.unique_id) != probe_id:
        raise RuntimeError("Exact debug probe unavailable")
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("Target not running; refusing resume")
        image = bytes(target.read_memory_block8(installed.link_address, installed.image_size))
        if image != installed.image:
            raise RuntimeError("Installed image changed; refusing session resume")
        if target.read8(addresses["workState"]) != 2:
            raise RuntimeError("Board not in safe EXTRA work state")
        boot_id = target.read32(addresses["bootId"])
        if not boot_id or target.read32(addresses["otaRuntimeBootId"]) != boot_id:
            raise RuntimeError("Map/boot token mismatch")
        raw = bytes(target.read_memory_block8(addresses["otaSession"], 44))
        active, session_id, version, image_size, crc, offset = struct.unpack_from(
            "<B3xIIIII", raw
        )
        nonce_bytes = raw[24:40]
        if active != 1 or not session_id or version != package.firmware_version:
            raise RuntimeError("No matching active OTA session")
        if image_size != package.image_size or crc != package.image_crc32:
            raise RuntimeError("Active OTA session does not match package")
        if not 0 < offset < package.image_size or offset % 4:
            raise RuntimeError("Invalid persisted OTA offset")
        if raw[40] != 0 or any(ch not in b"0123456789ABCDEF" for ch in nonce_bytes):
            raise RuntimeError("Invalid persisted OTA nonce")
        nonce = nonce_bytes.decode("ascii")
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("Target state changed during inspection")
    return VerifiedResume(nonce, offset, session_id, boot_id)


__all__ = ["VerifiedResume", "read_verified_resume"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    args = parser.parse_args()
    candidate = load_package(args.package)
    result = read_verified_resume(
        probe_id=args.probe, package=candidate,
        installed_package=args.installed, map_path=args.map_path,
    )
    print(f"verified=1 offset={result.offset}/{candidate.image_size} "
          f"session_id={result.session_id:08X} boot_id={result.boot_id:08X}")
