"""Fast non-halting Wi-Fi state trace after an authenticated OTA reboot.

This diagnostic reads only metadata, VTOR, a short application identity prefix,
and whitelisted non-secret SRAM scalars. It never reads credentials or writes
the target. Full-image readback remains a separate post-OTA acceptance check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import time

from elftools.elf.elffile import ELFFile
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import load_package
from inspect_wifi_ram_readonly import FIELDS


TRACE_FIELDS = (
    "clockMilliseconds", "wifiState", "wifiConnected", "wifiLastError",
    "retryExponent", "uartTxBusy", "uartErrorPending", "lanSocketId",
    "lanSocketResetPending", "lanStartupCleanupAttempts",
    "lanStartupStaleChildId", "lanStartupServerStatus",
    "lanServerDeleteVerifyAttempts", "lanListenerCreateVerifyAttempts",
    "lanCreateAttempts", "lanCreateErrors", "lanCreateLastResult",
    "lanCreateLastErrorCode", "lanQueryRows", "lanQuerySeeds",
    "lanQueryParents",
    "lanClientActive", "lanCleanupOverflow", "lanCleanupSocketId",
)


def symbols_from_elf(path: Path) -> dict[str, tuple[int, int, bool]]:
    symbols: dict[str, tuple[int, int, bool]] = {}
    with path.open("rb") as handle:
        table = ELFFile(handle).get_section_by_name(".symtab")
        if table is None:
            raise ValueError("ELF has no symbols")
        for symbol in table.iter_symbols():
            name = symbol.name
            if name not in TRACE_FIELDS:
                continue
            size, signed = FIELDS[name]
            address = int(symbol["st_value"])
            if (name in symbols or int(symbol["st_size"]) != size or
                    not 0x20000000 <= address < address + size <= 0x20020000):
                raise ValueError(f"invalid RAM symbol: {name}")
            symbols[name] = address, size, signed
    if set(symbols) != set(TRACE_FIELDS):
        raise ValueError("missing whitelisted RAM symbols")
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--elf", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=170.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.duration <= 300:
        raise ValueError("duration must be 1..300 seconds")
    symbols = symbols_from_elf(args.elf)
    package = load_package(args.package)
    options = {
        "target_override": "cortex_m", "connect_mode": "attach",
        "auto_unlock": False, "resume_on_disconnect": False,
        "no_config": True, "frequency": 1_000_000,
        "cache.enable_memory": False, "cache.enable_register": False,
        "cache.read_code_from_elf": False,
    }
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=args.probe, options=options
    )
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("exact debug probe unavailable")

    started = time.monotonic()
    events: list[dict] = []
    previous: tuple | None = None
    with session:
        target = session.target
        while time.monotonic() - started < args.duration:
            now = time.monotonic()
            event = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(now - started, 3),
            }
            try:
                state = target.get_state()
                event["cpu_state"] = str(state)
                if state != Target.State.RUNNING:
                    event["phase"] = "not_running"
                else:
                    metadata = bytes(target.read_memory_block8(0x08006000, 64))
                    words = struct.unpack("<16I", metadata)
                    vtor = target.read32(0xE000ED08)
                    event["metadata_magic"] = f"{words[0]:08X}"
                    event["metadata_version"] = f"{words[3]:08X}"
                    event["vtor"] = f"{vtor:08X}"
                    installed = (
                        words[0] == 0 and
                        words[3] == package.firmware_version and
                        vtor == package.link_address and
                        bytes(target.read_memory_block8(
                            package.link_address, 16
                        )) == package.image[:16]
                    )
                    if not installed:
                        event["phase"] = "boot_or_installing"
                    else:
                        event["phase"] = "application"
                        event["scalars"] = {
                            name: int.from_bytes(bytes(target.read_memory_block8(
                                address, size)), "little", signed=signed)
                            for name, (address, size, signed) in symbols.items()
                        }
            except Exception as exc:
                event["phase"] = "read_unavailable"
                event["error_type"] = type(exc).__name__
            signature = (
                event.get("phase"), event.get("metadata_magic"),
                event.get("vtor"),
                tuple(event.get("scalars", {}).get(name) for name in TRACE_FIELDS[1:]),
            )
            if signature != previous:
                print(json.dumps(event), flush=True)
                previous = signature
            events.append(event)
            time.sleep(max(0.0, 1.0 - (time.monotonic() - now)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": "wifi_boot_fast_readonly_v1",
        "probe": args.probe,
        "package_version": package.version_text,
        "events": events,
        "read_only": True,
        "full_image_verified": False,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
