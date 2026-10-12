"""Trigger only the running firmware's existing LAN-listener recycle path.

This is an explicit one-byte *RAM* write, not a flash update or reset.  It is
guarded by an exact application-image match and idle/dark OTA state checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import load_package


MAP_SYMBOL = re.compile(r"^\s*(\w+)\s+(0x[0-9a-fA-F]+)\s+Data\s+(\d+)\s+")
REQUIRED = {
    "workState", "otaSession", "otaRuntimeBootId", "bootId", "wifiState",
    "wifiConnected", "mqttConnected", "uartTxBusy", "localControlActive",
    "lanClientActive", "lanSocketId", "lanCleanupSocketId",
    "lanSocketResetPending", "lanStartupServerStatus", "nextLanRetryMs",
}


def map_symbols(path: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MAP_SYMBOL.match(line)
        if match and match.group(1) in REQUIRED:
            result[match.group(1)] = (int(match.group(2), 16), int(match.group(3)))
    if set(result) != REQUIRED:
        raise RuntimeError(f"missing symbols: {sorted(REQUIRED - set(result))}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--read-only", action="store_true",
                        help="Only report live state; never request listener recycle")
    args = parser.parse_args()
    package = load_package(args.package)
    symbols = map_symbols(args.map_path)
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False,
        unique_id=args.probe,
        options={
            "target_override": "cortex_m", "connect_mode": "attach",
            "auto_unlock": False, "resume_on_disconnect": False,
            "no_config": True, "frequency": 1_000_000,
            "cache.enable_memory": False, "cache.enable_register": False,
        },
    )
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("exact requested probe unavailable")

    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("CPU is not running; refusing RAM write")
        actual = bytes(target.read_memory_block8(package.link_address, package.image_size))
        if actual != package.image:
            raise RuntimeError("running application differs from expected package")

        def scalar(name: str, *, signed: bool = False) -> int:
            address, width = symbols[name]
            if name == "otaSession":
                width = 1  # first struct byte is active
            return int.from_bytes(
                bytes(target.read_memory_block8(address, width)),
                "little", signed=signed,
            )

        before = {
            "workState": scalar("workState"),
            "otaActive": scalar("otaSession"),
            "otaRuntimeBootId": scalar("otaRuntimeBootId"),
            "bootId": scalar("bootId"),
            "wifiState": scalar("wifiState"),
            "wifiConnected": scalar("wifiConnected"),
            "mqttConnected": scalar("mqttConnected"),
            "uartTxBusy": scalar("uartTxBusy"),
            "localControlActive": scalar("localControlActive"),
            "lanClientActive": scalar("lanClientActive"),
            "lanSocketId": scalar("lanSocketId", signed=True),
            "lanCleanupSocketId": scalar("lanCleanupSocketId", signed=True),
            "lanSocketResetPending": scalar("lanSocketResetPending"),
            "lanStartupServerStatus": scalar("lanStartupServerStatus", signed=True),
            "nextLanRetryMs": scalar("nextLanRetryMs"),
        }
        print(json.dumps({"preflight": before}, ensure_ascii=False), flush=True)
        if args.read_only:
            return 0
        required = {
            "workState": 2, "otaActive": 0, "wifiState": 30,
            "wifiConnected": 1, "mqttConnected": 1, "uartTxBusy": 0,
            "localControlActive": 0, "lanClientActive": 0,
            "lanSocketId": -1, "lanCleanupSocketId": -1,
            "lanSocketResetPending": 0, "lanStartupServerStatus": 3,
            "nextLanRetryMs": 0,
        }
        if any(before[name] != value for name, value in required.items()):
            raise RuntimeError("board is not in the exact idle LAN state required")
        if not before["bootId"] or before["otaRuntimeBootId"] != before["bootId"]:
            raise RuntimeError("symbol-map boot token mismatch")

        flag_address, flag_width = symbols["lanSocketResetPending"]
        if flag_width != 1:
            raise RuntimeError("unexpected reset flag width")
        target.write8(flag_address, 1)
        target.flush()
        print(json.dumps({
            "ram_write": {"symbol": "lanSocketResetPending", "value": 1},
            "flash_writes": 0,
        }, ensure_ascii=False), flush=True)
        time.sleep(8.0)
        after = {
            "wifiState": scalar("wifiState"),
            "lanSocketResetPending": scalar("lanSocketResetPending"),
            "lanStartupServerStatus": scalar("lanStartupServerStatus", signed=True),
            "lanSocketId": scalar("lanSocketId", signed=True),
            "cpuState": str(target.get_state()),
        }
        print(json.dumps({"after_8s": after}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
