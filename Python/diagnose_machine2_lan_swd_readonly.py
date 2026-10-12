"""Observe one LAN HELLO and selected STM32 state via read-only SWD.

This does not send OTA commands, read credentials, reset the target, or write
target memory.  The symbol map must be validated against the running build
before interpreting the resulting scalar values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
import threading
import time

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target


FIELDS = (
    "wifiState", "wifiConnected", "mqttConnected", "wifiLastError",
    "localControlActive", "lanClientActive", "lanRawMode",
    "lanSocketId", "lanCleanupSocketId", "lanDeleteTargetId",
    "lanSocketResetPending", "lanCleanupCanPromote", "lanCleanupOverflow",
    "lanReadPending", "lanDeleteKind", "lanStartupServerStatus",
    "lanStartupStaleChildId", "lanStartupCleanupAttempts",
    "lanServerDeleteVerifyAttempts", "lanServerDeleteVerifyStartup",
    "uartTxBusy", "uartTxCompleteCount", "uartRxByteCount",
)
SIGNED = {
    "lanSocketId", "lanCleanupSocketId", "lanDeleteTargetId",
    "lanStartupStaleChildId", "lanStartupServerStatus",
}
SYMBOL = re.compile(r"^\s*(\w+)\s+(0x[0-9a-fA-F]+)\s+Data\s+(\d+)\s+")


def symbols_from_map(path: Path) -> dict[str, tuple[int, int, bool]]:
    result: dict[str, tuple[int, int, bool]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SYMBOL.match(line)
        if match and match.group(1) in FIELDS:
            name = match.group(1)
            result[name] = (
                int(match.group(2), 16), int(match.group(3)), name in SIGNED
            )
    missing = set(FIELDS) - set(result)
    if missing:
        raise RuntimeError(f"map missing fields: {sorted(missing)}")
    return result


def snapshot(target, symbols: dict[str, tuple[int, int, bool]]) -> dict[str, int]:
    # The linker keeps these scalars in one .data and one .bss cluster.  Two
    # bounded reads keep a short-lived socket transition observable without
    # touching unrelated RAM or copying the credential-bearing config.
    groups: dict[int, list[tuple[str, int, int, bool]]] = {0: [], 1: []}
    for name, (address, width, signed) in symbols.items():
        groups[0 if address < 0x20000100 else 1].append(
            (name, address, width, signed)
        )
    values: dict[str, int] = {}
    for entries in groups.values():
        start = min(item[1] for item in entries)
        end = max(item[1] + item[2] for item in entries)
        block = bytes(target.read_memory_block8(start, end - start))
        for name, address, width, signed in entries:
            values[name] = int.from_bytes(
                block[address - start:address - start + width],
                "little", signed=signed,
            )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    symbols = symbols_from_map(args.map_path)
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

    connection_result: dict[str, object] = {}
    started = time.monotonic()

    def connect_once() -> None:
        time.sleep(1.0)
        sock = None
        try:
            sock = socket.create_connection((args.host, 45670), timeout=2.0)
            sock.settimeout(2.0)
            connection_result["connected_s"] = round(time.monotonic() - started, 3)
            sock.sendall(b"FBGL1|HELLO")
            received = sock.recv(4096)
            connection_result["received_bytes"] = len(received)
            connection_result["first_bytes_hex"] = received[:16].hex()
        except OSError as exc:
            connection_result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
            connection_result["completed_s"] = round(time.monotonic() - started, 3)

    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("CPU is not running; refusing to alter it")
        baseline = snapshot(target, symbols)
        if (baseline["wifiState"] > 44 or
                any(baseline[name] not in (0, 1) for name in (
                    "wifiConnected", "mqttConnected", "localControlActive",
                    "lanClientActive", "lanSocketResetPending"))):
            raise RuntimeError("symbol map is incompatible with running image")
        print(json.dumps({"at_s": 0, "state": baseline}, ensure_ascii=False), flush=True)
        worker = threading.Thread(target=connect_once, daemon=True)
        worker.start()
        previous = baseline
        while time.monotonic() - started < max(3.0, args.duration):
            current = snapshot(target, symbols)
            if current != previous:
                changed = {k: v for k, v in current.items() if v != previous[k]}
                print(json.dumps({
                    "at_s": round(time.monotonic() - started, 3),
                    "changed": changed,
                }, ensure_ascii=False), flush=True)
                previous = current
            time.sleep(0.05)
        worker.join(timeout=3.0)
        print(json.dumps({
            "connection": connection_result,
            "cpu_state_after": str(target.get_state()),
            "final_state": snapshot(target, symbols),
        }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
