"""Read-only LAN-owner trace using the exact current AXF map and SWD probe.

This diagnostic never opens USB serial, writes target memory, or controls the
laser/CNC.  It correlates one RAW TCP probe with the STM32 LAN state variables.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import time
from pathlib import Path

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target


SYMBOLS = (
    "lanSocketId",
    "lanCleanupSocketId",
    "lanDeleteTargetId",
    "wifiState",
    "localControlActive",
    "lanClientActive",
    "lanRawMode",
    "lanSocketResetPending",
    "lanCleanupCanPromote",
    "lanReadPending",
    "lanDeleteKind",
    "wifiLastError",
)


def load_symbols(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    pattern = re.compile(r"^\s*(\w+)\s+(0x[0-9a-fA-F]+)\s+Data\s+")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match and match.group(1) in SYMBOLS:
            result[match.group(1)] = int(match.group(2), 16)
    missing = sorted(set(SYMBOLS) - set(result))
    if missing:
        raise RuntimeError(f"Current map is missing symbols: {missing}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--port", type=int, default=45670)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--map", dest="map_file", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    symbols = load_symbols(args.map_file)
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False,
        unique_id=args.probe,
        options={
            "target_override": "cortex_m",
            "connect_mode": "attach",
            "auto_unlock": False,
            "resume_on_disconnect": True,
            "no_config": True,
            "frequency": 1_000_000,
            "cache.enable_memory": False,
            "cache.enable_register": False,
        },
    )
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("Exact debug probe is unavailable")

    report: dict[str, object] = {
        "read_only": True,
        "host": args.host,
        "port": args.port,
        "probe": args.probe,
        "events": [],
    }
    connection: socket.socket | None = None
    started = time.monotonic()
    last_state: dict[str, int] | None = None
    next_hello = 0.0

    def read_state(target: Target) -> dict[str, int]:
        state: dict[str, int] = {}
        for name, address in symbols.items():
            width = 2 if name == "wifiLastError" else 1
            raw = bytes(target.read_memory_block8(address, width))
            state[name] = int.from_bytes(raw, "little", signed=False)
        return state

    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("CPU must already be running for read-only trace")
        baseline = read_state(target)
        report["events"].append({"elapsed_s": 0.0, "phase": "before_connect", "state": baseline})
        last_state = baseline
        try:
            connection = socket.create_connection((args.host, args.port), timeout=3.0)
            connection.settimeout(0.02)
            while time.monotonic() - started < args.seconds:
                elapsed = time.monotonic() - started
                if elapsed >= next_hello:
                    try:
                        connection.sendall(b"FBGL1|RAW")
                    except OSError as exc:
                        report["socket_error"] = f"{type(exc).__name__}: {exc}"
                        break
                    next_hello = elapsed + 1.0
                try:
                    received = connection.recv(4096)
                    if received:
                        report.setdefault("received", []).append(received.hex())
                    else:
                        report["socket_error"] = "peer closed"
                        break
                except socket.timeout:
                    pass
                except OSError as exc:
                    report["socket_error"] = f"{type(exc).__name__}: {exc}"
                    break

                state = read_state(target)
                if state != last_state:
                    report["events"].append(
                        {"elapsed_s": round(elapsed, 4), "state": state}
                    )
                    last_state = state
                time.sleep(0.02)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        final_state = read_state(target)
        if final_state != last_state:
            report["events"].append(
                {
                    "elapsed_s": round(time.monotonic() - started, 4),
                    "phase": "after_socket_exit",
                    "state": final_state,
                }
            )
        report["cpu_state_after"] = str(target.get_state())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
