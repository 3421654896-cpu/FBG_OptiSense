"""Short hardware verification for the same-WiFi transport.

The script requests one stress frame and one temperature frame, then returns
the controller to EXTRA (local idle/debug) mode through COM6 before exiting.
It is intentionally separate from the normal graphical application.
"""

from __future__ import annotations

import argparse
import time

import serial

from fbg_lan_transport import FbgLanTransport
from fbg_remote_protocol import MODE_STRESS, MODE_TEMPERATURE


def _return_to_local_idle(port: str) -> None:
    command = bytearray(808)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
    command[8] = 2  # EXTRA_STATE: local idle/debug mode
    with serial.Serial(port, 115200, timeout=0.2) as device:
        device.dtr = True
        time.sleep(0.25)
        device.reset_input_buffer()
        device.write(command)
        device.flush()
        time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="192.168.3.46")
    parser.add_argument("--serial", default="COM6")
    args = parser.parse_args()

    metadata_modes: set[int] = set()
    acknowledgements = []
    transport = FbgLanTransport(
        board_host=args.board,
        on_connection=lambda connected, detail: print(
            f"connection={connected}: {detail}", flush=True
        ),
        on_metadata=lambda frame: (
            metadata_modes.add(frame.mode),
            print(
                f"metadata mode={frame.mode} points={len(frame.wavelength_pm)}",
                flush=True,
            ),
        ),
        on_ack=lambda ack: (
            acknowledgements.append(ack),
            print(f"ack={ack}", flush=True),
        ),
    )

    results = {}
    transport.start()
    try:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and metadata_modes != {0, 3}:
            time.sleep(0.05)
        if metadata_modes != {0, 3}:
            raise RuntimeError(f"metadata incomplete: {sorted(metadata_modes)}")

        for command_id, mode, timeout_s in (
            (0xA5010001, MODE_STRESS, 8.0),
            (0xA5010002, MODE_TEMPERATURE, 15.0),
        ):
            transport.publish_mode(mode, command_id)
            deadline = time.monotonic() + timeout_s
            frame = None
            while time.monotonic() < deadline:
                candidate = transport.take_latest_telemetry()
                if candidate is not None and candidate.mode == mode:
                    frame = candidate
                    break
                time.sleep(0.02)
            if frame is None:
                raise RuntimeError(f"no telemetry frame for mode {mode}")
            results[mode] = frame
            print(
                "telemetry "
                f"mode={frame.mode} sequence={frame.seq} "
                f"points={len(frame.samples)} "
                f"channels={len(frame.samples[0]) if frame.samples else 0}",
                flush=True,
            )
    finally:
        transport.stop()
        _return_to_local_idle(args.serial)
        print("board returned to local EXTRA/idle mode", flush=True)

    if set(results) != {MODE_STRESS, MODE_TEMPERATURE}:
        return 1
    if not acknowledgements:
        raise RuntimeError("mode acknowledgement was not received")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
