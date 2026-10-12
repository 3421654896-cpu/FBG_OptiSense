"""Verify that an open COM6 local session silences both network data paths."""

from __future__ import annotations

import argparse
import time

import serial

from fbg_lan_transport import FbgLanTransport
from fbg_remote_protocol import MODE_STRESS


def _wait_until(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="192.168.3.46")
    parser.add_argument("--serial", default="COM6")
    args = parser.parse_args()

    metadata = []
    acknowledgements = []
    transport = FbgLanTransport(
        board_host=args.board,
        on_metadata=lambda frame: metadata.append(frame),
        on_ack=lambda ack: acknowledgements.append(ack),
        on_connection=lambda connected, detail: print(
            f"connection={connected}: {detail}", flush=True
        ),
    )
    transport.start()
    try:
        if not _wait_until(lambda: len(metadata) >= 2, 12.0):
            raise RuntimeError("LAN bootstrap metadata was not received")
        print(f"LAN bootstrap metadata={len(metadata)}", flush=True)

        with serial.Serial(args.serial, 115200, timeout=0.2) as device:
            device.dtr = True
            time.sleep(0.35)
            idle = bytearray(808)
            idle[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
            idle[8] = 2
            device.write(idle)
            device.flush()
            time.sleep(0.35)

            metadata_before = len(metadata)
            ack_before = len(acknowledgements)
            while transport.take_latest_telemetry() is not None:
                pass
            try:
                transport.publish_mode(MODE_STRESS, 0xA5020001)
            except (OSError, RuntimeError):
                # The firmware may close the LAN seed as soon as COM6 wins.
                pass
            time.sleep(3.0)

            if len(metadata) != metadata_before:
                raise RuntimeError("LAN metadata was sent during local control")
            if len(acknowledgements) != ack_before:
                raise RuntimeError("network mode command was accepted during local control")
            if transport.take_latest_telemetry() is not None:
                raise RuntimeError("raw network telemetry was sent during local control")
            print("COM6 open: no LAN data and no network command ACK", flush=True)

        metadata_before_resume = len(metadata)
        if not _wait_until(lambda: len(metadata) >= metadata_before_resume + 2, 12.0):
            raise RuntimeError("LAN did not recover after COM6 closed")
        print("COM6 closed: LAN metadata resumed automatically", flush=True)
    finally:
        transport.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
