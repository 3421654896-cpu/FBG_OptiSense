"""Read BW20 transport health over native USB without changing credentials."""

from __future__ import annotations

import argparse
import json
import time

import serial

from benchmark_stress_realtime import _close_shutter, _safe_disarm


COMMAND_SIZE = 808
STATUS_COMMAND = 0x21
STATUS_VERSION = 1
STATE_NAMES = {
    0: "disabled",
    1: "backoff",
    2: "wait_at",
    3: "wait_echo_off",
    4: "wait_mode",
}
EXTERNAL_STATE_NAMES = {
    0: "disabled",
    1: "starting",
    2: "joining_wifi",
    3: "joining_mqtt",
    4: "online",
    5: "backoff",
}


def build_status_query() -> bytes:
    frame = bytearray(COMMAND_SIZE)
    frame[:4] = b"\xff\xff\x01\x21"
    return bytes(frame)


def find_status_reply(buffer: bytes) -> dict | None:
    for offset in range(max(0, len(buffer) - 4096), max(0, len(buffer) - 19)):
        frame = buffer[offset : offset + 20]
        if (
            len(frame) != 20
            or frame[:2] != b"\xff\xff"
            or frame[3] != STATUS_COMMAND
            or frame[4] != STATUS_VERSION
            or frame[18:] != b"\xff\xef"
        ):
            continue
        flags = frame[12]
        external_state = frame[5]
        raw_state = frame[10]
        return {
            "protocol_version": frame[4],
            "work_state": frame[2],
            "external_state": external_state,
            "external_state_name": EXTERNAL_STATE_NAMES.get(
                external_state, f"unknown_{external_state}"
            ),
            "config_valid": bool(frame[6]),
            "enabled": bool(frame[7]),
            "last_error": int.from_bytes(frame[8:10], "big"),
            "raw_state": raw_state,
            "raw_state_name": STATE_NAMES.get(raw_state, f"state_{raw_state}"),
            "retry_exponent": frame[11],
            "wifi_connected": bool(flags & 0x01),
            "mqtt_connected": bool(flags & 0x02),
            "uart_tx_busy": bool(flags & 0x04),
            "rx_overflow": bool(flags & 0x08),
            "uart_error_pending": bool(flags & 0x10),
            "lan_client_active": bool(flags & 0x20),
            "ota_active": bool(flags & 0x40),
            "rx_ring_used": frame[13],
            "uart_tx_complete_count": int.from_bytes(frame[14:16], "big"),
            "uart_rx_byte_count": int.from_bytes(frame[16:18], "big"),
        }
    return None


def query(device, *, timeout_s: float = 2.0) -> dict:
    if not _safe_disarm(device) or not _close_shutter(device):
        raise RuntimeError("initial exact disarm/shutter ACK missing")
    try:
        device.reset_input_buffer()
        device.write(build_status_query())
        device.flush()
        received = bytearray()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            chunk = device.read(4096)
            if chunk:
                received.extend(chunk)
                status = find_status_reply(received)
                if status is not None:
                    return status
        raise TimeoutError("BW20 USB status reply missing")
    finally:
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("final exact disarm/shutter ACK missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    with serial.Serial(
        args.port, args.baud, timeout=0.05, write_timeout=1.0
    ) as device:
        device.dtr = True
        time.sleep(0.15)
        result = query(device, timeout_s=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
