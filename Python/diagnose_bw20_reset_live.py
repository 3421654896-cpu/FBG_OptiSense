"""Measure a BW20 software reset through the temporary USB/UART bridge.

The script never sends laser, ADC, DAC, or CNC commands.  Wi-Fi passwords are
not printed: the WJAP response is reduced to status, SSID, BSSID and IP only.
"""

from __future__ import annotations

import argparse
import re
import time

import serial


WJAP_RE = re.compile(
    rb"\+WJAP:(\d+),([^,]*),[^,]*,([^,]*),[^,]*,[^,]*,[^,]*,([^,\r\n]*)"
)


def collect(port: serial.Serial, duration_s: float) -> bytes:
    deadline = time.monotonic() + duration_s
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            chunks.append(chunk)
        else:
            time.sleep(0.02)
    return b"".join(chunks)


def command(port: serial.Serial, payload: bytes, wait_s: float = 0.8) -> bytes:
    port.reset_input_buffer()
    port.write(payload + b"\r\n")
    port.flush()
    return collect(port, wait_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="COM6")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    with serial.Serial(args.serial, 115200, timeout=0.05, write_timeout=1.0) as port:
        initial = command(port, b"AT")
        if b"OK" not in initial:
            raise RuntimeError("BW20 did not answer the initial AT probe")

        before = command(port, b"AT+WJAP?", 1.2)
        before_match = WJAP_RE.search(before)
        if before_match:
            print(
                "before_reset="
                f"status:{before_match.group(1).decode('ascii', 'replace')} "
                f"ssid:{before_match.group(2).decode('utf-8', 'replace')} "
                f"bssid:{before_match.group(3).decode('ascii', 'replace')} "
                f"ip:{before_match.group(4).decode('ascii', 'replace')}"
            )
        else:
            print("before_reset=wjan_unparsed")

        port.reset_input_buffer()
        started = time.monotonic()
        port.write(b"AT+RST\r\n")
        port.flush()
        reset_reply = collect(port, 2.0)
        print(f"reset_ack={int(b'OK' in reset_reply)}")

        ready_elapsed: float | None = None
        deadline = started + args.timeout
        while time.monotonic() < deadline:
            reply = command(port, b"AT", 0.35)
            if b"OK" in reply:
                ready_elapsed = time.monotonic() - started
                break
            time.sleep(0.40)
        if ready_elapsed is None:
            print(f"at_ready=0 timeout_s={args.timeout:.1f}")
            return 1
        print(f"at_ready=1 elapsed_s={ready_elapsed:.3f}")

        association = command(port, b"AT+WJAP?", 1.5)
        match = WJAP_RE.search(association)
        if match:
            print(
                "after_reset="
                f"status:{match.group(1).decode('ascii', 'replace')} "
                f"ssid:{match.group(2).decode('utf-8', 'replace')} "
                f"bssid:{match.group(3).decode('ascii', 'replace')} "
                f"ip:{match.group(4).decode('ascii', 'replace')}"
            )
        else:
            print("after_reset=wjan_unparsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
