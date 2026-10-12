"""Inspect BW20 TCP-seed lifecycle through the temporary USB/UART bridge.

This tool never sends a laser command.  It removes only disconnected TCPSeed
entries belonging to the fixed LAN server ConID 9, then records the exact URCs
emitted for one fresh client connect, payload, and disconnect sequence.
"""

from __future__ import annotations

import argparse
import re
import socket
import time

import serial


SEED_RE = re.compile(
    rb"(?m)^(\d+),5,(-?\d+),[^,]*,-?\d+,-?\d+,9\r?$"
)


def exchange(port: serial.Serial, command: bytes, wait_s: float = 0.8) -> bytes:
    port.reset_input_buffer()
    port.write(command + b"\r\n")
    port.flush()
    time.sleep(wait_s)
    return port.read_all()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="COM6")
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--port", type=int, default=45670)
    args = parser.parse_args()

    with serial.Serial(args.serial, 115200, timeout=0.15, write_timeout=1.0) as uart:
        table = exchange(uart, b"AT+SOCKET?")
        print("socket_table_before:")
        print(table.decode("utf-8", "replace"))
        for match in SEED_RE.finditer(table):
            con_id = int(match.group(1))
            status = int(match.group(2))
            # Combo-AT documents state 3 as connected; 0/4/127 are stale,
            # failed, or deleting seeds and must be reaped explicitly.
            if status == 3:
                continue
            response = exchange(uart, f"AT+SOCKETDEL={con_id}".encode("ascii"))
            print(f"delete_stale_seed={con_id} status={status}")
            print(response.decode("utf-8", "replace"))

        print("socket_table_clean:")
        print(exchange(uart, b"AT+SOCKET?").decode("utf-8", "replace"))

        uart.reset_input_buffer()
        client = socket.create_connection((args.host, args.port), timeout=2.0)
        client.settimeout(0.5)
        time.sleep(0.5)
        print("events_after_connect:")
        print(uart.read_all().decode("utf-8", "replace"))

        client.sendall(b"FBGL1|RAW")
        time.sleep(0.5)
        print("events_after_payload:")
        print(uart.read_all().decode("utf-8", "replace"))
        print("socket_table_while_connected:")
        print(exchange(uart, b"AT+SOCKET?").decode("utf-8", "replace"))

        client.shutdown(socket.SHUT_RDWR)
        client.close()
        time.sleep(1.0)
        print("events_after_disconnect:")
        print(uart.read_all().decode("utf-8", "replace"))

        print("socket_table_after_disconnect:")
        print(exchange(uart, b"AT+SOCKET?").decode("utf-8", "replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
