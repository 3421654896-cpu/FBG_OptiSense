"""Read BW20 AT identity/socket table and one TCP-connect event via bridge.

No AT reset, socket deletion, laser command, OTA upload, or credential query.
"""

from __future__ import annotations

import argparse
import socket
import time

import serial


def exchange(port: serial.Serial, command: bytes, wait: float = 0.8) -> bytes:
    port.reset_input_buffer()
    port.write(command + b"\r\n")
    port.flush()
    time.sleep(wait)
    return port.read_all()


def show(label: str, data: bytes) -> None:
    print(label, repr(data.decode("utf-8", "replace")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="COM16")
    parser.add_argument("--host", default="192.168.4.23")
    parser.add_argument("--port", type=int, default=45670)
    args = parser.parse_args()

    with serial.Serial(args.serial, 115200, timeout=0.1, write_timeout=1.0) as uart:
        uart.dtr = True
        time.sleep(0.25)
        show("AT", exchange(uart, b"AT"))
        show("GMR", exchange(uart, b"AT+GMR"))
        association = exchange(uart, b"AT+WJAP?")
        association_line = next(
            (line for line in association.decode("utf-8", "replace").splitlines()
             if line.startswith("+WJAP:")), ""
        )
        fields = association_line.removeprefix("+WJAP:").split(",")
        if len(fields) >= 9:
            print(
                "WJAP_SANITIZED",
                {"status": fields[0], "mac": fields[5], "channel": fields[6],
                 "ip": fields[7], "gateway": fields[8]},
                flush=True,
            )
        else:
            print("WJAP_SANITIZED", "unavailable", flush=True)
        show("SOCKET_BEFORE", exchange(uart, b"AT+SOCKET?"))
        uart.reset_input_buffer()
        try:
            client = socket.create_connection((args.host, args.port), timeout=3.0)
            client.settimeout(0.5)
            time.sleep(0.25)
            show("CONNECT_URC", uart.read_all())
            try:
                client.sendall(b"FBGL1|HELLO\n")
                time.sleep(0.25)
                show("PAYLOAD_URC", uart.read_all())
            except OSError as exc:
                print("TCP_SEND_ERROR", type(exc).__name__, str(exc), flush=True)
            show("SOCKET_DURING", exchange(uart, b"AT+SOCKET?"))
            try:
                received = client.recv(64)
                show("TCP_RECEIVED", received)
            except OSError as exc:
                print("TCP_RECV_ERROR", type(exc).__name__, str(exc), flush=True)
            client.close()
        except OSError as exc:
            print("TCP_CONNECT_ERROR", type(exc).__name__, str(exc), flush=True)
        time.sleep(0.25)
        show("DISCONNECT_URC", uart.read_all())
        show("SOCKET_AFTER", exchange(uart, b"AT+SOCKET?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
