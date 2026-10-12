"""Read-only acceptance check for the LAN path used after an OTA reboot.

This tool never sends ``FOTA1|BEGIN/DATA/END`` and never sends a native
DAC/ADC command.  It only opens the board TCP server and requests the existing
``FBGL1|RAW`` ready handshake.  Repeating that handshake catches the stale
BW20 client/ConID failure which previously required a manual power cycle.

The uploader itself validates the package and ``REBOOT`` version receipt,
closes the old half-open socket locally, and checks the new boot token carried
by RAW READY.  This script is a non-invasive, separately runnable transport
smoke check.
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from fbg_lan_serial import (
    RAW_HELLO,
    RAW_KIND_READY,
    RAW_MAX_PAYLOAD,
    RAW_RESPONSE_HEADER,
    RAW_RESPONSE_MAGIC,
    RAW_RESPONSE_VERSION,
)
from fbg_ota_protocol import OTA_PORT, load_package


class RawReadyError(RuntimeError):
    """The board did not complete a valid, CRC-protected RAW handshake."""


@dataclass(frozen=True)
class RawReadyResult:
    host: str
    port: int
    attempts: int
    hello_count: int
    elapsed_s: float
    client_close_elapsed_s: float
    boot_id: int


def _consume_ready_packet(buffer: bytearray) -> int | None:
    """Consume complete FBR1 packets and return an empty READY's boot token."""

    while buffer:
        start = buffer.find(RAW_RESPONSE_MAGIC)
        if start < 0:
            # Retain only a possible partial magic suffix.
            if len(buffer) > len(RAW_RESPONSE_MAGIC) - 1:
                del buffer[: -(len(RAW_RESPONSE_MAGIC) - 1)]
            return None
        if start:
            del buffer[:start]
        if len(buffer) < RAW_RESPONSE_HEADER.size:
            return None
        magic, version, kind, _sequence, payload_length = (
            RAW_RESPONSE_HEADER.unpack_from(buffer)
        )
        if (
            magic != RAW_RESPONSE_MAGIC
            or version != RAW_RESPONSE_VERSION
            or payload_length > RAW_MAX_PAYLOAD
        ):
            del buffer[0]
            continue
        packet_length = RAW_RESPONSE_HEADER.size + payload_length + 4
        if len(buffer) < packet_length:
            return None
        packet = bytes(buffer[:packet_length])
        del buffer[:packet_length]
        expected_crc = int.from_bytes(packet[-4:], "big")
        if (zlib.crc32(packet[:-4]) & 0xFFFFFFFF) != expected_crc:
            continue
        payload = packet[RAW_RESPONSE_HEADER.size : -4]
        if kind == RAW_KIND_READY and not payload:
            return int(_sequence)
    return None


def _close_like_lan_client(sock: socket.socket) -> float:
    """Release the socket exactly like production ``FbgLanSerial.close``.

    A half-close followed by additional reads is a distinct lifecycle which
    the normal application never performs.  On BW20 it can change the order of
    SocketDown/SocketDisconnect events, so the verifier must not introduce it.
    Cleanup time is provided outside the closed connection between cycles.
    """

    started = time.monotonic()
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
    return time.monotonic() - started


def wait_for_raw_ready(
    host: str,
    *,
    port: int = OTA_PORT,
    timeout_s: float = 20.0,
    connect_timeout_s: float = 0.5,
    response_timeout_s: float = 0.25,
    retry_interval_s: float = 1.0,
    heartbeat_s: float = 1.0,
) -> RawReadyResult:
    """Wait for RAW READY without opening a new TCP socket per heartbeat.

    Once TCP connects, all RAW retries stay on that same connection.  This is
    essential during post-OTA recovery: after the 3.5 s ownership timeout the
    next heartbeat on the already accepted connection lets STM32 relearn its
    ConID, whereas opening another socket each second can exhaust BW20 seeds.
    """

    normalized_host = str(ipaddress.ip_address(str(host).strip()))
    if not 1 <= int(port) <= 65535:
        raise ValueError("port must be in 1..65535")
    started = time.monotonic()
    deadline = started + max(0.1, float(timeout_s))
    attempts = 0
    hello_count = 0
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        attempts += 1
        sock: socket.socket | None = None
        try:
            remaining = max(0.05, deadline - time.monotonic())
            sock = socket.create_connection(
                (normalized_host, int(port)),
                timeout=min(max(0.05, float(connect_timeout_s)), remaining),
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buffer = bytearray()
            next_hello_s = 0.0
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_hello_s:
                    sock.sendall(RAW_HELLO)
                    hello_count += 1
                    next_hello_s = now + max(0.1, float(heartbeat_s))
                receive_wait = min(
                    max(0.01, float(response_timeout_s)),
                    max(0.01, deadline - time.monotonic()),
                    max(0.01, next_hello_s - time.monotonic()),
                )
                sock.settimeout(receive_wait)
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    raise ConnectionResetError("board closed before RAW READY")
                buffer.extend(chunk)
                boot_id = _consume_ready_packet(buffer)
                if boot_id is not None:
                    ready_elapsed_s = time.monotonic() - started
                    client_close_elapsed_s = _close_like_lan_client(sock)
                    sock = None
                    return RawReadyResult(
                        normalized_host,
                        int(port),
                        attempts,
                        hello_count,
                        ready_elapsed_s,
                        client_close_elapsed_s,
                        boot_id,
                    )
        except (OSError, RawReadyError) as exc:
            last_error = exc
        finally:
            if sock is not None:
                _close_like_lan_client(sock)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(max(0.25, float(retry_interval_s)), remaining))

    raise RawReadyError(
        f"{normalized_host}:{int(port)} did not return RAW READY in "
        f"{max(0.1, float(timeout_s)):.1f}s after {attempts} attempts: "
        f"{last_error or 'no response'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only repeated RAW READY acceptance check"
    )
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--port", type=int, default=OTA_PORT)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--cycle-gap", type=float, default=3.75)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 20:
        raise ValueError("--cycles must be in 1..20")

    if args.package is not None:
        package = load_package(args.package)
        print(
            "package_preflight=PASS "
            f"version={package.version_text} size={package.image_size} "
            f"crc32={package.image_crc32:08X}",
            flush=True,
        )

    for cycle in range(1, args.cycles + 1):
        result = wait_for_raw_ready(
            args.host,
            port=args.port,
            timeout_s=args.timeout,
        )
        print(
            f"raw_ready_cycle={cycle} attempts={result.attempts} "
            f"hellos={result.hello_count} elapsed_s={result.elapsed_s:.3f} "
            f"client_close=1 close_s={result.client_close_elapsed_s:.3f} "
            f"boot_id={result.boot_id:08X} "
            f"host={result.host} "
            f"port={result.port}",
            flush=True,
        )
        if cycle < args.cycles:
            # Even if a particular AT firmware fails to return peer EOF, the
            # next cycle cannot start inside the STM32's 3.5 s owner timeout.
            time.sleep(max(3.5, args.cycle_gap))

    print("ota_lan_reconnect_path=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
