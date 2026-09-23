"""Direct same-WiFi TCP transport for the FBG interrogator.

The PC initiates the connection, so Windows treats it as outbound traffic and
does not require a public-network inbound firewall rule. The stream contains
the exact same CRC-protected FBG1/FBGM/FACK1 payloads used by MQTT.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from fbg_remote_protocol import (
    CRC_SIZE,
    HEADER_SIZE,
    MAX_POINT_COUNT,
    decode_ack,
    decode_metadata,
    decode_telemetry,
    encode_mode_command,
)


LAN_BOARD_PORT = 45670
LAN_HELLO = b"FBGL1|HELLO"
_ACK_PATTERN = re.compile(
    rb"^FACK1\|[0-9A-F]{8}\|(ACCEPTED|APPLIED|REJECTED)\|"
    rb"(STRESS|TEMPERATURE)\|[0-9A-F]{8}"
)
_MAGICS = (b"FBG1", b"FBGM", b"FACK1|")
_MAX_TELEMETRY_PACKET_SIZE = HEADER_SIZE + MAX_POINT_COUNT * 4 * 2 + CRC_SIZE
_MAX_METADATA_PACKET_SIZE = 12 + MAX_POINT_COUNT * 4 + CRC_SIZE
_MAX_ACK_PACKET_SIZE = len(b"FACK1|FFFFFFFF|ACCEPTED|TEMPERATURE|FFFFFFFF")

AckCallback = Callable[[object], None]
StatusCallback = Callable[[str], None]
ConnectionCallback = Callable[[bool, str], None]
MetadataCallback = Callable[[object], None]


def _local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 53))
            addresses.append(str(probe.getsockname()[0]))
        finally:
            probe.close()
    except OSError:
        pass
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.append(str(item[4][0]))
    except OSError:
        pass
    return list(
        dict.fromkeys(
            address
            for address in addresses
            if not address.startswith("127.") and not address.startswith("169.254.")
        )
    )


def _candidate_hosts() -> list[str]:
    """Return peers on the workstation's local /24 networks."""

    candidates: list[str] = []
    for local in _local_ipv4_addresses():
        try:
            network = ipaddress.ip_network(f"{local}/24", strict=False)
        except ValueError:
            continue
        candidates.extend(str(host) for host in network.hosts() if str(host) != local)
    return list(dict.fromkeys(candidates))


class FbgLanTransport:
    """Background TCP client with discovery and latest-frame backpressure."""

    def __init__(
        self,
        *,
        on_ack: Optional[AckCallback] = None,
        on_status: Optional[StatusCallback] = None,
        on_connection: Optional[ConnectionCallback] = None,
        on_metadata: Optional[MetadataCallback] = None,
        board_port: int = LAN_BOARD_PORT,
        board_host: Optional[str] = None,
        heartbeat_s: float = 1.0,
        discovery_interval_s: float = 2.0,
        connect_timeout_s: float = 0.12,
    ) -> None:
        self.on_ack = on_ack
        self.on_status = on_status
        self.on_connection = on_connection
        self.on_metadata = on_metadata
        if isinstance(board_port, bool) or not isinstance(board_port, int):
            raise ValueError("board_port 必须是整数")
        if not 1 <= board_port <= 65535:
            raise ValueError("board_port 必须在 1..65535 之间")
        self.board_port = board_port
        self.board_host = str(
            board_host or os.environ.get("FBG_LAN_BOARD_IP", "")
        ).strip() or None
        self.heartbeat_s = float(heartbeat_s)
        self.discovery_interval_s = float(discovery_interval_s)
        self.connect_timeout_s = float(connect_timeout_s)

        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[object] = None
        self._send_lock = threading.Lock()
        self._board_ip: Optional[str] = None
        self._buffer = bytearray()
        self._last_boot_id: Optional[int] = None

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def board_ip(self) -> Optional[str]:
        return self._board_ip

    def _notify_connection(self, connected: bool, detail: str) -> None:
        if self.on_connection is not None:
            try:
                self.on_connection(bool(connected), str(detail))
            except Exception:
                pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._connected.clear()
        self._thread = threading.Thread(
            target=self._run, name="fbg-lan-tcp", daemon=True
        )
        self._thread.start()
        self._notify_connection(False, "局域网正在寻找板卡")

    def stop(self) -> None:
        self._stop.set()
        self._close_socket()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self._connected.clear()
        self._board_ip = None
        with self._latest_lock:
            self._latest_frame = None

    def _close_socket(self) -> None:
        with self._send_lock:
            sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def _try_connect(self, host: str) -> Optional[socket.socket]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout_s)
        try:
            sock.connect((host, self.board_port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.25)
            return sock
        except OSError:
            sock.close()
            return None

    def _discover(self) -> tuple[Optional[socket.socket], Optional[str]]:
        hosts = [self.board_host] if self.board_host else _candidate_hosts()
        hosts = [host for host in hosts if host]
        if not hosts:
            return None, None
        if len(hosts) == 1:
            return self._try_connect(hosts[0]), hosts[0]

        executor = ThreadPoolExecutor(max_workers=min(48, len(hosts)))
        futures = {executor.submit(self._try_connect, host): host for host in hosts}
        found_socket = None
        found_host = None
        try:
            for future in as_completed(futures):
                sock = future.result()
                if sock is not None:
                    found_socket = sock
                    found_host = futures[future]
                    break
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return found_socket, found_host

    def _run(self) -> None:
        while not self._stop.is_set():
            sock, host = self._discover()
            if sock is None or host is None:
                self._stop.wait(self.discovery_interval_s)
                continue

            with self._send_lock:
                if self._stop.is_set():
                    sock.close()
                    return
                self._socket = sock
            self._board_ip = host
            self._buffer.clear()
            self._connected.set()
            self._notify_connection(True, f"局域网已连接板卡 {host}")
            next_hello = 0.0

            while not self._stop.is_set() and self._socket is sock:
                now = time.monotonic()
                if now >= next_hello:
                    try:
                        self._send(LAN_HELLO)
                    except OSError:
                        break
                    next_hello = now + self.heartbeat_s
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                self._buffer.extend(chunk)
                self._drain_frames()

            self._close_socket()
            if self._connected.is_set():
                self._connected.clear()
                self._notify_connection(False, "局域网连接断开，正在重新发现板卡")
            self._board_ip = None
            self._last_boot_id = None
            self._stop.wait(self.discovery_interval_s)

    def _packet_length(self) -> Optional[int]:
        data = self._buffer
        if data.startswith(b"FBG1"):
            if len(data) < 12:
                return None
            header_length = int.from_bytes(data[8:10], "big")
            payload_length = int.from_bytes(data[10:12], "big")
            total = header_length + payload_length + 4
            return total if 44 <= total <= _MAX_TELEMETRY_PACKET_SIZE else -1
        if data.startswith(b"FBGM"):
            if len(data) < 12:
                return None
            point_count = int.from_bytes(data[6:8], "big")
            total = 12 + point_count * 4 + 4
            return total if 20 <= total <= _MAX_METADATA_PACKET_SIZE else -1
        if data.startswith(b"FACK1|"):
            match = _ACK_PATTERN.match(bytes(data))
            if match is not None:
                return len(match.group(0))
            # A partial ACK must remain buffered, but a malformed ACK must not
            # permanently block a valid binary packet that follows it in the
            # same TCP stream.  Valid ACKs have a fixed maximum width.
            if any(data.find(magic, 1) >= 0 for magic in _MAGICS):
                return -1
            return -1 if len(data) >= _MAX_ACK_PACKET_SIZE else None
        return -1

    def _drain_frames(self) -> None:
        while self._buffer:
            length = self._packet_length()
            if length is None or (length > 0 and len(self._buffer) < length):
                return
            if length < 0:
                next_positions = [self._buffer.find(magic, 1) for magic in _MAGICS]
                next_positions = [position for position in next_positions if position >= 0]
                if not next_positions:
                    self._buffer.clear()
                    return
                del self._buffer[: min(next_positions)]
                continue
            payload = bytes(self._buffer[:length])
            del self._buffer[:length]
            self._handle_packet(payload)

    def _handle_packet(self, payload: bytes) -> None:
        try:
            if payload.startswith(b"FBG1"):
                decoded = decode_telemetry(payload)
                with self._latest_lock:
                    self._latest_frame = decoded
                if decoded.boot_id != self._last_boot_id and self.on_status is not None:
                    self._last_boot_id = decoded.boot_id
                    self.on_status(
                        f"ONLINE|{'STRESS' if decoded.mode == 0 else 'TEMPERATURE'}|"
                        f"{decoded.boot_id:08X}"
                    )
            elif payload.startswith(b"FBGM"):
                decoded = decode_metadata(payload)
                if self.on_metadata is not None:
                    self.on_metadata(decoded)
            elif payload.startswith(b"FACK1|"):
                decoded = decode_ack(payload)
                if self.on_ack is not None:
                    self.on_ack(decoded)
        except Exception as exc:
            self._notify_connection(
                self.connected, f"收到无效的局域网数据：{exc}"
            )

    def _send(self, payload: bytes) -> None:
        with self._send_lock:
            sock = self._socket
            if sock is None:
                raise RuntimeError("局域网连接尚未建立")
            sock.sendall(bytes(payload))

    def take_latest_telemetry(self) -> Optional[object]:
        with self._latest_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def publish_mode(self, mode: int, command_id: int) -> None:
        self._send(encode_mode_command(command_id, mode))


__all__ = ["FbgLanTransport", "LAN_BOARD_PORT", "LAN_HELLO"]
