"""PySerial-compatible byte stream carried by the board's same-LAN TCP port.

The desktop acquisition pages intentionally know nothing about Wi-Fi.  This
adapter transports their existing 808-byte commands and returns the original
USB response bytes, allowing LAN and USB to share one UI and one set of signal
processing code.
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from fbg_lan_transport import LAN_BOARD_PORT, _candidate_hosts
from frame_telemetry import RAW_SCAN_MAX_POINTS, extract_raw_scan_frames


RAW_HELLO = b"FBGL1|RAW"
RAW_COMPACT_ENABLE = b"FBGL1|CAPS|CS1"
RAW_COMMAND_PREFIX = b"FUSB1|"
RAW_RESPONSE_MAGIC = b"FBR1"
RAW_RESPONSE_HEADER = struct.Struct(">4sBBIH")
RAW_RESPONSE_VERSION = 1
RAW_KIND_DATA = 0
RAW_KIND_COMMAND_ACK = 1
RAW_KIND_READY = 2
RAW_KIND_COMPACT_SCAN = 3
RAW_KIND_COMPACT_F45 = 4
RAW_KIND_COMPACT_F45_BATCH = 5
RAW_COMPACT_SCAN_HEADER = struct.Struct(">BBHH")
RAW_COMPACT_SCAN_VERSION = 1
RAW_COMPACT_F45_SIZE = 128
RAW_NATIVE_F45_SIZE = 453
RAW_HEARTBEAT_S = 2.0
RAW_COMMAND_SIZE = 808
RAW_CHUNK_SIZE = 200
RAW_MAX_PAYLOAD = 2048


def expand_compact_scan(payload: bytes) -> bytes | None:
    """Restore one negotiated CS1 payload to the exact native USB frame."""

    if len(payload) < RAW_COMPACT_SCAN_HEADER.size:
        return None
    version, channel_mask, point_count, trailer_length = (
        RAW_COMPACT_SCAN_HEADER.unpack_from(payload)
    )
    if (
        version != RAW_COMPACT_SCAN_VERSION
        or point_count <= 0
        or point_count > RAW_SCAN_MAX_POINTS
        or channel_mask == 0
        or channel_mask & 0xF0
    ):
        return None
    channels = [
        channel for channel in range(4) if channel_mask & (1 << channel)
    ]
    sample_length = point_count * len(channels) * 2
    expected_length = (
        RAW_COMPACT_SCAN_HEADER.size + sample_length + trailer_length
    )
    if expected_length != len(payload):
        return None
    trailer = payload[RAW_COMPACT_SCAN_HEADER.size + sample_length :]
    if len(trailer) < 3 or trailer[:1] != b"\xAB" or trailer[-2:] != b"\xFF\xEF":
        return None

    native = bytearray(b"\xEE\xEE" + point_count.to_bytes(2, "big"))
    native.extend(point_count * 8 * b"\x00")
    source = RAW_COMPACT_SCAN_HEADER.size
    for point in range(point_count):
        destination = 4 + point * 8
        for channel in channels:
            native[destination + channel * 2 : destination + channel * 2 + 2] = (
                payload[source : source + 2]
            )
            source += 2
    native.extend(trailer)

    # Reject a syntactically compact packet if its preserved trailer cannot
    # form exactly one complete native scan.  This keeps malformed metadata
    # out of the shared USB/LAN parser without guessing a frame boundary.
    probe = bytearray(native)
    frames = extract_raw_scan_frames(probe)
    if len(frames) != 1 or probe or frames[0] != bytes(native):
        return None
    return bytes(native)


def expand_compact_f45(payload: bytes) -> bytes | None:
    """Restore a LAN first-code F45 packet to the shared native frame shape.

    Bit 1 in the restored flags records that only each point's physical first
    ADC code crossed the network.  The duplicated second-code and evenly
    distributed timestamps are compatibility fields, never physical samples.
    """

    if (
        len(payload) != RAW_COMPACT_F45_SIZE
        or payload[0] != RAW_COMPACT_SCAN_VERSION
        or payload[1] & ~0x01
        or payload[26] != 1
        or payload[27] != 45
    ):
        return None
    elapsed_us = int.from_bytes(payload[18:22], "big")
    if elapsed_us < 100:
        return None
    codes = payload[38:]
    if len(codes) != 90:
        return None

    native = bytearray(RAW_NATIVE_F45_SIZE)
    native[:3] = b"\xD9\x9D\x03"
    native[3] = payload[1] | 0x02
    native[4:40] = payload[2:38]
    native[40:85] = bytes(range(45))
    for point in range(45):
        code = codes[point * 2 : point * 2 + 2]
        if int.from_bytes(code, "big") > 4095:
            return None
        write_end = ((2 * point + 1) * elapsed_us) // 91
        second_end = ((2 * point + 2) * elapsed_us) // 91
        if not 0 <= write_end < second_end <= 0xFFFF:
            return None
        destination = 85 + point * 8
        native[destination : destination + 2] = code
        native[destination + 2 : destination + 4] = code
        native[destination + 4 : destination + 6] = write_end.to_bytes(2, "big")
        native[destination + 6 : destination + 8] = second_end.to_bytes(2, "big")
    native[-6:-2] = (zlib.crc32(native[:-6]) & 0xFFFFFFFF).to_bytes(4, "big")
    native[-2:] = b"\x9D\xD9"
    return bytes(native)


def expand_compact_f45_batch(payload: bytes) -> tuple[bytes, ...] | None:
    if len(payload) < 2 or payload[0] != 1 or not 1 <= payload[1] <= 4:
        return None
    count = payload[1]
    if len(payload) != 2 + count * RAW_COMPACT_F45_SIZE:
        return None
    frames = []
    for index in range(count):
        start = 2 + index * RAW_COMPACT_F45_SIZE
        frame = expand_compact_f45(
            payload[start : start + RAW_COMPACT_F45_SIZE]
        )
        if frame is None:
            return None
        frames.append(frame)
    return tuple(frames)


class FbgLanSerial:
    """Small subset of ``serial.Serial`` used by ``app_JDSU``."""

    def __init__(
        self,
        *,
        board_host: str | None = None,
        board_port: int = LAN_BOARD_PORT,
        timeout: float = 0.2,
        write_timeout: float = 2.0,
        connect_timeout: float = 0.12,
        discovery_interval: float = 1.0,
        on_connection: Optional[Callable[[bool, str], None]] = None,
    ):
        self.port = "LAN"
        self.baudrate = 2_000_000
        self.timeout = float(timeout)
        self.write_timeout = float(write_timeout)
        self.dtr = False
        self.board_host = str(
            board_host or os.environ.get("FBG_LAN_BOARD_IP", "")
        ).strip() or None
        self.board_port = int(board_port)
        self.connect_timeout = float(connect_timeout)
        self.discovery_interval = float(discovery_interval)
        self.on_connection = on_connection

        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._raw_ready = threading.Event()
        self._send_lock = threading.Lock()
        self._receive_condition = threading.Condition()
        self._receive_buffer = bytearray()
        self._network_buffer = bytearray()
        self._frame_arrivals: deque[tuple[int, int, float, int]] = deque(
            maxlen=1024
        )
        self._receive_batch_sequence = 0
        self._ack_offsets: dict[int, int] = {}
        self._request_id = int(time.time_ns()) & 0xFFFFFFFF
        self._board_ip: str | None = None
        self._compact_capability_sent = False
        self._wire_data_bytes = 0
        self._native_data_bytes = 0
        self._raw_data_packets = 0
        self._compact_scan_packets = 0
        self._invalid_compact_packets = 0
        self._compact_f45_packets = 0
        self._invalid_compact_f45_packets = 0

    @property
    def is_open(self) -> bool:
        return self._thread is not None and not self._stop.is_set()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def board_ip(self) -> str | None:
        return self._board_ip

    @property
    def transport_stats(self) -> dict[str, int | float | None]:
        """Return receive-side counters without affecting stream delivery."""

        with self._receive_condition:
            ratio = (
                self._wire_data_bytes / self._native_data_bytes
                if self._native_data_bytes
                else None
            )
            mean_wire_packet_bytes = (
                self._wire_data_bytes / self._raw_data_packets
                if self._raw_data_packets
                else None
            )
            return {
                "wire_packet_bytes": self._wire_data_bytes,
                "native_stream_bytes": self._native_data_bytes,
                "wire_to_native_ratio": ratio,
                "data_packets": self._raw_data_packets,
                "mean_wire_packet_bytes": mean_wire_packet_bytes,
                "uart_serialization_ms_at_115200": (
                    mean_wire_packet_bytes * 10.0 / 115_200.0 * 1000.0
                    if mean_wire_packet_bytes is not None
                    else None
                ),
                "compact_scan_packets": self._compact_scan_packets,
                "invalid_compact_packets": self._invalid_compact_packets,
                "compact_f45_packets": self._compact_f45_packets,
                "invalid_compact_f45_packets": self._invalid_compact_f45_packets,
            }

    @property
    def in_waiting(self) -> int:
        with self._receive_condition:
            return len(self._receive_buffer)

    def take_frame_arrival(self, frame: bytes) -> float | None:
        """Return the socket-thread arrival time for this exact native frame."""

        arrival = self.take_frame_arrival_info(frame)
        return arrival[0] if arrival is not None else None

    def take_frame_arrival_info(self, frame: bytes) -> tuple[float, int] | None:
        """Return ``(arrival_time, socket_receive_batch)`` for a frame."""

        key = (len(frame), zlib.crc32(frame) & 0xFFFFFFFF)
        with self._receive_condition:
            for index, (length, crc, arrived_at, batch_id) in enumerate(
                self._frame_arrivals
            ):
                if (length, crc) != key:
                    continue
                for _ in range(index + 1):
                    self._frame_arrivals.popleft()
                return arrived_at, batch_id
        return None

    def reset_transport_stats(self) -> None:
        """Reset diagnostic counters without changing the byte stream."""

        with self._receive_condition:
            self._wire_data_bytes = 0
            self._native_data_bytes = 0
            self._raw_data_packets = 0
            self._compact_scan_packets = 0
            self._invalid_compact_packets = 0
            self._compact_f45_packets = 0
            self._invalid_compact_f45_packets = 0

    def _notify(self, connected: bool, detail: str) -> None:
        if self.on_connection is not None:
            try:
                self.on_connection(bool(connected), str(detail))
            except Exception:
                pass

    def open(self) -> None:
        if self.is_open:
            return
        self._stop.clear()
        self._connected.clear()
        self._raw_ready.clear()
        with self._receive_condition:
            self._frame_arrivals.clear()
        self.reset_transport_stats()
        self._thread = threading.Thread(
            target=self._run, name="fbg-lan-serial", daemon=True
        )
        self._thread.start()
        self._notify(False, "局域网正在寻找板卡")

    def close(self) -> None:
        self._stop.set()
        self._close_socket()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self._connected.clear()
        self._board_ip = None
        with self._receive_condition:
            self._receive_condition.notify_all()

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

    def _try_connect(self, host: str) -> socket.socket | None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((host, self.board_port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.20)
            return sock
        except OSError:
            sock.close()
            return None

    def _discover(self) -> tuple[socket.socket | None, str | None]:
        hosts = [self.board_host] if self.board_host else _candidate_hosts()
        hosts = [host for host in hosts if host]
        if not hosts:
            return None, None
        if len(hosts) == 1:
            return self._try_connect(hosts[0]), hosts[0]
        executor = ThreadPoolExecutor(max_workers=min(48, len(hosts)))
        futures = {executor.submit(self._try_connect, host): host for host in hosts}
        found = (None, None)
        try:
            for future in as_completed(futures):
                sock = future.result()
                if sock is not None:
                    found = (sock, futures[future])
                    break
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return found

    def _run(self) -> None:
        while not self._stop.is_set():
            sock, host = self._discover()
            if sock is None or host is None:
                self._stop.wait(self.discovery_interval)
                continue
            with self._send_lock:
                if self._stop.is_set():
                    sock.close()
                    return
                self._socket = sock
            self._network_buffer.clear()
            self._board_ip = host
            self._raw_ready.clear()
            self._compact_capability_sent = False
            next_hello = 0.0
            while not self._stop.is_set() and self._socket is sock:
                now = time.monotonic()
                if now >= next_hello:
                    try:
                        self._send(RAW_HELLO)
                    except OSError:
                        break
                    next_hello = now + RAW_HEARTBEAT_S
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                received_at = time.perf_counter()
                self._network_buffer.extend(chunk)
                self._drain_packets(received_at=received_at)
                if self._raw_ready.is_set() and not self._connected.is_set():
                    self._connected.set()
                    self._notify(True, f"局域网全功能通道已连接板卡 {host}")
            self._close_socket()
            if self._connected.is_set():
                self._connected.clear()
                self._notify(False, "局域网连接断开，正在重新发现板卡")
            self._board_ip = None
            self._raw_ready.clear()
            self._stop.wait(self.discovery_interval)

    def _send(self, payload: bytes) -> None:
        with self._send_lock:
            if self._socket is None:
                raise OSError("局域网连接尚未建立")
            self._socket.sendall(payload)

    def _drain_packets(self, *, received_at: float | None = None) -> None:
        buffer = self._network_buffer
        if received_at is None:
            received_at = time.perf_counter()
        self._receive_batch_sequence += 1
        receive_batch = self._receive_batch_sequence
        while buffer:
            request_compact = False
            start = buffer.find(RAW_RESPONSE_MAGIC)
            if start < 0:
                # Retain a possible partial magic suffix.
                if len(buffer) > 3:
                    del buffer[:-3]
                return
            if start:
                del buffer[:start]
            if len(buffer) < RAW_RESPONSE_HEADER.size:
                return
            magic, version, kind, sequence, payload_length = (
                RAW_RESPONSE_HEADER.unpack_from(buffer)
            )
            if (
                magic != RAW_RESPONSE_MAGIC
                or version != RAW_RESPONSE_VERSION
                or payload_length > RAW_MAX_PAYLOAD
            ):
                del buffer[0]
                continue
            total = RAW_RESPONSE_HEADER.size + payload_length + 4
            if len(buffer) < total:
                return
            packet = bytes(buffer[:total])
            del buffer[:total]
            expected = int.from_bytes(packet[-4:], "big")
            if (zlib.crc32(packet[:-4]) & 0xFFFFFFFF) != expected:
                continue
            payload = packet[RAW_RESPONSE_HEADER.size:-4]
            compact_scan = False
            if kind == RAW_KIND_COMPACT_SCAN:
                expanded = expand_compact_scan(payload)
                if expanded is None:
                    with self._receive_condition:
                        self._invalid_compact_packets += 1
                    continue
                payload = expanded
                kind = RAW_KIND_DATA
                compact_scan = True
            compact_f45_frames: tuple[bytes, ...] = ()
            if kind == RAW_KIND_COMPACT_F45:
                expanded = expand_compact_f45(payload)
                if expanded is None:
                    with self._receive_condition:
                        self._invalid_compact_f45_packets += 1
                    continue
                payload = expanded
                kind = RAW_KIND_DATA
                compact_f45_frames = (expanded,)
            elif kind == RAW_KIND_COMPACT_F45_BATCH:
                expanded_batch = expand_compact_f45_batch(payload)
                if expanded_batch is None:
                    with self._receive_condition:
                        self._invalid_compact_f45_packets += 1
                    continue
                compact_f45_frames = expanded_batch
                payload = b"".join(expanded_batch)
                kind = RAW_KIND_DATA
            with self._receive_condition:
                if kind == RAW_KIND_DATA:
                    self._receive_buffer.extend(payload)
                    arrival_frames = compact_f45_frames or (payload,)
                    for arrival_frame in arrival_frames:
                        self._frame_arrivals.append(
                            (
                                len(arrival_frame),
                                zlib.crc32(arrival_frame) & 0xFFFFFFFF,
                                received_at,
                                receive_batch,
                            )
                        )
                    self._wire_data_bytes += total
                    self._native_data_bytes += len(payload)
                    self._raw_data_packets += 1
                    if compact_scan:
                        self._compact_scan_packets += 1
                    if compact_f45_frames:
                        self._compact_f45_packets += len(compact_f45_frames)
                elif kind == RAW_KIND_COMMAND_ACK and len(payload) == 2:
                    self._ack_offsets[sequence] = int.from_bytes(payload, "big")
                elif kind == RAW_KIND_READY and not payload:
                    self._raw_ready.set()
                    if not self._compact_capability_sent:
                        self._compact_capability_sent = True
                        request_compact = True
                self._receive_condition.notify_all()
            if request_compact:
                try:
                    self._send(RAW_COMPACT_ENABLE)
                except OSError:
                    # A reconnect resets this flag and retries negotiation.
                    self._compact_capability_sent = False

    def write(self, data: bytes) -> int:
        raw = bytes(data)
        if len(raw) != RAW_COMMAND_SIZE:
            raise ValueError(f"LAN控制命令必须为{RAW_COMMAND_SIZE}字节")
        if not self.is_open:
            raise OSError("局域网全功能通道尚未打开")
        if not self._connected.wait(timeout=max(0.1, self.write_timeout)):
            raise OSError("局域网尚未连接到板卡")

        meaningful = raw.rstrip(b"\x00")
        if not meaningful:
            meaningful = raw[:1]
        request_id = self._request_id = (self._request_id + 1) & 0xFFFFFFFF
        total = len(meaningful)
        offset = 0
        while offset < total:
            chunk = meaningful[offset : offset + RAW_CHUNK_SIZE]
            command = (
                RAW_COMMAND_PREFIX
                + f"{request_id:08X}|{offset:04X}|{total:04X}|".encode("ascii")
                + chunk.hex().upper().encode("ascii")
            )
            end_offset = offset + len(chunk)
            delivered = False
            for _attempt in range(3):
                self._send(command)
                deadline = time.monotonic() + max(0.15, self.write_timeout / 3.0)
                with self._receive_condition:
                    while time.monotonic() < deadline:
                        if self._ack_offsets.get(request_id, 0) >= end_offset:
                            delivered = True
                            break
                        self._receive_condition.wait(
                            timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                        )
                if delivered:
                    break
            if not delivered:
                raise OSError(
                    f"局域网命令第{offset}～{end_offset}字节未被板卡确认"
                )
            offset = end_offset
        with self._receive_condition:
            self._ack_offsets.pop(request_id, None)
        return len(raw)

    def read(self, size: int = 1) -> bytes:
        size = max(1, int(size))
        deadline = time.monotonic() + max(0.0, self.timeout)
        with self._receive_condition:
            while not self._receive_buffer and self.is_open:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return b""
                self._receive_condition.wait(timeout=remaining)
            data = bytes(self._receive_buffer[:size])
            del self._receive_buffer[:size]
            return data

    def reset_input_buffer(self) -> None:
        with self._receive_condition:
            self._receive_buffer.clear()
            self._frame_arrivals.clear()
            self._wire_data_bytes = 0
            self._native_data_bytes = 0
            self._raw_data_packets = 0
            self._compact_scan_packets = 0
            self._invalid_compact_packets = 0

    def reset_output_buffer(self) -> None:
        return

    def flush(self) -> None:
        return


__all__ = [
    "FbgLanSerial",
    "RAW_CHUNK_SIZE",
    "RAW_COMPACT_ENABLE",
    "RAW_COMMAND_SIZE",
    "RAW_HELLO",
    "RAW_KIND_COMPACT_F45_BATCH",
    "RAW_KIND_COMPACT_F45",
    "RAW_KIND_COMPACT_SCAN",
    "RAW_RESPONSE_HEADER",
    "expand_compact_f45",
    "expand_compact_f45_batch",
    "expand_compact_scan",
]
