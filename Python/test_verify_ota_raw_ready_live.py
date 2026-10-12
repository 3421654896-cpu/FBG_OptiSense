from __future__ import annotations

import socket
import threading
import time
import unittest
import zlib
from unittest import mock

from fbg_lan_serial import (
    RAW_HELLO,
    RAW_KIND_READY,
    RAW_RESPONSE_HEADER,
    RAW_RESPONSE_MAGIC,
    RAW_RESPONSE_VERSION,
)
from verify_ota_raw_ready_live import RawReadyError, wait_for_raw_ready


def _ready_packet(*, valid_crc: bool = True) -> bytes:
    packet = RAW_RESPONSE_HEADER.pack(
        RAW_RESPONSE_MAGIC, RAW_RESPONSE_VERSION, RAW_KIND_READY, 0, 0
    )
    checksum = zlib.crc32(packet) & 0xFFFFFFFF
    if not valid_crc:
        checksum ^= 1
    return packet + checksum.to_bytes(4, "big")


class _ReadyServer(threading.Thread):
    def __init__(self, packets: list[bytes]):
        super().__init__(daemon=True)
        self.packets = packets
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(len(packets))
        self.port = self.listener.getsockname()[1]
        self.error: Exception | None = None
        self.accepted_connections = 0
        self.hello_count = 0

    def run(self) -> None:
        try:
            for packet in self.packets:
                connection, _ = self.listener.accept()
                self.accepted_connections += 1
                with connection:
                    request = connection.recv(64)
                    self.hello_count += request.count(RAW_HELLO)
                    if RAW_HELLO not in request:
                        raise RuntimeError(f"unexpected request {request!r}")
                    # Exercise garbage resynchronization and fragmented TCP.
                    payload = b"stale" + packet
                    connection.sendall(payload[:7])
                    connection.sendall(payload[7:])
                    # The verifier follows production SHUT_RDWR + close.
                    connection.settimeout(1.0)
                    try:
                        while connection.recv(64):
                            pass
                    except (ConnectionResetError, ConnectionAbortedError):
                        pass
        except Exception as exc:
            self.error = exc
        finally:
            self.listener.close()


class RawReadyLiveTests(unittest.TestCase):
    def test_two_fresh_connections_each_require_crc_valid_ready(self) -> None:
        server = _ReadyServer([_ready_packet(), _ready_packet()])
        server.start()
        first = wait_for_raw_ready(
            "127.0.0.1", port=server.port, timeout_s=1.0, retry_interval_s=0.01
        )
        second = wait_for_raw_ready(
            "127.0.0.1", port=server.port, timeout_s=1.0, retry_interval_s=0.01
        )
        server.join(1.0)
        self.assertIsNone(server.error)
        self.assertEqual(first.attempts, 1)
        self.assertEqual(second.attempts, 1)

    def test_bad_crc_is_not_accepted_as_ready(self) -> None:
        server = _ReadyServer([_ready_packet(valid_crc=False)])
        server.start()
        with self.assertRaises(RawReadyError):
            wait_for_raw_ready(
                "127.0.0.1",
                port=server.port,
                timeout_s=0.20,
                response_timeout_s=0.05,
                retry_interval_s=0.01,
            )
        server.join(1.0)
        self.assertIsNone(server.error)

    def test_ignored_first_hello_retries_on_same_tcp_connection(self) -> None:
        ready = _ready_packet()
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        port = listener.getsockname()[1]
        observed = {"connections": 0, "hellos": 0, "error": None}

        def serve() -> None:
            try:
                connection, _ = listener.accept()
                observed["connections"] += 1
                buffer = bytearray()
                with connection:
                    connection.settimeout(1.0)
                    while observed["hellos"] < 2:
                        buffer.extend(connection.recv(128))
                        observed["hellos"] = bytes(buffer).count(RAW_HELLO)
                    connection.sendall(ready)
                    try:
                        while connection.recv(64):
                            pass
                    except (ConnectionResetError, ConnectionAbortedError):
                        pass
            except Exception as exc:
                observed["error"] = exc
            finally:
                listener.close()

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        result = wait_for_raw_ready(
            "127.0.0.1",
            port=port,
            timeout_s=1.0,
            heartbeat_s=0.05,
            response_timeout_s=0.01,
        )
        server.join(1.0)
        self.assertIsNone(observed["error"])
        self.assertEqual(observed["connections"], 1)
        self.assertGreaterEqual(observed["hellos"], 2)
        self.assertEqual(result.attempts, 1)
        self.assertGreaterEqual(result.hello_count, 2)

    def test_silent_connected_peer_does_not_create_connection_storm(self) -> None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        port = listener.getsockname()[1]
        observed = {"connections": 0, "hellos": 0, "error": None}

        def serve() -> None:
            try:
                connection, _ = listener.accept()
                observed["connections"] += 1
                with connection:
                    connection.settimeout(1.0)
                    while True:
                        chunk = connection.recv(256)
                        if not chunk:
                            break
                        observed["hellos"] += chunk.count(RAW_HELLO)
            except Exception as exc:
                observed["error"] = exc
            finally:
                listener.close()

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        with self.assertRaises(RawReadyError):
            wait_for_raw_ready(
                "127.0.0.1",
                port=port,
                timeout_s=0.20,
                heartbeat_s=0.04,
                response_timeout_s=0.01,
            )
        server.join(1.0)
        self.assertIsNone(observed["error"])
        self.assertEqual(observed["connections"], 1)
        self.assertGreaterEqual(observed["hellos"], 2)

    def test_success_matches_production_full_shutdown_without_extra_read(self) -> None:
        class FakeSocket:
            def __init__(self):
                self.recv_count = 0
                self.shutdown_calls = []
                self.closed = False

            def setsockopt(self, *_args):
                pass

            def settimeout(self, _value):
                pass

            def sendall(self, payload):
                self.sent = payload

            def recv(self, _size):
                self.recv_count += 1
                if self.recv_count == 1:
                    return _ready_packet()
                raise AssertionError("verifier read again after READY")

            def shutdown(self, how):
                self.shutdown_calls.append(how)

            def close(self):
                self.closed = True

        fake = FakeSocket()
        with mock.patch(
            "verify_ota_raw_ready_live.socket.create_connection",
            return_value=fake,
        ):
            result = wait_for_raw_ready("127.0.0.1", timeout_s=0.2)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(fake.recv_count, 1)
        self.assertEqual(fake.shutdown_calls, [socket.SHUT_RDWR])
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
