import base64
import hashlib
import hmac
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fbg_ota_protocol import (
    OTA_CHUNK_SIZE,
    OTA_LINK_ADDRESS,
    OtaClient,
    build_package,
    crc32,
    encode_version,
)


PASSWORD = "offline-stability-secret"
SOURCE_BOOT_ID = 0x11223344
NEW_BOOT_ID = 0x55667788


def _valid_image(length=4096):
    image = bytearray((index * 73 + 19) & 0xFF for index in range(length))
    struct.pack_into("<II", image, 0, 0x20001800, OTA_LINK_ADDRESS + 0x101)
    return bytes(image)


class _CommandStream:
    """Parse the delimiter-free command stream exactly as the OTA client emits it."""

    def __init__(self, connection):
        self.connection = connection
        self.buffer = bytearray()

    def _fill(self, size):
        while len(self.buffer) < size:
            chunk = self.connection.recv(4096)
            if not chunk:
                raise ConnectionError("client disconnected before a complete command")
            self.buffer.extend(chunk)

    def literal(self, value):
        self._fill(len(value))
        if bytes(self.buffer[: len(value)]) != value:
            raise RuntimeError(f"expected {value!r}, got {bytes(self.buffer)!r}")
        del self.buffer[: len(value)]

    def fields(self, field_count, final_field_length):
        while True:
            parts = bytes(self.buffer).split(b"|", field_count - 1)
            if len(parts) == field_count and len(parts[-1]) >= final_field_length:
                command_length = sum(len(part) for part in parts[:-1])
                command_length += field_count - 1 + final_field_length
                raw = bytes(self.buffer[:command_length])
                del self.buffer[:command_length]
                return raw.decode("ascii").split("|")
            chunk = self.connection.recv(4096)
            if not chunk:
                raise ConnectionError("client disconnected before a complete command")
            self.buffer.extend(chunk)


class _ResumableOtaBoard(threading.Thread):
    """Loopback board model with persistent flash and deterministic wire faults."""

    def __init__(
        self,
        package,
        *,
        drop_after_store=(),
        wrong_ack_after_store=(),
        malformed_ready_once=False,
        drop_after_commit_once=False,
    ):
        super().__init__(daemon=True)
        self.package = package
        self.drop_after_store = set(drop_after_store)
        self.wrong_ack_after_store = set(wrong_ack_after_store)
        self.malformed_ready_once = malformed_ready_once
        self.drop_after_commit_once = drop_after_commit_once
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.listener.settimeout(5.0)
        self.port = self.listener.getsockname()[1]
        self.received = bytearray()
        self.session = 0xA5A50123
        self.nonce = None
        self.connection_count = 0
        self.resume_offsets = []
        self.injected_faults = []
        self.raw_ready_received = False
        self.commit_count = 0
        self.error = None

    @staticmethod
    def _fragmented_send(connection, payload):
        # Deliberately split each reply at awkward boundaries.  TCP is free to
        # coalesce them again, but the client must also handle true fragments.
        widths = (1, 2, 5, 3, 8, 13)
        offset = 0
        index = 0
        while offset < len(payload):
            end = min(len(payload), offset + widths[index % len(widths)])
            connection.sendall(payload[offset:end])
            offset = end
            index += 1
            time.sleep(0.0002)

    @staticmethod
    def _reset_connection(connection):
        linger = struct.pack("hh", 1, 0)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)

    def _read_begin(self, stream, *, retry_connection=False):
        if retry_connection:
            # A real BW20 connection can be accepted as a candidate while the
            # previous OTA owner is still being reaped.  The retrying client
            # proves promotion by obtaining RAW READY on the same socket before
            # sending BEGIN.
            stream.literal(b"FBGL1|RAW")
            header = struct.pack(
                ">4sBBIH", b"FBR1", 1, 2, SOURCE_BOOT_ID, 0
            )
            self._fragmented_send(
                stream.connection, header + crc32(header).to_bytes(4, "big")
            )
            stream.literal(b"FBGL1|HELLO")
        else:
            stream.literal(b"FBGL1|HELLO")
        fields = stream.fields(8, 64)
        if fields[:3] != ["FOTA1", "BEGIN", "F205RE"]:
            raise RuntimeError(f"invalid BEGIN header: {fields[:3]!r}")
        prefix = "|".join(fields[:-1])
        expected = hmac.new(
            PASSWORD.encode("utf-8"), prefix.encode("ascii"), hashlib.sha256
        ).hexdigest().upper()
        if not hmac.compare_digest(fields[-1], expected):
            raise RuntimeError("invalid BEGIN HMAC")
        if int(fields[3], 16) != self.package.firmware_version:
            raise RuntimeError("wrong version")
        if int(fields[4], 16) != self.package.image_size:
            raise RuntimeError("wrong image size")
        if int(fields[5], 16) != self.package.image_crc32:
            raise RuntimeError("wrong image CRC")
        if self.nonce is None:
            self.nonce = fields[6]
        elif fields[6] != self.nonce:
            raise RuntimeError("retry did not preserve the authenticated nonce")

    def _serve_ota_connection(self, connection, *, retry_connection=False):
        stream = _CommandStream(connection)
        self._read_begin(stream, retry_connection=retry_connection)
        self.resume_offsets.append(len(self.received))
        if self.malformed_ready_once:
            self.malformed_ready_once = False
            self.injected_faults.append("malformed-ready")
            self._fragmented_send(
                connection, b"FOTA1|READY|NOT_HEX!|00000000|11223344\n"
            )
            return False
        self._fragmented_send(
            connection,
            (
                f"FOTA1|READY|{self.session:08X}|{len(self.received):08X}|"
                f"{SOURCE_BOOT_ID:08X}\n"
            ).encode("ascii"),
        )

        while len(self.received) < self.package.image_size:
            offset = len(self.received)
            block_length = min(
                OTA_CHUNK_SIZE, self.package.image_size - offset
            )
            encoded_length = 4 * ((block_length + 2) // 3)
            fields = stream.fields(6, encoded_length)
            if fields[:2] != ["FOTA1", "DATA"]:
                raise RuntimeError("expected DATA")
            if int(fields[2], 16) != self.session:
                raise RuntimeError("wrong DATA session")
            if int(fields[3], 16) != offset:
                raise RuntimeError("wrong DATA offset")
            block = base64.b64decode(fields[5], validate=True)
            if len(block) != block_length or int(fields[4], 16) != crc32(block):
                raise RuntimeError("invalid DATA block")
            expected = self.package.image[offset : offset + block_length]
            if block != expected:
                raise RuntimeError("DATA differs from package")
            self.received.extend(block)

            if offset in self.drop_after_store:
                self.drop_after_store.remove(offset)
                self.injected_faults.append(f"drop-after-{offset}")
                self._reset_connection(connection)
                return False
            if offset in self.wrong_ack_after_store:
                self.wrong_ack_after_store.remove(offset)
                self.injected_faults.append(f"wrong-ack-after-{offset}")
                self._fragmented_send(
                    connection,
                    (
                        f"FOTA1|ACK|{self.session:08X}|"
                        f"{len(self.received) - 1:08X}\n"
                    ).encode("ascii"),
                )
                return False
            self._fragmented_send(
                connection,
                f"FOTA1|ACK|{self.session:08X}|{len(self.received):08X}\n".encode(
                    "ascii"
                ),
            )

        fields = stream.fields(5, 8)
        if fields[:2] != ["FOTA1", "END"]:
            raise RuntimeError("expected END")
        if int(fields[2], 16) != self.session:
            raise RuntimeError("wrong END session")
        if int(fields[3], 16) != self.package.image_size:
            raise RuntimeError("wrong END size")
        if int(fields[4], 16) != self.package.image_crc32:
            raise RuntimeError("wrong END CRC")
        if crc32(self.received) != self.package.image_crc32:
            raise RuntimeError("persisted image CRC mismatch")
        self.commit_count += 1
        if self.drop_after_commit_once:
            self.drop_after_commit_once = False
            self.injected_faults.append("drop-after-commit")
            self._reset_connection(connection)
            return False
        self._fragmented_send(
            connection,
            f"FOTA1|REBOOT|{self.package.firmware_version:08X}\n".encode("ascii"),
        )
        return True

    def _serve_raw_ready(self):
        connection, _ = self.listener.accept()
        self.connection_count += 1
        with connection:
            connection.settimeout(1.0)
            request = bytearray()
            while b"FBGL1|RAW" not in request:
                chunk = connection.recv(2048)
                if not chunk:
                    raise RuntimeError("RAW client disconnected")
                request.extend(chunk)
            header = struct.pack(">4sBBIH", b"FBR1", 1, 2, NEW_BOOT_ID, 0)
            self._fragmented_send(
                connection, header + crc32(header).to_bytes(4, "big")
            )
            self.raw_ready_received = True

    def run(self):
        try:
            committed = False
            while not committed:
                connection, _ = self.listener.accept()
                self.connection_count += 1
                with connection:
                    connection.settimeout(2.0)
                    committed = self._serve_ota_connection(
                        connection,
                        retry_connection=self.connection_count > 1,
                    )
            self._serve_raw_ready()
        except Exception as exc:
            self.error = exc
        finally:
            self.listener.close()


class OtaTcpStabilityTests(unittest.TestCase):
    def _package(self, root):
        image_path = root / "offline_stability.bin"
        package_path = root / "offline_stability.fbgfw"
        image_path.write_bytes(_valid_image())
        return build_package(
            image_path,
            package_path,
            encode_version(1, 0, 58),
            build_timestamp=1,
        )

    def test_repeated_fragmented_tcp_uploads_resume_after_persisted_blocks(self):
        patterns = (
            {0},
            {OTA_CHUNK_SIZE},
            {5 * OTA_CHUNK_SIZE},
            {0, 7 * OTA_CHUNK_SIZE},
            {3 * OTA_CHUNK_SIZE, 11 * OTA_CHUNK_SIZE},
            {22 * OTA_CHUNK_SIZE},
        )
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary))
            for iteration in range(18):
                drops = patterns[iteration % len(patterns)]
                with self.subTest(iteration=iteration, drops=sorted(drops)):
                    board = _ResumableOtaBoard(
                        package,
                        drop_after_store=drops,
                    )
                    board.start()
                    progress = []
                    OtaClient(
                        "127.0.0.1",
                        PASSWORD,
                        port=board.port,
                        connect_timeout_s=0.5,
                        response_timeout_s=0.5,
                        retries=8,
                        reconnect_delay_s=0.0,
                        recovery_quiet_s=0.0,
                        reboot_timeout_s=1.0,
                        reboot_poll_s=0.05,
                        raw_connection_hold_s=0.2,
                        raw_hello_interval_s=0.05,
                        raw_release_cooldown_s=0.0,
                    ).upload(package, progress=progress.append)
                    board.join(2.0)
                    self.assertFalse(board.is_alive())
                    self.assertIsNone(board.error)
                    self.assertEqual(bytes(board.received), package.image)
                    self.assertEqual(crc32(board.received), package.image_crc32)
                    self.assertTrue(board.raw_ready_received)
                    self.assertEqual(board.resume_offsets[0], 0)
                    self.assertEqual(
                        len(board.resume_offsets), len(drops) + 1
                    )
                    self.assertIn("RAW READY", progress[-1].message)

    def test_protocol_faults_recover_without_rewriting_persisted_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary))
            board = _ResumableOtaBoard(
                package,
                malformed_ready_once=True,
                wrong_ack_after_store={4 * OTA_CHUNK_SIZE},
                drop_after_commit_once=True,
            )
            board.start()
            OtaClient(
                "127.0.0.1",
                PASSWORD,
                port=board.port,
                connect_timeout_s=0.5,
                response_timeout_s=0.5,
                retries=6,
                reconnect_delay_s=0.0,
                recovery_quiet_s=0.0,
                reboot_timeout_s=1.0,
                reboot_poll_s=0.05,
                raw_connection_hold_s=0.2,
                raw_hello_interval_s=0.05,
                raw_release_cooldown_s=0.0,
            ).upload(package)
            board.join(2.0)
            self.assertFalse(board.is_alive())
            self.assertIsNone(board.error)
            self.assertEqual(bytes(board.received), package.image)
            self.assertEqual(
                board.injected_faults,
                [
                    "malformed-ready",
                    f"wrong-ack-after-{4 * OTA_CHUNK_SIZE}",
                    "drop-after-commit",
                ],
            )
            self.assertEqual(
                board.resume_offsets,
                [0, 0, 5 * OTA_CHUNK_SIZE, package.image_size],
            )
            self.assertEqual(board.commit_count, 2)

    def test_wifi_record_has_a_compile_time_ota_preserve_guard(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU"
            / "Core"
            / "Src"
            / "wifi_transport.c"
        ).read_text(encoding="utf-8")
        self.assertIn("WifiStoredConfigMustFitOtaPreserve", source)
        self.assertIn(
            "sizeof(WifiStoredConfig) <= OTA_CONFIG_PRESERVE_SIZE", source
        )

    def test_ota_owner_lease_still_expires_when_disconnect_urc_is_lost(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU"
            / "Core"
            / "Src"
            / "wifi_transport.c"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if(lanSocketId >= 0 && !lanSocketResetPending &&\n"
            "       lanClientDeadlineMs != 0U &&",
            source,
        )
        self.assertNotIn(
            "!OtaUpdate_IsActive() && lanClientDeadlineMs != 0U",
            source,
        )
        self.assertIn(
            "if(OtaUpdate_IsActive() || lanCleanupOverflow) canPromote = 0U;",
            source,
        )
        self.assertIn("if(otaWasActive)\n    {\n        lanCleanupOverflow = 1U;", source)
        self.assertNotIn(
            "lanStartupServerStatus == 3 && lanStartupCleanupAttempts == 0U",
            source,
        )


if __name__ == "__main__":
    unittest.main()
