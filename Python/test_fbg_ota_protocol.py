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
from unittest import mock

from fbg_ota_protocol import (
    OtaClient,
    OtaCancelled,
    OtaError,
    OTA_LINK_ADDRESS,
    build_package,
    crc32,
    encode_version,
    load_package,
)


def _valid_image(length=620):
    image = bytearray((index * 37 + 11) & 0xFF for index in range(length))
    struct.pack_into("<II", image, 0, 0x20001000, OTA_LINK_ADDRESS + 0x101)
    return bytes(image)


class FirmwarePackageTests(unittest.TestCase):
    def test_version_components_must_be_exact_integers(self):
        with self.assertRaises(ValueError):
            encode_version(True, 0, 0)
        with self.assertRaises(ValueError):
            encode_version(1, 2.5, 0)

    def test_build_and_load_checks_vectors_crc_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "app.bin"
            package = root / "app.fbgfw"
            image.write_bytes(_valid_image())
            built = build_package(image, package, encode_version(2, 4, 1), build_timestamp=12)
            loaded = load_package(package)
            self.assertEqual(loaded.image, image.read_bytes())
            self.assertEqual(loaded.version_text, "2.4.1")
            self.assertEqual(loaded.image_crc32, built.image_crc32)

    def test_tampered_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "app.bin"
            package = root / "app.fbgfw"
            image.write_bytes(_valid_image())
            build_package(image, package, encode_version(1, 0, 0))
            raw = bytearray(package.read_bytes())
            raw[-3] ^= 0x5A
            package.write_bytes(raw)
            with self.assertRaisesRegex(OtaError, "CRC"):
                load_package(package)

    def test_package_builder_rejects_wrapped_version_or_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "app.bin"
            package = root / "app.fbgfw"
            image.write_bytes(_valid_image())
            for version in (-1, 0x1_0000_0000, 1.5, True):
                with self.assertRaises(ValueError):
                    build_package(image, package, version)
            with self.assertRaises(ValueError):
                build_package(image, package, 1, build_timestamp=1.5)
            with self.assertRaisesRegex(ValueError, "覆盖"):
                build_package(image, image, 1)

    def test_reset_vector_must_point_inside_the_packaged_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "app.bin"
            package = root / "app.fbgfw"
            invalid = bytearray(_valid_image())
            struct.pack_into(
                "<I", invalid, 4, OTA_LINK_ADDRESS + len(invalid) + 1
            )
            image.write_bytes(invalid)
            with self.assertRaisesRegex(OtaError, "映像文件之外"):
                build_package(image, package, 1)

    def test_bootloader_never_jumps_after_a_failed_pending_install(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Bootloader" / "ota_bootloader.c"
        ).read_text(encoding="utf-8")
        main = source[source.index("int main(void)") :]

        self.assertIn("FLASH_SR_SOP", source)
        self.assertIn("if(!InstallPendingImage(metadata))", main)
        self.assertIn("while(1) __WFI();", main)
        self.assertLess(
            main.index("if(!InstallPendingImage(metadata))"),
            main.index("if(VectorValid(OTA_APPLICATION_ADDRESS))"),
        )

    def test_bootloader_clean_resets_after_a_successful_pending_install(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Bootloader" / "ota_bootloader.c"
        ).read_text(encoding="utf-8")
        main = source[source.index("int main(void)") :]
        pending = main[
            main.index("if(MetadataValid(metadata))") :
            main.index("if(VectorValid(OTA_APPLICATION_ADDRESS))")
        ]

        self.assertIn("NVIC_SystemReset();", pending)
        self.assertLess(
            pending.index("if(!InstallPendingImage(metadata))"),
            pending.index("NVIC_SystemReset();"),
        )


class _FakeOtaBoard(threading.Thread):
    def __init__(self, password):
        super().__init__(daemon=True)
        self.password = password
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(2)
        self.port = self.listener.getsockname()[1]
        self.received = bytearray()
        self.error = None
        self.raw_ready_received = False

    def _recv_command(self, connection):
        data = connection.recv(2048)
        if not data:
            raise RuntimeError("client disconnected")
        return data.decode("ascii")

    def run(self):
        try:
            connection, _ = self.listener.accept()
            with connection:
                hello_and_more = connection.recv(2048)
                self.assert_hello = hello_and_more.startswith(b"FBGL1|HELLO")
                remainder = hello_and_more[len(b"FBGL1|HELLO") :]
                command = remainder.decode("ascii") if remainder else self._recv_command(connection)
                fields = command.split("|")
                prefix = "|".join(fields[:-1])
                expected = hmac.new(
                    self.password.encode(), prefix.encode(), hashlib.sha256
                ).hexdigest().upper()
                if fields[-1] != expected:
                    raise RuntimeError("bad hmac")
                session = 0xA5A50123
                connection.sendall(
                    f"FOTA1|READY|{session:08X}|00000000|11111111\n".encode()
                )
                total = int(fields[4], 16)
                while len(self.received) < total:
                    data_fields = self._recv_command(connection).split("|")
                    block = base64.b64decode(data_fields[5])
                    if int(data_fields[4], 16) != crc32(block):
                        raise RuntimeError("bad block crc")
                    self.received.extend(block)
                    connection.sendall(
                        f"FOTA1|ACK|{session:08X}|{len(self.received):08X}\n".encode()
                    )
                end_fields = self._recv_command(connection).split("|")
                if end_fields[1] != "END":
                    raise RuntimeError("missing end")
                connection.sendall(b"FOTA1|REBOOT|00010000\n")
            raw_connection, _ = self.listener.accept()
            with raw_connection:
                raw_hello = raw_connection.recv(2048)
                if raw_hello != b"FBGL1|RAW":
                    raise RuntimeError(f"missing raw hello: {raw_hello!r}")
                header = struct.pack(">4sBBIH", b"FBR1", 1, 2, 0x22222222, 0)
                raw_connection.sendall(header + crc32(header).to_bytes(4, "big"))
                self.raw_ready_received = True
        except Exception as exc:
            self.error = exc
        finally:
            self.listener.close()


class _DelayedRawBoard(threading.Thread):
    """Ignore early RAW probes and answer only on the same persistent seed."""

    def __init__(self, required_hellos=3):
        super().__init__(daemon=True)
        self.required_hellos = required_hellos
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(2)
        self.listener.settimeout(2.0)
        self.port = self.listener.getsockname()[1]
        self.connection_count = 0
        self.hello_count = 0
        self.error = None

    def run(self):
        try:
            connection, _ = self.listener.accept()
            self.connection_count += 1
            with connection:
                connection.settimeout(1.0)
                received = bytearray()
                while self.hello_count < self.required_hellos:
                    chunk = connection.recv(2048)
                    if not chunk:
                        raise RuntimeError("persistent RAW client disconnected early")
                    received.extend(chunk)
                    self.hello_count = received.count(b"FBGL1|RAW")
                header = struct.pack(">4sBBIH", b"FBR1", 1, 2, 0, 0)
                connection.sendall(header + crc32(header).to_bytes(4, "big"))
        except Exception as exc:
            self.error = exc
        finally:
            self.listener.close()


class _RepeatedRawBoard(threading.Thread):
    """Serve two READY cycles and record when each client actually closes."""

    def __init__(self):
        super().__init__(daemon=True)
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(2)
        self.listener.settimeout(2.0)
        self.port = self.listener.getsockname()[1]
        self.accepted_at = []
        self.closed_at = []
        self.error = None

    def run(self):
        try:
            for _ in range(2):
                connection, _ = self.listener.accept()
                self.accepted_at.append(time.monotonic())
                with connection:
                    connection.settimeout(1.0)
                    request = connection.recv(2048)
                    if b"FBGL1|RAW" not in request:
                        raise RuntimeError(f"missing raw hello: {request!r}")
                    header = struct.pack(">4sBBIH", b"FBR1", 1, 2, 0, 0)
                    connection.sendall(header + crc32(header).to_bytes(4, "big"))
                    try:
                        while connection.recv(2048):
                            pass
                    except (ConnectionResetError, OSError):
                        pass
                    self.closed_at.append(time.monotonic())
        except Exception as exc:
            self.error = exc
        finally:
            self.listener.close()


class OtaClientTests(unittest.TestCase):
    def test_verified_resume_nonce_is_used_for_authenticated_begin(self):
        package = type(
            "Package",
            (),
            {
                "firmware_version": 0x0001005A,
                "image_size": 8,
                "image_crc32": crc32(b"12345678"),
                "image": b"12345678",
            },
        )()
        client = OtaClient("127.0.0.1", "secret", retries=1)
        with mock.patch.object(client, "_connect", return_value=mock.Mock()), mock.patch.object(
            client, "_send"
        ) as send, mock.patch.object(client, "_response", side_effect=OtaError("test stop")):
            with self.assertRaisesRegex(OtaError, "test stop"):
                client.upload(package, verified_resume_nonce="a1b2c3d4e5f60708")
        begin = send.call_args.args[1]
        prefix, signature = begin.rsplit("|", 1)
        self.assertTrue(prefix.endswith("|A1B2C3D4E5F60708"))
        self.assertEqual(
            signature,
            hmac.new(b"secret", prefix.encode("ascii"), hashlib.sha256)
            .hexdigest()
            .upper(),
        )

    def test_verified_resume_nonce_rejects_malformed_values_before_connect(self):
        client = OtaClient("127.0.0.1", "secret", retries=1)
        with mock.patch.object(client, "_connect") as connect:
            for nonce in ("", "1234", "G" * 16, "1" * 17):
                with self.subTest(nonce=nonce), self.assertRaises(ValueError):
                    client.upload(mock.Mock(), verified_resume_nonce=nonce)
        connect.assert_not_called()

    def test_default_retry_budget_covers_real_bw20_socket_churn(self):
        client = OtaClient("127.0.0.1", "secret")
        self.assertGreaterEqual(client.retries, 12)
        self.assertGreaterEqual(client.reconnect_delay_s, 29.75)
        self.assertGreaterEqual(client.recovery_quiet_s, 135.0)
        self.assertGreaterEqual(client.retry_owner_timeout_s, 8.0)
        self.assertLessEqual(client.raw_hello_interval_s, 0.5)
        self.assertEqual(client.reboot_timeout_s, 600.0)
        self.assertGreaterEqual(client.reboot_initial_delay_s, 12.0)
        self.assertGreater(client.raw_connection_hold_s, 3.5)
        self.assertGreaterEqual(client.raw_release_cooldown_s, 5.75)
        self.assertGreaterEqual(client.reboot_poll_s, 10.0)

    def test_raw_ready_timeout_reports_the_configured_boot_window(self):
        client = OtaClient("127.0.0.1", "secret")
        with mock.patch(
            "fbg_ota_protocol.time.monotonic", side_effect=(0.0, 601.0)
        ):
            with self.assertRaisesRegex(OtaError, "600 秒内恢复全功能 LAN"):
                client.wait_for_raw_ready()

    def test_raw_ready_retry_backoff_remains_cancellable(self):
        client = OtaClient("127.0.0.1", "secret")
        cancel_checks = iter((False, True))
        with mock.patch.object(
            client, "query_raw_ready", side_effect=OtaError("尚未启动")
        ) as query:
            with self.assertRaisesRegex(OtaCancelled, "用户取消"):
                client.wait_for_raw_ready(cancelled=lambda: next(cancel_checks))
        query.assert_called_once()

    def test_raw_ready_rejects_old_boot_token_then_accepts_new_one(self):
        client = OtaClient(
            "127.0.0.1", "secret", reboot_timeout_s=1.0, reboot_poll_s=0.01
        )
        with mock.patch.object(
            client, "query_raw_ready", side_effect=(0x11111111, 0x22222222)
        ):
            boot_id = client.wait_for_raw_ready(
                previous_boot_id=0x11111111,
                require_boot_identity=True,
            )
        self.assertEqual(boot_id, 0x22222222)

    def test_lost_reboot_reply_accepts_changed_committed_boot_token(self):
        package = type(
            "Package",
            (),
            {
                "firmware_version": 0x00010045,
                "image_size": 8,
                "image_crc32": crc32(b"12345678"),
                "image": b"12345678",
            },
        )()
        client = OtaClient(
            "127.0.0.1",
            "secret",
            retries=2,
            reconnect_delay_s=0.0,
            recovery_quiet_s=0.0,
        )
        first_socket = mock.Mock()
        recovery_socket = mock.Mock()
        responses = (
            ["FOTA1", "READY", "12345678", "00000000", "11111111"],
            ["FOTA1", "ACK", "12345678", "00000008"],
            OtaError("REBOOT reply lost"),
        )
        progress = []
        with mock.patch.object(
            client, "_connect", side_effect=(first_socket, recovery_socket)
        ) as connect, mock.patch.object(
            client, "_response", side_effect=responses
        ) as response, mock.patch.object(
            client, "_wait_for_raw_ready", return_value=0x22222222
        ), mock.patch.object(client, "_cancelable_sleep"):
            client.upload(package, progress=progress.append)

        self.assertEqual(client.source_boot_id, 0x11111111)
        self.assertEqual(client.final_boot_id, 0x22222222)
        self.assertEqual(response.call_count, 3)
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(connect.call_args_list[1], mock.call(send_hello=False))
        recovery_socket.sendall.assert_not_called()
        self.assertIn("新启动标识", progress[-1].message)

    def test_authenticated_acknowledged_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "app.bin"
            package_path = root / "app.fbgfw"
            image_path.write_bytes(_valid_image())
            package = build_package(
                image_path, package_path, encode_version(1, 0, 0)
            )
            board = _FakeOtaBoard("secret")
            board.start()
            progress = []
            OtaClient(
                "127.0.0.1",
                "secret",
                port=board.port,
                response_timeout_s=2,
                reboot_initial_delay_s=0.0,
                raw_release_cooldown_s=0.0,
            ).upload(package, progress=progress.append)
            board.join(2)
            self.assertIsNone(board.error)
            self.assertTrue(board.assert_hello)
            self.assertEqual(bytes(board.received), package.image)
            self.assertTrue(board.raw_ready_received)
            self.assertEqual(progress[-1].sent, package.image_size)
            self.assertIn("RAW READY", progress[-1].message)

    def test_raw_wait_repeats_on_one_persistent_connection(self):
        board = _DelayedRawBoard(required_hellos=3)
        board.start()
        client = OtaClient(
            "127.0.0.1",
            "secret",
            port=board.port,
            connect_timeout_s=0.2,
            response_timeout_s=0.2,
            reboot_timeout_s=1.0,
            reboot_poll_s=0.02,
            raw_connection_hold_s=0.5,
            raw_hello_interval_s=0.02,
            raw_release_cooldown_s=0.0,
        )
        client.wait_for_raw_ready()
        board.join(1.0)
        self.assertIsNone(board.error)
        self.assertEqual(board.connection_count, 1)
        self.assertGreaterEqual(board.hello_count, 3)

    def test_successful_raw_probe_cools_down_before_second_connection(self):
        board = _RepeatedRawBoard()
        board.start()
        cooldown_s = 0.08
        client = OtaClient(
            "127.0.0.1",
            "secret",
            port=board.port,
            connect_timeout_s=0.2,
            response_timeout_s=0.2,
            reboot_timeout_s=1.0,
            reboot_poll_s=0.01,
            raw_connection_hold_s=0.3,
            raw_hello_interval_s=0.05,
            raw_release_cooldown_s=cooldown_s,
        )

        client.wait_for_raw_ready()
        client.wait_for_raw_ready()

        board.join(1.0)
        self.assertIsNone(board.error)
        self.assertEqual(len(board.accepted_at), 2)
        self.assertEqual(len(board.closed_at), 2)
        self.assertGreaterEqual(
            board.accepted_at[1] - board.closed_at[0], cooldown_s * 0.8
        )

    def test_malformed_ready_hex_is_reported_as_ota_error(self):
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "image_size": 8,
                "image_crc32": 0,
                "image": b"12345678",
            },
        )()
        client = OtaClient("127.0.0.1", "secret", retries=1)
        fake_socket = mock.Mock()
        with mock.patch.object(client, "_connect", return_value=fake_socket), mock.patch.object(
            client,
            "_response",
            return_value=["FOTA1", "READY", "NOT_HEX!", "00000000"],
        ):
            with self.assertRaisesRegex(OtaError, "会话编号"):
                client.upload(package)

    def test_unexpected_begin_reply_is_preserved_in_error(self):
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "image_size": 8,
                "image_crc32": 0,
                "image": b"12345678",
            },
        )()
        client = OtaClient("127.0.0.1", "secret", retries=1)
        fake_socket = mock.Mock()
        with mock.patch.object(
            client, "_connect", return_value=fake_socket
        ), mock.patch.object(
            client,
            "_response",
            return_value=["FOTA1", "ACK", "12345678", "00000008"],
        ):
            with self.assertRaisesRegex(
                OtaError, r"无效的 READY 回执.*FOTA1\|ACK\|12345678\|00000008"
            ):
                client.upload(package)

    def test_retry_ignores_stale_ack_before_ready(self):
        image = b"12345678"
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "version_text": "0.0.1",
                "image_size": len(image),
                "image_crc32": crc32(image),
                "image": image,
            },
        )()
        first_socket = mock.Mock()
        recovery_socket = mock.Mock()
        client = OtaClient(
            "127.0.0.1",
            "secret",
            retries=2,
            reconnect_delay_s=0.0,
            recovery_quiet_s=0.0,
        )
        progress = []
        with mock.patch.object(
            client, "_connect", side_effect=(first_socket, recovery_socket)
        ), mock.patch.object(
            client, "_wait_for_raw_ready", return_value=0x11111111
        ), mock.patch.object(
            client, "_cancelable_sleep"
        ), mock.patch.object(
            client,
            "_response",
            side_effect=(
                OtaError("first socket stalled"),
                ["FOTA1", "ACK", "A5A50123", "00000008"],
                ["FOTA1", "READY", "A5A50123", "00000000", "11111111"],
                ["FOTA1", "ACK", "A5A50123", "00000008"],
                ["FOTA1", "REBOOT", "00000001"],
            ),
        ) as response:
            client.upload(package, progress=progress.append, verify_reboot=False)

        self.assertEqual(response.call_count, 5)
        self.assertTrue(any("迟到 ACK" in item.message for item in progress))
        recovery_socket.sendall.assert_any_call(b"FBGL1|HELLO")

    def test_retry_accepts_delayed_reboot_before_ready(self):
        image = b"12345678"
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "version_text": "0.0.1",
                "image_size": len(image),
                "image_crc32": crc32(image),
                "image": image,
            },
        )()
        first_socket = mock.Mock()
        recovery_socket = mock.Mock()
        client = OtaClient(
            "127.0.0.1",
            "secret",
            retries=2,
            reconnect_delay_s=0.0,
            recovery_quiet_s=0.0,
        )
        progress = []
        with mock.patch.object(
            client, "_connect", side_effect=(first_socket, recovery_socket)
        ), mock.patch.object(
            client, "_wait_for_raw_ready", return_value=0x11111111
        ), mock.patch.object(
            client, "_cancelable_sleep"
        ), mock.patch.object(
            client,
            "_response",
            side_effect=(
                ["FOTA1", "READY", "A5A50123", "00000000", "11111111"],
                ["FOTA1", "ACK", "A5A50123", "00000008"],
                OtaError("REBOOT reply lost"),
                ["FOTA1", "REBOOT", "00000001"],
            ),
        ) as response:
            client.upload(package, progress=progress.append, verify_reboot=False)

        self.assertEqual(response.call_count, 4)
        self.assertTrue(any("迟到 REBOOT" in item.message for item in progress))
        recovery_socket.sendall.assert_any_call(b"FBGL1|HELLO")

    def test_adaptive_pacing_slows_after_a_transfer_break(self):
        image = b"A" * 360
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "version_text": "0.0.1",
                "image_size": len(image),
                "image_crc32": crc32(image),
                "image": image,
            },
        )()
        first_socket = mock.Mock()
        recovery_socket = mock.Mock()
        client = OtaClient(
            "127.0.0.1",
            "secret",
            retries=2,
            reconnect_delay_s=0.0,
            recovery_quiet_s=0.0,
            chunk_delay_s=0.2,
            adaptive_chunk_pacing=True,
            chunk_delay_max_s=0.45,
            chunk_delay_backoff=1.5,
            chunk_delay_stable_blocks=999,
        )
        progress = []
        sleeps = []
        with mock.patch.object(
            client, "_connect", side_effect=(first_socket, recovery_socket)
        ), mock.patch.object(
            client, "_wait_for_raw_ready", return_value=0x11111111
        ), mock.patch.object(
            client,
            "_cancelable_sleep",
            side_effect=lambda seconds, _cancelled: sleeps.append(seconds),
        ), mock.patch.object(
            client,
            "_response",
            side_effect=(
                ["FOTA1", "READY", "A5A50123", "00000000", "11111111"],
                ["FOTA1", "ACK", "A5A50123", "000000B4"],
                OtaError("lost block ACK"),
                ["FOTA1", "READY", "A5A50123", "000000B4", "11111111"],
                ["FOTA1", "ACK", "A5A50123", "00000168"],
                ["FOTA1", "REBOOT", "00000001"],
            ),
        ):
            client.upload(package, progress=progress.append, verify_reboot=False)

        self.assertEqual(client.recovery_count, 1)
        self.assertAlmostEqual(client.final_chunk_delay_s, 0.3)
        self.assertIn(0.2, sleeps)
        self.assertTrue(any(abs(value - 0.3) < 1e-9 for value in sleeps))
        self.assertTrue(any("自适应为 300 ms" in item.message for item in progress))

    def test_post_reboot_callback_can_release_a_debugger_halt(self):
        recovery = mock.Mock(return_value="已继续运行被调试器截停的引导")
        client = OtaClient(
            "127.0.0.1",
            "secret",
            reboot_timeout_s=10.0,
            reboot_poll_s=0.01,
            raw_release_cooldown_s=0.0,
            post_reboot_recovery=recovery,
            post_reboot_recovery_interval_s=1.0,
        )
        messages = []
        monotonic_values = iter((0.0, 0.1, 0.2, 2.0, 2.0, 2.1, 2.2, 2.3))
        with mock.patch.object(
            client,
            "query_raw_ready",
            side_effect=(OtaError("boot halted"), 0x22222222),
        ), mock.patch.object(
            client, "_cancelable_sleep"
        ), mock.patch(
            "fbg_ota_protocol.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            boot_id = client.wait_for_raw_ready(
                progress=messages.append,
                previous_boot_id=0x11111111,
                require_boot_identity=True,
            )

        self.assertEqual(boot_id, 0x22222222)
        recovery.assert_called_once_with()
        self.assertTrue(any("调试器截停" in message for message in messages))

    def test_cancel_before_connect_is_not_retried(self):
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "image_size": 8,
                "image_crc32": 0,
                "image": b"12345678",
            },
        )()
        client = OtaClient(
            "127.0.0.1", "secret", retries=4, reboot_initial_delay_s=0.0
        )
        with mock.patch.object(client, "_connect") as connect:
            with self.assertRaises(OtaCancelled):
                client.upload(package, cancelled=lambda: True)
        connect.assert_not_called()

    def test_retry_closes_failed_socket_before_backoff(self):
        image = b"12345678"
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "image_size": len(image),
                "image_crc32": crc32(image),
                "image": image,
            },
        )()
        first_socket = mock.Mock()
        second_socket = mock.Mock()
        client = OtaClient(
            "127.0.0.1",
            "secret",
            retries=2,
            reconnect_delay_s=0.25,
        )
        close_state_during_backoff = []
        with mock.patch.object(
            client, "_connect", side_effect=(first_socket, second_socket)
        ), mock.patch.object(
            client, "_wait_for_raw_ready", return_value=0x12345678
        ), mock.patch.object(
            client,
            "_cancelable_sleep",
            side_effect=lambda _seconds, _cancelled: close_state_during_backoff.append(
                first_socket.close.called
            ),
        ), mock.patch.object(
            client,
            "_response",
            side_effect=(
                OtaError("first socket stalled"),
                ["FOTA1", "READY", "A5A50123", "00000000"],
                ["FOTA1", "ACK", "A5A50123", "00000008"],
                ["FOTA1", "REBOOT", "00000001"],
            ),
        ), mock.patch(
            "fbg_ota_protocol.time.sleep"
        ) as text_mode_sleep:
            client.upload(package, verify_reboot=False)

        self.assertEqual(close_state_during_backoff, [True])
        text_mode_sleep.assert_called_once_with(0.08)
        first_socket.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        second_socket.sendall.assert_any_call(b"FBGL1|HELLO")
        second_socket.close.assert_called_once_with()

    def test_reboot_reply_must_match_package_version(self):
        image = b"12345678"
        package = type(
            "Package",
            (),
            {
                "firmware_version": 0x00010002,
                "version_text": "1.0.2",
                "image_size": len(image),
                "image_crc32": crc32(image),
                "image": image,
            },
        )()
        client = OtaClient("127.0.0.1", "secret", retries=1)
        fake_socket = mock.Mock()
        with mock.patch.object(client, "_connect", return_value=fake_socket), mock.patch.object(
            client,
            "_response",
            side_effect=(
                ["FOTA1", "READY", "A5A50123", "00000000"],
                ["FOTA1", "ACK", "A5A50123", "00000008"],
                ["FOTA1", "REBOOT", "00010001"],
            ),
        ):
            with self.assertRaisesRegex(OtaError, "待重启版本"):
                client.upload(package, verify_reboot=False)

    def test_post_commit_half_open_socket_is_closed_locally_without_retry(self):
        image = b"12345678"
        package = type(
            "Package",
            (),
            {
                "firmware_version": 1,
                "version_text": "0.0.1",
                "image_size": len(image),
                "image_crc32": crc32(image),
                "image": image,
            },
        )()
        client = OtaClient(
            "127.0.0.1", "secret", retries=4, reboot_initial_delay_s=0.0
        )
        fake_socket = mock.Mock()
        with mock.patch.object(client, "_connect", return_value=fake_socket) as connect, mock.patch.object(
            client,
            "_response",
            side_effect=(
                ["FOTA1", "READY", "A5A50123", "00000000", "11111111"],
                ["FOTA1", "ACK", "A5A50123", "00000008"],
                ["FOTA1", "REBOOT", "00000001"],
            ),
        ), mock.patch.object(
            client,
            "wait_for_reboot_disconnect",
            side_effect=OtaError("old socket stayed open"),
        ) as old_disconnect_wait, mock.patch.object(
            client, "wait_for_raw_ready", return_value=0x22222222
        ) as raw_ready:
            client.upload(package)
        connect.assert_called_once_with()
        old_disconnect_wait.assert_not_called()
        fake_socket.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        fake_socket.close.assert_called_once_with()
        raw_ready.assert_called_once_with(
            progress=mock.ANY,
            cancelled=None,
            previous_boot_id=0x11111111,
            require_boot_identity=True,
        )


if __name__ == "__main__":
    unittest.main()
