import base64
import hashlib
import hmac
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fbg_ota_protocol import OtaError, OtaCancelled, build_package, crc32
from fbg_usb_ota import UsbOtaClient, build_usb_update_frame

SERIAL = "207A33813232"


class FakeBoard:
    def __init__(self):
        self.version, self.boot, self.offset = 0x10069, 7, 0
        self.port = "COM20"
        self.reply_serial = SERIAL
        self.nonce = None
        self.commands = []
        self.drop_data_once = False
        self.bad_ack = False
        self.bad_end = False
        self.keep_boot = False
        self.drop_end_ack = False
        self.bootloader_ready = 1

    def ports(self):
        return [SimpleNamespace(device=self.port, vid=0x0483, pid=0x5740,
                                serial_number=SERIAL)]

    def factory(self, *_args, **_kwargs):
        board = self

        class FakePort:
            dtr = True
            def __init__(self): self.reply = bytearray()
            @property
            def in_waiting(self): return len(self.reply)
            def reset_input_buffer(self): self.reply.clear()
            def flush(self): pass
            def close(self): pass
            def read(self, _):
                # Fragment all responses to exercise the stream decoder.
                data = bytes(self.reply[:7]); del self.reply[:7]
                return data
            def write(self, frame):
                assert frame[:4] == b"\xff\xff\x02\x22"
                assert crc32(frame[:804]) == int.from_bytes(frame[804:], "big")
                length = int.from_bytes(frame[6:8], "big")
                command = frame[8:8 + length].decode()
                board.commands.append(command.split("|")[1])
                fields = command.split("|")
                if command == "FUSB1|INFO":
                    response = (f"FUSB1|INFO|{board.reply_serial}|{board.version:08X}|"
                                f"{board.boot:08X}|00038000|{board.bootloader_ready:08X}\n")
                elif command == "FUSB1|ABORT":
                    board.offset = 0
                    response = "FUSB1|ABORTED\n"
                elif fields[1] == "BEGIN":
                    prefix = "|".join(fields[:-1])
                    expected = hmac.new(f"USB:{SERIAL}".encode(), prefix.encode(), hashlib.sha256).hexdigest().upper()
                    assert fields[-1] == expected
                    if board.nonce != fields[6]: board.offset = 0
                    board.nonce = fields[6]
                    board.target_version = int(fields[3], 16)
                    response = f"FOTA1|READY|00000042|{board.offset:08X}|{board.boot:08X}\n"
                elif fields[1] == "DATA":
                    block = base64.b64decode(fields[5])
                    assert crc32(block) == int(fields[4], 16)
                    assert int(fields[3], 16) == board.offset
                    board.offset += len(block)
                    if board.drop_data_once:
                        board.drop_data_once = False
                        return len(frame)
                    offset = board.offset + int(board.bad_ack)
                    response = f"FOTA1|ACK|00000042|{offset:08X}\n"
                elif fields[1] == "END":
                    if board.bad_end:
                        response = "FOTA1|ERROR|VERIFY|whole-image-invalid\n"
                    else:
                        board.version = board.target_version
                        if not board.keep_boot: board.boot += 1
                        board.port = "COM22"
                        if board.drop_end_ack:
                            return len(frame)
                        response = f"FOTA1|REBOOT|{board.version:08X}\n"
                else: raise AssertionError(command)
                self.reply.extend(b"unrelated telemetry\n" + response.encode())
                return len(frame)
        return FakePort()


class UsbUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        image = struct.pack("<II", 0x20010000, 0x08008021) + bytes(range(256)) * 2
        (root / "app.bin").write_bytes(image)
        self.package = build_package(root / "app.bin", root / "app.fbgfw", 0x10069)
        self.board = FakeBoard()

    def client(self, **kwargs):
        return UsbOtaClient("COM20", SERIAL, serial_factory=self.board.factory,
                            ports_provider=self.board.ports, timeout_s=.015,
                            reboot_timeout_s=.03, **kwargs)

    def test_frame_length_version_and_crc(self):
        frame = build_usb_update_frame("FUSB1|INFO")
        self.assertEqual(len(frame), 808)
        self.assertEqual(frame[4:6], b"\x01\x00")
        self.assertEqual(int.from_bytes(frame[6:8], "big"), 10)
        self.assertEqual(int.from_bytes(frame[804:], "big"), crc32(frame[:804]))

    def test_invalid_commands_rejected(self):
        for text in ("", "x" * 500, "x\n", "x\0"):
            with self.assertRaises(ValueError): build_usb_update_frame(text)

    def test_descriptor_identity_mismatch_sends_nothing(self):
        self.board.ports = lambda: [SimpleNamespace(device="COM20", vid=0x0483,
            pid=0x5740, serial_number="395736523033")]
        with self.assertRaises(OtaError):
            with self.client(): pass
        self.assertEqual(self.board.commands, [])

    def test_mcu_identity_mismatch_never_begins(self):
        self.board.reply_serial = "395736523033"
        with self.client() as client:
            with self.assertRaises(OtaError): client.install(self.package)
        self.assertNotIn("BEGIN", self.board.commands)

    def test_missing_bootloader_never_begins(self):
        self.board.bootloader_ready = 0
        with self.client() as client:
            with self.assertRaises(OtaError): client.install(self.package)
        self.assertNotIn("BEGIN", self.board.commands)

    def test_complete_same_version_update_verifies_reenumerated_identity(self):
        events = []
        with self.client() as client:
            result = client.install(self.package, progress=events.append)
        self.assertEqual(result.firmware_version, 0x10069)
        self.assertEqual(result.boot_id, 8)
        self.assertEqual(client.port, "COM22")
        self.assertTrue(client.reboot_acknowledged)
        self.assertTrue(any("不能取消" in p.message for p in events))

    def test_missing_data_ack_resumes_at_authoritative_offset(self):
        self.board.drop_data_once = True
        with patch("fbg_usb_ota.time.sleep"):
            with self.client() as client: result = client.install(self.package)
        self.assertEqual(result.boot_id, 8)
        self.assertEqual(self.board.commands.count("BEGIN"), 2)
        self.assertEqual(self.board.commands.count("DATA"), 3)

    def test_reconnect_waits_for_same_device_not_another_board(self):
        attempts = [0]
        real_ports = self.board.ports
        def ports():
            attempts[0] += 1
            return [] if attempts[0] < 3 else real_ports()
        with self.client() as client:
            client.close()
            client.ports_provider = ports
            with patch("fbg_usb_ota.time.sleep"):
                client._reconnect()
            self.assertEqual(client.query_info().serial_number, SERIAL)

    def test_old_package_without_usb_protocol_never_begins(self):
        old = build_package(self.package.path.parent / "app.bin",
                            self.package.path.parent / "old.fbgfw", 0x10065)
        with self.client() as client:
            with self.assertRaisesRegex(OtaError, "不支持 USB"):
                client.install(old)
        self.assertNotIn("BEGIN", self.board.commands)

    def test_bad_block_ack_never_commits(self):
        self.board.bad_ack = True
        with self.client() as client:
            with self.assertRaises(OtaError): client.install(self.package)
        self.assertNotIn("END", self.board.commands)
        self.assertIn("ABORT", self.board.commands)

    def test_cancel_before_commit_is_acknowledged(self):
        flag = [False]
        def progress(p):
            if p.sent: flag[0] = True
        with self.client() as client:
            with self.assertRaisesRegex(OtaCancelled, "已确认取消"):
                client.install(self.package, cancelled=lambda: flag[0], progress=progress)
        self.assertIn("ABORT", self.board.commands)
        self.assertNotIn("END", self.board.commands)

    def test_crc_rejection_never_claims_installed(self):
        self.board.bad_end = True
        with self.client() as client:
            with self.assertRaisesRegex(OtaError, "VERIFY"): client.install(self.package)
        self.assertEqual(self.board.boot, 7)

    def test_lost_final_reply_verifies_runtime_without_a_second_begin(self):
        self.board.drop_end_ack = True
        with self.client() as client:
            result = client.install(self.package)
        self.assertEqual(result.boot_id, 8)
        self.assertFalse(client.reboot_acknowledged)
        self.assertEqual(self.board.commands.count("BEGIN"), 1)

    def test_cancel_after_commit_does_not_abort_committed_image(self):
        flag = [False]
        def progress(event):
            if "不能取消" in event.message:
                flag[0] = True
        with self.client() as client:
            result = client.install(self.package, progress=progress, cancelled=lambda: flag[0])
        self.assertEqual(result.boot_id, 8)
        self.assertNotIn("ABORT", self.board.commands)

    def test_unchanged_boot_token_never_claims_success(self):
        self.board.keep_boot = True
        with patch("fbg_usb_ota.time.sleep"):
            with self.client() as client:
                with self.assertRaisesRegex(OtaError, "未确认新版本"): client.install(self.package)

    def test_changed_package_never_sends_begin(self):
        data = bytearray(self.package.path.read_bytes()); data[-1] ^= 1
        self.package.path.write_bytes(data)
        with self.client() as client:
            with self.assertRaises(OtaError): client.install(self.package)
        self.assertNotIn("BEGIN", self.board.commands)

    def test_firmware_physical_origin_and_main_loop_contract(self):
        root = Path(__file__).parents[1] / "JDSU"
        usb = (root / "USB_DEVICE/App/usbd_cdc_if.c").read_text(encoding="utf-8")
        main = (root / "Core/Src/main.c").read_text(encoding="utf-8")
        ota = (root / "Core/Src/ota_update.c").read_text(encoding="utf-8")
        wifi = (root / "Core/Src/wifi_transport.c").read_text(encoding="utf-8")
        self.assertIn("OtaUpdate_QueueUsbFrame(usbAssemblyBuffer", usb)
        self.assertNotIn("OtaUpdate_QueueUsbFrame", wifi)
        self.assertLess(main.index("OtaUpdate_ProcessUsb();"), main.index("if(WifiTransport_IsOtaActive())"))
        self.assertIn("if(usbUpdateReboot) NVIC_SystemReset();", ota)
        self.assertIn("if(cdc == NULL || cdc->TxState != 0U) return;", ota)
        self.assertIn("different-transport-owner", ota)
        self.assertIn("EXTRA alone does not gate optical output", ota)


if __name__ == "__main__": unittest.main()
