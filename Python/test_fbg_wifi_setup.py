import os
import unittest
import zlib
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

from fbg_wifi_setup import (
    PROVISION_PAYLOAD_END,
    ProvisionWindow,
    ProvisioningSettings,
    build_provisioning_frame,
    find_provisioning_ack,
)


class ProvisioningPacketTests(unittest.TestCase):
    def settings(self, **changes):
        values = dict(
            ssid="lab-wifi",
            wifi_password="wifi-secret",
            broker_host="mqtt.example.test",
            broker_port=8883,
            client_id="fbg-board-1",
            mqtt_username="board-1",
            mqtt_password="mqtt-secret",
            topic_prefix="fbg/v1",
            device_id="fbg-board1",
        )
        values.update(changes)
        return ProvisioningSettings(**values)

    def test_packet_layout_and_crc(self):
        packet = build_provisioning_frame(self.settings())
        self.assertEqual(len(packet), 808)
        self.assertEqual(packet[:5], b"\xff\xff\x01\x20\x01")
        self.assertEqual(int.from_bytes(packet[14:16], "big"), 8883)
        self.assertEqual(packet[16], 2)
        self.assertEqual(
            int.from_bytes(packet[804:808], "big"),
            zlib.crc32(packet[:PROVISION_PAYLOAD_END]) & 0xFFFFFFFF,
        )
        self.assertIn(b"mqtt.example.test", packet)
        self.assertEqual(packet[5], 0x03)

    def test_lan_only_does_not_require_a_broker_or_tls(self):
        packet = build_provisioning_frame(
            self.settings(lan_only=True, broker_host="", tls=False)
        )
        self.assertEqual(packet[5], 0x07)
        self.assertEqual(packet[8], 0)  # empty MQTT host field
        self.assertEqual(packet[16], 1)
        self.assertEqual(
            int.from_bytes(packet[804:808], "big"),
            zlib.crc32(packet[:PROVISION_PAYLOAD_END]) & 0xFFFFFFFF,
        )

    def test_invalid_port_and_oversized_field(self):
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(broker_port=0))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(ssid="x" * 33))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(mqtt_password='bad"password'))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(topic_prefix="fbg,v1"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(wifi_password="bad,password"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(topic_prefix="/"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(topic_prefix="fbg/+/v1"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(device_id="bad/device"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(device_id="fbg-board-1"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(device_id="fbg-board_1"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(device_id="board1"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(broker_port=8883.5))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(tls="false"))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(enabled=1))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(lan_only=1))
        with self.assertRaisesRegex(ValueError, "TLS"):
            build_provisioning_frame(self.settings(tls=False))
        with self.assertRaises(ValueError):
            build_provisioning_frame(self.settings(device_id="._-"))

    def test_ack_can_be_found_in_mixed_stream(self):
        ack = bytearray(20)
        ack[:4] = b"\xff\xff\x03\x20"
        ack[4:7] = bytes((0, 5, 1))
        ack[8:12] = bytes.fromhex("12345678")
        ack[15] = 2
        ack[18:20] = b"\xff\xef"
        self.assertEqual(
            find_provisioning_ack(
                b"telemetry" + ack + b"tail", expected_crc32=0x12345678
            ),
            (0, 5, 1, 0x12345678),
        )

    def test_ack_rejects_bad_tail_and_stale_success(self):
        ack = bytearray(20)
        ack[:4] = b"\xff\xff\x03\x20"
        ack[4:7] = bytes((0, 4, 1))
        ack[8:12] = bytes.fromhex("12345678")
        ack[15] = 2
        self.assertIsNone(find_provisioning_ack(bytes(ack)))

        ack[18:20] = b"\xff\xef"
        self.assertIsNone(
            find_provisioning_ack(bytes(ack), expected_crc32=0x87654321)
        )


class ProvisioningPortSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    @staticmethod
    def port(device, vid, pid, description):
        return SimpleNamespace(
            device=device,
            vid=vid,
            pid=pid,
            description=description,
        )

    def test_native_cdc_is_preferred_over_programmer_port(self):
        ports = [
            self.port("COM7", 0x1A86, 0x55D4, "USB programmer UART"),
            self.port("COM9", 0x0483, 0x5740, "STM32 Virtual COM Port"),
        ]
        with mock.patch("fbg_wifi_setup.list_ports.comports", return_value=ports):
            window = ProvisionWindow()
        self.assertEqual(window.port.currentData(), "COM9")
        window.close()

    def test_programmer_port_is_not_automatically_selected(self):
        programmer = self.port("COM7", 0x1A86, 0x55D4, "USB programmer UART")
        with mock.patch(
            "fbg_wifi_setup.list_ports.comports", return_value=[programmer]
        ):
            window = ProvisionWindow()
            self.assertEqual(window.port.currentIndex(), -1)

            # Once the operator explicitly chooses an available port, refresh
            # must preserve that deliberate choice.
            window.port.setCurrentIndex(0)
            window.refresh_ports()
            self.assertEqual(window.port.currentData(), "COM7")
        window.close()

    def test_device_id_is_not_silently_trimmed(self):
        with mock.patch("fbg_wifi_setup.list_ports.comports", return_value=[]):
            window = ProvisionWindow()
        window.ssid.setText("lab-wifi")
        window.host.setText("mqtt.example.test")
        window.device_id.setText(" fbg-board1 ")
        self.assertEqual(window._settings().device_id, " fbg-board1 ")
        with self.assertRaisesRegex(ValueError, "device_id"):
            build_provisioning_frame(window._settings())
        window.close()


if __name__ == "__main__":
    unittest.main()
