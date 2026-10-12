import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from fbg_ota_protocol import OtaClient
from validate_ota_hardware_stability import (
    FAULT_PLANS,
    FaultInjectingOtaClient,
    load_private_wifi_password,
    wait_for_wifi_online,
)


def _config_record(password="secret"):
    raw = bytearray(b"\xff" * 1024)
    size = 572
    raw[:size] = b"\0" * size
    struct.pack_into("<IHHI", raw, 0, 0x57464331, 1, size, 0x12345678)
    raw[49 : 49 + len(password)] = password.encode("utf-8")
    struct.pack_into("<I", raw, size - 4, zlib.crc32(raw[: size - 4]) & 0xFFFFFFFF)
    return bytes(raw)


class HardwareOtaAcceptanceTests(unittest.TestCase):
    def test_plan_covers_all_ten_cycles_with_normal_endpoints(self):
        self.assertEqual(sorted(FAULT_PLANS), list(range(1, 11)))
        self.assertEqual(FAULT_PLANS[1], {"kind": "normal"})
        self.assertEqual(FAULT_PLANS[10], {"kind": "normal"})

    def test_private_password_is_validated_and_not_returned_with_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.bin"
            path.write_bytes(_config_record("not-printed"))
            password, preserved = load_private_wifi_password(path)
            self.assertEqual(password, "not-printed")
            self.assertEqual(len(preserved), 1024)
            damaged = bytearray(path.read_bytes())
            damaged[20] ^= 1
            path.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "CRC"):
                load_private_wifi_password(path)

    def test_fault_client_drops_only_the_requested_data_command(self):
        client = FaultInjectingOtaClient(
            "127.0.0.1",
            "secret",
            drop_after_data_command=2,
            raw_release_cooldown_s=0,
        )
        sock = mock.Mock()
        with mock.patch.object(OtaClient, "_send") as parent_send, mock.patch.object(
            client, "_close"
        ) as close:
            client._send(sock, "FOTA1|DATA|one")
            client._send(sock, "FOTA1|DATA|two")
            client._send(sock, "FOTA1|DATA|three")
        self.assertEqual(parent_send.call_count, 3)
        close.assert_called_once_with(sock)
        self.assertTrue(client.fault_injected)

    def test_wifi_wait_accepts_only_stable_online_transition(self):
        transitional = {
            "external_state_name": "joining_mqtt",
            "config_valid": True,
            "enabled": True,
            "wifi_connected": True,
            "mqtt_connected": True,
            "ota_active": False,
            "last_error": 0,
        }
        online = dict(transitional, external_state_name="online")
        fake_serial = mock.MagicMock()
        fake_serial.__enter__.return_value = fake_serial
        with mock.patch(
            "validate_ota_hardware_stability.serial.Serial",
            return_value=fake_serial,
        ), mock.patch(
            "validate_ota_hardware_stability.query_wifi_status",
            side_effect=(transitional, online),
        ), mock.patch("validate_ota_hardware_stability.time.sleep"):
            status, observations = wait_for_wifi_online(timeout_s=5)
        self.assertEqual(status["external_state_name"], "online")
        self.assertEqual(
            [item["external_state"] for item in observations],
            ["joining_mqtt", "online"],
        )


if __name__ == "__main__":
    unittest.main()
