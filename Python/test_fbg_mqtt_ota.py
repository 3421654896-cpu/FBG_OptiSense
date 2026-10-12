import hashlib
import hmac
import unittest
from pathlib import Path
from types import SimpleNamespace

from fbg_mqtt_ota import MqttOtaClient, parse_pong, signed_abort, signed_ping
from fbg_mqtt_transport import RemoteMqttConfig
from fbg_ota_protocol import FirmwarePackage, OtaCancelled, OtaError, crc32


SECRET = "test-device-ota-secret"


class MqttOtaWireTests(unittest.TestCase):
    def test_paho_v2_suback_reason_code_enables_only_granted_subscription(self):
        config = RemoteMqttConfig(device_id="fbg-TEST1", host="example.invalid")
        client = MqttOtaClient(config, SECRET)
        client._on_subscribe(None, None, 1, [SimpleNamespace(is_failure=False)], None)
        self.assertTrue(client._subscribed.is_set())
        client._subscribed.clear()
        client._on_subscribe(None, None, 2, [SimpleNamespace(is_failure=True)], None)
        self.assertFalse(client._subscribed.is_set())

    def test_mqtt_ack_requires_a_valid_device_mac(self):
        config = RemoteMqttConfig(device_id="fbg-TEST1", host="example.invalid")
        client = MqttOtaClient(config, SECRET)
        client._subscribed.set()

        class Publisher:
            def publish(self, topic, payload, qos, retain):
                self_topic = config.ack_topic
                unsigned = "FOTA1|ACK|AABBCCDD|000000B4"
                wrong = SimpleNamespace(
                    topic=self_topic, retain=False,
                    payload=(unsigned + "|" + "0" * 64).encode(),
                )
                mac = hmac.new(SECRET.encode(), unsigned.encode(), hashlib.sha256).hexdigest().upper()
                valid = SimpleNamespace(
                    topic=self_topic, retain=False,
                    payload=(unsigned + "|" + mac).encode(),
                )
                client._on_message(None, None, wrong)
                client._on_message(None, None, valid)
                return SimpleNamespace(rc=0)

        client._client = Publisher()
        result = client._exchange("FOTA1|DATA|example", ("ACK",), tries=1)
        self.assertEqual(result, ["FOTA1", "ACK", "AABBCCDD", "000000B4"])

    def test_signed_identity_challenge_rejects_wrong_device_or_mac(self):
        nonce = "0123456789ABCDEF"
        request = signed_ping(nonce, SECRET)
        self.assertTrue(request.startswith("FOTA1|PING|" + nonce + "|"))
        payload = f"FOTA1|PONG|{nonce}|12345678|00010065"
        mac = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest().upper()
        self.assertEqual(parse_pong(payload + "|" + mac, nonce, SECRET), (0x12345678, 0x10065))
        with self.assertRaises(OtaError):
            parse_pong(payload + "|" + mac, "AAAAAAAAAAAAAAAA", SECRET)
        with self.assertRaises(OtaError):
            parse_pong(payload + "|" + "0" * 64, nonce, SECRET)

    def test_upload_has_authenticated_begin_and_whole_image_end(self):
        image = bytes(range(256)) * 2
        package = FirmwarePackage(Path("test.fbgfw"), 0xF2050052, 0x08008000,
                                  0x00010065, 0, crc32(image), image)
        config = RemoteMqttConfig(device_id="fbg-TEST1", host="example.invalid")
        client = MqttOtaClient(config, SECRET)
        commands = []
        boots = iter([(0x11111111, 0x00010064), (0x22222222, 0x00010065)])
        client.ping = lambda: next(boots)

        def exchange(command, accepted, **kwargs):
            commands.append(command)
            parts = command.split("|")
            if parts[1] == "BEGIN":
                return ["FOTA1", "READY", "AABBCCDD", "00000000", "11111111"]
            if parts[1] == "DATA":
                offset = int(parts[3], 16)
                import base64
                size = len(base64.b64decode(parts[5]))
                return ["FOTA1", "ACK", "AABBCCDD", f"{offset + size:08X}"]
            if parts[1] == "END":
                return ["FOTA1", "REBOOT", "00010065"]
            raise AssertionError(command)

        client._exchange = exchange
        import unittest.mock
        with unittest.mock.patch("fbg_mqtt_ota.time.sleep"):
            client.install(package)
        begin, *_, end = commands
        expected_begin_mac = hmac.new(SECRET.encode(), "|".join(begin.split("|")[:-1]).encode(), hashlib.sha256).hexdigest().upper()
        self.assertEqual(begin.split("|")[-1], expected_begin_mac)
        self.assertEqual(end.split("|")[-1], hmac.new(SECRET.encode(), image, hashlib.sha256).hexdigest().upper())
        self.assertEqual(sum(line.startswith("FOTA1|DATA|") for line in commands), 3)

    def test_cancel_before_commit_sends_signed_abort(self):
        image = b"abcde123"
        package = FirmwarePackage(Path("test.fbgfw"), 0xF2050052, 0x08008000,
                                  0x00010065, 0, crc32(image), image)
        client = MqttOtaClient(RemoteMqttConfig(device_id="fbg-TEST1", host="example.invalid"), SECRET)
        client.ping = lambda: (0x11111111, 0x00010064)
        commands = []

        def exchange(command, accepted, **kwargs):
            commands.append(command)
            if command.startswith("FOTA1|BEGIN|"):
                return ["FOTA1", "READY", "AABBCCDD", "00000000", "11111111"]
            if command.startswith("FOTA1|ABORT|"):
                return ["FOTA1", "ABORTED", command.split("|")[2]]
            raise AssertionError(command)

        client._exchange = exchange
        with self.assertRaises(OtaCancelled):
            client.install(package, cancelled=lambda: True)
        abort = next(c for c in commands if c.startswith("FOTA1|ABORT|"))
        signed_prefix, mac = abort.rsplit("|", 1)
        self.assertEqual(mac, hmac.new(SECRET.encode(), signed_prefix.encode(), hashlib.sha256).hexdigest().upper())

    def test_lost_end_ack_never_restarts_a_possibly_committed_image(self):
        image = b"abcde123"
        package = FirmwarePackage(Path("test.fbgfw"), 0xF2050052, 0x08008000,
                                  0x00010065, 0, crc32(image), image)
        client = MqttOtaClient(RemoteMqttConfig(device_id="fbg-TEST1", host="example.invalid"), SECRET)
        boots = iter([(0x11111111, 0x00010064), (0x22222222, 0x00010065)])
        client.ping = lambda: next(boots)
        commands = []

        def exchange(command, accepted, **kwargs):
            commands.append(command)
            if command.startswith("FOTA1|BEGIN|"):
                return ["FOTA1", "READY", "AABBCCDD", "00000000", "11111111"]
            if command.startswith("FOTA1|DATA|"):
                return ["FOTA1", "ACK", "AABBCCDD", "00000008"]
            if command.startswith("FOTA1|END|"):
                raise OtaError("设备未在限定重试内回复 OTA 命令")
            raise AssertionError(command)

        client._exchange = exchange
        import unittest.mock
        with unittest.mock.patch("fbg_mqtt_ota.time.sleep"):
            client.install(package)
        self.assertEqual(sum(c.startswith("FOTA1|BEGIN|") for c in commands), 1)

    def test_rejects_wan_downgrade_before_begin(self):
        config = RemoteMqttConfig(device_id="fbg-TEST1", host="example.invalid")
        client = MqttOtaClient(config, SECRET)
        client.ping = lambda: (1, 0x00010065)
        package = FirmwarePackage(Path("test.fbgfw"), 0xF2050052, 0x08008000,
                                  0x00010064, 0, 0, b"abc")
        with self.assertRaisesRegex(OtaError, "相同或更新"):
            client.install(package)


if __name__ == "__main__":
    unittest.main()
