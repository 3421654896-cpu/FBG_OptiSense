import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fbg_mqtt_transport import (
    FbgMqttTransport,
    RemoteMqttConfig,
    load_remote_config,
    validate_device_id,
)


class LatestTelemetrySlotTests(unittest.TestCase):
    def test_bw20_compatible_command_topic_has_no_slash(self):
        config = RemoteMqttConfig(
            device_id="fbg-395736563033", host="mqtt.example.test"
        )
        self.assertEqual(config.command_topic, "fbgcmd395736563033")
        self.assertNotIn("/", config.command_topic)

    def test_only_newest_frame_is_visible_to_gui(self):
        transport = FbgMqttTransport(
            RemoteMqttConfig(device_id="fbg-board1", host="mqtt.example.test")
        )
        transport._put_latest("old")
        transport._put_latest("new")
        self.assertEqual(transport.take_latest_telemetry(), "new")
        self.assertIsNone(transport.take_latest_telemetry())

    def test_remote_topic_identity_is_strict(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "remote.yaml"
            path.write_text(
                "device_id: bad/device\n"
                "topic_prefix: fbg/v1\n"
                "broker:\n  host: mqtt.example.test\n"
                "tls:\n  enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_remote_config(path)

            path.write_text(
                "device_id: fbg-board1\n"
                "topic_prefix: fbg/v1\n"
                "broker:\n  host: mqtt.example.test\n  port: 8883.5\n"
                "tls:\n  enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "broker.port.*整数"):
                load_remote_config(path)

            path.write_text(
                "device_id: fbg-board1\n"
                "topic_prefix: fbg/v1\n"
                "broker:\n  host: mqtt.example.test\n  keepalive_s: true\n"
                "tls:\n  enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "keepalive_s.*整数"):
                load_remote_config(path)

            path.write_text(
                "device_id: fbg-board1\n"
                "topic_prefix: fbg/v1\n"
                "broker:\n  host: mqtt.example.test\n"
                "tls:\n  enabled: 'false'\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "true 或 false"):
                load_remote_config(path)

            path.write_text(
                "device_id: fbg-board1\n"
                "topic_prefix: fbg/+/v1\n"
                "broker:\n  host: mqtt.example.test\n"
                "tls:\n  enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_remote_config(path)

        with self.assertRaises(ValueError):
            validate_device_id("bad/device")
        with self.assertRaises(ValueError):
            validate_device_id("x" * 33)
        with self.assertRaises(ValueError):
            validate_device_id("._-")
        for colliding_or_noncanonical in (
            "jdsu001",
            "jdsu-001",
            "fbg-board-1",
            "fbg-board_1",
            " fbg-board1",
        ):
            with self.subTest(device_id=colliding_or_noncanonical), self.assertRaises(
                ValueError
            ):
                validate_device_id(colliding_or_noncanonical)

    def test_connected_only_after_successful_suback(self):
        events = []
        transport = FbgMqttTransport(
            RemoteMqttConfig(device_id="fbg-board1", host="mqtt.example.test"),
            on_connection=lambda connected, detail: events.append((connected, detail)),
        )
        client = object()
        transport._client = client
        transport._subscribe_mid = 1
        transport._on_subscribe(client, None, 1, [1, 1, 1, 1, 1], None)
        self.assertTrue(transport.connected)
        self.assertTrue(events[-1][0])

        transport._subscribe_mid = 2
        transport._on_subscribe(client, None, 2, [1, 1, 0x80, 1, 1], None)
        self.assertFalse(transport.connected)
        self.assertFalse(events[-1][0])
        self.assertIn("ACL", events[-1][1])

        transport._subscribe_mid = 3
        transport._on_subscribe(client, None, 3, [1, 1, 1], None)
        self.assertFalse(transport.connected)
        self.assertIn("不完整", events[-1][1])

    def test_callbacks_from_stopped_client_are_ignored(self):
        events = []
        transport = FbgMqttTransport(
            RemoteMqttConfig(device_id="fbg-board1", host="mqtt.example.test"),
            on_connection=lambda connected, detail: events.append((connected, detail)),
        )
        old_client = object()
        transport._client = old_client
        transport._connected.set()
        transport._client = None
        transport._connected.clear()

        transport._on_subscribe(old_client, None, 1, [1], None)
        transport._on_message(
            old_client,
            None,
            SimpleNamespace(topic=transport.config.status_topic, payload=b"online"),
        )

        self.assertFalse(transport.connected)
        self.assertEqual(events, [])

    def test_mode_command_values_are_not_silently_truncated(self):
        transport = FbgMqttTransport(
            RemoteMqttConfig(device_id="fbg-board1", host="mqtt.example.test")
        )
        with self.assertRaises(ValueError):
            transport.publish_mode(0.0, 1)
        with self.assertRaises(ValueError):
            transport.publish_mode(0, 1.5)


if __name__ == "__main__":
    unittest.main()
