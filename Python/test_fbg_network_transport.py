import threading
import unittest
from unittest.mock import patch

from fbg_network_transport import FbgSelectableTransport, NETWORK_LAN, NETWORK_WAN


class _FakeTransport:
    def __init__(self, name, events, **callbacks):
        self.name = name
        self.events = events
        self.callbacks = callbacks
        self.connected = False
        self.latest = None
        self.board_ip = "192.168.3.46" if name == "lan" else None

    def start(self):
        self.events.append((self.name, "start"))
        self.connected = True

    def stop(self):
        self.events.append((self.name, "stop"))
        self.connected = False

    def take_latest_telemetry(self):
        frame, self.latest = self.latest, None
        return frame

    def publish_mode(self, mode, command_id):
        self.events.append((self.name, "mode", mode, command_id))


class SelectableTransportTests(unittest.TestCase):
    def test_switch_stops_old_path_before_starting_new_path(self):
        events = []

        def lan_factory(**callbacks):
            return _FakeTransport("lan", events, **callbacks)

        def wan_factory(config, **callbacks):
            self.assertEqual(config, "mqtt-config")
            return _FakeTransport("wan", events, **callbacks)

        with patch("fbg_network_transport.FbgLanTransport", lan_factory), patch(
            "fbg_network_transport.FbgMqttTransport", wan_factory
        ):
            transport = FbgSelectableTransport(
                "mqtt-config", initial_mode=NETWORK_LAN
            )
            transport.start()
            transport.select_network(NETWORK_WAN)

        self.assertEqual(
            events,
            [("lan", "start"), ("lan", "stop"), ("wan", "start")],
        )
        self.assertEqual(transport.network_mode, NETWORK_WAN)

    def test_missing_mqtt_config_does_not_stop_running_lan(self):
        events = []

        def lan_factory(**callbacks):
            return _FakeTransport("lan", events, **callbacks)

        with patch("fbg_network_transport.FbgLanTransport", lan_factory):
            transport = FbgSelectableTransport(None, initial_mode=NETWORK_LAN)
            transport.start()
            with self.assertRaises(RuntimeError):
                transport.select_network(NETWORK_WAN)

        self.assertEqual(events, [("lan", "start")])
        self.assertEqual(transport.network_mode, NETWORK_LAN)

    def test_callbacks_from_stopped_path_are_ignored(self):
        events = []
        statuses = []
        instances = []

        def factory(name):
            def build(*args, **callbacks):
                instance = _FakeTransport(name, events, **callbacks)
                instances.append(instance)
                return instance

            return build

        with patch(
            "fbg_network_transport.FbgLanTransport", factory("lan")
        ), patch("fbg_network_transport.FbgMqttTransport", factory("wan")):
            transport = FbgSelectableTransport(
                "mqtt-config", initial_mode=NETWORK_LAN, on_status=statuses.append
            )
            transport.start()
            old_callback = instances[0].callbacks["on_status"]
            transport.select_network(NETWORK_WAN)
            old_callback("stale")
            instances[1].callbacks["on_status"]("current")

        self.assertEqual(statuses, ["current"])

    def test_board_ip_is_exposed_only_by_active_lan_path(self):
        events = []

        with patch(
            "fbg_network_transport.FbgLanTransport",
            lambda **callbacks: _FakeTransport("lan", events, **callbacks),
        ), patch(
            "fbg_network_transport.FbgMqttTransport",
            lambda config, **callbacks: _FakeTransport("wan", events, **callbacks),
        ):
            transport = FbgSelectableTransport("mqtt-config")
            transport.start()
            self.assertEqual(transport.board_ip, "192.168.3.46")
            transport.select_network(NETWORK_WAN)
            self.assertIsNone(transport.board_ip)

    def test_slow_old_selection_cannot_resurrect_after_stop(self):
        events = []
        start_entered = threading.Event()
        allow_start = threading.Event()

        class SlowTransport(_FakeTransport):
            def start(self):
                events.append((self.name, "start-enter"))
                start_entered.set()
                self.assert_released = allow_start.wait(2.0)
                self.connected = True
                events.append((self.name, "start-exit"))

        with patch(
            "fbg_network_transport.FbgLanTransport",
            lambda **callbacks: SlowTransport("lan", events, **callbacks),
        ):
            transport = FbgSelectableTransport(None)
            worker = threading.Thread(target=transport.start)
            worker.start()
            self.assertTrue(start_entered.wait(1.0))
            transport.stop()
            allow_start.set()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertIsNone(transport.network_mode)
        self.assertFalse(transport.connected)
        self.assertEqual(events[-1], ("lan", "stop"))

    def test_mode_values_are_validated_before_delegation(self):
        events = []
        transport = FbgSelectableTransport(None)
        transport._transport = _FakeTransport("lan", events)
        with self.assertRaises(ValueError):
            transport.publish_mode(False, 1)
        with self.assertRaises(ValueError):
            transport.publish_mode(0, 2.5)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
