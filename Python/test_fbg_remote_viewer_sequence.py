import unittest
from collections import deque
from types import SimpleNamespace

from fbg_network_transport import NETWORK_LAN, NETWORK_WAN
from fbg_remote_viewer import MODE_TEMPERATURE, RemoteViewer


def sequence_state(*, boot_id=7, seq=100, uptime_ms=100_000):
    state = SimpleNamespace(
        last_boot_id=boot_id,
        last_seq=seq,
        last_uptime_ms=uptime_ms,
        restart_candidate=None,
        missing_sequences=0,
        out_of_order_sequences=0,
        arrival_times=deque((1.0, 2.0)),
        reset_count=0,
    )

    def reset_tracking():
        state.reset_count += 1

    state._reset_tracking = reset_tracking
    return state


class SequenceAcceptanceTests(unittest.TestCase):
    def test_same_boot_id_with_uptime_rollback_is_restart(self):
        state = sequence_state()
        first = SimpleNamespace(boot_id=7, seq=0, uptime_ms=100)
        second = SimpleNamespace(boot_id=7, seq=1, uptime_ms=1100)

        self.assertFalse(RemoteViewer._accept_sequence(state, first))
        self.assertEqual(state.last_seq, 100)
        self.assertTrue(RemoteViewer._accept_sequence(state, second))
        self.assertEqual(state.last_seq, 1)
        self.assertEqual(state.reset_count, 1)
        self.assertEqual(len(state.arrival_times), 0)

    def test_duplicate_or_old_frame_is_rejected(self):
        state = sequence_state()
        frame = SimpleNamespace(boot_id=7, seq=99, uptime_ms=99_900)
        self.assertFalse(RemoteViewer._accept_sequence(state, frame))
        self.assertEqual(state.last_seq, 100)
        self.assertEqual(state.out_of_order_sequences, 1)
        self.assertEqual(state.reset_count, 0)

    def test_forward_gap_counts_missing_frames(self):
        state = sequence_state()
        frame = SimpleNamespace(boot_id=7, seq=103, uptime_ms=100_300)
        self.assertTrue(RemoteViewer._accept_sequence(state, frame))
        self.assertEqual(state.missing_sequences, 2)
        self.assertEqual(state.last_seq, 103)

    def test_old_frame_with_large_uptime_rollback_does_not_reset_session(self):
        state = sequence_state()
        old = SimpleNamespace(boot_id=7, seq=90, uptime_ms=96_000)
        current = SimpleNamespace(boot_id=7, seq=101, uptime_ms=101_000)

        self.assertFalse(RemoteViewer._accept_sequence(state, old))
        self.assertEqual(state.last_seq, 100)
        self.assertEqual(state.reset_count, 0)
        self.assertTrue(RemoteViewer._accept_sequence(state, current))
        self.assertEqual(state.last_seq, 101)
        self.assertEqual(state.reset_count, 0)

    def test_restart_candidate_requires_consecutive_forward_frame(self):
        state = sequence_state()
        first = SimpleNamespace(boot_id=7, seq=0, uptime_ms=100)
        nonconsecutive = SimpleNamespace(boot_id=7, seq=2, uptime_ms=300)
        backward = SimpleNamespace(boot_id=7, seq=1, uptime_ms=200)

        self.assertFalse(RemoteViewer._accept_sequence(state, first))
        self.assertFalse(RemoteViewer._accept_sequence(state, nonconsecutive))
        self.assertFalse(RemoteViewer._accept_sequence(state, backward))
        self.assertEqual(state.last_seq, 100)
        self.assertEqual(state.reset_count, 0)


class InitialLanModeTests(unittest.TestCase):
    @staticmethod
    def link_state(network_mode, *, connected=True, data_online=False):
        state = SimpleNamespace(
            transport=SimpleNamespace(network_mode=network_mode),
            mqtt_connected=connected,
            device_status_online=data_online,
            local_control_active=False,
            lan_control_active=False,
        )
        state._lan_link_connected = lambda: RemoteViewer._lan_link_connected(state)
        return state

    def test_lan_can_send_first_mode_command_before_adc_telemetry(self):
        state = self.link_state(NETWORK_LAN)
        self.assertTrue(RemoteViewer._mode_command_link_ready(state))

    def test_wan_still_requires_device_online_status(self):
        state = self.link_state(NETWORK_WAN)
        self.assertFalse(RemoteViewer._mode_command_link_ready(state))
        state.device_status_online = True
        self.assertTrue(RemoteViewer._mode_command_link_ready(state))

    def test_idle_lan_connection_starts_selected_temperature_mode(self):
        requested = []
        state = self.link_state(NETWORK_LAN)
        state.pending_command_id = None
        state.current_mode = MODE_TEMPERATURE
        state._request_mode = requested.append

        RemoteViewer._start_selected_lan_mode_if_idle(state)

        self.assertEqual(requested, [MODE_TEMPERATURE])


if __name__ == "__main__":
    unittest.main()
