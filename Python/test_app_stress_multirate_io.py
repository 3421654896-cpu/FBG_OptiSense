from __future__ import annotations

import unittest
from unittest.mock import Mock

import app_JDSU as app


class ScanReadTests(unittest.TestCase):
    def test_v5_request_requires_v5_ack_and_map(self):
        for ack_version, map_version in ((5, 5), (4, 5), (5, 4)):
            with self.subTest(ack=ack_version, map=map_version):
                native = _native_frame(_extension(MULTIRATE_PROFILE_MAP, tuple(range(45)), version=map_version))
                port = _ScriptedPort(arm_ack=_ack(enabled=True, version=ack_version), scan_frame=native)
                if ack_version == map_version == 5:
                    schedule = app.enter_stress_multirate_and_verify_first_map(port, schedule_version=5)
                    self.assertEqual(schedule.version, 5)
                    self.assertEqual(port.writes[0][10], 5)
                else:
                    with self.assertRaises(RuntimeError):
                        app.enter_stress_multirate_and_verify_first_map(port, schedule_version=5)
        app.frames_queue.clear()

    def test_partial_tail_is_not_held_for_64_bytes(self):
        port = Mock(in_waiting=17)
        app.read_available_scan_bytes(port)
        port.read.assert_called_once_with(17)

    def test_empty_port_waits_for_first_byte(self):
        port = Mock(in_waiting=0)
        app.read_available_scan_bytes(port)
        port.read.assert_called_once_with(1)

    def test_backlog_is_drained_in_bounded_batch(self):
        port = Mock(in_waiting=100000)
        app.read_available_scan_bytes(port)
        port.read.assert_called_once_with(65536)

from frame_telemetry import (
    MULTIRATE_EXTENSION_VERSION,
    MULTIRATE_EXTENSION_VERSION_V3,
    MULTIRATE_EXTENSION_VERSION_V4,
    MULTIRATE_PROFILE_MAP,
)
from stress_multirate_protocol import (
    MultirateConfiguration,
    build_multirate_arm_command,
    decode_multirate_ack,
    validate_current_arm_ack,
    validate_current_disarm_ack,
)
from test_stress_multirate import _extension, _native_frame


def _ack(
    *,
    enabled: bool,
    version: int = MULTIRATE_EXTENSION_VERSION,
    map_period: int = 0,
    activate: int = 40,
    release: int = 20,
) -> bytes:
    frame = bytearray(20)
    frame[:4] = b"\xFF\xFF\x02\x05"
    frame[4] = int(enabled)
    frame[5] = int(map_period)
    frame[6:8] = int(activate).to_bytes(2, "big")
    frame[8:10] = int(release).to_bytes(2, "big")
    frame[10] = app.ACK_VALUE
    frame[11] = int(version)
    return bytes(frame)


class _ScriptedPort:
    def __init__(self, *, arm_ack: bytes, scan_frame: bytes | None = None):
        self._arm_ack = bytes(arm_ack)
        self._scan_frame = scan_frame
        self._rx = bytearray()
        self.writes: list[bytes] = []
        self.is_open = True

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def write(self, payload: bytes) -> int:
        data = bytes(payload)
        self.writes.append(data)
        if data[:4] == b"\xFF\xFF\x01\x08":
            self._rx.extend(self._arm_ack)
        elif (
            data[:4] == b"\xFF\xFF\x01\x02"
            and len(data) > 8
            and data[8] == 0
            and self._scan_frame is not None
        ):
            self._rx.extend(self._scan_frame)
        return len(data)

    def read(self, size: int) -> bytes:
        count = min(int(size), len(self._rx))
        result = bytes(self._rx[:count])
        del self._rx[:count]
        return result


class AppStressMultirateIoTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.frames_queue.clear(reset_counters=True)

    def test_arm_then_stress_requires_and_retains_first_v4_map(self) -> None:
        extension = _extension(
            MULTIRATE_PROFILE_MAP,
            tuple(range(45)),
            version=MULTIRATE_EXTENSION_VERSION,
        )
        native = _native_frame(extension)
        port = _ScriptedPort(arm_ack=_ack(enabled=True), scan_frame=native)

        schedule = app.enter_stress_multirate_and_verify_first_map(port)

        self.assertTrue(schedule.is_complete_map)
        self.assertEqual(port.writes[0][:4], b"\xFF\xFF\x01\x08")
        self.assertEqual(port.writes[1][:4], b"\xFF\xFF\x01\x02")
        retained = app.frames_queue.take(latest_only=False)
        self.assertIsNotNone(retained)
        assert retained is not None
        self.assertEqual(retained.payload, native)

    def test_old_ack_never_enters_stress(self) -> None:
        port = _ScriptedPort(
            arm_ack=_ack(enabled=True, version=2),
            scan_frame=b"unused",
        )

        with self.assertRaisesRegex(RuntimeError, "v4"):
            app.enter_stress_multirate_and_verify_first_map(port, timeout_s=0.02)

        self.assertEqual(len(port.writes), 1)
        self.assertEqual(port.writes[0][:4], b"\xFF\xFF\x01\x08")

    def test_full_reference_requires_exact_period_ack_and_gap_flag(self):
        config = MultirateConfiguration(map_period_frames=1)
        for period, gap, succeeds in ((0, False, False), (1, False, False), (1, True, True)):
            with self.subTest(period=period, gap=gap):
                frame = _native_frame(_extension(MULTIRATE_PROFILE_MAP, tuple(range(45)),
                                                 bandwidth_discontinuity=gap))
                port = _ScriptedPort(arm_ack=_ack(enabled=True, map_period=period), scan_frame=frame)
                if succeeds:
                    schedule = app.enter_stress_multirate_and_verify_first_map(port, config)
                    self.assertTrue(schedule.is_complete_map)
                    self.assertTrue(schedule.bandwidth_discontinuity)
                else:
                    with self.assertRaises(RuntimeError):
                        app.enter_stress_multirate_and_verify_first_map(port, config)
                    if period != 1:
                        self.assertEqual(len(port.writes), 1)

    def test_disarm_accepts_retained_scheduler_parameters(self) -> None:
        port = _ScriptedPort(
            arm_ack=_ack(
                enabled=False,
                map_period=30,
                activate=80,
                release=35,
            )
        )

        ack = app.set_stress_multirate_and_verify(
            port,
            MultirateConfiguration(enabled=False),
            timeout_s=0.02,
        )

        self.assertFalse(ack.configuration.enabled)
        self.assertEqual(ack.configuration.map_period_frames, 30)

    def test_protocol_can_explicitly_negotiate_and_validate_v3_compatibility(self):
        configuration = MultirateConfiguration()
        command = build_multirate_arm_command(
            configuration, schedule_version=MULTIRATE_EXTENSION_VERSION_V3
        )
        self.assertEqual(command[10], MULTIRATE_EXTENSION_VERSION_V3)

        decoded = decode_multirate_ack(
            _ack(
                enabled=True,
                version=MULTIRATE_EXTENSION_VERSION_V3,
            )
        )
        self.assertIs(
            validate_current_arm_ack(
                decoded,
                configuration,
                requested_version=MULTIRATE_EXTENSION_VERSION_V3,
            ),
            decoded,
        )

        with self.assertRaisesRegex(ValueError, "requested schedule version v4"):
            validate_current_arm_ack(
                decoded,
                configuration,
                requested_version=MULTIRATE_EXTENSION_VERSION_V4,
            )

        disabled = decode_multirate_ack(
            _ack(
                enabled=False,
                version=MULTIRATE_EXTENSION_VERSION_V3,
            )
        )
        self.assertIs(
            validate_current_disarm_ack(
                disabled,
                requested_version=MULTIRATE_EXTENSION_VERSION_V3,
            ),
            disabled,
        )


if __name__ == "__main__":
    unittest.main()
