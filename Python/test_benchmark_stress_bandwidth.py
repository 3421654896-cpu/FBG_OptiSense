import unittest
from dataclasses import replace
from unittest.mock import patch

from benchmark_stress_realtime import (
    _extract_multirate_ack_frames,
    _require_initial_current_map,
    _safe_disarm,
    _strict_bandwidth_acceptance,
    _summarize_schedule_freshness,
    _wait_multirate_ack,
)
from frame_telemetry import (
    MULTIRATE_EXTENSION_VERSION,
    MULTIRATE_PROFILE_MAP,
    MULTIRATE_PROFILE_SURVEY,
    MULTIRATE_PROFILE_TRACK_SINGLE,
    MULTIRATE_PROFILE_TRACK_WIDE,
    MULTIRATE_SENTINEL_LOCAL_OFFSETS,
    BoardFrameSchedule,
    BoardFrameTiming,
)
from stress_multirate_protocol import (
    CURRENT_SCHEDULE_VERSION,
    MultirateConfiguration,
    build_multirate_arm_command,
    decode_multirate_ack,
    default_profile_budgets,
)


def _fresh_indices(profile):
    if profile == MULTIRATE_PROFILE_MAP:
        return tuple(range(45))
    if profile == MULTIRATE_PROFILE_SURVEY:
        return tuple(
            segment * 5 + MULTIRATE_SENTINEL_LOCAL_OFFSETS[segment]
            for segment in range(9)
        )
    if profile == MULTIRATE_PROFILE_TRACK_WIDE:
        points = []
        for segment in range(9):
            base = segment * 5
            if segment in (0, 1):
                points.extend((base + 1, base + 2, base + 3))
            else:
                points.append(base + MULTIRATE_SENTINEL_LOCAL_OFFSETS[segment])
        return tuple(points)
    if profile == MULTIRATE_PROFILE_TRACK_SINGLE:
        points = []
        for segment in range(9):
            base = segment * 5
            if segment == 0:
                points.extend((base + 1, base + 2, base + 3))
            else:
                points.append(base + MULTIRATE_SENTINEL_LOCAL_OFFSETS[segment])
        return tuple(points)
    raise AssertionError(profile)


def _schedule(start_ms, profile=MULTIRATE_PROFILE_MAP):
    fresh = _fresh_indices(profile)
    offsets = [None] * 45
    for ordinal, point in enumerate(fresh, 1):
        offsets[point] = ordinal * 20
    return BoardFrameSchedule(
        version=MULTIRATE_EXTENSION_VERSION,
        profile=profile,
        primary_segment=(
            0
            if profile in (
                MULTIRATE_PROFILE_TRACK_WIDE,
                MULTIRATE_PROFILE_TRACK_SINGLE,
            )
            else None
        ),
        secondary_segment=(
            1 if profile == MULTIRATE_PROFILE_TRACK_WIDE else None
        ),
        fresh_indices=fresh,
        map_age_frames=0 if profile == MULTIRATE_PROFILE_MAP else 1,
        frame_start_ms=start_ms,
        sample_offset_us=tuple(offsets),
        bandwidth_discontinuity=False,
    )


def _period_summary(period_ms):
    return {
        "intervals": 10,
        "mean": period_ms,
        "median": period_ms,
        "p95": period_ms,
        "p99": period_ms,
        "min": period_ms,
        "max": period_ms,
        "mean_hz": 1000.0 / period_ms,
    }


def _ack_for(configuration, *, accepted=True):
    command = build_multirate_arm_command(configuration)
    ack = bytearray(20)
    ack[:4] = b"\xFF\xFF\x02\x05"
    ack[4:10] = command[4:10]
    ack[10] = 0x21 if accepted else 0xE1
    ack[11] = CURRENT_SCHEDULE_VERSION
    return bytes(ack)


class _FakeDevice:
    def __init__(self, *chunks):
        self.chunks = list(chunks)
        self.writes = []

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, _size):
        return self.chunks.pop(0) if self.chunks else b""

    def write(self, payload):
        self.writes.append(bytes(payload))

    def reset_input_buffer(self):
        return None


class StressPhysicalBandwidthTests(unittest.TestCase):
    def test_default_budget_exposes_track11_and_wide_without_dropping_legacy_key(self):
        budgets = {item.profile: item for item in default_profile_budgets()}
        self.assertEqual(budgets["TRACK11_SINGLE"].fresh_points, 11)
        self.assertEqual(budgets["TRACK11_SINGLE"].estimated_period_us, 29_355)
        self.assertEqual(budgets["TRACK13_WIDE"].fresh_points, 13)
        self.assertEqual(
            budgets["TRACK13"].estimated_period_us,
            budgets["TRACK13_WIDE"].estimated_period_us,
        )

    def test_profile_and_fresh_counts_are_reported_without_cached_rows(self):
        summary = _summarize_schedule_freshness(
            [
                _schedule(100, MULTIRATE_PROFILE_MAP),
                _schedule(132, MULTIRATE_PROFILE_SURVEY),
                _schedule(164, MULTIRATE_PROFILE_TRACK_WIDE),
                _schedule(196, MULTIRATE_PROFILE_TRACK_SINGLE),
            ]
        )

        self.assertEqual(
            summary["profile_distribution"],
            {
                "MAP45": 1,
                "SURVEY9": 1,
                "TRACK11_SINGLE": 1,
                "TRACK13_WIDE": 1,
            },
        )
        self.assertEqual(
            summary["fresh_point_count_distribution"],
            {"9": 1, "11": 1, "13": 1, "45": 1},
        )
        self.assertEqual(summary["current_version_frames"], 4)
        self.assertEqual(summary["invalid_or_legacy_schedule_frames"], 0)
        for grating in range(1, 10):
            item = summary["per_grating"][f"G{grating}"]
            self.assertEqual(item["fresh_events"], 4)
            self.assertEqual(item["fresh_inter_sample_period_ms"]["intervals"], 3)

    def test_fifteen_fps_is_not_accepted_as_fifteen_hz_signal_bandwidth(self):
        freshness = _summarize_schedule_freshness(
            [_schedule(start) for start in (0, 64, 128, 192)]
        )
        hard, recommendation = _strict_bandwidth_acceptance(
            signal_bandwidth_hz=15.0,
            measured_frame_hz=15.625,
            host_exact_period=_period_summary(64.0),
            board_frame_period=_period_summary(64.0),
            schedule_freshness=freshness,
        )

        self.assertFalse(hard["mean_frame_rate_meets_nyquist"])
        self.assertFalse(hard["host_p99_meets_nyquist_interval"])
        self.assertFalse(hard["board_p99_meets_nyquist_interval"])
        self.assertFalse(hard["all_gratings_mean_fresh_rate_meets_nyquist"])
        self.assertFalse(hard["all_gratings_fresh_p99_meets_nyquist_interval"])
        self.assertFalse(hard["all_gratings_fresh_max_gap_meets_nyquist_interval"])
        self.assertEqual(recommendation["target_fresh_rate_hz"], 60.0)

    def test_thirty_hz_nyquist_can_pass_while_sixty_hz_margin_is_only_reported(self):
        freshness = _summarize_schedule_freshness(
            [_schedule(start) for start in (0, 32, 64, 96)]
        )
        hard, recommendation = _strict_bandwidth_acceptance(
            signal_bandwidth_hz=15.0,
            measured_frame_hz=31.25,
            host_exact_period=_period_summary(32.0),
            board_frame_period=_period_summary(32.0),
            schedule_freshness=freshness,
        )

        self.assertTrue(all(hard.values()))
        self.assertFalse(recommendation["mean_frame_rate_reaches_4x_margin"])
        self.assertFalse(recommendation["host_p99_reaches_4x_margin"])
        self.assertFalse(
            recommendation["all_gratings_mean_fresh_rate_reaches_4x_margin"]
        )

    def test_periodic_long_map_gap_fails_p99_instead_of_being_averaged_away(self):
        freshness = _summarize_schedule_freshness(
            [_schedule(start) for start in (0, 30, 60, 112, 142, 172)]
        )
        hard, _ = _strict_bandwidth_acceptance(
            signal_bandwidth_hz=15.0,
            measured_frame_hz=30.0,
            host_exact_period=_period_summary(52.0),
            board_frame_period=_period_summary(52.0),
            schedule_freshness=freshness,
        )

        self.assertFalse(hard["host_p99_meets_nyquist_interval"])
        self.assertFalse(hard["board_p99_meets_nyquist_interval"])
        self.assertFalse(hard["all_gratings_fresh_p99_meets_nyquist_interval"])
        self.assertFalse(hard["all_gratings_fresh_max_gap_meets_nyquist_interval"])

    def test_initial_timed_frame_must_be_current_version_map45(self):
        timing = BoardFrameTiming(sequence=7, acquisition_duration_us=30_000)
        _require_initial_current_map(_schedule(10), timing)
        with self.assertRaisesRegex(RuntimeError, "MAP45"):
            _require_initial_current_map(
                _schedule(10, MULTIRATE_PROFILE_SURVEY), timing
            )
        with self.assertRaisesRegex(RuntimeError, "\u65f6\u5e8f"):
            _require_initial_current_map(_schedule(10), None)

    def test_ack_extractor_and_decoder_accept_only_exact_0205_contract(self):
        configuration = MultirateConfiguration()
        ack = _ack_for(configuration)
        buffer = bytearray(b"noise\xFF\xFF\x02\x04" + bytes(16) + ack)

        frames = _extract_multirate_ack_frames(buffer)
        self.assertEqual(frames, [ack])
        decoded = decode_multirate_ack(frames[0])
        self.assertIsNotNone(decoded)
        self.assertTrue(decoded.accepted)
        self.assertEqual(decoded.configuration, configuration)

    def test_wait_ack_rejects_a_different_echoed_configuration(self):
        expected = MultirateConfiguration(map_period_frames=30)
        returned = MultirateConfiguration(map_period_frames=31)
        device = _FakeDevice(_ack_for(returned))
        with self.assertRaisesRegex(RuntimeError, "ACK\u672a\u901a\u8fc7"):
            _wait_multirate_ack(device, expected, timeout_s=0.1)

    def test_old_ack_without_v3_echo_is_diagnostic_only_and_rejected(self):
        expected = MultirateConfiguration()
        old_ack = bytearray(_ack_for(expected))
        old_ack[11] = 0
        device = _FakeDevice(bytes(old_ack))
        with self.assertRaisesRegex(RuntimeError, "ACK\u672a\u901a\u8fc7"):
            _wait_multirate_ack(device, expected, timeout_s=0.1)

    def test_forced_map_bandwidth_gap_flag_is_a_hard_failure(self):
        schedules = [_schedule(start) for start in (0, 32, 64, 96)]
        schedules[2] = replace(schedules[2], bandwidth_discontinuity=True)
        freshness = _summarize_schedule_freshness(schedules)
        hard, _ = _strict_bandwidth_acceptance(
            signal_bandwidth_hz=15.0,
            measured_frame_hz=31.25,
            host_exact_period=_period_summary(32.0),
            board_frame_period=_period_summary(32.0),
            schedule_freshness=freshness,
        )

        self.assertEqual(freshness["bandwidth_discontinuity_frames"], 1)
        self.assertFalse(hard["no_reported_bandwidth_discontinuity"])

    def test_safe_disarm_is_extra_first_and_requires_disabled_ack(self):
        # Firmware intentionally preserves the preceding period/thresholds
        # while proving enabled=false; disarm must not mistake that for a bad
        # echo of the all-default disarm request.
        disabled = MultirateConfiguration(
            enabled=False,
            map_period_frames=30,
            activate_threshold_codes=55,
            release_threshold_codes=21,
        )
        device = _FakeDevice(_ack_for(disabled))
        with patch("benchmark_stress_realtime.time.sleep"):
            self.assertTrue(_safe_disarm(device))

        self.assertEqual(device.writes[0][:4], b"\xFF\xFF\x01\x02")
        self.assertEqual(device.writes[0][8], 2)
        self.assertEqual(device.writes[1][:4], b"\xFF\xFF\x01\x08")
        self.assertEqual(device.writes[1][4], 0)


if __name__ == "__main__":
    unittest.main()
