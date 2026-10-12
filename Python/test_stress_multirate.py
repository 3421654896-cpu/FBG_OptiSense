from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from fbg_remote_protocol import TelemetryFrame, decode_telemetry, encode_telemetry
from frame_telemetry import (
    BOARD_STATUS_V1_FRAME_START_LENGTH,
    BOARD_STATUS_V1_MULTIRATE_LENGTH,
    BOARD_STATUS_V1_MULTIRATE_V1_LENGTH,
    MULTIRATE_BANDWIDTH_GAP_FLAG,
    MULTIRATE_EXTENSION_VERSION,
    MULTIRATE_EXTENSION_VERSION_V1,
    MULTIRATE_EXTENSION_VERSION_V2,
    MULTIRATE_PROFILE_MAP,
    MULTIRATE_PROFILE_SURVEY,
    MULTIRATE_PROFILE_TRACK,
    current_frame_fresh_indices,
    decode_board_frame_start_ms,
    decode_board_frame_schedule,
    extract_raw_scan_frames,
)
from stress_multirate_protocol import (
    DEFAULT_MAP_PERIOD,
    MODE_EXTRA,
    MODE_STRESS,
    MultirateConfiguration,
    build_multirate_arm_command,
    build_multirate_disarm_command,
    build_safe_arm_command_sequence,
    build_safe_disarm_command_sequence,
    decode_multirate_ack,
    default_profile_budgets,
    estimate_survey_peak_shifts,
    select_current_samples,
    select_current_timed_samples,
    validate_current_arm_ack,
    validate_current_disarm_ack,
)
from stress_multirate_reference import (
    BASELINE_EMA_SHIFT,
    STARTUP_MAX_FRAMES,
    STARTUP_STABLE_FRAMES,
    MAX_MAP_DEFERRAL_FRAMES,
    SENTINEL_LOCAL_OFFSETS,
    BaselineStatus,
    Profile,
    Scheduler,
    flank_indices,
    sentinel_index,
    track_indices,
    triplet_indices,
)


ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "JDSU" / "Core" / "Src" / "ms5614t.c"
SCHEDULER = ROOT / "JDSU" / "Core" / "Src" / "stress_multirate.c"
PROJECT = ROOT / "JDSU" / "MDK-ARM" / "JDSU.uvprojx"
CERTIFIED_SELECTION = (
    ROOT
    / "Python"
    / "artifacts"
    / "ch1_measured_template_selection_v4_certified_20260905.json"
)


def _extension(
    profile: int,
    indices: tuple[int, ...],
    *,
    primary: int = 0xFF,
    secondary: int = 0xFF,
    map_age: int = 0,
    start_ms: int = 1234,
    version: int = MULTIRATE_EXTENSION_VERSION,
    acquisition_us: int = 100_000,
    bandwidth_discontinuity: bool = False,
) -> bytes:
    length = (
        BOARD_STATUS_V1_MULTIRATE_LENGTH
        if version >= MULTIRATE_EXTENSION_VERSION_V2
        else BOARD_STATUS_V1_MULTIRATE_V1_LENGTH
    )
    payload = bytearray(length)
    payload[32:36] = int(acquisition_us).to_bytes(4, "big")
    payload[36] = version
    payload[37] = profile
    payload[38] = primary
    payload[39] = secondary
    payload[40] = len(indices)
    payload[41] = map_age | (
        MULTIRATE_BANDWIDTH_GAP_FLAG if bandwidth_discontinuity else 0
    )
    for index in indices:
        payload[42 + index // 8] |= 1 << (index & 7)
    payload[48:52] = int(start_ms).to_bytes(4, "big")
    if version >= MULTIRATE_EXTENSION_VERSION_V2:
        payload[52:] = b"\xFF" * (len(payload) - 52)
        for ordinal, index in enumerate(indices, start=1):
            ticks = ordinal * 100
            payload[52 + 2 * index : 54 + 2 * index] = ticks.to_bytes(2, "big")
    return bytes(payload)


def _native_frame(extension: bytes) -> bytes:
    point_count = 45
    frame = bytearray(b"\xEE\xEE" + point_count.to_bytes(2, "big"))
    frame.extend(bytes(point_count * 8))
    frame.extend(b"\xAB\x00\x00\x00\x00")
    frame.extend(bytes(4))
    frame.extend(bytes(4))
    frame.extend((0xB5, 0x4D, 0x01, len(extension)))
    frame.extend(extension)
    frame.extend(b"\xFF\xEF")
    return bytes(frame)


class SchedulerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = [1000 + index for index in range(45)]

    def _complete(self, scheduler: Scheduler, values: list[int]) -> None:
        plan = scheduler.begin_frame()
        scheduler.record_plan(values[index] for index in plan.fresh_indices)
        scheduler.end_frame()

    def _learn_startup(self, scheduler: Scheduler, values: list[int] | None = None) -> None:
        source = self.baseline if values is None else values
        for _ in range(STARTUP_STABLE_FRAMES):
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.SURVEY)
            scheduler.record_plan(source[index] for index in plan.fresh_indices)
            scheduler.end_frame()
        self.assertIs(scheduler.baseline_status, BaselineStatus.READY)

    def test_sparse_startup_baseline_absorbs_map_to_survey_offset_only(self) -> None:
        scheduler = Scheduler()
        self._complete(scheduler, self.baseline)
        sparse_steady = list(self.baseline)
        for segment in range(9):
            sparse_steady[sentinel_index(segment)] += 120

        # The first sparse frame is far beyond the contact threshold, but it
        # is a repeatable path-state offset rather than a press.  It must not
        # latch TRACK while startup learning is explicit and bounded.
        for _ in range(STARTUP_STABLE_FRAMES + 1):
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.SURVEY)
            scheduler.record_plan(
                sparse_steady[index] for index in plan.fresh_indices
            )
            scheduler.end_frame()
            self.assertFalse(scheduler.activity_latched)
        self.assertIs(scheduler.baseline_status, BaselineStatus.READY)

        # A real change after qualification remains fully detectable.
        contact = list(sparse_steady)
        contact[sentinel_index(4)] += 100
        contact[sentinel_index(7)] += 80
        plan = scheduler.begin_frame()
        scheduler.record_plan(contact[index] for index in plan.fresh_indices)
        scheduler.end_frame()
        self.assertTrue(scheduler.activity_latched)
        self.assertEqual(scheduler.active_pair, (4, 7))

    def test_unstable_startup_grace_has_a_hard_bound_and_recovers(self) -> None:
        scheduler = Scheduler()
        self._complete(scheduler, self.baseline)
        for frame in range(STARTUP_MAX_FRAMES):
            unstable = list(self.baseline)
            offset = 100 if frame % 2 == 0 else 0
            for segment in range(9):
                unstable[sentinel_index(segment)] += offset
            plan = scheduler.begin_frame()
            scheduler.record_plan(unstable[index] for index in plan.fresh_indices)
            scheduler.end_frame()
            self.assertFalse(scheduler.activity_latched)
            if frame + 1 < STARTUP_MAX_FRAMES:
                self.assertIs(scheduler.baseline_status, BaselineStatus.LEARNING)

        self.assertIs(scheduler.baseline_status, BaselineStatus.FORCED_READY)
        self.assertEqual(scheduler.startup_baseline_frames, STARTUP_MAX_FRAMES)

        # Forced-ready is observable, and normal quiet qualification upgrades
        # it without reopening an unbounded learning window.
        for _ in range(4):
            plan = scheduler.begin_frame()
            scheduler.record_plan(self.baseline[index] for index in plan.fresh_indices)
            scheduler.end_frame()
        self.assertIs(scheduler.baseline_status, BaselineStatus.READY)

    def test_hard_bound_keeps_learning_on_survey_then_arms_next_frame(self) -> None:
        scheduler = Scheduler(map_period_frames=0)
        self._complete(scheduler, self.baseline)

        for frame in range(STARTUP_MAX_FRAMES):
            unstable = list(self.baseline)
            unstable[sentinel_index(3)] += 100 if frame % 2 == 0 else 0
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.SURVEY)
            scheduler.record_plan(unstable[index] for index in plan.fresh_indices)
            scheduler.end_frame()
            self.assertFalse(scheduler.activity_latched)

        self.assertIs(scheduler.baseline_status, BaselineStatus.FORCED_READY)

        # The bounded grace is not extended by another sparse frame: contact
        # detection is live immediately after the 32nd learning-only SURVEY.
        contact = list(self.baseline)
        contact[sentinel_index(4)] += 100
        plan = scheduler.begin_frame()
        self.assertIs(plan.profile, Profile.SURVEY)
        scheduler.record_plan(contact[index] for index in plan.fresh_indices)
        scheduler.end_frame()
        self.assertTrue(scheduler.activity_latched)

    def test_captured_static_g4_settling_does_not_false_latch_track(self) -> None:
        """Replay the CH1/G4 sentinel values from the static USB trace.

        Four quiet frames occur briefly before the laser/ADC path has really
        settled.  The old 4/12-frame qualifier armed on that quiet pocket and
        then interpreted 292 -> 337 codes as contact.  Requiring the longer
        bounded qualification keeps this no-contact trace in SURVEY while a
        subsequent real-size step is still detected immediately.
        """

        scheduler = Scheduler(map_period_frames=0)
        initial = list(self.baseline)
        initial[sentinel_index(3)] = 114
        self._complete(scheduler, initial)

        captured_g4 = (
            114, 124, 155, 230, 279, 354, 359, 360,
            324, 308, 306, 292, 273, 277, 281, 315,
            309, 300, 329, 304, 337, 273, 268, 278,
            279, 282, 267, 266, 277, 275, 276, 274,
        )

        for code in captured_g4:
            values = list(initial)
            values[sentinel_index(3)] = code
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.SURVEY)
            scheduler.record_plan(values[index] for index in plan.fresh_indices)
            scheduler.end_frame()
            self.assertFalse(scheduler.activity_latched)

        self.assertIs(scheduler.baseline_status, BaselineStatus.READY)

        contact = list(initial)
        contact[sentinel_index(3)] = captured_g4[-1] + 100
        plan = scheduler.begin_frame()
        scheduler.record_plan(contact[index] for index in plan.fresh_indices)
        scheduler.end_frame()
        self.assertTrue(scheduler.activity_latched)

    def test_initial_frame_is_map_and_initializes_every_cache_slot(self) -> None:
        scheduler = Scheduler()
        plan = scheduler.begin_frame()
        self.assertIs(plan.profile, Profile.MAP)
        self.assertEqual(plan.fresh_indices, tuple(range(45)))
        with self.assertRaises(RuntimeError):
            scheduler.cached_frame()
        scheduler.record_plan(self.baseline)
        scheduler.end_frame()
        self.assertEqual(scheduler.cached_frame(), tuple(self.baseline))

    def test_survey9_covers_one_certified_sentinel_for_every_peak(self) -> None:
        scheduler = Scheduler()
        self._complete(scheduler, self.baseline)
        plan = scheduler.begin_frame()
        self.assertIs(plan.profile, Profile.SURVEY)
        self.assertEqual(len(plan.fresh_indices), 9)
        for segment in range(9):
            self.assertEqual(
                tuple(index for index in plan.fresh_indices if index // 5 == segment),
                (sentinel_index(segment),),
            )

    def test_confirmed_activity_switches_to_track13_full_spatial_watch(self) -> None:
        scheduler = Scheduler()
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        survey = scheduler.begin_frame()
        changed = list(self.baseline)
        changed[sentinel_index(4)] += 120
        changed[sentinel_index(7)] += 80
        scheduler.record_plan(changed[index] for index in survey.fresh_indices)
        scheduler.end_frame()
        track = scheduler.begin_frame()
        self.assertIs(track.profile, Profile.TRACK)
        self.assertEqual((track.primary_segment, track.secondary_segment), (4, 7))
        self.assertEqual(track.fresh_indices, track_indices(4, 7))
        self.assertEqual(len(track.fresh_indices), 13)
        for segment in range(9):
            segment_points = tuple(
                index for index in track.fresh_indices if index // 5 == segment
            )
            expected = (
                triplet_indices(segment)
                if segment in (4, 7)
                else (sentinel_index(segment),)
            )
            self.assertEqual(segment_points, expected)

    def test_periodic_map_occurs_exactly_every_thirtieth_frame(self) -> None:
        scheduler = Scheduler(map_period_frames=30)
        profiles = []
        for _ in range(61):
            plan = scheduler.begin_frame()
            profiles.append(plan.profile)
            scheduler.record_plan(self.baseline[index] for index in plan.fresh_indices)
            scheduler.end_frame()
        self.assertEqual(
            [index for index, profile in enumerate(profiles) if profile is Profile.MAP],
            [0, 30, 60],
        )

    def test_track_sentinel_can_promote_a_new_spatial_roi(self) -> None:
        scheduler = Scheduler(map_period_frames=100)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        survey = scheduler.begin_frame()
        active = list(self.baseline)
        active[sentinel_index(4)] += 100
        active[sentinel_index(7)] += 80
        scheduler.record_plan(active[index] for index in survey.fresh_indices)
        scheduler.end_frame()

        track = scheduler.begin_frame()
        self.assertEqual((track.primary_segment, track.secondary_segment), (4, 7))
        moved = list(self.baseline)
        moved[sentinel_index(2)] += 140
        for index in triplet_indices(4):
            moved[index] += 60
        scheduler.record_plan(moved[index] for index in track.fresh_indices)
        scheduler.end_frame()
        promoted = scheduler.begin_frame()
        self.assertEqual((promoted.primary_segment, promoted.secondary_segment), (2, 4))

    def test_quiet_baseline_ema_updates_fresh_rows_only(self) -> None:
        scheduler = Scheduler(map_period_frames=100)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        original_q6 = tuple(scheduler.baseline_q6 or ())
        quiet_drift = [value + 16 for value in self.baseline]
        for _ in range(4):
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.SURVEY)
            scheduler.record_plan(quiet_drift[index] for index in plan.fresh_indices)
            scheduler.end_frame()
        assert scheduler.baseline_q6 is not None
        self.assertEqual(
            scheduler.baseline_q6[sentinel_index(0)],
            original_q6[sentinel_index(0)] + 16,
        )
        self.assertEqual(
            scheduler.baseline_q6[2], original_q6[2]
        )
        self.assertEqual(BASELINE_EMA_SHIFT, 6)

    def test_contact_frames_never_train_the_quiet_baseline(self) -> None:
        scheduler = Scheduler(map_period_frames=100)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        original_q6 = tuple(scheduler.baseline_q6 or ())
        survey = scheduler.begin_frame()
        contact = list(self.baseline)
        contact[sentinel_index(3)] += 100
        scheduler.record_plan(contact[index] for index in survey.fresh_indices)
        scheduler.end_frame()
        self.assertEqual(tuple(scheduler.baseline_q6 or ()), original_q6)

    def test_disabled_policy_is_real_map_not_cached_map(self) -> None:
        scheduler = Scheduler(enabled=False)
        for _ in range(3):
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.MAP)
            self.assertEqual(len(plan.fresh_indices), 45)
            scheduler.record_plan(self.baseline)
            scheduler.end_frame()

    def test_new_session_discards_every_cached_value_and_forces_map(self) -> None:
        scheduler = Scheduler()
        self._complete(scheduler, self.baseline)
        self.assertEqual(scheduler.cached_frame(), tuple(self.baseline))
        scheduler.start_session()
        with self.assertRaises(RuntimeError):
            scheduler.cached_frame()
        self.assertIs(scheduler.begin_frame().profile, Profile.MAP)

    def test_track_releases_only_after_four_consecutive_quiet_frames(self) -> None:
        scheduler = Scheduler(map_period_frames=100)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        survey = scheduler.begin_frame()
        active = list(self.baseline)
        active[sentinel_index(4)] += 100
        active[sentinel_index(7)] += 80
        scheduler.record_plan(active[index] for index in survey.fresh_indices)
        scheduler.end_frame()
        for quiet_frame in range(4):
            plan = scheduler.begin_frame()
            self.assertIs(plan.profile, Profile.TRACK)
            scheduler.record_plan(self.baseline[index] for index in plan.fresh_indices)
            scheduler.end_frame()
            self.assertEqual(scheduler.activity_latched, quiet_frame < 3)
        self.assertIs(scheduler.begin_frame().profile, Profile.SURVEY)

    def test_active_contact_defers_routine_map_until_maximum_age(self) -> None:
        scheduler = Scheduler(map_period_frames=3)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        survey = scheduler.begin_frame()
        active = list(self.baseline)
        active[sentinel_index(2)] += 120
        active[sentinel_index(6)] += 80
        scheduler.record_plan(active[index] for index in survey.fresh_indices)
        scheduler.end_frame()

        while scheduler.frames_since_map < MAX_MAP_DEFERRAL_FRAMES - 1:
            track = scheduler.begin_frame()
            self.assertIs(track.profile, Profile.TRACK)
            self.assertFalse(track.bandwidth_discontinuity)
            scheduler.record_plan(active[index] for index in track.fresh_indices)
            scheduler.end_frame()

        forced = scheduler.begin_frame()
        self.assertIs(forced.profile, Profile.MAP)
        self.assertTrue(forced.bandwidth_discontinuity)
        self.assertEqual(forced.map_age_frames, 0x7F if MAX_MAP_DEFERRAL_FRAMES > 128 else MAX_MAP_DEFERRAL_FRAMES - 1)

    def test_continuous_period_zero_never_inserts_an_automatic_map(self) -> None:
        scheduler = Scheduler(map_period_frames=0)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        survey = scheduler.begin_frame()
        active = list(self.baseline)
        active[sentinel_index(2)] += 120
        active[sentinel_index(6)] += 80
        scheduler.record_plan(active[index] for index in survey.fresh_indices)
        scheduler.end_frame()
        for _ in range(MAX_MAP_DEFERRAL_FRAMES + 5):
            track = scheduler.begin_frame()
            self.assertIs(track.profile, Profile.TRACK)
            self.assertFalse(track.bandwidth_discontinuity)
            scheduler.record_plan(active[index] for index in track.fresh_indices)
            scheduler.end_frame()

    def test_explicit_force_map_marks_gap_only_while_contact_is_active(self) -> None:
        scheduler = Scheduler(map_period_frames=100)
        self._complete(scheduler, self.baseline)
        self._learn_startup(scheduler)
        scheduler.force_map()
        quiet_map = scheduler.begin_frame()
        self.assertIs(quiet_map.profile, Profile.MAP)
        self.assertFalse(quiet_map.bandwidth_discontinuity)
        scheduler.record_plan(self.baseline)
        scheduler.end_frame()

        survey = scheduler.begin_frame()
        active = list(self.baseline)
        active[sentinel_index(1)] += 120
        active[sentinel_index(8)] += 80
        scheduler.record_plan(active[index] for index in survey.fresh_indices)
        scheduler.end_frame()
        scheduler.force_map()
        active_map = scheduler.begin_frame()
        self.assertIs(active_map.profile, Profile.MAP)
        self.assertTrue(active_map.bandwidth_discontinuity)


class ProtocolFreshnessTests(unittest.TestCase):
    def test_map_period_zero_is_dynamic_and_one_is_explicit_reference(self) -> None:
        self.assertEqual(MultirateConfiguration().validate().map_period_frames, 0)
        self.assertEqual(MultirateConfiguration(map_period_frames=1).validate().map_period_frames, 1)
        for invalid in (2, 101):
            with self.assertRaises(ValueError):
                MultirateConfiguration(map_period_frames=invalid).validate()

    def test_command_and_ack_round_trip(self) -> None:
        config = MultirateConfiguration()
        command = build_multirate_arm_command(config)
        self.assertEqual(len(command), 808)
        self.assertEqual(DEFAULT_MAP_PERIOD, 0)
        self.assertEqual(
            command[:11],
            b"\xFF\xFF\x01\x08\x01\x00\x00\x28\x00\x14\x04",
        )
        ack = bytearray(20)
        ack[:4] = b"\xFF\xFF\x02\x05"
        ack[4:10] = command[4:10]
        ack[10] = 0x21
        ack[11] = MULTIRATE_EXTENSION_VERSION
        decoded = decode_multirate_ack(bytes(ack))
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertTrue(decoded.accepted)
        self.assertEqual(decoded.configuration, config)
        self.assertEqual(decoded.schedule_version, MULTIRATE_EXTENSION_VERSION)
        self.assertIs(validate_current_arm_ack(decoded, config), decoded)

        ack[11] = 0  # Legacy ACK is visible, but cannot safely arm v3.
        legacy = decode_multirate_ack(bytes(ack))
        self.assertIsNotNone(legacy)
        assert legacy is not None
        self.assertIsNone(legacy.schedule_version)
        with self.assertRaisesRegex(ValueError, "version v4"):
            validate_current_arm_ack(legacy, config)

        ack[4] = 0
        ack[5] = 30  # Firmware retains the prior period while disarmed.
        ack[11] = MULTIRATE_EXTENSION_VERSION
        disarmed = decode_multirate_ack(bytes(ack))
        self.assertIsNotNone(disarmed)
        self.assertIs(validate_current_disarm_ack(disarmed), disarmed)

    def test_safe_arm_and_disarm_command_order(self) -> None:
        enter_extra, arm, enter_stress = build_safe_arm_command_sequence()
        self.assertEqual(enter_extra[:4], b"\xFF\xFF\x01\x02")
        self.assertEqual(enter_extra[8], MODE_EXTRA)
        self.assertEqual(arm, build_multirate_arm_command())
        self.assertEqual(enter_stress[8], MODE_STRESS)
        leave_stress, disarm = build_safe_disarm_command_sequence()
        self.assertEqual(leave_stress, enter_extra)
        self.assertEqual(disarm, build_multirate_disarm_command())
        self.assertEqual(disarm[4], 0)

    def test_plain_map40_frame_start_is_not_mistaken_for_a_schedule(self) -> None:
        extension = bytearray(BOARD_STATUS_V1_FRAME_START_LENGTH)
        extension[36:40] = (0xFFFF_FFFE).to_bytes(4, "big")
        self.assertIsNone(decode_board_frame_schedule(bytes(extension)))
        self.assertEqual(decode_board_frame_start_ms(bytes(extension)), 0xFFFF_FFFE)

    def test_v3_extension_has_exact_compact_length(self) -> None:
        extension = _extension(MULTIRATE_PROFILE_MAP, tuple(range(45)))
        self.assertEqual(len(extension), 142)
        self.assertEqual(extension[36], MULTIRATE_EXTENSION_VERSION)
        self.assertEqual(len(extension[52:142]), 45 * 2)

    def test_legacy_extension_is_full_map(self) -> None:
        self.assertEqual(
            current_frame_fresh_indices(bytes(36), point_count=45),
            tuple(range(45)),
        )

    def test_track_bitmap_strictly_excludes_cached_rows(self) -> None:
        indices = track_indices(4, 7)
        extension = _extension(
            MULTIRATE_PROFILE_TRACK, indices, primary=4, secondary=7
        )
        schedule = decode_board_frame_schedule(extension)
        self.assertEqual(schedule.fresh_indices, indices)
        selected = select_current_samples(tuple(range(45)), extension=extension)
        self.assertEqual(selected, tuple((index, index) for index in indices))
        timed = select_current_timed_samples(tuple(range(45)), extension=extension)
        self.assertEqual(
            timed,
            tuple((index, index, (ordinal + 1) * 200)
                  for ordinal, index in enumerate(indices)),
        )

    def test_track13_rejects_wrong_sentinel_or_missing_peak(self) -> None:
        indices = list(track_indices(4, 7))
        indices.remove(sentinel_index(2))
        indices.append(2 * 5 + (1 if SENTINEL_LOCAL_OFFSETS[2] == 3 else 3))
        extension = _extension(
            MULTIRATE_PROFILE_TRACK,
            tuple(sorted(indices)),
            primary=4,
            secondary=7,
        )
        with self.assertRaisesRegex(ValueError, "ROI/sentinel"):
            decode_board_frame_schedule(extension)

    def test_old_52_byte_schedule_keeps_freshness_but_has_no_fake_timing(self) -> None:
        indices = triplet_indices(2) + triplet_indices(5)
        extension = _extension(
            MULTIRATE_PROFILE_TRACK,
            indices,
            primary=2,
            secondary=5,
            version=MULTIRATE_EXTENSION_VERSION_V1,
        )
        schedule = decode_board_frame_schedule(extension)
        self.assertEqual(schedule.version, MULTIRATE_EXTENSION_VERSION_V1)
        self.assertEqual(schedule.fresh_indices, indices)
        self.assertFalse(schedule.has_point_timing)
        with self.assertRaises(ValueError):
            select_current_timed_samples(tuple(range(45)), extension=extension)

    def test_v2_survey18_and_track13_remain_backward_compatible(self) -> None:
        survey = tuple(
            point for segment in range(9) for point in flank_indices(segment)
        )
        survey_schedule = decode_board_frame_schedule(
            _extension(
                MULTIRATE_PROFILE_SURVEY,
                survey,
                version=MULTIRATE_EXTENSION_VERSION_V2,
            )
        )
        self.assertEqual(survey_schedule.fresh_indices, survey)
        self.assertFalse(survey_schedule.bandwidth_discontinuity)

        track = track_indices(2, 5)
        track_schedule = decode_board_frame_schedule(
            _extension(
                MULTIRATE_PROFILE_TRACK,
                track,
                primary=2,
                secondary=5,
                version=MULTIRATE_EXTENSION_VERSION_V2,
            )
        )
        self.assertEqual(track_schedule.fresh_indices, track)
        self.assertTrue(track_schedule.has_point_timing)

    def test_v2_timing_requires_fresh_values_only_and_strict_order(self) -> None:
        survey = tuple(point for segment in range(9) for point in flank_indices(segment))
        extension = bytearray(
            _extension(
                MULTIRATE_PROFILE_SURVEY,
                survey,
                version=MULTIRATE_EXTENSION_VERSION_V2,
            )
        )

        # A cached row cannot claim a current-frame acquisition time.
        extension[52:54] = (1).to_bytes(2, "big")
        with self.assertRaises(ValueError):
            decode_board_frame_schedule(bytes(extension))

        extension = bytearray(
            _extension(
                MULTIRATE_PROFILE_SURVEY,
                survey,
                version=MULTIRATE_EXTENSION_VERSION_V2,
            )
        )
        first = survey[0]
        extension[52 + 2 * first : 54 + 2 * first] = b"\xFF\xFF"
        with self.assertRaises(ValueError):
            decode_board_frame_schedule(bytes(extension))

        # A uint16 wrap inside one frame is never silently unwrapped.
        extension = bytearray(
            _extension(
                MULTIRATE_PROFILE_SURVEY,
                survey,
                acquisition_us=131_068,
                version=MULTIRATE_EXTENSION_VERSION_V2,
            )
        )
        first, second = survey[:2]
        extension[52 + 2 * first : 54 + 2 * first] = (0xFFFE).to_bytes(2, "big")
        extension[52 + 2 * second : 54 + 2 * second] = (0).to_bytes(2, "big")
        with self.assertRaises(ValueError):
            decode_board_frame_schedule(bytes(extension))

    def test_v2_timestamp_upper_limit_and_frame_tick_wrap_are_explicit(self) -> None:
        indices = tuple(range(45))
        extension = bytearray(
            _extension(
                MULTIRATE_PROFILE_MAP,
                indices,
                acquisition_us=131_068,
                start_ms=0xFFFF_FFFE,
                version=MULTIRATE_EXTENSION_VERSION_V2,
            )
        )
        extension[52 + 2 * 44 : 54 + 2 * 44] = (0xFFFE).to_bytes(2, "big")
        schedule = decode_board_frame_schedule(bytes(extension))
        self.assertEqual(schedule.frame_start_ms, 0xFFFF_FFFE)
        self.assertEqual(schedule.sample_offset_us[44], 131_068)

    def test_dual_flanks_recover_small_shift_and_common_power_change(self) -> None:
        survey = tuple(point for segment in range(9) for point in flank_indices(segment))
        extension = _extension(
            MULTIRATE_PROFILE_SURVEY,
            survey,
            version=MULTIRATE_EXTENSION_VERSION_V2,
        )
        wavelengths = [1525.0 + 0.02 * index for index in range(45)]
        baseline: list[float] = []
        for _segment in range(9):
            baseline.extend((100.0, 300.0, 500.0, 300.0, 100.0))
        current = list(baseline)
        expected_shift = 0.003
        expected_common = 7.0
        for segment in range(9):
            base = segment * 5
            left, right = base + 1, base + 3
            slope_left = (baseline[base + 2] - baseline[base]) / 0.04
            slope_right = (baseline[base + 4] - baseline[base + 2]) / 0.04
            current[left] += -slope_left * expected_shift + expected_common
            current[right] += -slope_right * expected_shift + expected_common
        estimates = estimate_survey_peak_shifts(
            wavelengths, baseline, current, extension=extension
        )
        self.assertEqual(tuple(item.segment for item in estimates), tuple(range(9)))
        for estimate in estimates:
            self.assertAlmostEqual(estimate.shift_nm, expected_shift, places=9)
            self.assertAlmostEqual(estimate.common_mode_codes, expected_common, places=9)

    def test_v3_survey9_does_not_claim_dual_flank_shift(self) -> None:
        survey = tuple(sentinel_index(segment) for segment in range(9))
        extension = _extension(MULTIRATE_PROFILE_SURVEY, survey)
        values = [100.0] * 45
        wavelengths = [1525.0 + 0.02 * index for index in range(45)]
        with self.assertRaisesRegex(ValueError, "v2 SURVEY18"):
            estimate_survey_peak_shifts(
                wavelengths,
                values,
                values,
                extension=extension,
            )

    def test_forced_active_map_gap_flag_is_explicit_and_not_dynamic_data(self) -> None:
        extension = _extension(
            MULTIRATE_PROFILE_MAP,
            tuple(range(45)),
            map_age=119,
            bandwidth_discontinuity=True,
        )
        schedule = decode_board_frame_schedule(extension)
        self.assertEqual(schedule.map_age_frames, 119)
        self.assertTrue(schedule.bandwidth_discontinuity)
        with self.assertRaisesRegex(ValueError, "bandwidth discontinuity"):
            select_current_timed_samples(tuple(range(45)), extension=extension)

        survey = tuple(sentinel_index(segment) for segment in range(9))
        malformed = _extension(
            MULTIRATE_PROFILE_SURVEY,
            survey,
            bandwidth_discontinuity=True,
        )
        with self.assertRaisesRegex(ValueError, "only on a forced MAP"):
            decode_board_frame_schedule(malformed)

    def test_malformed_bitmap_never_falls_back_to_full_fresh(self) -> None:
        extension = bytearray(
            _extension(
                MULTIRATE_PROFILE_SURVEY,
                tuple(sentinel_index(segment) for segment in range(9)),
            )
        )
        extension[40] = 17
        with self.assertRaises(ValueError):
            current_frame_fresh_indices(bytes(extension), point_count=45)

    def test_future_length_native_parser_and_remote_crc_remain_compatible(self) -> None:
        survey = tuple(sentinel_index(segment) for segment in range(9))
        extension = _extension(MULTIRATE_PROFILE_SURVEY, survey, map_age=3)
        raw = _native_frame(extension)
        buffer = bytearray(b"noise" + raw)
        self.assertEqual(extract_raw_scan_frames(buffer), [raw])
        self.assertEqual(buffer, bytearray())

        telemetry = TelemetryFrame(
            mode=0,
            flags=0,
            boot_id=1,
            seq=2,
            uptime_ms=3,
            temperature_mC=25_000,
            table_crc32=4,
            gainmask0=0,
            gainmask1=0,
            channel_mask=0x02,
            samples=tuple((index,) for index in range(45)),
        )
        self.assertEqual(decode_telemetry(encode_telemetry(telemetry)), telemetry)


class TimingAndFirmwareContractTests(unittest.TestCase):
    def test_theoretical_profiles_show_track_and_survey9_nominal_margin(self) -> None:
        budgets = {budget.profile: budget for budget in default_profile_budgets()}
        self.assertEqual(budgets["MAP45"].estimated_period_us, 52_305)
        self.assertEqual(budgets["SURVEY9"].estimated_period_us, 28_005)
        self.assertEqual(budgets["TRACK13"].estimated_period_us, 30_705)
        self.assertGreater(budgets["TRACK13"].estimated_rate_hz, 30.0)
        self.assertLess(budgets["TRACK13"].estimated_rate_hz, 35.0)
        self.assertGreater(budgets["SURVEY9"].estimated_rate_hz, 35.0)
        self.assertLess(budgets["MAP45"].estimated_rate_hz, 25.0)
        # No offline estimate is allowed to certify the physical 15 Hz goal;
        # deployment must use the board's p99 acquisition telemetry.

    def test_certified_sentinel_offsets_are_slope_selected_and_reliable(self) -> None:
        report = json.loads(CERTIFIED_SELECTION.read_text(encoding="utf-8"))
        peaks = report["stress"]["peaks"]
        self.assertEqual(len(peaks), 9)
        derived = []
        for peak in peaks:
            weights = peak["selected_reliability_weight"]
            slopes = peak["selected_template_slope_v_per_nm"]
            self.assertGreaterEqual(weights[1], 0.9)
            self.assertGreaterEqual(weights[3], 0.9)
            derived.append(1 if abs(slopes[1]) >= abs(slopes[3]) else 3)
        self.assertEqual(tuple(derived), SENTINEL_LOCAL_OFFSETS)

    def test_firmware_replays_full_dac_path_before_sampling_decision(self) -> None:
        source = FIRMWARE.read_text(encoding="utf-8")
        loop = source.index("for (i = tableStartIndex")
        hidden = source.index(
            "activeScanPointIndex == STRESS_PATH_PRECONDITION_POINT_INDEX", loop
        )
        normal_apply = source.index(
            "PI11210_ApplyStressRowCalibrationOrder", hidden
        )
        sample_decision = source.index("StressMultirate_ShouldSample", normal_apply)
        self.assertLess(hidden, normal_apply)
        self.assertLess(normal_apply, sample_decision)
        self.assertIn("Stress_Path_Precondition_DAC", source[hidden:normal_apply])

    def test_firmware_is_opt_in_ch1_only_and_exports_freshness(self) -> None:
        source = FIRMWARE.read_text(encoding="utf-8")
        self.assertRegex(source, r"stressMultirateArmed\s*=\s*0U")
        self.assertIn("STRESS_MULTIRATE_CHANNEL_MASK", source)
        self.assertIn("plan->freshBitmap[byte]", source)
        self.assertIn("STRESS_MULTIRATE_SAMPLE_TIME_UNIT_US", source)
        self.assertIn("stressSampleTimingOverflow", source)
        self.assertIn("stressMultirateDisarmAndClear", source)
        self.assertRegex(source, r"workState == EXTRA_STATE\s*&&\s*enable <= 1U")
        self.assertIn("requestedVersion == STRESS_MULTIRATE_EXTENSION_VERSION_V4", source)
        self.assertIn("plan->bandwidthDiscontinuity", source)
        self.assertIn("stress_multirate.c", PROJECT.read_text(encoding="utf-8"))
        self.assertIn("TRACK13", SCHEDULER.read_text(encoding="utf-8"))

    def test_c_scheduler_declares_all_three_profiles_and_cache_guard(self) -> None:
        source = SCHEDULER.read_text(encoding="utf-8")
        for token in (
            "buildMapPlan",
            "buildSurveyPlan",
            "buildTrackPlan",
            "StressMultirate_AllCached",
            "StressMultirate_GetCachedCh1",
        ):
            self.assertIn(token, source)
        self.assertRegex(
            source,
            re.compile(
                r"!StressMultirate_AllCached\(state\).*?buildMapPlan\(plan\)", re.S
            ),
        )
        self.assertIn("STRESS_MULTIRATE_MAX_MAP_DEFERRAL_FRAMES", source)
        self.assertIn("buildSurveyPlan", source)
        self.assertIn("STRESS_MULTIRATE_STARTUP_MAX_FRAMES", source)
        self.assertIn("STRESS_MULTIRATE_STARTUP_STABLE_FRAMES", source)
        self.assertIn("StressMultirate_BaselineStatus", source)


if __name__ == "__main__":
    unittest.main()
