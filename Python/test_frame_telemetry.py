import unittest

from frame_telemetry import (
    BOARD_STATUS_V1_FRAME_START_LENGTH,
    BOARD_STATUS_V1_LEGACY_LENGTH,
    BOARD_STATUS_V1_MULTIRATE_LENGTH,
    BOARD_STATUS_V1_MULTIRATE_V1_LENGTH,
    BOARD_STATUS_V1_TIMING_LENGTH,
    BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH,
    MULTIRATE_BANDWIDTH_GAP_FLAG,
    MULTIRATE_EXTENSION_VERSION,
    MULTIRATE_EXTENSION_VERSION_V3,
    MULTIRATE_EXTENSION_VERSION_V4,
    MULTIRATE_PROFILE_MAP,
    MULTIRATE_PROFILE_SURVEY,
    MULTIRATE_PROFILE_TRACK,
    MULTIRATE_PROFILE_TRACK_SINGLE,
    MULTIRATE_PROFILE_TRACK_WIDE,
    MULTIRATE_SENTINEL_LOCAL_OFFSETS,
    MULTIRATE_TRACK_POINT_COUNT,
    MULTIRATE_TRACK_SINGLE_POINT_COUNT,
    MULTIRATE_TRACK_WIDE_POINT_COUNT,
    BoardSequenceTracker,
    decode_board_runtime_identity,
    decode_board_frame_schedule,
    decode_board_frame_start_ms,
    decode_board_frame_timing,
    extract_raw_scan_frames,
)


def _extension(sequence=0, acquisition_us=0):
    payload = bytearray(BOARD_STATUS_V1_TIMING_LENGTH)
    payload[:BOARD_STATUS_V1_LEGACY_LENGTH] = bytes(
        range(BOARD_STATUS_V1_LEGACY_LENGTH)
    )
    payload[28:32] = int(sequence).to_bytes(4, "big")
    payload[32:36] = int(acquisition_us).to_bytes(4, "big")
    return bytes(payload)


def _identity(*, table_crc=0xA1B2C3D4, boot_id=0x10203040, uptime_ms=12345):
    payload = bytearray(BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH)
    payload[:4] = b"\xC7\x49\x01\x0C"
    payload[4:8] = int(table_crc).to_bytes(4, "big")
    payload[8:12] = int(boot_id).to_bytes(4, "big")
    payload[12:16] = int(uptime_ms).to_bytes(4, "big")
    return bytes(payload)


def _v4_schedule_extension(
    profile,
    indices,
    *,
    primary=0xFF,
    secondary=0xFF,
    version=MULTIRATE_EXTENSION_VERSION_V4,
    map_age=2,
    bandwidth_discontinuity=False,
):
    payload = bytearray(BOARD_STATUS_V1_MULTIRATE_LENGTH)
    payload[32:36] = (100_000).to_bytes(4, "big")
    payload[36] = int(version)
    payload[37] = int(profile)
    payload[38] = int(primary)
    payload[39] = int(secondary)
    payload[40] = len(indices)
    payload[41] = int(map_age) | (
        MULTIRATE_BANDWIDTH_GAP_FLAG if bandwidth_discontinuity else 0
    )
    for index in indices:
        payload[42 + index // 8] |= 1 << (index & 7)
    payload[48:52] = (1234).to_bytes(4, "big")
    payload[52:] = b"\xFF\xFF" * 45
    for ordinal, index in enumerate(indices, 1):
        payload[52 + 2 * index : 54 + 2 * index] = (ordinal * 10).to_bytes(
            2, "big"
        )
    return bytes(payload)


def _sentinel_index(segment):
    return segment * 5 + MULTIRATE_SENTINEL_LOCAL_OFFSETS[segment]


def _track_single_indices(primary):
    points = {_sentinel_index(segment) for segment in range(9)}
    points.discard(_sentinel_index(primary))
    points.update((primary * 5 + 1, primary * 5 + 2, primary * 5 + 3))
    return tuple(sorted(points))


def _track_wide_indices(primary, secondary):
    points = {_sentinel_index(segment) for segment in range(9)}
    for segment in (primary, secondary):
        points.discard(_sentinel_index(segment))
        points.update((segment * 5 + 1, segment * 5 + 2, segment * 5 + 3))
    return tuple(sorted(points))


class BoardFrameTimingTests(unittest.TestCase):
    def test_frame_start_decoder_distinguishes_base_and_multirate_layouts(self):
        base = bytearray(BOARD_STATUS_V1_FRAME_START_LENGTH)
        # A normal 40-byte frame may coincidentally start with a multirate
        # version byte. Its length still makes offset 36 unambiguous.
        base[36:40] = (0x01234567).to_bytes(4, "big")
        multirate_v1 = bytearray(BOARD_STATUS_V1_MULTIRATE_V1_LENGTH)
        multirate_v1[36] = 1
        multirate_v1[48:52] = (0x89ABCDEF).to_bytes(4, "big")
        multirate_v2 = bytearray(BOARD_STATUS_V1_MULTIRATE_LENGTH)
        multirate_v2[36] = 2
        multirate_v2[48:52] = (0x01020304).to_bytes(4, "big")
        multirate_v3 = bytearray(BOARD_STATUS_V1_MULTIRATE_LENGTH)
        multirate_v3[36] = MULTIRATE_EXTENSION_VERSION_V3
        multirate_v3[48:52] = (0x11223344).to_bytes(4, "big")
        multirate_v4 = bytearray(BOARD_STATUS_V1_MULTIRATE_LENGTH)
        multirate_v4[36] = MULTIRATE_EXTENSION_VERSION_V4
        multirate_v4[48:52] = (0x55667788).to_bytes(4, "big")

        self.assertEqual(decode_board_frame_start_ms(base), 0x01234567)
        self.assertEqual(decode_board_frame_start_ms(multirate_v1), 0x89ABCDEF)
        self.assertEqual(decode_board_frame_start_ms(multirate_v2), 0x01020304)
        self.assertEqual(decode_board_frame_start_ms(multirate_v3), 0x11223344)
        self.assertEqual(decode_board_frame_start_ms(multirate_v4), 0x55667788)
        self.assertIsNone(
            decode_board_frame_start_ms(bytes(BOARD_STATUS_V1_TIMING_LENGTH))
        )

    def test_tagged_identity_does_not_turn_a_long_base_frame_into_multirate(self):
        base = bytearray(BOARD_STATUS_V1_FRAME_START_LENGTH)
        # The first tick byte deliberately equals a valid schedule version.
        base[36:40] = (0x01234567).to_bytes(4, "big")
        extension = bytes(base) + _identity()

        self.assertEqual(decode_board_frame_start_ms(extension), 0x01234567)
        self.assertIsNone(decode_board_frame_schedule(extension))
        identity = decode_board_runtime_identity(extension)
        self.assertEqual(identity.table_crc32, 0xA1B2C3D4)
        self.assertEqual(identity.boot_id, 0x10203040)
        self.assertEqual(identity.uptime_ms, 12345)

    def test_identity_suffix_coexists_with_full_multirate_schedule(self):
        extension = _v4_schedule_extension(
            MULTIRATE_PROFILE_MAP, tuple(range(45))
        ) + _identity(table_crc=7, boot_id=8, uptime_ms=9)

        schedule = decode_board_frame_schedule(extension)
        identity = decode_board_runtime_identity(extension)
        self.assertTrue(schedule.is_complete_map)
        self.assertEqual(schedule.frame_start_ms, 1234)
        self.assertEqual(identity.table_crc32, 7)
        self.assertEqual(identity.boot_id, 8)
        self.assertEqual(identity.uptime_ms, 9)

    def test_malformed_tagged_identity_fails_closed(self):
        malformed = bytearray(_identity())
        malformed[3] = 11
        extension = bytes(BOARD_STATUS_V1_FRAME_START_LENGTH) + bytes(malformed)
        with self.assertRaisesRegex(ValueError, "runtime identity length"):
            decode_board_runtime_identity(extension)
        with self.assertRaisesRegex(ValueError, "runtime identity length"):
            decode_board_frame_schedule(extension)

    def test_v4_adds_track_single_without_repurposing_track_wide(self):
        self.assertEqual(MULTIRATE_EXTENSION_VERSION, 4)
        self.assertEqual(MULTIRATE_PROFILE_TRACK, MULTIRATE_PROFILE_TRACK_WIDE)
        self.assertEqual(MULTIRATE_TRACK_POINT_COUNT, 13)
        self.assertEqual(MULTIRATE_TRACK_WIDE_POINT_COUNT, 13)
        self.assertEqual(MULTIRATE_TRACK_SINGLE_POINT_COUNT, 11)

        single = decode_board_frame_schedule(
            _v4_schedule_extension(
                MULTIRATE_PROFILE_TRACK_SINGLE,
                _track_single_indices(4),
                primary=4,
            )
        )
        self.assertEqual(len(single.fresh_indices), 11)
        self.assertEqual(single.primary_segment, 4)
        self.assertIsNone(single.secondary_segment)
        self.assertTrue(single.is_track_single)

        wide = decode_board_frame_schedule(
            _v4_schedule_extension(
                MULTIRATE_PROFILE_TRACK_WIDE,
                _track_wide_indices(2, 7),
                primary=2,
                secondary=7,
            )
        )
        self.assertEqual(len(wide.fresh_indices), 13)
        self.assertTrue(wide.is_track_wide)

    def test_track_single_requires_v4_one_roi_and_exact_bitmap(self):
        indices = _track_single_indices(4)
        with self.assertRaisesRegex(ValueError, "requires stress multirate v4"):
            decode_board_frame_schedule(
                _v4_schedule_extension(
                    MULTIRATE_PROFILE_TRACK_SINGLE,
                    indices,
                    primary=4,
                    version=MULTIRATE_EXTENSION_VERSION_V3,
                )
            )
        with self.assertRaisesRegex(ValueError, "secondary segment"):
            decode_board_frame_schedule(
                _v4_schedule_extension(
                    MULTIRATE_PROFILE_TRACK_SINGLE,
                    indices,
                    primary=4,
                    secondary=7,
                )
            )

        wrong = list(indices)
        wrong.remove(_sentinel_index(0))
        wrong.append(1)
        with self.assertRaisesRegex(ValueError, "fresh bitmap"):
            decode_board_frame_schedule(
                _v4_schedule_extension(
                    MULTIRATE_PROFILE_TRACK_SINGLE,
                    tuple(sorted(wrong)),
                    primary=4,
                )
            )

    def test_v3_survey9_and_gap_semantics_survive_current_v4(self):
        survey9 = tuple(_sentinel_index(segment) for segment in range(9))
        survey = decode_board_frame_schedule(
            _v4_schedule_extension(
                MULTIRATE_PROFILE_SURVEY,
                survey9,
                version=MULTIRATE_EXTENSION_VERSION_V3,
            )
        )
        self.assertEqual(survey.fresh_indices, survey9)

        complete = decode_board_frame_schedule(
            _v4_schedule_extension(
                MULTIRATE_PROFILE_MAP,
                tuple(range(45)),
                version=MULTIRATE_EXTENSION_VERSION_V3,
                bandwidth_discontinuity=True,
            )
        )
        self.assertTrue(complete.bandwidth_discontinuity)

    def test_legacy_28_byte_extension_remains_valid_without_timing(self):
        self.assertIsNone(
            decode_board_frame_timing(bytes(BOARD_STATUS_V1_LEGACY_LENGTH))
        )

    def test_new_extension_decodes_big_endian_timing_fields(self):
        timing = decode_board_frame_timing(
            _extension(sequence=0x89ABCDEF, acquisition_us=65_432)
        )
        self.assertEqual(timing.sequence, 0x89ABCDEF)
        self.assertEqual(timing.acquisition_duration_us, 65_432)
        self.assertAlmostEqual(timing.acquisition_duration_ms, 65.432)

    def test_decoder_ignores_future_extension_suffix(self):
        timing = decode_board_frame_timing(_extension(7, 12_345) + b"future")
        self.assertEqual(timing.sequence, 7)
        self.assertEqual(timing.acquisition_duration_us, 12_345)

    def test_length_parser_does_not_split_ffef_inside_timing_fields(self):
        point_count = 1
        payload = _extension(sequence=0xFFEF1234, acquisition_us=0x56FFEF78)
        frame = bytearray(b"\xEE\xEE\x00\x01" + bytes(point_count * 8))
        frame.extend(b"\xAB\x00\x00\x00\x00")
        frame.extend(bytes(4))
        frame.extend(bytes(4))
        frame.extend((0xB5, 0x4D, 0x01, len(payload)))
        frame.extend(payload)
        frame.extend(b"\xFF\xEF")
        buffer = bytearray(b"noise" + frame)
        self.assertEqual(extract_raw_scan_frames(buffer), [bytes(frame)])
        self.assertEqual(buffer, bytearray())

    def test_length_parser_keeps_runtime_identity_with_embedded_ffef(self):
        point_count = 1
        payload = (
            bytearray(BOARD_STATUS_V1_FRAME_START_LENGTH)
            + _identity(table_crc=0xFFEF1234, boot_id=0x56FFEF78)
        )
        frame = bytearray(b"\xEE\xEE\x00\x01" + bytes(point_count * 8))
        frame.extend(b"\xAB\x00\x00\x00\x00")
        frame.extend(bytes(4))
        frame.extend(bytes(4))
        frame.extend((0xB5, 0x4D, 0x01, len(payload)))
        frame.extend(payload)
        frame.extend(b"\xFF\xEF")
        buffer = bytearray(frame)
        self.assertEqual(extract_raw_scan_frames(buffer), [bytes(frame)])
        self.assertEqual(buffer, bytearray())


class BoardSequenceTrackerTests(unittest.TestCase):
    def test_counts_forward_gaps_and_duplicates(self):
        tracker = BoardSequenceTracker()
        for sequence in (10, 11, 14, 14, 15):
            tracker.observe(sequence)
        stats = tracker.stats()
        self.assertEqual(stats.received, 5)
        self.assertEqual(stats.missing, 2)
        self.assertEqual(stats.duplicates, 1)
        self.assertEqual(stats.resets, 0)
        self.assertEqual(stats.first_sequence, 10)
        self.assertEqual(stats.last_sequence, 15)

    def test_uint32_wrap_is_contiguous(self):
        tracker = BoardSequenceTracker()
        for sequence in (0xFFFFFFFE, 0xFFFFFFFF, 0, 1):
            tracker.observe(sequence)
        stats = tracker.stats()
        self.assertEqual(stats.missing, 0)
        self.assertEqual(stats.duplicates, 0)
        self.assertEqual(stats.resets, 0)

    def test_large_backward_jump_is_restart_not_loss(self):
        tracker = BoardSequenceTracker()
        for sequence in (1200, 1201, 0, 1):
            tracker.observe(sequence)
        stats = tracker.stats()
        self.assertEqual(stats.missing, 0)
        self.assertEqual(stats.resets, 1)
        self.assertEqual(stats.last_sequence, 1)

    def test_reset_clears_session_statistics(self):
        tracker = BoardSequenceTracker()
        tracker.observe(1)
        tracker.observe(4)
        tracker.reset()
        self.assertEqual(tracker.stats().received, 0)
        self.assertIsNone(tracker.stats().last_sequence)


if __name__ == "__main__":
    unittest.main()
