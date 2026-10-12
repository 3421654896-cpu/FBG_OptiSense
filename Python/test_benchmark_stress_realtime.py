import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_JDSU
from benchmark_stress_realtime import (
    _extract_scan_frames,
    _summarize_board_start_ticks,
    _summarize_board_telemetry,
    _summarize_host_delivery,
    summarize_sentinel_signal_health,
)
from frame_telemetry import (
    BOARD_STATUS_V1_FRAME_START_LENGTH,
    BOARD_STATUS_V1_LEGACY_LENGTH,
    BOARD_STATUS_V1_MULTIRATE_LENGTH,
    BOARD_STATUS_V1_TIMING_LENGTH,
    MULTIRATE_EXTENSION_VERSION_V4,
    MULTIRATE_PROFILE_SURVEY,
    MULTIRATE_PROFILE_TRACK_SINGLE,
    MULTIRATE_SENTINEL_LOCAL_OFFSETS,
    BoardFrameTiming,
    decode_board_frame_schedule,
)
from verify_runtime_firmware import board_extension


class SentinelHealthTests(unittest.TestCase):
    def test_timed_zero_trace_is_not_pressure_validation(self):
        trace = [{'fresh_ch1_codes': {str(g*5+offset): (0 if g == 7 else 100)
                   for g, offset in enumerate(MULTIRATE_SENTINEL_LOCAL_OFFSETS)}} for _ in range(10)]
        report = summarize_sentinel_signal_health(trace)
        self.assertEqual(report['review_gratings'], ['G8'])
        self.assertFalse(report['pressure_response_verified'])
        self.assertEqual(report['per_grating']['G8']['near_zero_fraction'], 1)

    def test_absent_trace_does_not_imply_good_signal(self):
        report = summarize_sentinel_signal_health([])
        self.assertFalse(report['available'])
        self.assertFalse(report['pressure_response_verified'])
        self.assertEqual(report['per_grating']['G8']['fresh_samples'], 0)


def _raw_frame(point_count, extension):
    frame = bytearray(b"\xEE\xEE")
    frame.extend(int(point_count).to_bytes(2, "big"))
    frame.extend(bytes(point_count * 8))
    frame.append(0xAB)
    frame.extend(b"\x00\x00\x00\x00")  # four zero peak counts
    frame.extend(bytes(4))  # temperature
    frame.extend(bytes(4))  # legacy gain masks
    frame.extend((0xB5, 0x4D, 0x01, len(extension)))
    frame.extend(extension)
    frame.extend(b"\xFF\xEF")
    return bytes(frame)


def _survey_extension(*, sequence=8, map_age=3, start_ms=456):
    extension = bytearray(BOARD_STATUS_V1_MULTIRATE_LENGTH)
    extension[16] = 0x02
    extension[22:24] = b"\x01\x01"
    extension[28:32] = int(sequence).to_bytes(4, "big")
    extension[32:36] = (20_000).to_bytes(4, "big")
    extension[36] = 2
    extension[37] = MULTIRATE_PROFILE_SURVEY
    extension[38:40] = b"\xFF\xFF"
    fresh = tuple(point for peak in range(9) for point in (peak * 5 + 1, peak * 5 + 3))
    extension[40] = len(fresh)
    extension[41] = int(map_age)
    for point in fresh:
        extension[42 + point // 8] |= 1 << (point % 8)
    extension[48:52] = int(start_ms).to_bytes(4, "big")
    extension[52:142] = b"\xFF\xFF" * 45
    for ordinal, point in enumerate(fresh, 1):
        extension[52 + 2 * point : 54 + 2 * point] = (ordinal * 10).to_bytes(
            2, "big"
        )
    return bytes(extension)


def _track11_extension(*, primary=4, sequence=9, start_ms=488):
    extension = bytearray(BOARD_STATUS_V1_MULTIRATE_LENGTH)
    extension[16] = 0x02
    extension[22:24] = b"\x01\x01"
    extension[28:32] = int(sequence).to_bytes(4, "big")
    extension[32:36] = (29_000).to_bytes(4, "big")
    extension[36] = MULTIRATE_EXTENSION_VERSION_V4
    extension[37] = MULTIRATE_PROFILE_TRACK_SINGLE
    extension[38] = int(primary)
    extension[39] = 0xFF
    fresh = []
    for segment in range(9):
        base = segment * 5
        if segment == primary:
            fresh.extend((base + 1, base + 2, base + 3))
        else:
            fresh.append(base + MULTIRATE_SENTINEL_LOCAL_OFFSETS[segment])
    extension[40] = len(fresh)
    extension[41] = 4
    for point in fresh:
        extension[42 + point // 8] |= 1 << (point % 8)
    extension[48:52] = int(start_ms).to_bytes(4, "big")
    extension[52:142] = b"\xFF\xFF" * 45
    for ordinal, point in enumerate(fresh, 1):
        extension[52 + 2 * point : 54 + 2 * point] = (ordinal * 10).to_bytes(
            2, "big"
        )
    return bytes(extension)


class StressRealtimeBenchmarkTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = app_JDSU.QtWidgets.QApplication.instance()
        if cls.qt_app is None:
            cls.qt_app = app_JDSU.QtWidgets.QApplication([])

    def test_frame_extractor_accepts_legacy_and_timed_v1_extensions(self):
        legacy = _raw_frame(2, bytes(BOARD_STATUS_V1_LEGACY_LENGTH))
        timed = _raw_frame(2, bytes(BOARD_STATUS_V1_TIMING_LENGTH))
        frame_start = _raw_frame(2, bytes(BOARD_STATUS_V1_FRAME_START_LENGTH))
        buffer = bytearray(b"noise" + legacy + timed + frame_start)
        frames = _extract_scan_frames(buffer, 2)
        self.assertEqual(len(frames), 3)
        self.assertEqual(len(board_extension(frames[0])), 28)
        self.assertEqual(len(board_extension(frames[1])), 36)
        self.assertEqual(len(board_extension(frames[2])), 40)
        self.assertEqual(buffer, bytearray())

    def test_frame_extractor_preserves_v4_track11_schedule(self):
        native = _raw_frame(45, _track11_extension(primary=4))
        frames = _extract_scan_frames(bytearray(native), 45)
        self.assertEqual(frames, [native])
        schedule = decode_board_frame_schedule(board_extension(frames[0]))
        self.assertTrue(schedule.is_track_single)
        self.assertEqual(schedule.primary_segment, 4)
        self.assertIsNone(schedule.secondary_segment)
        self.assertEqual(len(schedule.fresh_indices), 11)

    def test_summary_reports_board_duration_and_sequence_loss(self):
        report = _summarize_board_telemetry(
            [
                BoardFrameTiming(100, 50_000),
                BoardFrameTiming(101, 60_000),
                BoardFrameTiming(104, 70_000),
                BoardFrameTiming(104, 80_000),
            ]
        )
        self.assertEqual(report["available_frames"], 4)
        self.assertEqual(report["missing_telemetry_frames"], 0)
        self.assertEqual(report["sequence"]["missing"], 2)
        self.assertEqual(report["sequence"]["duplicates"], 1)
        self.assertAlmostEqual(report["acquisition_ms"]["mean"], 65.0)
        self.assertAlmostEqual(report["acquisition_ms"]["min"], 50.0)
        self.assertAlmostEqual(report["acquisition_ms"]["max"], 80.0)

    def test_summary_marks_legacy_frame_without_fabricating_timing(self):
        report = _summarize_board_telemetry([None])
        self.assertEqual(report["available_frames"], 0)
        self.assertEqual(report["missing_telemetry_frames"], 1)
        self.assertIsNone(report["acquisition_ms"])
        self.assertIsNone(report["sequence"]["first"])

    def test_board_start_ticks_report_true_period_and_wrap(self):
        report = _summarize_board_start_ticks(
            [0xFFFFFFC0, 4, 64, None, 124, 184]
        )
        self.assertEqual(report["intervals"], 3)
        self.assertEqual(report["min"], 60.0)
        self.assertEqual(report["max"], 68.0)
        self.assertAlmostEqual(report["mean"], 62.6666666667)
        self.assertAlmostEqual(report["mean_hz"], 15.9574468085)

    def test_board_start_ticks_do_not_invent_missing_intervals(self):
        self.assertIsNone(_summarize_board_start_ticks([1, None, 3]))

    def test_host_delivery_never_turns_one_socket_batch_into_zero_periods(self):
        report = _summarize_host_delivery(
            [1.000, 1.060, 1.120, 1.120, 1.180],
            [1, 2, 3, 3, 4],
        )

        self.assertEqual(report["unique_frames"], 5)
        self.assertEqual(report["socket_batches"], 4)
        self.assertEqual(report["multi_frame_batches"], 1)
        self.assertEqual(report["max_frames_per_batch"], 2)
        self.assertIsNone(report["exact_unique_frame_arrival_period_ms"])
        self.assertEqual(
            report["socket_batch_arrival_period_ms"]["intervals"], 3
        )
        self.assertGreater(
            report["socket_batch_arrival_period_ms"]["min"], 0.0
        )
        self.assertGreater(
            report["amortized_unique_frame_delivery_period_ms"]["min"], 0.0
        )
        self.assertAlmostEqual(report["effective_unique_frame_hz"], 22.2222222222)

    def test_host_delivery_exposes_exact_period_only_for_single_frame_batches(self):
        report = _summarize_host_delivery(
            [2.000, 2.064, 2.128],
            [10, 11, 12],
        )

        exact = report["exact_unique_frame_arrival_period_ms"]
        self.assertIsNotNone(exact)
        self.assertEqual(exact["intervals"], 2)
        self.assertAlmostEqual(exact["mean"], 64.0)
        self.assertAlmostEqual(report["effective_unique_frame_hz"], 15.625)

    def test_host_delivery_rejects_nonpositive_batch_time_as_exact_timing(self):
        report = _summarize_host_delivery(
            [3.000, 3.000, 3.064],
            [20, 21, 22],
        )

        self.assertEqual(report["nonpositive_inter_batch_intervals"], 1)
        self.assertIsNone(report["exact_unique_frame_arrival_period_ms"])
        self.assertGreater(report["socket_batch_arrival_period_ms"]["min"], 0.0)

    def test_graph_page_stores_new_timing_and_accepts_next_legacy_frame(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        point_count = len(page.wave_const)
        extension = bytearray(BOARD_STATUS_V1_TIMING_LENGTH)
        extension[16] = 0x02
        extension[22:24] = b"\x01\x01"
        extension[24:28] = (0x00010015).to_bytes(4, "big")
        extension[28:32] = (123).to_bytes(4, "big")
        extension[32:36] = (54_321).to_bytes(4, "big")

        app_JDSU.frames_queue.clear(reset_counters=True)
        app_JDSU.frames_queue.put(_raw_frame(point_count, bytes(extension)))
        page.new_frame_pending = False
        page.process_down = True
        page.process_frame()
        self.assertEqual(page.board_frame_sequence, 123)
        self.assertEqual(page.board_acquisition_duration_us, 54_321)
        self.assertEqual(page.board_sequence_tracker.stats().received, 1)

        legacy = bytearray(BOARD_STATUS_V1_LEGACY_LENGTH)
        legacy[16] = 0x02
        legacy[22:24] = b"\x01\x01"
        app_JDSU.frames_queue.put(_raw_frame(point_count, bytes(legacy)))
        page.new_frame_pending = False
        page.process_down = True
        page.process_frame()
        self.assertIsNone(page.board_frame_sequence)
        self.assertIsNone(page.board_acquisition_duration_us)
        self.assertEqual(page.board_sequence_tracker.stats().received, 1)
        page.deleteLater()

    def test_graph_page_exposes_multirate_freshness_and_sample_times(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        app_JDSU.frames_queue.clear(reset_counters=True)
        app_JDSU.frames_queue.put(
            _raw_frame(len(page.wave_const), _survey_extension())
        )
        page.new_frame_pending = False
        page.process_down = True
        page.process_frame()

        self.assertEqual(page.acquisition_profile, "SURVEY")
        self.assertEqual(len(page.fresh_point_mask), 45)
        self.assertEqual(sum(page.fresh_point_mask), 18)
        self.assertEqual(page.fresh_point_indices[:2], (1, 3))
        self.assertEqual(page.map_age_frames, 3)
        self.assertEqual(page.point_age_frames[0], 3.0)
        self.assertEqual(page.point_age_frames[1], 0.0)
        self.assertIsNone(page.sample_offset_us[0])
        self.assertEqual(page.sample_offset_us[1], 20)
        self.assertEqual(page.frame_start_device_ms, 456)
        self.assertFalse(page.current_frame_is_complete_map)
        page.deleteLater()

    def test_reduced_frame_never_enters_legacy_gaussian_fit(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        page.current_frame_is_complete_map = False
        page.acquisition_profile = "SURVEY"
        for channel in range(4):
            page.adc[channel].extend(np.zeros(45, dtype=float))
            page.data[channel].extend(np.zeros(45, dtype=int))
        page.new_frame_pending = True
        page.process_down = True
        with patch.object(app_JDSU, "fit_channel_segments") as fit:
            page.update_plot()
        fit.assert_not_called()
        page.deleteLater()

    def test_raw_ch1_baseline_accepts_only_twenty_stable_complete_maps(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        candidate = np.asarray(
            [300 + (point % 5) * 250 for point in range(45)], dtype=int
        )
        page.begin_contact_baseline_capture()
        page.current_frame_is_complete_map = False
        for _ in range(5):
            page._collect_contact_baseline_frame(candidate)
        self.assertEqual(page.contact_baseline_samples, [])

        page.current_frame_is_complete_map = True
        for frame in range(20):
            page._collect_contact_baseline_frame(candidate + frame % 2)
        self.assertFalse(page.contact_baseline_capture_active)
        self.assertEqual(len(page.contact_baseline_adc_codes), 45)
        self.assertEqual(page.contact_baseline_transimpedance_ohm, 40_000.0)
        page.deleteLater()

    def test_graph_converts_public_microseconds_back_to_wire_ticks_for_audit(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        evidence = object()
        estimate = object()
        page.contact_localizer = Mock()
        page.contact_localizer.extract_spectrum_evidence.return_value = evidence
        page.contact_tracker = Mock()
        page.contact_tracker.update_evidence.return_value = estimate
        page.contact_baseline_adc_codes = tuple([500] * 45)
        page.contact_baseline_transimpedance_ohm = 40_000.0
        page.board_frame_schedule = SimpleNamespace()
        page.acquisition_profile = "SURVEY"
        page.fresh_point_indices = (1, 3)
        page.sample_offset_us = tuple(
            20 if index == 1 else (40 if index == 3 else None)
            for index in range(45)
        )
        page.point_age_frames = tuple(
            0.0 if index in (1, 3) else 2.0 for index in range(45)
        )
        page.map_age_frames = 2
        page.board_frame_sequence = 77

        page._update_contact_pipeline(tuple([500] * 45), 123_000_000)
        kwargs = page.contact_localizer.extract_spectrum_evidence.call_args.kwargs
        self.assertEqual(kwargs["sample_offsets_2us"][1], 10)
        self.assertEqual(kwargs["sample_offsets_2us"][3], 20)
        self.assertEqual(kwargs["sample_offsets_2us"][0], 0xFFFF)
        self.assertNotIn("fresh_point_indices", kwargs)
        page.contact_tracker.update_evidence.assert_called_once_with(
            evidence, sequence=77
        )
        self.assertIs(page.contact_estimate, estimate)
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
