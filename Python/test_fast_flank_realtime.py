import unittest

from capture_fast_flank_stream import ROWS
from fast_flank_realtime import CADENCE_WINDOW, FastFlankRealtimeLocalizer


def rows_fixture():
    rows = []
    for index in range(45):
        local = index % 5
        rows.append(
            {
                "template_signal_v": 0.20 + local * 0.01,
                "template_slope_v_per_nm": 3.0 if local < 2 else -3.0,
            }
        )
    return rows


def frame(sequence, *, changed_grating=None, interval_us=21_000):
    records = []
    for row in ROWS:
        code = 200
        if changed_grating == "all" and row % 5 == 3:
            code = 270
        elif isinstance(changed_grating, int) and row == changed_grating * 5 + 3:
            code = 270
        records.append({"row": row, "second_code": code})
    return {
        "sequence": sequence,
        "cycle_start_us": sequence * interval_us,
        "minimum_warmup_complete": sequence >= 16,
        "records": records,
    }


def gradual_pair_frame(sequence, step_codes):
    result = frame(sequence)
    for grating in (4, 5):
        row = grating * 5 + 3
        next(item for item in result["records"] if item["row"] == row)[
            "second_code"
        ] = 200 + int(step_codes)
    return result


class FastFlankRealtimeTests(unittest.TestCase):
    def _ready(self):
        pipeline = FastFlankRealtimeLocalizer(rows_fixture(), baseline_frames=16)
        result = None
        for sequence in range(32):
            result = pipeline.update(frame(sequence), declared_no_contact=True)
        self.assertIsNotNone(result)
        self.assertTrue(result.baseline_ready)
        return pipeline

    def test_requires_frozen_baseline_before_contact_inference(self):
        pipeline = self._ready()
        result = pipeline.update(frame(32, changed_grating=4), declared_no_contact=False)
        self.assertFalse(result.contact_detected)
        self.assertIn("contact_phase_without_frozen_baseline", result.quality_flags)
        self.assertIsNone(result.physical_contact_area_mm2)

    def test_detects_provisional_position_but_never_invents_area(self):
        pipeline = self._ready()
        pipeline.freeze_for_press()
        result = pipeline.update(frame(32, changed_grating=4), declared_no_contact=False)
        self.assertTrue(result.contact_detected)
        self.assertEqual(
            result.detection_kind, "pressure_change_onset_not_static_contact_state"
        )
        self.assertEqual(result.dominant_sensor_id, 5)
        self.assertIsNotNone(result.provisional_x_mm)
        self.assertIsNotNone(result.provisional_y_mm)
        self.assertTrue(result.geometry_calibrated)
        self.assertFalse(result.position_accuracy_verified)
        self.assertIsNone(result.physical_contact_area_mm2)
        self.assertFalse(result.contact_area_verified)
        self.assertIsNotNone(result.one_frame_delta_pm[4])

    def test_cadence_gate_requires_a_full_recent_window(self):
        pipeline = self._ready()
        pipeline.freeze_for_press()
        result = pipeline.update(frame(32), declared_no_contact=False)
        self.assertTrue(result.sampling_gate_for_15hz)
        self.assertGreater(result.cadence_hz, 30.0)
        self.assertFalse(result.physical_15hz_verified)
        self.assertEqual(len(pipeline._intervals_us), CADENCE_WINDOW)

    def test_common_mode_change_is_not_localized_as_pressure(self):
        pipeline = self._ready()
        pipeline.freeze_for_press()
        result = pipeline.update(frame(32, changed_grating="all"), declared_no_contact=False)
        self.assertFalse(result.contact_detected)
        self.assertIn("below_pressure_change_onset", result.quality_flags)

    def test_eight_frame_branch_detects_gradual_spatial_contact(self):
        pipeline = self._ready()
        pipeline.freeze_for_press()
        result = None
        for offset, sequence in enumerate(range(32, 40), start=1):
            result = pipeline.update(
                gradual_pair_frame(sequence, offset * 4),
                declared_no_contact=False,
            )
        self.assertIsNotNone(result)
        self.assertTrue(result.contact_detected)
        self.assertIn(
            "pressure_change_accumulated_8_frames", result.quality_flags
        )
        self.assertLess(abs(result.one_frame_delta_pm[4]), 3.0)

    def test_slow_interval_fails_sampling_gate(self):
        pipeline = FastFlankRealtimeLocalizer(rows_fixture(), baseline_frames=16)
        result = None
        for sequence in range(49):
            result = pipeline.update(
                frame(sequence, interval_us=34_000), declared_no_contact=True
            )
        self.assertIsNotNone(result)
        self.assertFalse(result.sampling_gate_for_15hz)

    def test_sequence_gap_resets_baseline(self):
        pipeline = FastFlankRealtimeLocalizer(rows_fixture(), baseline_frames=16)
        pipeline.update(frame(0), declared_no_contact=True)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            pipeline.update(frame(2), declared_no_contact=True)
        self.assertFalse(pipeline.baseline.ready)
        self.assertEqual(len(pipeline._intervals_us), 0)


if __name__ == "__main__":
    unittest.main()
