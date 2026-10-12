import math
import unittest

import numpy as np
from ch1_contact_localizer import (
    AREA_KIND,
    SPECTRUM_EVIDENCE_KIND,
    CH1NinePeakLocalizer,
    ContactLocalizerConfig,
    MultiRateContactTracker,
    benchmark_localizer,
    load_contact_geometry,
    make_machine_training_record,
    peak_fresh_mask_from_point_indices,
)
from finger_dataset import ADC_CODE_COUNT, ADC_REFERENCE_V
from finger_features import PeakQuality
from stress_multirate_reference import sentinel_index, track_indices


def track_single_indices(primary):
    return tuple(
        sorted(
            {
                *(
                    sentinel_index(segment)
                    for segment in range(9)
                    if segment != int(primary)
                ),
                *(int(primary) * 5 + local for local in (1, 2, 3)),
            }
        )
    )


def synthetic_spectrum(
    shifts_pm=None,
    *,
    amplitude_v=0.70,
    transimpedance_scale=1.0,
):
    shifts = np.zeros(9) if shifts_pm is None else np.asarray(shifts_pm, dtype=float)
    centers = 1528.0 + 4.0 * np.arange(9)
    offsets = np.asarray((-0.040, -0.020, 0.0, 0.020, 0.040))
    wavelength = []
    voltage = []
    for center, shift_pm in zip(centers, shifts, strict=True):
        local = center + offsets
        shape = 0.03 + amplitude_v * np.exp(
            -0.5 * ((local - center - shift_pm / 1000.0) / 0.018) ** 2
        )
        wavelength.extend(local)
        voltage.extend(shape * float(transimpedance_scale))
    codes = np.clip(
        np.rint(np.asarray(voltage) * ADC_CODE_COUNT / ADC_REFERENCE_V),
        0,
        4095,
    ).astype(int)
    return tuple(float(value) for value in wavelength), tuple(
        int(value) for value in codes
    )


class CH1ContactLocalizerTests(unittest.TestCase):
    def setUp(self):
        self.geometry = load_contact_geometry()
        self.localizer = CH1NinePeakLocalizer(
            geometry=self.geometry,
            config=ContactLocalizerConfig(activation_pm=3.0),
        )

    def test_geometry_uses_current_ch1_layout_and_machine_calibration(self):
        self.assertTrue(self.geometry.position_calibrated)
        self.assertEqual(self.geometry.coordinate_frame, "mach3_work_mm")
        self.assertEqual(
            [site.sensor_id for site in self.geometry.sensors], list(range(1, 10))
        )
        self.assertEqual(
            [site.peak_index for site in self.geometry.sensors], list(range(9))
        )
        self.assertGreater(self.geometry.effective_sensor_cell_area_mm2, 0.0)
        self.assertGreater(
            self.geometry.silicone_planform_area_mm2,
            9.0 * self.geometry.effective_sensor_cell_area_mm2,
        )

    def test_one_peak_maps_to_its_calibrated_sensor_site(self):
        shifts = np.zeros(9)
        shifts[4] = 40.0
        result = self.localizer.estimate_peak_shifts(shifts, timestamp_ns=1)
        expected_norm = self.geometry.sensor_positions_norm_by_peak[4]
        expected_mm = self.geometry.sensor_positions_mm_by_peak[4]
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.dominant_sensor_id, 5)
        np.testing.assert_allclose((result.x_norm, result.y_norm), expected_norm)
        np.testing.assert_allclose((result.x_mm, result.y_mm), expected_mm)
        self.assertAlmostEqual(result.effective_sensor_count, 1.0)
        self.assertAlmostEqual(
            result.effective_response_area_mm2,
            self.geometry.effective_sensor_cell_area_mm2,
        )

    def test_two_equal_neighbors_map_to_midpoint_and_two_sensor_footprint(self):
        shifts = np.zeros(9)
        shifts[4:6] = 40.0
        result = self.localizer.estimate_peak_shifts(shifts, timestamp_ns=2)
        expected = np.mean(self.geometry.sensor_positions_mm_by_peak[4:6], axis=0)
        np.testing.assert_allclose((result.x_mm, result.y_mm), expected)
        self.assertAlmostEqual(result.effective_sensor_count, 2.0)
        self.assertAlmostEqual(
            result.effective_response_area_mm2,
            2.0 * self.geometry.effective_sensor_cell_area_mm2,
        )

    def test_common_mode_can_be_removed_before_spatial_localization(self):
        localizer = CH1NinePeakLocalizer(
            geometry=self.geometry,
            config=ContactLocalizerConfig(
                activation_pm=3.0,
                remove_common_mode=True,
            ),
        )
        shifts = np.full(9, 15.0)
        shifts[7] += 35.0
        result = localizer.estimate_peak_shifts(shifts, timestamp_ns=3)
        self.assertEqual(result.dominant_sensor_id, 8)
        self.assertAlmostEqual(result.common_mode_pm, 15.0)
        self.assertIn("median_common_mode_removed", result.quality_flags)

    def test_fatal_peak_quality_is_excluded_and_soft_quality_is_downweighted(self):
        shifts = np.zeros(9)
        shifts[3] = 80.0
        shifts[4] = 30.0
        masks = np.zeros(9, dtype=np.uint16)
        masks[3] = int(PeakQuality.ADC_OUT_OF_RANGE)
        masks[4] = int(PeakQuality.WIDTH_UNRESOLVED)
        result = self.localizer.estimate_peak_shifts(
            shifts,
            timestamp_ns=4,
            peak_quality_masks=masks,
        )
        self.assertEqual(result.dominant_sensor_id, 5)
        self.assertEqual(result.response_weights[3], 0.0)
        self.assertIn("fatal_peak_quality_excluded", result.quality_flags)
        self.assertIn("edge_or_width_quality_downweighted", result.quality_flags)

    def test_45point_primary_evidence_localizes_changed_peak(self):
        wavelength, baseline = synthetic_spectrum()
        shifts = np.zeros(9)
        shifts[4] = 12.0
        _, current = synthetic_spectrum(shifts)
        evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            timestamp_ns=10,
        )
        result = self.localizer.estimate_spectrum(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            timestamp_ns=10,
        )
        self.assertGreater(evidence.peak_response_scores[4], 1.0)
        self.assertLess(max(evidence.peak_response_scores[:4]), 0.05)
        self.assertEqual(len(evidence.normalized_profile_residual_45), 45)
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.dominant_sensor_id, 5)
        self.assertEqual(result.localization_evidence_kind, SPECTRUM_EVIDENCE_KIND)
        self.assertEqual(len(result.local_shape_features), 9)

    def test_template_fit_failure_is_audit_only_and_primary_still_localizes(self):
        wavelength, baseline = synthetic_spectrum()
        shifts = np.zeros(9)
        shifts[2] = 12.0
        _, current = synthetic_spectrum(shifts)
        localizer = CH1NinePeakLocalizer.from_project_defaults(
            config=ContactLocalizerConfig(activation_pm=3.0)
        )
        result = localizer.estimate_spectrum(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            timestamp_ns=11,
        )
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.dominant_sensor_id, 3)
        self.assertTrue(all(value is None for value in result.template_audit_shifts_pm))
        self.assertIn(
            "template_audit_unavailable_primary_evidence_retained",
            result.quality_flags,
        )

    def test_normalized_residual_is_invariant_to_matched_transimpedance_change(self):
        wavelength, baseline_low = synthetic_spectrum()
        shifts = np.zeros(9)
        shifts[6] = 10.0
        _, current_low = synthetic_spectrum(shifts)
        _, baseline_high = synthetic_spectrum(transimpedance_scale=2.0)
        _, current_high = synthetic_spectrum(shifts, transimpedance_scale=2.0)
        low = self.localizer.extract_spectrum_evidence(
            wavelength,
            current_low,
            baseline_adc_codes=baseline_low,
            transimpedance_ohm=20_000,
            baseline_transimpedance_ohm=20_000,
        )
        high = self.localizer.extract_spectrum_evidence(
            wavelength,
            current_high,
            baseline_adc_codes=baseline_high,
            transimpedance_ohm=40_000,
            baseline_transimpedance_ohm=40_000,
        )
        np.testing.assert_allclose(
            low.normalized_residual_rms,
            high.normalized_residual_rms,
            atol=0.003,
        )

    def test_effective_response_area_is_never_reported_as_contact_area(self):
        shifts = np.zeros(9)
        shifts[0] = 35.0
        result = self.localizer.estimate_peak_shifts(shifts, timestamp_ns=20)
        payload = result.to_dict()
        self.assertEqual(result.effective_response_area_kind, AREA_KIND)
        self.assertIsNone(result.physical_contact_area_mm2)
        self.assertFalse(result.physical_contact_area_calibrated)
        self.assertIsNone(payload["physical_contact_area_mm2"])
        self.assertNotAlmostEqual(result.effective_response_area_mm2, math.pi * 4.0**2)

    def test_sparse_fresh_point_masks_cover_survey18_and_track6(self):
        survey18 = [5 * peak + local for peak in range(9) for local in (1, 3)]
        self.assertEqual(
            peak_fresh_mask_from_point_indices(
                survey18,
                minimum_points_per_peak=2,
            ),
            (True,) * 9,
        )
        track6 = [5 * peak + local for peak in (4, 5) for local in (1, 2, 3)]
        self.assertEqual(
            peak_fresh_mask_from_point_indices(
                track6,
                minimum_points_per_peak=3,
            ),
            (False, False, False, False, True, True, False, False, False),
        )

    def test_survey18_metadata_preserves_freshness_age_and_2us_timing(self):
        wavelength, baseline = synthetic_spectrum()
        shifts = np.zeros(9)
        shifts[3] = 10.0
        _, current = synthetic_spectrum(shifts)
        offsets = np.full(45, 0xFFFF, dtype=int)
        survey18 = [5 * peak + local for peak in range(9) for local in (1, 3)]
        for order, point_index in enumerate(survey18):
            offsets[point_index] = order * 7
        ages = np.full(45, 4.0)
        ages[survey18] = 0.0
        evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            profile="SURVEY",
            point_age_frames=ages,
            sample_offsets_2us=offsets,
            map_age_frames=4,
            frame_start_timestamp_ns=1_000_000,
        )
        self.assertEqual(evidence.profile, "SURVEY")
        self.assertEqual(evidence.peak_fresh_mask, (True,) * 9)
        self.assertEqual(evidence.timestamp_ns, 1_000_000 + 17 * 7 * 2_000)
        self.assertEqual(evidence.sample_offsets_2us_45[0], None)
        self.assertEqual(evidence.sample_offsets_2us_45[1], 0)
        self.assertEqual(sum(evidence.fresh_point_mask_45), 18)

    def test_track6_evidence_updates_only_two_rois(self):
        wavelength, baseline = synthetic_spectrum()
        shifts = np.zeros(9)
        shifts[4] = 12.0
        _, current = synthetic_spectrum(shifts)
        track6 = [5 * peak + local for peak in (4, 5) for local in (1, 2, 3)]
        evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            profile="TRACK",
            fresh_point_indices=track6,
            map_age_frames=2,
            timestamp_ns=2_000_000,
        )
        self.assertEqual(
            evidence.peak_fresh_mask,
            (False, False, False, False, True, True, False, False, False),
        )
        self.assertEqual(evidence.peak_age_frames[0], 2.0)
        self.assertEqual(evidence.peak_age_frames[4], 0.0)
        tracker = MultiRateContactTracker(self.localizer)
        tracked = tracker.update_evidence(evidence, sequence=1)
        self.assertEqual(tracked.profile, "TRACK")
        self.assertEqual(tracked.fresh_peak_count, 2)
        self.assertEqual(tracked.measurement.acquisition_profile, "TRACK")
        self.assertEqual(sum(tracked.measurement.fresh_point_mask_45), 6)

    def test_track13_uses_all_sentinels_but_fits_centres_only_in_two_rois(self):
        wavelength, baseline = synthetic_spectrum()
        tracker = MultiRateContactTracker(self.localizer)
        map_evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            baseline,
            baseline_adc_codes=baseline,
            timestamp_ns=1,
            profile="MAP",
        )
        tracker.update_evidence(map_evidence, sequence=1)

        current = list(baseline)
        changed_segment = 2
        current[sentinel_index(changed_segment)] = min(
            4095, current[sentinel_index(changed_segment)] + 500
        )
        fresh = track_indices(4, 7)
        evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            timestamp_ns=2,
            profile="TRACK",
            fresh_point_indices=fresh,
            map_age_frames=1,
        )
        self.assertEqual(evidence.peak_activity_fresh_mask, (True,) * 9)
        self.assertEqual(
            evidence.peak_fresh_mask,
            (False, False, False, False, True, False, False, True, False),
        )
        tracked = tracker.update_evidence(evidence, sequence=2)
        self.assertEqual(tracked.fresh_peak_count, 9)
        self.assertEqual(tracked.measurement.dominant_sensor_id, 3)
        self.assertIn(
            "fresh_activity_sentinel_without_fitted_peak_center",
            tracked.quality_flags,
        )
        self.assertEqual(tracked.measurement.acquisition_profile, "TRACK")
        self.assertEqual(sum(tracked.measurement.fresh_point_mask_45), 13)

    def test_track11_single_uses_all_sentinels_but_fits_only_primary_roi(self):
        wavelength, baseline = synthetic_spectrum()
        tracker = MultiRateContactTracker(self.localizer)
        map_evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            baseline,
            baseline_adc_codes=baseline,
            timestamp_ns=1,
            profile="MAP",
        )
        tracker.update_evidence(map_evidence, sequence=1)

        current = list(baseline)
        changed_segment = 2
        current[sentinel_index(changed_segment)] = min(
            4095, current[sentinel_index(changed_segment)] + 500
        )
        fresh = track_single_indices(4)
        evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            timestamp_ns=2,
            profile="TRACK11-SINGLE",
            fresh_point_indices=fresh,
            map_age_frames=1,
        )
        self.assertEqual(evidence.profile, "TRACK11_SINGLE")
        self.assertEqual(evidence.peak_activity_fresh_mask, (True,) * 9)
        self.assertEqual(
            evidence.peak_fresh_mask,
            (False, False, False, False, True, False, False, False, False),
        )
        tracked = tracker.update_evidence(evidence, sequence=2)
        self.assertEqual(tracked.profile, "TRACK11_SINGLE")
        self.assertEqual(tracked.fresh_peak_count, 9)
        self.assertEqual(tracked.measurement.dominant_sensor_id, 3)
        self.assertEqual(
            tracked.measurement.acquisition_profile, "TRACK11_SINGLE"
        )
        self.assertEqual(sum(tracked.measurement.fresh_point_mask_45), 11)

    def test_survey9_sentinels_refresh_activity_without_any_peak_centre_fit(self):
        wavelength, baseline = synthetic_spectrum()
        tracker = MultiRateContactTracker(self.localizer)
        map_evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            baseline,
            baseline_adc_codes=baseline,
            timestamp_ns=10,
            profile="MAP",
        )
        tracker.update_evidence(map_evidence, sequence=10)

        current = list(baseline)
        changed_segment = 6
        current[sentinel_index(changed_segment)] = min(
            4095, current[sentinel_index(changed_segment)] + 500
        )
        survey9 = tuple(sentinel_index(segment) for segment in range(9))
        evidence = self.localizer.extract_spectrum_evidence(
            wavelength,
            current,
            baseline_adc_codes=baseline,
            timestamp_ns=11,
            profile="SURVEY",
            fresh_point_indices=survey9,
            map_age_frames=1,
        )
        self.assertEqual(evidence.peak_activity_fresh_mask, (True,) * 9)
        self.assertEqual(evidence.peak_fresh_mask, (False,) * 9)
        tracked = tracker.update_evidence(evidence, sequence=11)
        self.assertEqual(tracked.fresh_peak_count, 9)
        self.assertEqual(tracked.measurement.dominant_sensor_id, 7)
        self.assertIn(
            "fresh_activity_sentinel_without_fitted_peak_center",
            tracked.quality_flags,
        )

    def test_freshness_sources_are_not_allowed_to_disagree(self):
        wavelength, baseline = synthetic_spectrum()
        with self.assertRaisesRegex(ValueError, "choose fresh_point_indices"):
            self.localizer.extract_spectrum_evidence(
                wavelength,
                baseline,
                baseline_adc_codes=baseline,
                profile="TRACK",
                fresh_point_indices=(21, 22, 23),
                sample_offsets_2us=np.zeros(45, dtype=int),
            )

    def test_multirate_tracker_is_causal_and_accepts_exact_60hz(self):
        tracker = MultiRateContactTracker(self.localizer)
        shifts = np.zeros(9)
        shifts[4] = 25.0
        scores = np.zeros(9)
        scores[4] = 4.0
        first = tracker.update(
            shifts,
            timestamp_ns=0,
            profile="MAP",
            sequence=1,
            peak_response_scores=scores,
        )
        shifts[4] = 30.0
        scores[4] = 5.0
        track6 = [5 * peak + local for peak in (4, 5) for local in (1, 2, 3)]
        second = tracker.update(
            shifts,
            timestamp_ns=16_666_667,
            profile="TRACK",
            sequence=2,
            fresh_point_indices=track6,
            minimum_fresh_points_per_peak=3,
            peak_response_scores=scores,
        )
        self.assertEqual(first.track_state, "MEASURED")
        self.assertEqual(second.fresh_peak_count, 2)
        self.assertTrue(second.nyquist_met)
        self.assertTrue(second.recommended_rate_met)
        predicted = tracker.predict(33_333_334)
        self.assertEqual(predicted.track_state, "PREDICTED")
        self.assertIn("predicted_not_new_spectrum", predicted.quality_flags)
        lost = tracker.predict(200_000_000)
        self.assertEqual(lost.track_state, "LOST")
        self.assertEqual(lost.confidence, 0.0)

    def test_sequence_gap_is_explicit(self):
        tracker = MultiRateContactTracker(self.localizer)
        shifts = np.zeros(9)
        shifts[1] = 20.0
        tracker.update(shifts, timestamp_ns=0, sequence=10)
        result = tracker.update(shifts, timestamp_ns=20_000_000, sequence=12)
        self.assertIn("sequence_gap_aged_stale_peaks", result.quality_flags)

    def test_machine_training_record_enforces_indentation_limit_and_no_area_label(self):
        shifts = np.zeros(9)
        shifts[8] = 30.0
        estimate = self.localizer.estimate_peak_shifts(shifts, timestamp_ns=30)
        record = make_machine_training_record(
            estimate,
            session_id="session-a",
            event_id="event-a",
            labelled_work_x_mm=-0.4,
            labelled_work_y_mm=3.6,
            indentation_mm=0.0,
            profile="MAP",
        )
        self.assertEqual(record["sensor_channel"], 1)
        self.assertFalse(record["area_label"]["available"])
        self.assertIsNone(record["area_label"]["physical_contact_area_mm2"])
        with self.assertRaisesRegex(ValueError, "0..0.8"):
            make_machine_training_record(
                estimate,
                session_id="session-a",
                event_id="event-b",
                labelled_work_x_mm=0.0,
                labelled_work_y_mm=0.0,
                indentation_mm=0.81,
                profile="MAP",
            )

    def test_compute_benchmark_is_well_inside_15hz_frame_budget(self):
        shifts = np.arange(9, dtype=float) * 3.0
        result = benchmark_localizer(self.localizer, shifts, repetitions=50)
        self.assertTrue(result["meets_15hz_compute_budget"])
        self.assertLess(result["p99_ms"], result["budget_ms_at_15hz"])


if __name__ == "__main__":
    unittest.main()
