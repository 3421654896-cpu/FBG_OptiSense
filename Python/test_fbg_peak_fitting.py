import unittest

import numpy as np

from fbg_peak_fitting import (
    AdaptivePeakTracker,
    FBGPeakFit,
    PeakDisplayNormalizer,
    TemperaturePeakTracker,
    build_segments,
    detect_wavelength_gaps,
    fit_channel_segments,
    fit_dense_reflection_spectrum,
    fit_fbg_segment,
    mask_unconnected_fbg_segments,
    repair_isolated_outliers,
    required_display_gain,
    rounded_display_curve,
)


class FBGPeakFittingTests(unittest.TestCase):
    def test_gaussian_center_is_sub_sample_accurate(self):
        rng = np.random.default_rng(12)
        x = np.array(
            [1527.415, 1527.504, 1527.625, 1527.706, 1527.810,
             1527.907, 1528.027, 1528.105, 1528.199, 1528.298, 1528.410]
        )
        true_center = 1527.9107
        baseline = 1.05 + 0.03 * (x - np.mean(x))
        y = baseline + 0.72 * np.exp(-0.5 * ((x - true_center) / 0.41) ** 2)
        y += rng.normal(0.0, 0.0025, x.size)
        fit = fit_fbg_segment(x, y, min_prominence_v=0.02)
        self.assertTrue(fit.valid)
        self.assertLess(abs(fit.center_nm - true_center), 0.010)

    def test_dense_reflection_scan_finds_every_bragg_center(self):
        rng = np.random.default_rng(405)
        x = np.linspace(1525.0, 1565.0, 2001)
        centres = np.asarray((1532.47228, 1544.45520, 1559.91610))
        y = np.full_like(x, 0.08)
        for centre, amplitude, sigma in zip(
            centres, (0.42, 0.68, 0.51), (0.24, 0.31, 0.27), strict=True
        ):
            y += amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
        y += rng.normal(0.0, 0.0012, x.size)

        result = fit_dense_reflection_spectrum(x, y)
        fitted = np.asarray([fit.center_nm for fit in result.fits])
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.rejected_count, 0)
        self.assertEqual(len(fitted), 3)
        np.testing.assert_allclose(fitted, centres, atol=0.003)

    def test_stress_peak_may_be_fitted_at_a_section_edge(self):
        x = np.linspace(1543.9, 1544.9, 11)
        y = 0.12 + 0.85 * np.exp(-0.5 * ((x - x[-1]) / 0.31) ** 2)
        stable_fit = fit_fbg_segment(x, y, 0.02)
        stress_fit = fit_fbg_segment(x, y, 0.02, allow_edge_peak=True)
        self.assertFalse(stable_fit.valid)
        self.assertTrue(stress_fit.valid)
        self.assertLessEqual(abs(stress_fit.center_nm - x[-1]), 0.015)

    def test_segments_are_never_joined(self):
        x = np.array([1.0, 1.1, 1.2, 4.0, 4.1, 4.2])
        segments = build_segments(len(x), {2})
        self.assertEqual([(s.start, s.stop) for s in segments], [(0, 3), (3, 6)])

    def test_nonuniform_samples_do_not_split_a_peak(self):
        x = np.array([1.00, 1.08, 1.17, 1.34, 1.43, 4.50, 4.59, 4.71])
        gaps = detect_wavelength_gaps(x)
        self.assertEqual(gaps, {4})

    def test_optical_channel_mapping_masks_unconnected_sections(self):
        fits = [FBGPeakFit(valid=True) for _ in range(9)]
        mask_unconnected_fbg_segments(fits, 2)
        self.assertEqual([fit.valid for fit in fits], [True] * 3 + [False] * 6)

    def test_local_nine_grating_ch1_keeps_all_stress_sections(self):
        fits = [FBGPeakFit(valid=True) for _ in range(9)]
        mask_unconnected_fbg_segments(
            fits, 1, stress_ch1_all_segments=True
        )
        self.assertEqual([fit.valid for fit in fits], [True] * 9)

    def test_local_adaptive_stress_keeps_all_detected_sections_on_ch0(self):
        fits = [FBGPeakFit(valid=True) for _ in range(8)]
        mask_unconnected_fbg_segments(
            fits, 0, stress_all_segments=True
        )
        self.assertEqual([fit.valid for fit in fits], [True] * 8)

    def test_digital_gain_only_applies_at_maximum_hardware_gain(self):
        gain = required_display_gain(
            0.10, 0.20, hardware_is_maximum=True, target_peak_v=1.05
        )
        self.assertAlmostEqual(gain, 4.75)
        self.assertEqual(
            required_display_gain(
                0.10, 0.20, hardware_is_maximum=False, target_peak_v=1.05
            ),
            1.0,
        )

    def test_digital_gain_does_not_promote_adc_noise(self):
        self.assertEqual(
            required_display_gain(
                0.10, 0.002, hardware_is_maximum=True, target_peak_v=1.05
            ),
            1.0,
        )

    def test_strong_asymmetric_peak_is_not_lost_to_slight_residual_mismatch(self):
        x = np.linspace(1540.90, 1541.30, 9)
        main = 1.10 * np.exp(-0.5 * ((x - 1541.10) / 0.075) ** 2)
        shoulder = 0.32 * np.exp(-0.5 * ((x - 1541.19) / 0.035) ** 2)
        fit = fit_fbg_segment(x, 0.03 + main + shoulder, 0.03, allow_edge_peak=True)
        self.assertTrue(fit.valid)
        self.assertGreater(fit.amplitude_v, 0.15)
        self.assertLess(abs(fit.center_nm - 1541.10), 0.08)

    def test_display_amplitude_stays_inside_measured_adc_envelope(self):
        x = np.linspace(0.0, 1.0, 11)
        y = np.array([0.1, 0.4, 0.9, 1.6, 2.2, 2.5, 2.5, 2.1, 1.4, 0.7, 0.2])
        fit = fit_fbg_segment(x, y, 0.02)
        self.assertTrue(fit.valid)
        self.assertLessEqual(
            fit.display_baseline_v + fit.display_amplitude_v,
            np.max(y) + 0.15 * np.ptp(y) + 1e-12,
        )

    def test_low_gain_segments_render_with_equal_peak_excursions(self):
        x = np.r_[np.linspace(0.0, 1.0, 11), np.linspace(4.0, 5.0, 11)]
        y = np.r_[
            0.01 + 0.06 * np.exp(-0.5 * ((x[:11] - 0.5) / 0.25) ** 2),
            0.01 + 0.12 * np.exp(-0.5 * ((x[11:] - 4.5) / 0.25) ** 2),
        ]
        segments = [slice(0, 11), slice(11, 22)]
        fits = fit_channel_segments(x, y, segments, 0.005)
        px, py = rounded_display_curve(x, y, segments, fits, display_scale=20.0)
        groups = np.split(py, np.flatnonzero(np.isnan(py)))
        excursions = [float(np.nanmax(group) - min(f.display_baseline_v, np.nanmin(group))) for group, f in zip(groups, fits)]
        self.assertAlmostEqual(excursions[0], excursions[1], places=5)

    def test_high_gain_digital_scaling_preserves_relative_peak_heights(self):
        x = np.r_[np.linspace(0.0, 1.0, 11), np.linspace(4.0, 5.0, 11)]
        y = np.r_[
            0.10 + 0.08 * np.exp(-0.5 * ((x[:11] - 0.5) / 0.25) ** 2),
            0.10 + 0.16 * np.exp(-0.5 * ((x[11:] - 4.5) / 0.25) ** 2),
        ]
        segments = [slice(0, 11), slice(11, 22)]
        fits = fit_channel_segments(x, y, segments, 0.005)
        _px, py = rounded_display_curve(
            x,
            y,
            segments,
            fits,
            display_scale=5.0,
            normalize_segment_heights=False,
        )
        groups = np.split(py, np.flatnonzero(np.isnan(py)))
        excursions = [
            float(np.nanmax(group) - fit.display_baseline_v)
            for group, fit in zip(groups, fits)
        ]
        self.assertGreater(excursions[1], 1.8 * excursions[0])

    def test_isolated_switching_dropout_is_repaired(self):
        y = np.array([0.35, 0.73, 0.03, 0.97, 1.31, 1.41, 1.17])
        repaired = repair_isolated_outliers(y)
        self.assertEqual(repaired[1], y[1])
        self.assertGreater(repaired[2], 0.80)

    def test_static_tracker_and_transient_response(self):
        tracker = AdaptivePeakTracker(1)
        rng = np.random.default_rng(8)
        static_output = []
        for value in 1544.455 + rng.normal(0.0, 0.007, 35):
            fit = fit_fbg_segment(
                np.linspace(value - 0.5, value + 0.5, 11),
                1.0 + np.exp(-0.5 * (np.linspace(-0.5, 0.5, 11) / 0.35) ** 2),
                0.02,
            )
            fit.center_nm = float(value)
            fit.valid = True
            static_output.append(tracker.update([fit])[0])
        self.assertLess(np.std(static_output[-15:]), 0.010)

        previous = static_output[-1]
        fit.center_nm = 1544.555
        stepped = tracker.update([fit])[0]
        self.assertGreater(stepped - previous, 0.075)

    def test_tracker_rejects_one_frame_outlier_but_confirms_a_real_shift(self):
        tracker = AdaptivePeakTracker(1)
        fit = FBGPeakFit(
            center_nm=1540.0,
            center_std_pm=3.0,
            valid=True,
        )
        self.assertEqual(tracker.update([fit])[0], 1540.0)

        fit.center_nm = 1540.070
        self.assertAlmostEqual(tracker.update([fit])[0], 1540.0)
        fit.center_nm = 1540.0
        self.assertAlmostEqual(tracker.update([fit])[0], 1540.0)

        fit.center_nm = 1540.045
        self.assertAlmostEqual(tracker.update([fit])[0], 1540.0)
        confirmed = tracker.update([fit])[0]
        self.assertGreater(confirmed, 1540.035)

    def test_immediate_stress_tracker_publishes_same_frame(self):
        tracker = AdaptivePeakTracker(1, immediate_response=True)
        fit = FBGPeakFit(
            center_nm=1540.000,
            center_std_pm=3.0,
            r_squared=0.97,
            valid=True,
        )
        self.assertEqual(tracker.update([fit])[0], 1540.000)

        # Even a lower-quality, sub-90 pm movement is the current force signal
        # and must not wait for a confirmation frame.
        fit.center_nm = 1540.063
        fit.r_squared = 0.82
        moved = tracker.update([fit])[0]
        self.assertAlmostEqual(moved, 1540.063)
        self.assertTrue(tracker.measurement_accepted[0])

    def test_temperature_tracker_rejects_pm_scale_static_jitter(self):
        rng = np.random.default_rng(21)
        tracker = TemperaturePeakTracker(1)
        outputs = []
        for center in 1544.455 + rng.normal(0.0, 0.006, 40):
            fit = FBGPeakFit(center_nm=float(center), r_squared=0.95, valid=True)
            outputs.append(tracker.update([fit])[0])
        self.assertLess(np.std(outputs[-20:]) * 1000.0, 2.0)

    def test_temperature_mode_keeps_all_sections_on_both_high_gain_channels(self):
        fits_ch0 = [FBGPeakFit(valid=True) for _ in range(9)]
        fits_ch1 = [FBGPeakFit(valid=True) for _ in range(9)]
        mask_unconnected_fbg_segments(fits_ch0, 0, temperature_mode=True)
        mask_unconnected_fbg_segments(fits_ch1, 1, temperature_mode=True)
        self.assertEqual([fit.valid for fit in fits_ch0], [True] * 9)
        self.assertEqual([fit.valid for fit in fits_ch1], [True] * 9)

    def test_each_temperature_segment_can_have_its_own_display_gain(self):
        x = np.r_[np.linspace(0.0, 1.0, 11), np.linspace(4.0, 5.0, 11)]
        y = np.r_[
            0.10 + 0.10 * np.exp(-0.5 * ((x[:11] - 0.5) / 0.25) ** 2),
            0.10 + 0.20 * np.exp(-0.5 * ((x[11:] - 4.5) / 0.25) ** 2),
        ]
        segments = [slice(0, 11), slice(11, 22)]
        fits = fit_channel_segments(x, y, segments, 0.005)
        _px, py = rounded_display_curve(
            x, y, segments, fits, display_scale=np.array([4.0, 2.0])
        )
        groups = np.split(py, np.flatnonzero(np.isnan(py)))
        excursions = [
            float(np.nanmax(group) - fit.display_baseline_v)
            for group, fit in zip(groups, fits)
        ]
        self.assertAlmostEqual(excursions[0], excursions[1], places=4)

    def test_tracker_prefers_the_clean_weak_peak_shape(self):
        tracker = AdaptivePeakTracker(1)
        distorted = FBGPeakFit(
            center_nm=1532.192,
            center_std_pm=28.0,
            r_squared=0.81,
            valid=True,
        )
        clean = FBGPeakFit(
            center_nm=1532.253,
            center_std_pm=22.0,
            r_squared=0.94,
            valid=True,
        )
        tracker.update([distorted])
        reacquired = tracker.update([clean])[0]
        self.assertGreater(reacquired, 1532.240)
        for _ in range(3):
            held = tracker.update([distorted])[0]
        self.assertGreater(held, 1532.240)

    def test_low_gain_channels_are_display_normalized(self):
        x = np.linspace(0.0, 1.0, 11)
        sections = [slice(0, 11)]
        fits = []
        for amplitude in (0.6, 0.8, 0.04, 0.05):
            y = 1.1 + amplitude * np.exp(-0.5 * ((x - 0.5) / 0.3) ** 2)
            fits.append(fit_channel_segments(x, y, sections, 0.005))
        scales = PeakDisplayNormalizer().update(fits)
        self.assertAlmostEqual(scales[0], 1.0)
        self.assertAlmostEqual(scales[1], 1.0)
        self.assertGreater(scales[2], 15.0)
        self.assertGreater(scales[3], 12.0)

    def test_rounded_curve_contains_gap_separator(self):
        x = np.array([1.0, 1.1, 1.2, 4.0, 4.1, 4.2])
        y = np.array([0.0, 1.0, 0.0, 0.0, 0.8, 0.0])
        segments = build_segments(len(x), {2})
        fits = fit_channel_segments(x, y, segments, 0.01)
        plot_x, plot_y = rounded_display_curve(x, y, segments, fits)
        self.assertEqual(np.count_nonzero(np.isnan(plot_x)), 1)
        self.assertEqual(np.count_nonzero(np.isnan(plot_y)), 1)

    def test_fitted_peak_is_monotonic_on_both_sides(self):
        x = np.linspace(1543.9, 1544.9, 11)
        y = 0.2 + 0.9 * np.exp(-0.5 * ((x - 1544.43) / 0.32) ** 2)
        y += np.array([0.0, 0.01, -0.015, 0.012, -0.008, 0.015,
                       -0.012, 0.01, -0.008, 0.006, 0.0])
        fits = fit_channel_segments(x, y, [slice(0, len(x))], 0.02)
        plot_x, plot_y = rounded_display_curve(
            x, y, [slice(0, len(x))], fits, fitted=True
        )
        summit = int(np.argmax(plot_y))
        self.assertTrue(np.all(np.diff(plot_y[: summit + 1]) >= -1e-10))
        self.assertTrue(np.all(np.diff(plot_y[summit:]) <= 1e-10))


if __name__ == "__main__":
    unittest.main()
