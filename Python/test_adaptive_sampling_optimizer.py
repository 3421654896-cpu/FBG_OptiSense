import unittest

import numpy as np

from adaptive_sampling_optimizer import (
    PeakRegion,
    build_peak_template,
    equally_spaced_peak_indices,
    estimate_plan_shifts,
    estimate_template_shift,
    optimize_ch1_sampling,
)


def _gaussian(x: np.ndarray) -> np.ndarray:
    return 0.04 + 0.80 * np.exp(-0.5 * (x / 0.16) ** 2)


def _asymmetric(x: np.ndarray) -> np.ndarray:
    peak = 0.68 * np.exp(-0.5 * (x / 0.18) ** 2)
    return 0.03 + peak * (1.0 + 0.42 * np.tanh(x / 0.12))


def _rippled(x: np.ndarray) -> np.ndarray:
    envelope = 0.70 * np.exp(-0.5 * (x / 0.19) ** 2)
    return 0.04 + envelope * (1.0 + 0.08 * np.sin(2.0 * np.pi * x / 0.075))


class AdaptiveSamplingOptimizerTests(unittest.TestCase):
    def test_default_nine_peak_plan_is_ordered_unique_and_has_45_points(self):
        x = np.linspace(1525.0, 1565.0, 4001)
        centres = np.linspace(1527.0, 1563.0, 9)
        y = 0.025 + 0.0002 * (x - np.mean(x))
        regions = []
        for number, centre in enumerate(centres):
            local = x - centre
            y += (0.50 + 0.025 * number) * np.exp(
                -0.5 * (local / (0.14 + 0.004 * number)) ** 2
            ) * (1.0 + 0.08 * np.tanh(local / 0.08))
            regions.append(PeakRegion(centre - 0.65, centre + 0.65))

        plan = optimize_ch1_sampling(x, y, regions)

        self.assertEqual(plan.total_points, 45)
        self.assertEqual(len(plan.peaks), 9)
        self.assertTrue(np.all(np.diff(plan.selected_indices) > 0))
        self.assertEqual(np.unique(plan.selected_indices).size, 45)
        for peak in plan.peaks:
            self.assertEqual(peak.selected_indices.size, 5)
            self.assertIn(peak.peak_index, peak.selected_indices)
            self.assertTrue(np.any(peak.selected_indices < peak.peak_index))
            self.assertTrue(np.any(peak.selected_indices > peak.peak_index))
            self.assertGreater(peak.projected_shift_information, 0.0)

    def test_configurable_budget_keeps_five_anchors_and_adds_unique_points(self):
        x = np.linspace(-0.65, 0.65, 301)
        y = _gaussian(x)
        five = optimize_ch1_sampling(
            x, y, [(-0.65, 0.65)], expected_peak_count=1
        )
        seven = optimize_ch1_sampling(
            x,
            y,
            [(-0.65, 0.65)],
            points_per_peak=7,
            expected_peak_count=1,
        )
        self.assertEqual(seven.total_points, 7)
        self.assertTrue(
            set(five.selected_indices).issubset(set(seven.selected_indices))
        )
        self.assertTrue(np.all(np.diff(seven.selected_indices) > 0))

    def test_unstable_and_high_noise_samples_are_avoided_when_neighbors_exist(self):
        x = np.linspace(-0.65, 0.65, 401)
        y = _rippled(x)
        stable = np.ones(x.size, dtype=bool)
        stable[(x > -0.19) & (x < -0.14)] = False
        stable[(x > 0.09) & (x < 0.13)] = False
        noise = np.full(x.size, 0.002)
        noise[(x > 0.16) & (x < 0.21)] = 0.08

        plan = optimize_ch1_sampling(
            x,
            y,
            [(-0.65, 0.65)],
            expected_peak_count=1,
            stable_mask=stable,
            sample_noise_std=noise,
        )

        selected = plan.selected_indices
        self.assertTrue(np.all(stable[selected]))
        self.assertFalse(np.any(noise[selected] == 0.08))

    def test_per_sample_noise_is_supported_by_shift_estimator(self):
        x = np.linspace(-0.65, 0.65, 401)
        y = _asymmetric(x)
        peak = optimize_ch1_sampling(
            x, y, [(-0.65, 0.65)], expected_peak_count=1
        ).peaks[0]
        true_shift = 0.021
        observed = np.interp(peak.selected_wavelength_nm - true_shift, x, y)
        noise = np.array([0.002, 0.003, 0.002, 0.004, 0.002])
        estimate = estimate_template_shift(
            peak,
            observed,
            noise_std=noise,
            shift_bounds_nm=(-0.08, 0.08),
        )
        self.assertTrue(estimate.accepted, estimate.reason)
        self.assertLess(abs(estimate.shift_nm - true_shift), 0.001)
        self.assertGreater(estimate.projected_fisher_information, 0.0)

    def test_full_plan_estimator_preserves_peak_order(self):
        x = np.linspace(-3.0, 3.0, 1201)
        centres = (-2.0, 0.0, 2.0)
        y = np.full_like(x, 0.03)
        regions = []
        for centre in centres:
            y += 0.60 * np.exp(-0.5 * ((x - centre) / 0.16) ** 2)
            regions.append((centre - 0.55, centre + 0.55))
        plan = optimize_ch1_sampling(
            x, y, regions, expected_peak_count=3
        )
        shifts = (-0.018, 0.007, 0.026)
        frame = np.concatenate(
            [
                np.interp(peak.selected_wavelength_nm - shift, x, y)
                for peak, shift in zip(plan.peaks, shifts, strict=True)
            ]
        )
        estimates = estimate_plan_shifts(
            plan,
            frame,
            noise_std=0.002,
            shift_bounds_nm=(-0.06, 0.06),
        )
        self.assertEqual(len(estimates), 3)
        self.assertTrue(all(item.accepted for item in estimates))
        np.testing.assert_allclose(
            [item.shift_nm for item in estimates], shifts, atol=0.001
        )

    def test_estimator_rejects_unobservable_flat_measurement(self):
        x = np.linspace(-0.65, 0.65, 301)
        y = _gaussian(x)
        peak = optimize_ch1_sampling(
            x, y, [(-0.65, 0.65)], expected_peak_count=1
        ).peaks[0]
        estimate = estimate_template_shift(
            peak,
            np.full(5, 0.12),
            noise_std=0.002,
            shift_bounds_nm=(-0.08, 0.08),
        )
        self.assertFalse(estimate.accepted)
        self.assertNotEqual(estimate.reason, "ok")

    def test_optimized_points_match_or_beat_equal_spacing_on_measured_shapes(self):
        x = np.linspace(-0.65, 0.65, 401)
        region = PeakRegion(-0.65, 0.65)
        shapes = {
            "gaussian": _gaussian,
            "asymmetric": _asymmetric,
            "multi_ripple": _rippled,
        }
        improvement_count = 0
        for shape_number, (name, shape) in enumerate(shapes.items()):
            with self.subTest(shape=name):
                teacher = shape(x)
                optimized = optimize_ch1_sampling(
                    x, teacher, [region], expected_peak_count=1
                ).peaks[0]
                equal_indices = equally_spaced_peak_indices(x, region, 5)
                equal = build_peak_template(x, teacher, region, equal_indices)
                rng = np.random.default_rng(8100 + shape_number)
                optimized_error = []
                equal_error = []
                for _ in range(100):
                    true_shift = float(rng.uniform(-0.035, 0.035))
                    optimized_observed = np.interp(
                        optimized.selected_wavelength_nm - true_shift,
                        x,
                        teacher,
                    ) + rng.normal(0.0, 0.0025, 5)
                    equal_observed = np.interp(
                        equal.selected_wavelength_nm - true_shift,
                        x,
                        teacher,
                    ) + rng.normal(0.0, 0.0025, 5)
                    optimized_fit = estimate_template_shift(
                        optimized,
                        optimized_observed,
                        noise_std=0.0025,
                        shift_bounds_nm=(-0.08, 0.08),
                    )
                    equal_fit = estimate_template_shift(
                        equal,
                        equal_observed,
                        noise_std=0.0025,
                        shift_bounds_nm=(-0.08, 0.08),
                    )
                    self.assertTrue(optimized_fit.accepted, optimized_fit.reason)
                    self.assertTrue(equal_fit.accepted, equal_fit.reason)
                    optimized_error.append((optimized_fit.shift_nm - true_shift) ** 2)
                    equal_error.append((equal_fit.shift_nm - true_shift) ** 2)
                optimized_rmse = float(np.sqrt(np.mean(optimized_error)))
                equal_rmse = float(np.sqrt(np.mean(equal_error)))
                self.assertLessEqual(optimized_rmse, 1.05 * equal_rmse)
                if optimized_rmse < 0.85 * equal_rmse:
                    improvement_count += 1
        self.assertGreaterEqual(improvement_count, 2)

    def test_invalid_peak_count_and_too_small_budget_are_rejected(self):
        x = np.linspace(-1.0, 1.0, 101)
        y = _gaussian(x)
        with self.assertRaisesRegex(ValueError, "expected 9"):
            optimize_ch1_sampling(x, y, [(-0.8, 0.8)])
        with self.assertRaisesRegex(ValueError, "at least five"):
            optimize_ch1_sampling(
                x,
                y,
                [(-0.8, 0.8)],
                points_per_peak=4,
                expected_peak_count=1,
            )


if __name__ == "__main__":
    unittest.main()
