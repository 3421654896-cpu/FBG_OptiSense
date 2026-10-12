import unittest

import numpy as np

from calibrate_fast_flank_position_model import (
    _feature_families,
    _metrics,
    _raw_adc_feature_families,
    _ramp_feature_families,
    _resolve_template_inputs,
    _resolve_validation_inputs,
)


class FastFlankPositionCalibrationTests(unittest.TestCase):
    def test_features_are_finite_and_common_mode_invariant(self):
        base = np.arange(15 * 9, dtype=float).reshape(15, 9)
        shifted = base + 123.0
        first = _feature_families(base)
        second = _feature_families(shifted)
        for name in (
            "settled_centered",
            "settled_unit",
            "temporal_centered",
            "temporal_unit",
            "robust_shape",
            "robust_shape_unit",
        ):
            np.testing.assert_allclose(first[name], second[name])
            self.assertTrue(np.all(np.isfinite(first[name])))

    def test_metrics_use_true_euclidean_position_error(self):
        predicted = np.array([[3.0, 4.0], [0.0, 0.0]])
        truth = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = _metrics(predicted, truth)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["maximum_error_mm"], 5.0)
        self.assertAlmostEqual(result["mean_error_mm"], 2.5)
        self.assertAlmostEqual(result["within_4mm_percent"], 50.0)

    def test_validation_inputs_are_a_distinct_later_repeat(self):
        values = _resolve_validation_inputs(__import__("pathlib").Path("captures"), 4)
        self.assertEqual(len(values), 17)
        self.assertTrue(str(values[0][0]).endswith("p01_0mm_clean_validation_r4_20260906.json"))
        self.assertEqual(values[-1][1], 4)

    def test_explicit_template_keeps_depth_and_repeat_names_unambiguous(self):
        values = _resolve_template_inputs(
            __import__("pathlib").Path("captures"),
            "contact_p{point:02d}_0p10mm_physical_r{repeat}.json",
            (2, 4),
        )
        self.assertEqual(len(values), 34)
        self.assertTrue(str(values[0][0]).endswith("contact_p01_0p10mm_physical_r2.json"))
        self.assertEqual(values[16][1], 2)
        self.assertTrue(str(values[17][0]).endswith("contact_p01_0p10mm_physical_r4.json"))
        self.assertEqual(values[-1][1], 4)

    def test_raw_fractional_features_ignore_common_adc_scale(self):
        baseline = np.tile(np.arange(20.0, 38.0), (64, 1))
        hold = np.tile(np.arange(22.0, 40.0), (20, 1))
        apparent = _feature_families(np.arange(20 * 9, dtype=float).reshape(20, 9))
        first = _raw_adc_feature_families(baseline, hold, apparent)
        second = _raw_adc_feature_families(baseline * 2.0, hold * 2.0, apparent)
        for name in (
            "adc_fractional",
            "adc_fractional_unit",
            "adc_log_ratio",
            "adc_log_ratio_unit",
            "adc_pair_contrast",
            "adc_pair_contrast_unit",
        ):
            np.testing.assert_allclose(first[name], second[name], atol=1e-12)

    def test_ramp_features_keep_nine_sensor_shape(self):
        baseline = np.tile(np.arange(20.0, 38.0), (64, 1))
        ramp = np.tile(np.arange(21.0, 39.0), (16, 1))
        apparent = np.arange(16 * 9, dtype=float).reshape(16, 9)
        result = _ramp_feature_families(baseline, ramp, apparent)
        self.assertEqual(result["ramp_adc_log_ratio_peak"].shape, (9,))
        self.assertEqual(result["ramp_apparent_peak"].shape, (9,))
        self.assertEqual(result["ramp_combined_unit"].shape, (45,))
        self.assertTrue(np.all(np.isfinite(result["ramp_combined_unit"])))


if __name__ == "__main__":
    unittest.main()
