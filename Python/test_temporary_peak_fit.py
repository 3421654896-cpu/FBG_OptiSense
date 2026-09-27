import unittest

import numpy as np

from temporary_peak_fit import fit_five_points


class FivePointCubicFitTests(unittest.TestCase):
    def test_asymmetric_peak_uses_five_point_makima_center(self):
        """An asymmetric peak must not be pulled to a symmetric Gaussian centre."""

        x = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        true_center = 0.30
        sigma = np.where(x < true_center, 0.80, 1.40)
        y = 100.0 + 1000.0 * np.exp(
            -0.5 * ((x - true_center) / sigma) ** 2
        )

        result = fit_five_points(x, y)

        self.assertTrue(result['fit_available'])
        self.assertEqual(result['fit_method'], 'makima_piecewise_cubic_5point')
        self.assertLess(abs(result['center_nm'] - true_center), 0.08)
        self.assertEqual(result['input_point_count'], 5)

    def test_rendered_curve_passes_through_all_five_measured_points(self):
        """The displayed cubic peak must preserve the five physical samples."""

        x = np.asarray([1528.54, 1528.78, 1528.92, 1529.12, 1529.30])
        y = np.asarray([1882.0, 3676.0, 4049.0, 3111.0, 1651.0])

        result = fit_five_points(x, y)
        curve_x = np.asarray(result['curve_x_nm'], dtype=float)
        curve_y = np.asarray(result['curve_adc'], dtype=float)
        rendered_at_samples = np.interp(x, curve_x, curve_y)

        np.testing.assert_allclose(rendered_at_samples, y, atol=0.5, rtol=0.0)
        self.assertGreater(result['center_nm'], x[1])
        self.assertLess(result['center_nm'], x[3])
        self.assertGreater(result['peak_adc'], max(y[0], y[-1]))

    def test_five_point_frame_keeps_one_result_per_peak(self):
        from temporary_peak_fit import fit_temporary_frame

        rows = [
            {'measured_nm': 1530.0 + 0.1 * index}
            for index in range(10)
        ]
        values = [100, 300, 900, 500, 150, 80, 260, 700, 420, 120]

        results = fit_temporary_frame(rows, values, warmed_up=True)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result['input_point_count'] == 5 for result in results))
        self.assertTrue(all(result['fit_method'] == 'makima_piecewise_cubic_5point'
                            for result in results))

    def test_irregular_shape_retains_nontrivial_quality_diagnostic(self):
        """Exact interpolation must not turn every malformed peak into R²=1."""

        result = fit_five_points(
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [68.0, 314.0, 211.0, 54.0, 180.0],
        )

        self.assertLess(result['r_squared'], 0.90)
        self.assertGreater(result['rmse_adc'], 0.0)
        self.assertFalse(result['valid'])


if __name__ == '__main__':
    unittest.main()
