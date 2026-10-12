"""Regression checks for the optional, non-certified Makima display fitter."""

import unittest

import numpy as np

from temporary_peak_fit import fit_five_points as fit_gaussian
from temporary_peak_fit_makima import fit_five_points as fit_makima


class MakimaDisplayTests(unittest.TestCase):
    def test_interpolates_five_samples_without_mutating_input(self):
        x = 1535.0 + np.linspace(-0.4, 0.4, 5)
        y = np.array([45.0, 250.0, 720.0, 500.0, 130.0])
        original = y.copy()
        fit = fit_makima(x, y)
        self.assertTrue(fit['fit_available'])
        self.assertEqual(fit['fit_method'], 'makima_piecewise_cubic_5point')
        self.assertFalse(fit['optical_accuracy_verified'])
        self.assertTrue(x[0] <= fit['center_nm'] <= x[-1])
        np.testing.assert_allclose(
            np.interp(x, fit['curve_x_nm'], fit['curve_adc']), y, atol=0.3
        )
        np.testing.assert_array_equal(y, original)

    def test_default_gaussian_remains_separate_and_more_accurate_on_gaussian(self):
        x = 1535.0 + np.linspace(-0.6, 0.6, 5)
        actual_center = 1534.78
        y = 30.0 + 700.0 * np.exp(-0.5 * ((x - actual_center) / 0.25) ** 2)
        gaussian = fit_gaussian(x, y)
        makima = fit_makima(x, y)
        self.assertAlmostEqual(gaussian['center_nm'], actual_center, delta=0.002)
        self.assertGreater(abs(makima['center_nm'] - actual_center), 0.01)
        self.assertFalse(makima['optical_accuracy_verified'])


if __name__ == '__main__':
    unittest.main()
