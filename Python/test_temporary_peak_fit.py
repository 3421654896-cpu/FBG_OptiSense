import unittest
import numpy as np
from temporary_peak_fit import fit_five_points, fit_temporary_frame


class PeakFitTests(unittest.TestCase):
    def test_shifted_gaussian_not_pinned_to_middle(self):
        x = 1535+np.linspace(-.6, .6, 5)
        for shift in (-.22, 0., .18):
            y = 30+700*np.exp(-.5*((x-(1535+shift))/.25)**2)
            saved = y.copy()
            fit = fit_five_points(x, y)
            self.assertTrue(fit['valid'], fit)
            self.assertAlmostEqual(fit['center_nm'], 1535+shift, delta=.002)
            self.assertFalse(fit['optical_accuracy_verified'])
            np.testing.assert_array_equal(y, saved)
            self.assertGreaterEqual(min(fit['curve_x_nm']), x[0])
            self.assertLessEqual(max(fit['curve_x_nm']), x[-1])

    def test_unusual_finite_shapes_still_have_a_flagged_fit(self):
        x = np.arange(5)*.3+1535
        for y in ([5]*5, [10,100,4090,100,10], [10,100,200,300,500],
                  [10,300,50,400,10], [0,2,10,3,0]):
            result = fit_five_points(x, y)
            self.assertFalse(result['valid'])
            self.assertTrue(result['fit_available'])
            self.assertIsNotNone(result['center_nm'])
            self.assertEqual(len(result['curve_x_nm']), 101)
            self.assertIn('可能有问题', result['reason'])

    def test_nonfinite_data_is_the_only_kind_without_a_fit(self):
        x = np.arange(5)*.3+1535
        result = fit_five_points(x, [10,100,np.nan,100,10])
        self.assertFalse(result['valid'])
        self.assertFalse(result['fit_available'])
        self.assertIsNone(result['center_nm'])

    def test_unsorted_wavelength_points_are_sorted_and_flagged(self):
        x = 1535 + np.array([-.6, .3, 0., -.3, .6])
        y = 30 + 700*np.exp(-.5*((x-1535.1)/.25)**2)
        result = fit_five_points(x, y)
        self.assertTrue(result['fit_available'])
        self.assertFalse(result['valid'])
        self.assertAlmostEqual(result['center_nm'], 1535.1, delta=.005)
        self.assertIn('波长点顺序已重排', result['reason'])

    def test_raw_scale_invariance_and_frame_groups(self):
        x = np.linspace(1534.4,1535.6,5)
        y = 30+400*np.exp(-.5*((x-1535.1)/.25)**2)
        a, b = fit_five_points(x,y), fit_five_points(x,y*2)
        self.assertTrue(a['valid'] and b['valid'])
        self.assertAlmostEqual(a['center_nm'],b['center_nm'])
        rows = [dict(measured_nm=float(v+g*2)) for g in range(9) for v in x]
        values = np.tile(y,9)
        result = fit_temporary_frame(rows, values)
        self.assertEqual(len(result),9)
        self.assertTrue(all(r['valid'] for r in result))
        warmup = fit_temporary_frame(rows, values, False)
        self.assertTrue(all(not r['valid'] and r['center_nm'] is None for r in warmup))


if __name__ == '__main__':
    unittest.main()

