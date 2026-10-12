import unittest
import numpy as np
from audit_candidate27_null import summarize


class NullControlTests(unittest.TestCase):
    def test_constant_control_has_zero_delta(self):
        result = summarize(np.full((256, 27), 100), np.arange(256)*.0272)
        self.assertEqual(result['delta_rms_codes'], 0)
        self.assertEqual(result['shape_delta_rms'], 0)

    def test_late_drift_is_not_hidden(self):
        t = np.arange(256)*.0272
        v = np.full((256, 27), 100)
        v[t >= 4] += 10
        self.assertEqual(summarize(v, t)['delta_rms_codes'], 10)

    def test_incomplete_time_invalid_code_and_unsorted_rejected(self):
        t = np.arange(256)*.0272; v = np.full((256, 27), 100.0)
        for values, times in ((v[:180], t[:180]), (v, t[::-1]), (v*np.nan, t), (v+4095, t)):
            with self.assertRaises(ValueError): summarize(values, times)


if __name__ == '__main__': unittest.main()
