import copy
import unittest
from causal_transient_features import extract


class CausalFeaturesTests(unittest.TestCase):
    def curve(self):
        return [dict(start_us=i*100, end_us=i*100+50,
                     direct_adc=[i]*6) for i in range(5)]

    def test_future_samples_cannot_change_features(self):
        curve = self.curve()
        before = extract(curve, 300)
        curve[3]['direct_adc'] = [4000]*6
        curve[4]['direct_adc'] = [3000]*6
        self.assertEqual(before, extract(curve, 300))
        self.assertEqual(before['available_us'], 250)

    def test_conversion_start_is_not_availability(self):
        with self.assertRaises(ValueError):
            extract(self.curve(), 225)

    def test_nonmonotonic_rejected(self):
        curve = self.curve()
        curve[2]['start_us'] = 140
        with self.assertRaises(ValueError):
            extract(curve, 300)

    def test_input_not_modified(self):
        curve = self.curve()
        original = copy.deepcopy(curve)
        extract(curve, 300)
        self.assertEqual(curve, original)
