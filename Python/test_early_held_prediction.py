import copy
import unittest
import numpy as np
from evaluate_early_held_prediction import features


class CausalFeatureTests(unittest.TestCase):
    def test_future_and_teacher_cannot_change_input(self):
        record = {'curve':[{'end_us':t,'direct_adc':[i]*6}
                           for i,t in enumerate([55,305,705,755,10055])],
                  'teacher':{'median':[40]*6}}
        before = features(record)
        changed = copy.deepcopy(record)
        changed['teacher']['median'] = [400]*6
        for row in changed['curve']:
            if row['end_us'] > 750: row['direct_adc'] = [4095]*6
        np.testing.assert_array_equal(before[0],features(changed)[0])
        self.assertEqual(before[2],705)

    def test_no_early_samples_fails(self):
        with self.assertRaises(ValueError):
            features({'curve':[{'end_us':800,'direct_adc':[1]*6}]})


if __name__ == '__main__': unittest.main()
