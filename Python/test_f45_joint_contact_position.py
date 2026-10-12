import unittest
import numpy as np
from evaluate_f45_joint_contact_position import decode_contact_probability


class DecodeTest(unittest.TestCase):
    def test_split_contact_vote_and_class_order(self):
        p = np.array([[.3, .4, .3], [.05, .9, .05]])
        np.testing.assert_array_equal(decode_contact_probability(p, [3, 0, 7], .5), [3, 0])

    def test_invalid_inputs(self):
        for p, classes, threshold in [([[.5, .5]], [1, 2], .5),
                ([[.2, .2]], [0, 1], .5), ([[1, 0]], [0, 1], -1),
                ([[float('nan'), 0]], [0, 1], .5)]:
            with self.assertRaises(ValueError):
                decode_contact_probability(p, classes, threshold)


if __name__ == '__main__':
    unittest.main()
