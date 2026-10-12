import unittest
import numpy as np
from causal_scan_alignment import align


class AlignmentTests(unittest.TestCase):
    def test_linear_signal_exact_and_delayed(self):
        pt=np.array([0, 2, 4]); ct=pt+10
        result=align(pt, 3*pt+1, ct, 3*ct+1, max_gap_us=11)
        np.testing.assert_allclose(result['values'], [31,31,31])
        self.assertEqual(result['target_us'],10)
        self.assertEqual(result['available_us'],14)
        self.assertEqual(result['inherent_delay_us'],4)

    def test_no_extrapolation_or_missing_frames(self):
        with self.assertRaises(ValueError): align([0,12],[0,0],[10,14],[0,0],max_gap_us=20)
        with self.assertRaises(ValueError): align([0,2],[0,0],[30,32],[0,0],max_gap_us=20)
        with self.assertRaises(ValueError): align([0,2],[0,0],[10,9],[0,0],max_gap_us=20)

    def test_channels_are_not_blended(self):
        result=align([0,2],[7,20],[10,12],[7,20],max_gap_us=11)
        np.testing.assert_allclose(result['values'],[7,20])


if __name__=='__main__': unittest.main()
