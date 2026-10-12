import unittest
import numpy as np
from evaluate_position_depth_segments import infer


class TestSegments(unittest.TestCase):
    def test_intermediate_depth(self):
        r=infer(np.ones(45),[(8,np.zeros(45),np.ones(45)*2,.1,.3),(9,np.ones(45)*5,np.ones(45)*6,.1,.3)])
        self.assertEqual(r['point'],8)
        self.assertAlmostEqual(r['estimated_indentation_mm'],.2)

    def test_no_extrapolation(self):
        r=infer(np.ones(45)*3,[(8,np.zeros(45),np.ones(45)*2,.1,.3)])
        self.assertAlmostEqual(r['estimated_indentation_mm'],.3)

    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError): infer(np.ones(45),[(8,np.zeros(45),np.zeros(45),.1,.3)])


if __name__=='__main__': unittest.main()
