import unittest
import numpy as np
from evaluate_separate_contact_position import shape


class PositionShapeTests(unittest.TestCase):
    def test_scale_invariance_and_zero(self):
        x=np.arange(270,dtype=float).reshape(2,135)
        np.testing.assert_allclose(shape(x),shape(x*8))
        np.testing.assert_array_equal(shape(np.zeros((1,135))),np.zeros((1,90)))


if __name__=='__main__': unittest.main()
