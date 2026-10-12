import unittest
import numpy as np
from evaluate_f45_causal_alignment import align_previous_current


class AlignmentTest(unittest.TestCase):
    def test_linear_signal_exact_and_prefix_invariant(self):
        t=np.arange(10)[:,None]+np.array([0,.2,.8])[None,:]
        target,result=align_previous_current(t,3*t+2)
        np.testing.assert_allclose(result,np.repeat((3*target+2)[:,None],3,axis=1))
        _,prefix=align_previous_current(t[:5],3*t[:5]+2)
        np.testing.assert_allclose(prefix,result[:4])

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            align_previous_current([[0,2],[1,3]],[[0,0],[0,0]])


if __name__=='__main__': unittest.main()
