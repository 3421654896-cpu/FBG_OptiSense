import unittest
import numpy as np
from f45_shape_amplitude import features


class ShapeAmplitudeTests(unittest.TestCase):
    def test_gain_is_retained_separate_from_shape(self):
        base=np.tile([10.,20.,50.,30.,15.],18)
        result=features((2*base+3)[None,:],np.tile(base,(64,1)))
        self.assertEqual(result.shape,(1,126))
        np.testing.assert_allclose(result[0,:90],0,atol=1e-12)
        np.testing.assert_allclose(result[0,90:108],1)
        self.assertTrue(np.all(result[0,108:]>0))

    def test_asymmetry_and_shared_change_retained(self):
        base=np.tile([10.,20.,50.,30.,15.],18)
        changed=base.copy(); changed[::5]+=4
        result=features(changed[None,:],np.tile(base,(64,1)))
        self.assertGreater(np.linalg.norm(result[0,:90]),0)
        self.assertTrue(np.all(result[0,108:]>0))

    def test_flat_baseline_safe(self):
        result=features(np.zeros((1,90)),np.zeros((64,90)))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result,0)


if __name__=='__main__': unittest.main()
