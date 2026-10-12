import unittest
import numpy as np
from evaluate_current_three_positions import nearest, spectral_feature


class TestNearest(unittest.TestCase):
    def test_uses_features_without_position_labels(self):
        index, distances = nearest([[1,2],[5,6],[9,10]], [5.1,6.2])
        self.assertEqual(index,1)
        self.assertLess(distances[1],distances[0])

    def test_rejects_invalid_features(self):
        for query in ([1], [float('nan'),2]):
            with self.assertRaises(ValueError): nearest([[1,2]],query)

    def test_shape_ignores_uniform_gain_and_offset(self):
        baseline=np.tile([3,10,25,15,5],9)
        np.testing.assert_allclose(spectral_feature(baseline,baseline*3+7,'shape_delta'),0,atol=1e-12)

    def test_shape_retains_spectral_change(self):
        baseline=np.tile([3,10,25,15,5],9)
        shifted=np.tile([3,5,15,25,10],9)
        self.assertGreater(np.linalg.norm(spectral_feature(baseline,shifted,'shape_delta')),0.1)

    def test_raw_unchanged(self):
        baseline=np.arange(45)
        np.testing.assert_array_equal(spectral_feature(baseline,baseline+2,'raw_delta'),np.full(45,2))


if __name__=='__main__': unittest.main()
