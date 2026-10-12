import unittest

import numpy as np

from evaluate_candidate27_feature_variants import _features


class Candidate27PositionFeatureTests(unittest.TestCase):
    def test_delta_plus_energy_is_finite_and_scale_invariant(self):
        baseline = np.arange(1.0, 28.0)[None, :].repeat(8, axis=0)
        delta = np.linspace(-3.0, 5.0, 27)
        first = _features(baseline, baseline[:2] + delta)["delta_plus_energy"]
        second = _features(baseline, baseline[:2] + 4.0 * delta)["delta_plus_energy"]
        self.assertEqual(first.shape, (2, 36))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_energy_half_uses_nine_fbg_groups(self):
        baseline = np.full((8, 27), 100.0)
        sample = baseline[:1].copy()
        sample[0, 12:15] += 20.0
        feature = _features(baseline, sample)["delta_plus_energy"][0]
        energy = feature[27:]
        self.assertEqual(int(np.argmax(energy)), 4)
        self.assertAlmostEqual(float(np.linalg.norm(energy)), 1.0)


if __name__ == "__main__":
    unittest.main()
