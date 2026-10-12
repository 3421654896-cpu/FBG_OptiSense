import unittest
import numpy as np
from idle_spectral_noise import whitening_matrix


class IdleNoiseTests(unittest.TestCase):
    def test_noisy_direction_is_downweighted(self):
        rng = np.random.default_rng(15)
        samples = rng.normal(size=(2000, 45))
        samples[:, 0] *= 20
        matrix = whitening_matrix(samples)
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0))
        self.assertLess(np.linalg.norm(matrix[0]), np.linalg.norm(matrix[1]) / 3)

    def test_constant_input_is_finite(self):
        self.assertTrue(np.isfinite(whitening_matrix(np.zeros((100, 45)))).all())

    def test_invalid_vectors_rejected(self):
        for samples in (np.zeros((99, 45)), np.zeros((100, 44)), np.full((100, 45), np.nan)):
            with self.assertRaises(ValueError):
                whitening_matrix(samples)


if __name__ == '__main__':
    unittest.main()
