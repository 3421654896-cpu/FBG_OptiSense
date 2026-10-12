import unittest

import numpy as np

from candidate27_grouped_regressor import build_regressor


class GroupedRegressorTests(unittest.TestCase):
    def test_regressors_return_xy(self):
        x = np.asarray([[0, 0], [0, .1], [1, 1], [1, .9], [2, 2], [2, 1.9]])
        y = np.asarray([[0, 0], [0, 0], [1, 2], [1, 2], [2, 4], [2, 4]])
        for kind in ("ridge_a1", "knn_k3", "extra_leaf1", "svr_c10"):
            with self.subTest(kind=kind):
                model = build_regressor(kind)
                model.fit(x, y)
                self.assertEqual(model.predict(x).shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
