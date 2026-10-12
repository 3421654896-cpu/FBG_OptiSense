import unittest

import numpy as np

from candidate27_grouped_classifier import _majority, build_estimator


class GroupedClassifierTests(unittest.TestCase):
    def test_majority_has_deterministic_tie_break(self):
        self.assertEqual(_majority(np.asarray([4, 2, 4, 2])), 2)

    def test_estimators_fit_small_multiclass_problem(self):
        x = np.asarray([[0, 0], [0, .1], [1, 1], [1, .9], [2, 2], [2, 1.9]])
        y = np.asarray([1, 1, 2, 2, 3, 3])
        for kind in ("lda_shrinkage", "logistic_c1", "rbf_c10"):
            with self.subTest(kind=kind):
                model = build_estimator(kind)
                model.fit(x, y)
                self.assertEqual(model.predict(x).shape, (6,))


if __name__ == "__main__":
    unittest.main()
