import copy
import unittest

import numpy as np

from analyze_certified_ch1_transients import (
    causal_read, causal_shape_features, held_group_gain, nested_shape_prediction, metrics, analyze,
)


class CausalAnalysisTests(unittest.TestCase):
    def test_group_must_finish_before_cutoff(self):
        row = {'sample_start_after_target_write_us': [0, 56, 111, 166],
               'direct_adc': [[0, x, 0, 0, 0, 0] for x in [10, 20, 999, 1000]]}
        self.assertEqual(causal_read(row, 111), (10, 20, 1, 111))
        changed = copy.deepcopy(row)
        changed['direct_adc'][2][1] = 1
        self.assertEqual(causal_read(changed, 111), causal_read(row, 111))
        np.testing.assert_array_equal(causal_shape_features(changed, 111)[1], causal_shape_features(row, 111)[1])
        with self.assertRaises(ValueError): causal_read(row, 55)

    def test_held_grating_label_not_used_for_its_predictor(self):
        first = np.arange(45, dtype=float) + 100
        last = first + 5
        truth = first + 15
        groups = np.arange(45) // 5
        valid = np.ones(45, dtype=bool)
        prediction, _ = held_group_gain(first, last, truth, groups, valid)
        np.testing.assert_allclose(prediction, truth)
        modified = truth.copy()
        modified[groups == 3] += 300
        changed, _ = held_group_gain(first, last, modified, groups, valid)
        np.testing.assert_allclose(changed[groups == 3], prediction[groups == 3])

    def test_nested_regularization_does_not_see_held_out_labels(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(90, 3)) * 10
        last = np.full(90, 300.)
        truth = last + features @ np.array([1., -2., .5])
        groups = np.arange(90) // 10
        valid = np.ones(90, dtype=bool)
        predicted, folds = nested_shape_prediction(features, last, truth, groups, valid)
        modified = truth.copy()
        modified[groups == 4] += 500
        changed, changed_folds = nested_shape_prediction(features, last, modified, groups, valid)
        np.testing.assert_allclose(changed[groups == 4], predicted[groups == 4])
        self.assertEqual(folds[4]['ridge_fraction'], changed_folds[4]['ridge_fraction'])

    def test_voltage_gain_does_not_improve_normalized_error(self):
        a = metrics(np.array([100, 200]), np.array([110, 180]), np.array([100, 100]), 5000)
        b = metrics(np.array([800, 1600]), np.array([880, 1440]), np.array([800, 800]), 40000)
        self.assertAlmostEqual(a['peak_span_normalized_rmse_percent'], b['peak_span_normalized_rmse_percent'])
        self.assertAlmostEqual(a['rmse_input_equivalent_ua'], b['rmse_input_equivalent_ua'])

    def test_unpaired_incomplete_or_unsafe_capture_rejected(self):
        for payload in [{}, {'complete': True},
                        {'complete': True, 'paired_stable_teacher': True}]:
            with self.assertRaises(ValueError): analyze(payload)


if __name__ == '__main__':
    unittest.main()
