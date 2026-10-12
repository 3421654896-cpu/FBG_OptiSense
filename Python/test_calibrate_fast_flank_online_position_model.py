import unittest
from types import SimpleNamespace

import numpy as np

from calibrate_fast_flank_online_position_model import (
    _online_features,
    _rolling_median,
    _training_rows,
)


class FastFlankOnlinePositionTests(unittest.TestCase):
    def test_rolling_window_is_causal(self):
        values = np.arange(5.0).reshape(5, 1)
        result = _rolling_median(values, 3)
        np.testing.assert_allclose(result[:, 0], [0.0, 0.5, 1.0, 2.0, 3.0])

    def test_online_features_preserve_one_output_per_input_frame(self):
        values = np.arange(8 * 9, dtype=float).reshape(8, 9)
        raw = np.arange(8 * 18, dtype=float).reshape(8, 18)
        features = _online_features(values, raw, values * 0.5)
        self.assertEqual(features["log_ratio_unit_amp_w3"].shape, (8, 10))
        self.assertEqual(features["raw_log_unit_amp_w2"].shape, (8, 19))
        self.assertEqual(features["ratio_common_unit_amp_w3"].shape, (8, 20))
        self.assertEqual(features["symmetric_common_unit_amp_w1"].shape, (8, 20))
        self.assertTrue(np.all(np.isfinite(features["log_ratio_unit_amp_w2"])))

    def test_event_quantiles_give_each_contact_equal_training_weight(self):
        short = SimpleNamespace(
            features={"f": np.arange(12.0).reshape(4, 3)},
            sample=SimpleNamespace(x_mm=1.0, y_mm=2.0),
        )
        long = SimpleNamespace(
            features={"f": np.arange(30.0).reshape(10, 3)},
            sample=SimpleNamespace(x_mm=3.0, y_mm=4.0),
        )
        features, labels = _training_rows(
            [short, long], "f", "event_quantiles"
        )
        self.assertEqual(features.shape, (10, 3))
        np.testing.assert_array_equal(labels[:5], np.tile([1.0, 2.0], (5, 1)))
        np.testing.assert_array_equal(labels[5:], np.tile([3.0, 4.0], (5, 1)))

if __name__ == "__main__":
    unittest.main()
