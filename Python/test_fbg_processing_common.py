import unittest
from collections import deque

import numpy as np

from fbg_processing_common import (
    adc_codes_to_voltage,
    precision_median_fuse,
    scan_min_prominence,
)


class SharedProcessingTests(unittest.TestCase):
    def test_adc_scaling_and_invalid_codes(self):
        values = adc_codes_to_voltage([0, 2048, 4095, 4096])
        np.testing.assert_allclose(values[:3], [0.0, 1.25, 4095 * 2.5 / 4096])
        self.assertTrue(np.isnan(values[3]))

    def test_mode_channel_thresholds(self):
        self.assertEqual(scan_min_prominence(0.03, 0, True), 0.005)
        self.assertEqual(scan_min_prominence(0.03, 1, False), 0.008)
        self.assertEqual(scan_min_prominence(0.03, 2, False), 0.00375)
        self.assertEqual(scan_min_prominence(0.03, 0, False), 0.03)

    def test_precision_fusion_uses_latest_five_frames(self):
        history = deque(maxlen=5)
        for value in (1.0, 9.0, 2.0, 4.0, 3.0):
            matrix = np.full((4, 2), value)
            fused = precision_median_fuse(history, matrix, (0, 1))
        np.testing.assert_allclose(fused[0], [3.0, 3.0])
        np.testing.assert_allclose(fused[1], [3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
