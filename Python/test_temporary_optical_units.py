import math
import unittest

import numpy as np

from temporary_optical_units import (
    adc_code_to_dbm,
    adc_code_to_optical_power_w,
    adc_code_to_voltage,
    format_physical_value,
    transimpedance_ohm,
)


class TemporaryOpticalUnitTests(unittest.TestCase):
    def test_adc_range1_uses_2p5_volt_over_4096(self):
        self.assertAlmostEqual(float(adc_code_to_voltage(2048)), 1.25)
        np.testing.assert_allclose(
            adc_code_to_voltage([0, 4095]),
            [0.0, 4095.0 * 2.5 / 4096.0],
        )

    def test_selector_mapping_matches_populated_feedback_network(self):
        self.assertEqual(
            [transimpedance_ohm(selector) for selector in range(4)],
            [2000.0, 40000.0, 5000.0, 20000.0],
        )

    def test_dbm_is_derived_from_voltage_transimpedance_and_responsivity(self):
        # 1 V / 5 kOhm / 0.85 A/W = 235.294 uW = -6.2839 dBm.
        code_for_one_volt = 4096.0 / 2.5
        power = float(adc_code_to_optical_power_w(code_for_one_volt, 2))
        self.assertAlmostEqual(power, 1.0 / 5000.0 / 0.85, places=12)
        expected = 10.0 * math.log10(power / 1e-3)
        self.assertAlmostEqual(
            float(adc_code_to_dbm(code_for_one_volt, 2)), expected, places=10
        )

    def test_zero_code_is_shown_as_below_plot_floor(self):
        self.assertTrue(format_physical_value(0, "dbm", 2).startswith("≤"))


if __name__ == "__main__":
    unittest.main()
