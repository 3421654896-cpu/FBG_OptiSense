import importlib
import unittest

from data_convert import to_int_str


MODULE_NAMES = ("ap6150b波长计_JDSU", "ap6150b_JDSU_多峰值版本")


class LegacyMeterUtilitySafetyTests(unittest.TestCase):
    def test_import_does_not_open_stm32_serial_port(self):
        for name in MODULE_NAMES:
            module = importlib.import_module(name)
            self.assertFalse(module.ser.is_open)
            self.assertIsNone(module.ser.port)

    def test_excel_dac_rows_enforce_laser_tuning_limits(self):
        valid = [None, None, None, None, 32768, 24576, 24576]
        invalid_rows = (
            [None, None, None, None, 32769, 24576, 24576],
            [None, None, None, None, 32768, 24577, 24576],
            [None, None, None, None, 32768, 24576, 24577],
            [None, None, None, None, -1, 0, 0],
            [None, None, None, None, 1.25, 0, 0],
            [None, None, None, None, float("nan"), 0, 0],
        )
        for name in MODULE_NAMES:
            module = importlib.import_module(name)
            result = module.excel_operate(iter([valid]))
            command = result[0] if isinstance(result, tuple) else result
            self.assertEqual(len(command), 8)
            for row in invalid_rows:
                with self.assertRaises(ValueError):
                    module.excel_operate(iter([row]))

    def test_table_converter_never_truncates_or_exceeds_dac_limits(self):
        self.assertEqual(to_int_str("58982.000", 58982), "58982")
        for value in ("58982.1", "nan", "not-a-code", "-1", "58983"):
            with self.assertRaises(ValueError):
                to_int_str(value, 58982)


if __name__ == "__main__":
    unittest.main()
