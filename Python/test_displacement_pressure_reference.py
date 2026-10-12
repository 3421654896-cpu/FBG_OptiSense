import unittest
from displacement_pressure_reference import from_clearance, from_machine


class DisplacementReferenceTests(unittest.TestCase):
    def test_actual_machine_coordinate_not_commanded_depth(self):
        result = from_machine({"contact_machine_z": 14.0, "depth_mm": 0.8},
                              {"connected": True, "machine_z": 13.7, "work_z": 999})
        self.assertAlmostEqual(result["indentation_mm"], 0.3)
        self.assertAlmostEqual(result["relative_pressure_percent"], 37.5)
        self.assertIsNone(result["force_n"])

    def test_retracted_is_zero_and_overtravel_not_hidden(self):
        self.assertEqual(from_clearance(4)["relative_pressure_percent"], 0)
        self.assertTrue(from_clearance(-0.9)["over_depth_limit"])
        self.assertGreater(from_clearance(-0.9)["relative_pressure_percent"], 100)

    def test_offline_and_unmapped_not_guessed(self):
        self.assertIsNone(from_machine({}, {"connected": True, "machine_z": 5}))
        self.assertIsNone(from_machine({"contact_machine_z": 4}, {}))


if __name__ == "__main__":
    unittest.main()
