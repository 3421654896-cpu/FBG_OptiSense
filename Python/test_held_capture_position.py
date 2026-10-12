import unittest
from held_capture_position import bracket, reference


class PositionBracketsTests(unittest.TestCase):
    def status(self):
        return dict(connected=True, stopped=True, moving=False, x_homed=True,
                    y_homed=True, z_homed=True, machine_x=1., machine_y=2., machine_z=13.7)

    def test_actual_depth_and_no_overclaim(self):
        result = bracket({'contact_machine_z': 14.}, self.status(), self.status())
        self.assertAlmostEqual(result['before']['pressure_reference']['indentation_mm'], .3)
        self.assertFalse(result['stationarity_verified_throughout'])

    def test_motion_and_depth_rejected(self):
        for change in ({'moving':True}, {'machine_z':13.1}, {'estop':True}, {'x_homed':False}):
            with self.assertRaises(ValueError):
                reference({'contact_machine_z':14.}, dict(self.status(), **change))

    def test_displacement_rejected(self):
        with self.assertRaises(ValueError):
            bracket({'contact_machine_z':14.}, self.status(), dict(self.status(), machine_x=1.1))
