import unittest
from commission_held_depth_teacher import validate_teacher_result


class HeldDepthTests(unittest.TestCase):
    def data(self):
        return dict(complete=True, safety_cleanup=dict(exact_disarm_ack=True, exact_soa_shutter_ack=True),
                    captures=[dict(teacher=dict(stable=True), machine_position_bracket={
                        phase: {'pressure_reference': {'indentation_mm': .2984375}}
                        for phase in ('before','after')})])

    def test_actual_position_required(self):
        self.assertAlmostEqual(validate_teacher_result(self.data(), .3), .2984375)
        with self.assertRaises(ValueError):
            validate_teacher_result(self.data(), .1)

    def test_shutter_and_stable_teacher_required(self):
        data = self.data()
        data['safety_cleanup']['exact_soa_shutter_ack'] = False
        with self.assertRaises(ValueError): validate_teacher_result(data, .3)
        data = self.data()
        data['captures'][0]['teacher']['stable'] = False
        with self.assertRaises(ValueError): validate_teacher_result(data, .3)
