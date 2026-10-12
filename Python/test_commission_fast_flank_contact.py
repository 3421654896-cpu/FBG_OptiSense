import unittest
from unittest.mock import patch, Mock

from commission_fast_flank_center_contact import coverage_target, resolve_work_target, validate_status
from commission_fast_flank_center_contact import open_verified_optical_port
from commission_fast_flank_center_contact import move_contact_descent
from commission_fast_flank_center_contact import main


class FastFlankContactTargetTests(unittest.TestCase):
    def test_optical_preflight_rejects_missing_ack_and_closes_port(self):
        device=Mock()
        with patch('commission_fast_flank_center_contact.serial.Serial',return_value=device), \
             patch('commission_fast_flank_center_contact._safe_disarm',return_value=True), \
             patch('commission_fast_flank_center_contact._close_shutter',return_value=False):
            with self.assertRaises(RuntimeError): open_verified_optical_port('COM6')
        device.close.assert_called_once()

    def test_rezero_preserves_absolute_target_and_labels(self):
        original=coverage_target(9,.3)
        offset=original['calibration_machine_minus_work']
        status=dict(connected=True,stopped=True,moving=False,x_homed=True,y_homed=True,z_homed=True)
        for axis,value,old in zip('xyz',[original['work_x'],original['work_y'],original['safe_z']],offset):
            status['machine_'+axis]=value+old
            status['work_'+axis]=value+100
        resolved=resolve_work_target(original,status)
        validate_status(status,resolved)
        self.assertEqual(resolved['calibration_work_xy_mm'],[original['work_x'],original['work_y']])
        self.assertAlmostEqual(resolved['press_z']+resolved['calibration_machine_minus_work'][2],
                               original['press_z']+offset[2])
        changed=dict(status,work_x=status['work_x']+1)
        with self.assertRaises(RuntimeError): validate_status(changed,resolved)

    def test_nonzero_depth_is_relative_to_calibrated_contact_plane(self):
        target = coverage_target(9, 0.10)
        self.assertAlmostEqual(target["contact_z"], 0.0)
        self.assertAlmostEqual(target["press_z"], -0.10)
        self.assertAlmostEqual(target["safe_z"], 4.0)
        self.assertAlmostEqual(target["depth_mm"], 0.10)

    def test_depth_above_hardware_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            coverage_target(9, 0.81)

    def test_deep_contact_path_is_compatible_with_half_mm_staging(self):
        controller = Mock()
        controller.move_relative_staged.return_value = [
            {"after": {"work_z": 0.0}},
            {"after": {"work_z": -0.5}},
            {"after": {"work_z": -0.8}},
        ]
        result = move_contact_descent(controller, -1.30)
        controller.move_relative_staged.assert_called_once_with(
            "Z",
            -1.30,
            step_mm=0.5,
            feed_mm_min=30.0,
            safe_zone_confirmed=True,
        )
        self.assertAlmostEqual(result["after"]["work_z"], -0.8)

    def test_candidate_validation_precedes_machine_and_serial_access(self):
        with patch('commission_fast_flank_center_contact.prepare_capture',side_effect=ValueError('invalid route')), \
             patch('commission_fast_flank_center_contact.Mach3Controller') as machine, \
             patch('commission_fast_flank_center_contact.open_verified_optical_port') as serial:
            with self.assertRaises(ValueError):
                main(['--operator-confirmed','--protocol','candidate27','--candidate','bad.json',
                      '--candidate-probe','probe.json','--cycles','640','--output','__unused_contact_fixture__.json'])
        machine.assert_not_called()
        serial.assert_not_called()

    def test_guarded_candidate_validation_precedes_machine_and_serial_access(self):
        with patch('commission_fast_flank_center_contact.prepare_guarded_candidate_capture',
                   side_effect=ValueError('invalid guarded route')), \
             patch('commission_fast_flank_center_contact.Mach3Controller') as machine, \
             patch('commission_fast_flank_center_contact.open_verified_optical_port') as serial:
            with self.assertRaises(ValueError):
                main(['--operator-confirmed','--protocol','candidate27_guarded',
                      '--candidate','bad.json','--candidate-probe','probe.json',
                      '--cycles','640','--output','__unused_guarded_contact_fixture__.json'])
        machine.assert_not_called()
        serial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
