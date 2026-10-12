"""Optical-tool safety reporting without opening a physical port."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import benchmark_temporary_wait_balance_live as live


class ThermalGateTests(unittest.TestCase):
    def exercise(self, values, *, changed_identity=False):
        identity = dict(serial_number='synthetic_not_a_board', boot_id=1,
                        firmware_version=0x10074)
        args = SimpleNamespace(port='none', serial_number='synthetic_not_a_board', cycles=32,
                               channel=0, feedback_selector=2, single_adc=True,
                               fast_dac_writes=True)
        device = MagicMock()
        device.__enter__.return_value = device

        def capture(*_args, **_kwargs):
            for value in values:
                device.thermal_observer.callback(value)
            return dict(complete=True, shutter_ack=True, disarm_ack=True)

        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'capture.json'
            after = dict(identity, boot_id=2) if changed_identity else identity
            with patch.object(live, 'verify_usb', side_effect=[identity, after]), \
                    patch.object(live, 'ThermalSerial', return_value=device), \
                    patch.object(live, 'execute_device', side_effect=capture):
                error = None
                try:
                    result = live.run_capture(args, {}, [850], output, identity)
                except RuntimeError as exc:
                    error, result = str(exc), None
            stored = json.loads(output.read_text()) if output.exists() else None
            return result, error, stored

    def telemetry(self, **changes):
        return dict(thermal_flags=17, temperature_c=34., sensor_age_ms=100,
                    fan_rpm=2300, fan_duty_permille=250, **changes)

    def test_valid_telemetry_keeps_actual_rpm_and_shutdown_evidence(self):
        result, error, stored = self.exercise([self.telemetry()])
        self.assertIsNone(error)
        self.assertEqual(result, stored)
        self.assertEqual(stored['thermal_summary']['valid_snapshots'], 1)
        self.assertEqual(stored['thermal_snapshots'][0]['fan_rpm'], 2300)
        self.assertTrue(stored['shutter_ack'])

    def test_missing_faulty_or_unsafe_data_refuses_recommendation_but_saves_evidence(self):
        for mode in ('missing', 'invalid', 'sensor_fault', 'tach_fault', 'pwm_fault', 'hot'):
            with self.subTest(mode=mode):
                values = [self.telemetry()]
                if mode == 'missing': values = []
                elif mode == 'invalid': values[0]['thermal_flags'] = 16
                elif mode == 'hot': values[0]['temperature_c'] = 85.
                else: values[0]['thermal_flags'] |= {'sensor_fault':4, 'tach_fault':8,
                                                      'pwm_fault':32}[mode]
                result, error, stored = self.exercise(values)
                self.assertIsNone(result)
                self.assertIn('thermal telemetry', error)
                self.assertEqual(stored['thermal_snapshots'], values)
                self.assertTrue(stored['disarm_ack'])
                self.assertTrue(stored['shutter_ack'])

    def test_boot_change_never_returns_a_qualified_capture(self):
        result, error, _ = self.exercise([self.telemetry()], changed_identity=True)
        self.assertIsNone(result)
        self.assertIn('changed during capture', error)


if __name__ == '__main__':
    unittest.main()
