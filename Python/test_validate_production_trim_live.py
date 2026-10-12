"""No-hardware checks for the dark production acceptance helper."""
from unittest import TestCase, mock
from pathlib import Path
from tempfile import TemporaryDirectory
import validate_production_trim_live as live


class DarkSmokeSafetyTests(TestCase):
    def test_feedback_matrix_uses_only_zero_codes_and_all_supported_gains(self):
        def execute(_device, plan, output, **kwargs):
            self.assertTrue(all(row['codes'] == [0] * 5 for row in plan['rows']))
            self.assertEqual(len(plan['rows']), 45)
            self.assertEqual(kwargs['cycles'], 32)
            self.assertEqual(kwargs['first_delay_us'], 850)
            self.assertEqual(kwargs['boundary_extra_us'], 0)
            self.assertEqual(kwargs['point_delays_us'], [850]*45)
            if kwargs['adc_channel'] >= 2:
                self.assertEqual(kwargs['feedback_selector'], 0)
            return dict(complete=True, received_frames=32, disarm_ack=True,
                        shutter_ack=True, mean_hz=18.0, terminal={})
        with TemporaryDirectory() as folder, \
             mock.patch.object(live, 'execute_device', side_effect=execute) as capture, \
             mock.patch('builtins.print'):
            results = live.temporary_feedback_zero_matrix(mock.Mock(), Path(folder)/'matrix.json')
        self.assertEqual(capture.call_count, 10)
        self.assertEqual([(v['channel'], v['feedback_selector']) for v in results],
                         [(0, i) for i in range(4)] + [(1, i) for i in range(4)] + [(2, 0), (3, 0)])

    def test_feedback_matrix_failure_stops_before_the_next_gain(self):
        with TemporaryDirectory() as folder, \
             mock.patch.object(live, 'execute_device', return_value=dict(
                 complete=True, received_frames=31, disarm_ack=True, shutter_ack=True)) as capture:
            with self.assertRaises(RuntimeError):
                live.temporary_feedback_zero_matrix(mock.Mock(), Path(folder)/'matrix.json')
        self.assertEqual(capture.call_count, 1)

    def test_single_source_command_contains_only_real_zero_codes(self):
        device = mock.Mock()
        frame = bytearray(20)
        frame[:4] = b'\xff\xff\x02\x00'
        frame[15] = 0x87
        def wait(_device, decode):
            return decode(bytes(frame))
        with mock.patch.object(live, 'wait_single', side_effect=wait):
            ack = live.zero_source(device)
        packet = device.write.call_args.args[0]
        self.assertEqual(packet[4:14], bytes(10))
        self.assertEqual(packet[14], 0)
        self.assertEqual(ack['codes'], (0,)*5)

    def test_single_source_never_accepts_nonzero_or_fault_ack(self):
        for channel in range(5):
            frame = bytearray(20)
            frame[:4] = b'\xff\xff\x02\x00'
            frame[15] = 0x87
            frame[5+channel*2] = 1
            def wait(_device, decode):
                return decode(bytes(frame))
            with mock.patch.object(live, 'wait_single', side_effect=wait):
                with self.assertRaises(RuntimeError):
                    live.zero_source(mock.Mock())

    def test_dense_never_loads_operator_calibration_or_nonzero_codes(self):
        device = mock.Mock()
        def acquire(_device, points, guards, **kwargs):
            self.assertEqual(len(points), 2001)
            self.assertFalse(guards)
            self.assertTrue(all(p.codes == (0,)*5 for p in points))
            self.assertEqual(kwargs['settle_s'], .002)
            return iter(dict(index=i) for i in range(2001))
        with mock.patch.object(live.app, 'EqualIntervalWorker'), \
             mock.patch.object(live.dense, 'probe', return_value={'kind': 0}), \
             mock.patch.object(live.dense, 'acquire', side_effect=acquire), \
             mock.patch.object(live.app, 'load_fullband_accuracy_table',
                               side_effect=AssertionError('Operator table must not be loaded')):
            result = live.dense_zero(device, 2001)
        self.assertEqual(result['points'], 2001)
        self.assertFalse(result['optical_accuracy_verified'])

    def test_legacy_fallback_is_observed_before_any_extra_override(self):
        device = mock.Mock()
        device.read.return_value = b'query response'
        with mock.patch.object(live, 'find_status_reply',
                               return_value={'work_state': 3}), \
             mock.patch.object(live, 'query_wifi',
                               side_effect=AssertionError('Must observe real work state')):
            with self.assertRaises(RuntimeError):
                live.reject_legacy(device, 3)
        mode = device.write.call_args_list[0].args[0]
        self.assertEqual(mode[8], 3)


if __name__ == '__main__':
    import unittest
    unittest.main()
