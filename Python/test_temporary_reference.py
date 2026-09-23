import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import app_JDSU as app
from temporary_reference import acquire_reference, acquire_checked_point


class ReferenceTests(unittest.TestCase):
    def test_hold_then_forward_recovery_and_channel_diagnostics(self):
        samples=[[100]*6 for _ in range(20)]
        for i in range(20):
            samples[i][3]=100+(30 if i%2 else -30)
        bad=dict(stable=False,ch1_saturated=False,monitor_samples=samples)
        good=dict(stable=True,ch1_saturated=False,monitor_samples=[[100]*6 for _ in range(20)])
        points = [MagicMock(index=i, codes=(i, i, i, i, i)) for i in range(8)]
        worker = MagicMock(points=points, guards={})
        worker._interruptible_wait.return_value = True
        with patch('capture_ch1_teacher_spectrum.acquire_settled_ch1',side_effect=[dict(bad),dict(bad),dict(good)]) as read, \
             patch('capture_ch1_teacher_spectrum.command_with_link_retry', return_value=0.0) as command:
            result=acquire_checked_point(worker,points[-1])
        self.assertTrue(result['stable'])
        self.assertFalse(result['optical_stability_verified'])
        self.assertEqual([c.args[2] for c in read.call_args_list],[.2,.5,.2])
        self.assertEqual([c.kwargs['apply_codes'] for c in read.call_args_list],[True,False,True])
        self.assertEqual([c.args[1] for c in command.call_args_list],
                         [p.codes for p in points[2:7]])
        self.assertEqual(result['attempts'][2]['replayed_indices'], [2,3,4,5,6])
        self.assertEqual(result['attempts'][0]['failed_channels'],['CH1'])

    def test_persistent_failure_is_not_accepted(self):
        point = MagicMock(index=0, codes=(0,0,0,0,0))
        worker = MagicMock(points=[point], guards={})
        with patch('capture_ch1_teacher_spectrum.acquire_settled_ch1',side_effect=lambda *a,**k:dict(stable=False)) as read:
            result=acquire_checked_point(worker,point)
        self.assertFalse(result['stable'])
        self.assertEqual([c.args[2] for c in read.call_args_list],[.2,.5,.2,.5])
        self.assertEqual([c.kwargs['apply_codes'] for c in read.call_args_list],
                         [True,False,True,True])

    def test_cancel_before_retry(self):
        with patch('capture_ch1_teacher_spectrum.acquire_settled_ch1',return_value=dict(stable=False)) as read:
            with self.assertRaises(InterruptedError):
                acquire_checked_point(MagicMock(),MagicMock(),should_stop=lambda:read.call_count>=1)
        self.assertEqual(read.call_count,1)

    def run_reference(self, *, cancel=False, stable=True):
        port = MagicMock(timeout=.03)
        point = MagicMock(index=0)
        worker = MagicMock()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'load_fullband_accuracy_table', return_value=[point]), \
             patch.object(app, 'load_fullband_transition_guards', return_value={}), \
             patch.object(app, 'EqualIntervalWorker', return_value=worker), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
             patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
             patch('capture_ch1_teacher_spectrum.establish_forward_state'), \
             patch('capture_ch1_teacher_spectrum.acquire_settled_ch1', return_value={'stable':stable}), \
             patch('temporary_reference.time.sleep'):
            output = Path(folder)/'reference.json'
            if cancel or not stable:
                with self.assertRaises(InterruptedError if cancel else RuntimeError):
                    acquire_reference(port, output, should_stop=lambda:cancel)
            else:
                result = acquire_reference(port, output)
                self.assertTrue(result['complete'])
            self.assertEqual(disarm.call_count, 2)
            self.assertEqual(shutter.call_count, 2)
            self.assertEqual(port.timeout, .03)
            self.assertTrue(output.exists())

    def test_complete_shutdown(self):
        self.run_reference()

    def test_cancel_shutdown(self):
        self.run_reference(cancel=True)

    def test_unstable_shutdown(self):
        self.run_reference(stable=False)

    def test_dense_reference_marks_unstable_point_and_continues(self):
        port = MagicMock(timeout=.03)
        points = [MagicMock(index=index, target_nm=1525 + .02 * index,
                            measured_nm=1525 + .02 * index,
                            codes=(index,) * 5)
                  for index in range(3)]
        worker = MagicMock()
        rows = [
            dict(index=index, target_wavelength_nm=point.target_nm,
                 measured_wavelength_nm=point.measured_nm,
                 dac_codes=list(point.codes), ch1_adc_code=100 + index,
                 stable=index != 1, ch1_saturated=False, attempts=[{}])
            for index, point in enumerate(points)
        ]
        observed = []
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'load_fullband_accuracy_table', return_value=points), \
             patch.object(app, 'load_fullband_transition_guards', return_value={}), \
             patch.object(app, 'EqualIntervalWorker', return_value=worker), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('capture_ch1_teacher_spectrum.establish_forward_state'), \
             patch('temporary_reference.acquire_checked_point', side_effect=rows), \
             patch('temporary_reference.time.sleep'):
            result = acquire_reference(
                port, Path(folder) / 'reference.json',
                continue_on_unstable=True,
                on_row=lambda row, n, total: observed.append((row, n, total)),
            )
        self.assertTrue(result['complete'])
        self.assertFalse(result['all_points_passed'])
        self.assertEqual(result['failed_point_count'], 1)
        self.assertEqual(len(result['rows']), 3)
        self.assertEqual(len(observed), 3)
        self.assertFalse(observed[1][0]['point_passed'])
        self.assertIn('波动/漂移未通过', observed[1][0]['failure_reason'])

    def test_dense_reference_can_return_safe_partial_result(self):
        port = MagicMock(timeout=.03)
        points = [MagicMock(index=index, target_nm=1525 + .02 * index,
                            measured_nm=1525 + .02 * index,
                            codes=(index,) * 5)
                  for index in range(3)]
        worker = MagicMock()
        calls = {'count': 0}

        def read_point(_worker, point, **_kwargs):
            calls['count'] += 1
            return dict(index=point.index,
                        target_wavelength_nm=point.target_nm,
                        measured_wavelength_nm=point.measured_nm,
                        dac_codes=list(point.codes), ch1_adc_code=100,
                        stable=True, ch1_saturated=False, attempts=[{}])

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'load_fullband_accuracy_table', return_value=points), \
             patch.object(app, 'load_fullband_transition_guards', return_value={}), \
             patch.object(app, 'EqualIntervalWorker', return_value=worker), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('capture_ch1_teacher_spectrum.establish_forward_state'), \
             patch('temporary_reference.acquire_checked_point', side_effect=read_point), \
             patch('temporary_reference.time.sleep'):
            result = acquire_reference(
                port, Path(folder) / 'reference.json',
                allow_partial_stop=True,
                should_stop=lambda: calls['count'] >= 1,
            )
        self.assertFalse(result['complete'])
        self.assertTrue(result['partial'])
        self.assertTrue(result['stopped_early'])
        self.assertEqual(len(result['rows']), 1)

    def test_equal_interval_single_frame_uses_selected_channel_and_stride(self):
        port = MagicMock(timeout=.03)
        points = [MagicMock(index=index, target_nm=1525 + .02 * index,
                            measured_nm=1525 + .02 * index,
                            codes=(index,) * 5)
                  for index in range(9)]
        worker = MagicMock()
        worker.settle_s = .15
        worker._feedback_command_and_ack.return_value = True
        worker._acquire_point.side_effect = [
            dict(adc_codes=(10., 20., 100. + index, 40.),
                 voltages=(.01, .02, .1, .04), saturated_channels=(),
                 monitor_codes=(1., 2.), settle_s=.15, elapsed_s=.16)
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'load_fullband_accuracy_table', return_value=points), \
             patch.object(app, 'load_fullband_transition_guards', return_value={}), \
             patch.object(app, 'EqualIntervalWorker', return_value=worker) as make_worker, \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('capture_ch1_teacher_spectrum.establish_forward_state') as establish, \
             patch('temporary_reference.time.sleep'):
            result = acquire_reference(
                port, Path(folder) / 'equal.json',
                dense_indices=[0, 2, 4, 6, 8],
                acquisition_method='equal_interval_single',
                settle_s=.15, signal_channel=2, feedback_selector=3,
                continue_on_unstable=True,
            )
        self.assertTrue(result['complete'])
        self.assertEqual(result['signal_channel_name'], 'CH2')
        self.assertEqual(result['feedback_selector'], 0)
        self.assertEqual(result['requested_point_count'], 5)
        self.assertEqual([row['index'] for row in result['rows']], [0, 2, 4, 6, 8])
        self.assertEqual([row['signal_adc_code'] for row in result['rows']],
                         [100., 101., 102., 103., 104.])
        self.assertTrue(all(row['stability_not_evaluated'] for row in result['rows']))
        make_worker.assert_called_once()
        self.assertEqual(make_worker.call_args.kwargs['feedback_selectors'], (0, 0))
        establish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
