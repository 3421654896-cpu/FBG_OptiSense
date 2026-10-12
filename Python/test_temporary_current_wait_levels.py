import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets
from temporary_current_wait_levels import (
    SCHEMA, SCAN_CYCLES, analyze_current_wait_levels, capture_current_wait_levels,
)
from temporary_test_widget import TemporaryTestWindow, CurrentWaitLevelWorker
from test_temporary_t90_waits import cycle_report, gated_report, voltage_probe
from test_temporary_symmetric_seven import selected
import test_temporary_test_widget as widget_tests


def level_report(rows=None, *, channel=1, target=1000, sampled=800, waits=None,
                 mode='uniform_and_cross_peak'):
    rows = copy.deepcopy(rows or selected()[-1]['rows'])
    waits = list(waits or [850] * len(rows))
    records, end = [], 0
    for wait in waits:
        write = end + 450
        end = write + wait + 40
        records.append(dict(adc_code=sampled, write_end_us=write, sample_end_us=end))
    frames = [dict(kind='data', sequence=i, adc_channel=channel,
                   feedback_selector=2 if channel < 2 else 0, single_sample=True,
                   fast_dac_writes=False, i2c_errors=0, spi_errors=0, firmware=115,
                   points_per_peak=7, cycle_start_us=i*(end+100), elapsed_us=end,
                   records=copy.deepcopy(records)) for i in range(SCAN_CYCLES)]
    scan = dict(complete=True, disarm_ack=True, shutter_ack=True,
                point_delays_us=waits, signal_channel=channel, feedback_selector=2 if channel < 2 else 0,
                single_sample=True, fast_dac_writes=False, plan=dict(rows=rows), frames=frames)
    probes = []
    for order, row in enumerate(rows):
        probe = voltage_probe(row, channel=channel, adc=target)
        probe['route_target_order'] = order
        probes.append(probe)
    return dict(schema=SCHEMA, complete=True, exact_disarm_ack=True, exact_soa_shutter_ack=True,
                route=rows, point_delays_us=waits, signal_channel=channel,
                feedback_selector=2 if channel < 2 else 0, fast_dac_writes=False,
                single_sample=True, machine_id='machine_1', wait_mode=mode,
                scan_capture=scan, voltage_probes=probes)


class CurrentWaitAnalysisTests(unittest.TestCase):
    def test_actual_last_cyclic_frame_not_interpolated_or_averaged(self):
        report = level_report()
        for frame in report['scan_capture']['frames'][:-1]:
            for record in frame['records']:
                record['adc_code'] = 400
        raw = copy.deepcopy(report)
        result = analyze_current_wait_levels(report)
        self.assertEqual(result['measured_sequence'], 31)
        self.assertEqual(result['points'][0]['source_order'], 62)
        self.assertEqual(result['points'][0]['percentage_of_final'], 80.)
        self.assertEqual(result['points'][0]['deviation_percent'], -20.)
        self.assertAlmostEqual(result['points'][0]['sample_voltage_v'], 800*2.5/4096)
        self.assertEqual(result['points'][0]['sample_end_after_target_write_us'], 890)
        self.assertFalse(result['points'][0]['within_10pct'])
        self.assertFalse(result['stability_verified'])
        self.assertEqual(report, raw)

    def test_overshoot_not_capped_and_all_channels_supported(self):
        for channel in range(4):
            with self.subTest(channel=channel):
                result = analyze_current_wait_levels(level_report(channel=channel, sampled=1200))
                self.assertEqual(result['points'][0]['percentage_of_final'], 120.)
                self.assertEqual(result['points'][0]['deviation_percent'], 20.)

    def test_low_voltage_is_actually_measured_not_replaced_with_300(self):
        result = analyze_current_wait_levels(level_report(target=100, sampled=80))
        self.assertEqual(result['points'][0]['percentage_of_final'], 80.)
        self.assertEqual(result['points'][0]['wait_us'], 850)
        self.assertEqual(result['points'][0]['reasons'], [])

    def test_zero_and_saturated_references_have_no_fake_percentage(self):
        for target, sampled, reason in ((0, 100, 'zero_reference'), (4080, 100, 'reference_clipped'),
                                        (1000, 4080, 'sample_clipped')):
            with self.subTest(reason=reason):
                point = analyze_current_wait_levels(level_report(target=target, sampled=sampled))['points'][0]
                self.assertIsNone(point['percentage_of_final'])
                self.assertIsNone(point['within_10pct'])
                self.assertIn(reason, point['reasons'])

    def test_unstable_tail_ratio_is_explicitly_unreliable(self):
        report = level_report()
        for record, adc in zip(report['voltage_probes'][0]['direct_adc'][-3:], [920, 1000, 1080]):
            record[1] = adc
        point = analyze_current_wait_levels(report)['points'][0]
        self.assertEqual(point['percentage_of_final'], 80.)
        self.assertIn('reference_unstable', point['reasons'])
        self.assertIsNone(point['within_10pct'])

    def test_identity_timing_incomplete_and_shutdown_failures_are_rejected(self):
        mutations = [lambda r: r.update(complete=False), lambda r: r.update(exact_disarm_ack=False),
                     lambda r: r['scan_capture'].update(shutter_ack=False),
                     lambda r: r['scan_capture']['frames'].pop(),
                     lambda r: r['scan_capture']['frames'][-1].update(sequence=32),
                     lambda r: r['scan_capture']['frames'][-1].update(adc_channel=0),
                     lambda r: r['voltage_probes'][0].update(source_fullband_index=0),
                     lambda r: r['voltage_probes'][0].update(route_target_order=1),
                     lambda r: r['scan_capture']['frames'][-1]['records'][0].update(sample_end_us=500),
                     lambda r: r.update(wait_mode='unknown')]
        for mutate in mutations:
            report = level_report()
            mutate(report)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                analyze_current_wait_levels(report)


class CurrentWaitWorkerTests(unittest.TestCase):
    def run_capture(self, folder, report, **kwargs):
        feedback = MagicMock()
        feedback._feedback_command_and_ack.return_value = True
        with patch('machine_profile.get_runtime_machine_id', return_value='machine_1'), \
             patch('temporary_current_wait_levels.execute_device', return_value=report['scan_capture']) as scan, \
             patch('capture_temporary_half_height_transients.capture_target_level',
                   side_effect=[copy.deepcopy(p) for p in report['voltage_probes']]) as probes, \
             patch('app_JDSU.EqualIntervalWorker', return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=kwargs.pop('disarm', True)) as disarm, \
             patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
             patch('temporary_current_wait_levels.time.sleep'):
            result = capture_current_wait_levels(MagicMock(timeout=.03), report['scan_capture']['plan'],
                Path(folder)/'levels.json', point_delays_us=report['point_delays_us'],
                wait_mode=report['wait_mode'], channel=1, selector=2, **kwargs)
            return result, scan, probes, disarm, shutter

    def test_real_capture_paths_use_same_waits_and_do_not_skip_low_voltage(self):
        report = level_report(target=100, sampled=80)
        with tempfile.TemporaryDirectory() as folder:
            result, scan, probes, disarm, shutter = self.run_capture(folder, report)
            self.assertEqual(scan.call_args.kwargs['point_delays_us'], [850]*63)
            self.assertTrue(scan.call_args.kwargs['single_sample'])
            self.assertEqual(probes.call_count, 63)
            self.assertTrue(result['complete'])
            self.assertEqual(result['current_wait_levels']['points'][0]['percentage_of_final'], 80.)
            self.assertTrue(json.loads((Path(folder)/'levels.json').read_text(encoding='utf-8'))['complete'])
            disarm.assert_called_once()
            shutter.assert_called_once()

    def test_cancelled_measurement_never_emits_a_partial_result(self):
        report = level_report()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(InterruptedError):
                self.run_capture(folder, report, should_stop=lambda: True)
            saved = json.loads((Path(folder)/'levels.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['complete'])
            self.assertTrue(saved['exact_disarm_ack'])
            self.assertTrue(saved['exact_soa_shutter_ack'])

    def test_shutdown_failure_retains_raw_but_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, '未确认'):
                self.run_capture(folder, level_report(), disarm=False)
            saved = json.loads((Path(folder)/'levels.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['complete'])
            self.assertEqual(len(saved['voltage_probes']), 63)

    def test_thread_does_not_emit_on_capture_error(self):
        worker = CurrentWaitLevelWorker(MagicMock(), {'current_wait_measurement_mode':'point_waits'},
                                        Path('unused.json'), 32, point_delays_us=[850]*63)
        emitted, messages = [], []
        worker.measured.connect(emitted.append)
        worker.message.connect(messages.append)
        with patch('temporary_current_wait_levels.capture_current_wait_levels', side_effect=OSError('disk full')):
            worker.run()
        self.assertEqual(emitted, [])
        self.assertIn('disk full', messages[-1])

    def test_unwritable_output_fails_before_scanning_but_attempts_both_shutdowns(self):
        report = level_report()
        with tempfile.TemporaryDirectory() as folder, \
             patch('capture_ch1_teacher_spectrum.save_checkpoint', side_effect=OSError('disk full')), \
             patch('temporary_current_wait_levels.execute_device') as scan, \
             patch('capture_temporary_half_height_transients.capture_target_level') as probe, \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
             patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter:
            with self.assertRaises(OSError):
                capture_current_wait_levels(MagicMock(timeout=.03), report['scan_capture']['plan'],
                    Path(folder)/'levels.json', point_delays_us=[850]*63,
                    wait_mode=report['wait_mode'], channel=1, selector=2)
            scan.assert_not_called()
            probe.assert_not_called()
            disarm.assert_called_once()
            shutter.assert_called_once()

    def test_cancel_after_last_probe_is_still_incomplete(self):
        stops = iter([False]*65 + [True])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(InterruptedError):
                self.run_capture(folder, level_report(), should_stop=lambda: next(stops))
            saved = json.loads((Path(folder)/'levels.json').read_text(encoding='utf-8'))
            self.assertFalse(saved['complete'])
            self.assertEqual(len(saved['voltage_probes']), 63)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'levels.json'
            output.touch()
            with self.assertRaises(FileExistsError):
                capture_current_wait_levels(MagicMock(), {'rows':[]}, output,
                    point_delays_us=[], wait_mode='point_waits', channel=1, selector=2)
            self.assertEqual(output.stat().st_size, 0)


class CurrentWaitWidgetTests(unittest.TestCase):
    setUpClass = widget_tests.WidgetTests.__dict__['setUpClass']
    setUp = widget_tests.WidgetTests.setUp

    def window(self):
        with patch.dict(os.environ, {'FBG_STUDIO_BETA':'1'}):
            window = TemporaryTestWindow(None)
        self.addCleanup(window.close)
        window.channel_combo.setCurrentIndex(window.channel_combo.findData(1))
        window.plan = selected()[-1]
        window.render_plan(ready=True)
        return window

    def test_unchecked_option_starts_with_visible_uniform_and_cross_peak_waits(self):
        import app_JDSU as app
        window = self.window()
        window.first_delay.setValue(500)
        window.boundary_delay.setValue(850)
        self.assertFalse(window.use_point_waits.isChecked())
        with patch.object(app, 'switch_mode_enable', True), patch.object(app, 'ser_open', False), \
             patch('temporary_test_widget.temporary_transport_supported', return_value=True), \
             patch.object(app, 'ensure_serial_open'), patch('temporary_test_widget.CurrentWaitLevelWorker') as worker:
            window.start_current_wait_levels()
            self.assertEqual(worker.call_args.kwargs['point_delays_us'], [850 if i%7==0 else 500 for i in range(63)])
            self.assertEqual(worker.call_args.args[1]['current_wait_measurement_mode'], 'uniform_and_cross_peak')
            worker.return_value.start.assert_called_once()
            self.assertFalse(window.measure_current_wait_levels.isEnabled())
        window.worker = None

    def test_checked_point_wait_table_is_used_without_changes(self):
        window = self.window()
        window.point_waits = [300 if i%2 else 850 for i in range(63)]
        window.point_wait_route = window.point_route_key()
        window.use_point_waits.setChecked(True)
        before = list(window.point_waits)
        window.worker = SimpleNamespace(output='levels.json')
        window.current_wait_levels_ready(level_report(window.plan['rows'], waits=before, mode='point_waits'))
        window.worker = None
        self.assertEqual(window.point_waits, before)
        self.assertTrue(window.use_point_waits.isChecked())
        self.assertIn('逐点等待表', window.status.text())

    def test_result_dialog_shows_voltages_actual_percent_and_stale_wait_is_blocked(self):
        window = self.window()
        window.worker = SimpleNamespace(output='levels.json')
        window.current_wait_levels_ready(level_report(window.plan['rows'], sampled=1200))
        window.worker = None
        window.refresh_wait_controls()
        self.assertFalse(window.use_point_waits.isChecked())
        self.assertIsNone(window.point_waits)
        self.assertTrue(window.current_wait_results.isEnabled())
        def inspect(dialog):
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.rowCount(), 63)
            self.assertEqual(table.item(0,5).text(), '120.00%')
            self.assertIn('+20.00%', table.item(0,6).text())
            self.assertEqual(table.item(0,4).text(), f'{1000*2.5/4096:.6f}')
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog,'exec_',inspect):
            window.show_current_wait_levels()
        window.first_delay.setValue(500)
        with patch.object(QtWidgets.QDialog,'exec_') as dialog:
            window.show_current_wait_levels()
            dialog.assert_not_called()
        self.assertIn('等待设置已改变',window.status.text())

    def test_t90_failures_have_distinct_markers_tooltips_and_unit_correct_positions(self):
        window = self.window()
        report = cycle_report(window.plan['rows'])
        for order in (0,7,62):
            report['captures'][order]['direct_adc'][-1][1] = 3000
        raw = copy.deepcopy(report)
        window.worker = SimpleNamespace(output='t90.json')
        window.switch_timing_ready(report)
        window.worker = None
        self.assertEqual(report, raw)
        markers = window.t90_failed_markers.points()
        self.assertEqual([m.data()['target_order'] for m in markers], [0,7,62])
        self.assertEqual(window.t90_failed_markers.opts['symbol'], 'd')
        self.assertIn('末段不稳定', markers[0].data()['tooltip'])
        self.assertIn('最大偏差', markers[0].data()['tooltip'])
        window._select_t90_failed_point(window.t90_failed_markers, [markers[1]], None)
        self.assertEqual(window.selected_selection_point, (1,0))
        for mode in ('adc','voltage','dbm'):
            window._set_display_mode(mode, persist=False)
            marker = window.t90_failed_markers.points()[0]
            row = window.plan['rows'][0]
            expected = window._plot_display_values([window._dense_value_at(row['measured_nm'])],selector=2)[0]
            self.assertAlmostEqual(marker.pos().y(),expected)
        self.assertFalse(window.apply_beta_t90_waits())

    def test_fixed_300us_low_voltage_points_are_not_marked_failed(self):
        window = self.window()
        window.worker = SimpleNamespace(output='t90.json')
        window.switch_timing_ready(gated_report([100]*63))
        window.worker = None
        self.assertEqual(len(window.t90_failed_markers.points()), 0)

    def test_stale_route_gain_and_channel_clear_failure_markers(self):
        for change in ('route','gain','channel'):
            window = self.window()
            report = cycle_report(window.plan['rows'])
            report['captures'][0]['direct_adc'][-1][1] = 3000
            window.worker = SimpleNamespace(output='t90.json')
            window.switch_timing_ready(report)
            window.worker = None
            self.assertEqual(len(window.t90_failed_markers.points()), 1)
            if change == 'route':
                window.plan['rows'][0]['index'] += 1
                window.redraw_dense_selection()
            elif change == 'gain':
                window.analog_gain.setCurrentIndex(window.analog_gain.findData(1))
            else:
                window.channel_combo.setCurrentIndex(window.channel_combo.findData(0))
            self.assertEqual(len(window.t90_failed_markers.points()), 0)

    def test_dac_option_explicitly_explains_why_it_cannot_be_selected(self):
        window = self.window()
        self.assertFalse(window.beta_fast_dac.isEnabled())
        self.assertIn('验证未通过',window.beta_fast_dac.text())

    def test_percentage_dialog_rows_are_compact_after_layout(self):
        window = self.window()
        window.worker = SimpleNamespace(output='levels.json')
        window.current_wait_levels_ready(level_report(window.plan['rows']))
        window.worker = None
        def inspect(dialog):
            dialog.show()
            for _ in range(8): QtWidgets.QApplication.processEvents()
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertLess(table.rowHeight(0),100)
            self.assertGreater(table.columnWidth(6),200)  # Font/theme-dependent column widths.
            dialog.close()
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog,'exec_',inspect):
            window.show_current_wait_levels()


if __name__ == '__main__':
    unittest.main()
