import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import unittest
from unittest.mock import Mock, patch
import numpy as np
import app_JDSU  # Match application import order before constructing QApplication.
from PyQt5 import QtCore, QtGui, QtWidgets
from temporary_test_widget import (
    TemporaryTestWindow, TemporaryWorker, ReferenceLineWorker,
    equal_interval_indices_for_range, load_forward_settle_hints,
    temporary_transport_supported,
)
from fbg_lan_serial import FbgLanSerial
from test_temporary_test_mode import reference, frame
from temporary_test_mode import select_points, decode, quality
from temporary_optical_units import adc_code_to_dbm
from machine_profile import (
    get_runtime_machine_id, runtime_machine_label, runtime_parameter_note,
    set_runtime_machine_id,
)


class WidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_machine_one_parameter_ownership_is_visible(self):
        window = TemporaryTestWindow(None)
        self.assertIn('一号机', window.title_label.text())
        self.assertIn('一号机', window.machine_profile_label.text())
        self.assertIn('9峰', window.machine_profile_label.text())
        self.assertIn('2001点', window.machine_profile_label.text())
        self.assertIn('3峰', window.machine_profile_label.text())
        window.close()

    def test_machine_switch_isolates_routes_and_saved_record_paths(self):
        previous = get_runtime_machine_id()
        set_runtime_machine_id('machine_1')
        window = TemporaryTestWindow(None)
        try:
            machine_one_session = window.last_session_path
            self.assertEqual(window.point_count(), 45)

            set_runtime_machine_id('machine_2')
            window.reload_machine_profile(
                runtime_machine_label(), runtime_parameter_note()
            )
            machine_two_session = window.last_session_path
            self.assertNotEqual(machine_one_session, machine_two_session)
            self.assertIn('machine_1', str(machine_one_session))
            self.assertIn('machine_2', str(machine_two_session))
            self.assertIsNone(window.plan)

            set_runtime_machine_id('machine_1')
            window.reload_machine_profile(
                runtime_machine_label(), runtime_parameter_note()
            )
            self.assertEqual(window.point_count(), 45)
        finally:
            window.close()
            set_runtime_machine_id(previous)

    def test_temporary_transport_accepts_usb_and_full_function_lan(self):
        import serial
        self.assertTrue(temporary_transport_supported(FbgLanSerial()))
        self.assertFalse(temporary_transport_supported(Mock()))
        usb = serial.Serial(port=None)
        try:
            self.assertTrue(temporary_transport_supported(usb))
        finally:
            usb.close()

    def test_gain_changes_only_display_not_points_or_raw_records(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        self.assertFalse(window.plan_ready)
        self.assertFalse(window.start_button.isEnabled())
        self.assertEqual([row['index'] for row in window.plan['rows'][:5]], [96, 113, 130, 147, 164])
        self.assertTrue(window.use_point_waits.isChecked())
        self.assertEqual(window.point_waits[22], 9000)
        self.assertEqual(window.point_wait_route, window.point_route_key())
        self.assertTrue(window.continuous.isChecked())
        self.assertFalse(window.cycles.isEnabled())
        window.continuous.setChecked(False)
        self.assertTrue(window.cycles.isEnabled())
        window.continuous.setChecked(True)
        table, x, y = reference()
        window.plan = select_points(table, x, y)
        window.render_plan()
        indices = [r['index'] for r in window.plan['rows']]
        data = decode(frame(), 123, 456)
        data['quality'] = quality(data)
        window.update_frame(data)
        window.digital_gain.setValue(10)
        np.testing.assert_array_equal(window.curves[0].getData()[1], [1020]*5)
        self.assertEqual(window.table.item(0, 4).text(), '102')
        self.assertEqual(data['records'][0]['second_code'], 102)
        window.analog_gain.setCurrentIndex(3)
        window.first_delay.setValue(600)
        self.assertEqual(indices, [r['index'] for r in window.plan['rows']])
        window.peak_view.setCurrentIndex(3)
        view = window.plot.getViewBox().viewRange()[0]
        self.assertLess(view[1]-view[0], 1.)
        self.assertLess(view[0], window.plan['rows'][10]['measured_nm'])
        self.assertGreater(view[1], window.plan['rows'][14]['measured_nm'])
        self.assertEqual(indices, [r['index'] for r in window.plan['rows']])
        self.assertFalse(window.plot.getViewBox().autoRangeEnabled()[1])
        window.close()

    def test_display_button_switches_adc_voltage_and_estimated_dbm(self):
        window = TemporaryTestWindow(None)
        self.assertEqual(window.display_mode, 'dbm')
        self.assertEqual(window.display_unit_button.text(), '显示：光功率 dBm')
        window._set_display_mode('adc')
        count = window.point_count()
        window.latest_values = [2048] * count
        window.latest_feedback_selector = 2
        window.dense_feedback_selector = 2
        window.redraw_values()

        self.assertEqual(window.display_unit_button.text(), '显示：ADC码')
        self.assertEqual(window.table.item(0, 4).text(), '2048')

        window.display_unit_button.click()
        self.assertEqual(window.display_mode, 'voltage')
        self.assertEqual(window.display_unit_button.text(), '显示：电压 V')
        self.assertEqual(window.table.item(0, 4).text(), '1.250000')
        np.testing.assert_allclose(window.curves[0].getData()[1], [1.25] * 5)
        self.assertFalse(window.digital_gain.isEnabled())

        window.display_unit_button.click()
        expected_dbm = float(adc_code_to_dbm(2048, 2))
        self.assertEqual(window.display_mode, 'dbm')
        self.assertEqual(window.display_unit_button.text(), '显示：光功率 dBm')
        self.assertEqual(window.table.item(0, 4).text(), f'{expected_dbm:.2f}')
        np.testing.assert_allclose(window.curves[0].getData()[1], [expected_dbm] * 5)

        # A newly selected gain must not reinterpret already captured samples.
        window.analog_gain.setCurrentIndex(window.analog_gain.findData(1))
        window.redraw_values()
        self.assertEqual(window.table.item(0, 4).text(), f'{expected_dbm:.2f}')
        np.testing.assert_allclose(window.curves[0].getData()[1], [expected_dbm] * 5)
        window.close()

    def test_worker_snapshots_hardware_settings(self):
        worker = TemporaryWorker(None, None, None, 32, first_delay_us=500,
                                 boundary_extra_us=1000, feedback_selector=3)
        self.assertEqual(worker.options, dict(first_delay_us=500, boundary_extra_us=1000, feedback_selector=3))
        self.assertEqual(worker.adc_channel, 1)

    def test_channel_selector_invalidates_old_route_and_updates_labels(self):
        window = TemporaryTestWindow(None)
        self.assertEqual(window.selected_channel(), 1)
        self.assertIsNotNone(window.plan)
        window.channel_combo.setCurrentIndex(window.channel_combo.findData(2))
        self.assertEqual(window.selected_channel(), 2)
        self.assertIsNone(window.plan)
        self.assertFalse(window.start_button.isEnabled())
        self.assertFalse(window.analog_gain.isEnabled())
        self.assertIn('CH2', window.title_label.text())
        self.assertIn('CH2', window.plot.getAxis('left').labelText)
        self.assertIn('CH2', window.status.text())
        window.close()

    def test_equal_interval_reference_builds_adjustable_table_stride(self):
        window = TemporaryTestWindow(None)
        button_texts = [
            button.text() for button in window.findChildren(QtWidgets.QPushButton)
        ]
        self.assertNotIn('重采密集光谱并选点', button_texts)
        self.assertIn('按等间隔重采密集光谱', button_texts)
        window.scan_start_nm.setValue(1530.00)
        window.scan_stop_nm.setValue(1540.00)
        window.equal_interval_nm.setValue(.06)
        window.equal_interval_settle.setValue(.125)
        with patch.object(window, 'start_scan') as start:
            window.start_equal_interval_reference()
        kwargs = start.call_args.kwargs
        self.assertTrue(kwargs['reference'])
        self.assertTrue(kwargs['equal_interval'])
        indices = kwargs['dense_indices']
        self.assertEqual(indices[:4], [250, 253, 256, 259])
        self.assertEqual(indices[-1], 750)
        self.assertLessEqual(max(np.diff(indices)), 3)
        window.close()

    def test_equal_interval_range_rejects_fewer_than_45_spectrum_points(self):
        table, _, _ = reference()
        with self.assertRaisesRegex(ValueError, '至少需要45个'):
            equal_interval_indices_for_range(table, 1530.0, 1531.0, .10)

    def test_point_wait_worker_copies_settings_and_route_is_bound(self):
        waits = [650]*45
        worker = TemporaryWorker(None, None, None, 32, point_delays_us=waits)
        waits[22] = 5000
        self.assertEqual(worker.options['point_delays_us'][22], 650)
        window = TemporaryTestWindow(None)
        key = window.point_route_key()
        window.use_point_waits.setChecked(True)
        self.assertFalse(window.first_delay.isEnabled())
        self.assertFalse(window.boundary_delay.isEnabled())
        self.assertEqual(key, window.point_route_key())
        window.use_point_waits.setChecked(False)
        self.assertTrue(window.first_delay.isEnabled())
        window.close()

    def test_enabling_point_waits_binds_current_route_and_keeps_start_gate(self):
        window = TemporaryTestWindow(None)
        window.plan_ready = True
        window.start_button.setEnabled(True)
        window.point_waits = None
        window.point_wait_route = None
        window.use_point_waits.setChecked(False)
        window.use_point_waits.setChecked(True)
        self.assertEqual(len(window.point_waits), 45)
        self.assertEqual(window.point_waits[0], 2150)
        self.assertEqual(window.point_waits[1], 350)
        self.assertEqual(window.point_wait_route, window.point_route_key())
        self.assertTrue(window.start_button.isEnabled())
        window.close()

    def test_point_wait_dialog_accepts_all_45_at_15ms(self):
        window = TemporaryTestWindow(None)
        def accept_max(dialog):
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.columnCount(), 5)
            self.assertEqual(table.horizontalHeaderItem(1).text(), '正向跳转波长 nm')
            self.assertEqual(table.horizontalHeaderItem(3).text(),
                             '正向跳转主边沿/稳定时间')
            self.assertIn('→', table.item(1, 1).text())
            self.assertTrue(table.item(0, 3).text())
            spins = dialog.findChildren(QtWidgets.QSpinBox)
            self.assertEqual(len(spins), 45)
            for spin in spins:
                self.assertEqual(spin.maximum(), 15000)
                spin.setValue(15000)
            buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
            buttons.button(QtWidgets.QDialogButtonBox.Ok).click()
            return QtWidgets.QDialog.Accepted
        with patch.object(QtWidgets.QDialog, 'exec_', accept_max):
            window.configure_point_waits()
        self.assertEqual(window.point_waits, [15000]*45)
        self.assertTrue(window.use_point_waits.isChecked())
        window.close()

    def test_three_peak_plan_resizes_ui_and_point_wait_editor_to_15(self):
        window = TemporaryTestWindow(None)
        table, x, _ = reference()
        y = sum(np.exp(-.5 * ((x - center) / .12) ** 2)
                for center in (1530., 1542., 1554.))
        window.plan = select_points(table, x, y)
        window.render_plan(ready=True)
        self.assertEqual(window.peak_count(), 3)
        self.assertEqual(window.point_count(), 15)
        self.assertEqual(window.table.rowCount(), 15)
        self.assertEqual(window.peak_fit_table.columnCount(), 3)
        self.assertIn('3峰×5点', window.title_label.text())
        self.assertIn('15个点', window.edit_point_waits.text())

        def inspect(dialog):
            table_widget = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table_widget.rowCount(), 15)
            self.assertEqual(len(dialog.findChildren(QtWidgets.QSpinBox)), 15)
            dialog.reject()
            return QtWidgets.QDialog.Rejected

        with patch.object(QtWidgets.QDialog, 'exec_', inspect):
            window.configure_point_waits()
        window.close()

    def test_forward_settle_diagnostics_are_keyed_by_physical_transition(self):
        hints = load_forward_settle_hints()
        if not hints['pairs']:
            self.assertIn('暂无', hints['source_label'])
            return
        self.assertIn('CH1', hints['source_label'])
        # Captured route: physical index 178 was followed by 333.
        item = hints['pairs'][(178, 333)]
        self.assertEqual(item['source_index'], 178)
        self.assertEqual(item['target_index'], 333)
        self.assertIn('settle_us', item)
        self.assertIn('edge_us', item)

    def test_realtime_fit_preserves_raw_points_and_keeps_flagged_wavelength(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        table, x, y = reference()
        window.plan = select_points(table, x, y)
        window.render_plan()
        data = decode(frame(),123,456)
        for g in range(9):
            rows = window.plan['rows'][g*5:g*5+5]
            xx = np.array([r['measured_nm'] for r in rows])
            yy = 30+600*np.exp(-.5*((xx-xx[2])/(xx[1]-xx[0]))**2)
            for j, val in enumerate(yy):
                data['records'][g*5+j]['second_code'] = int(val)
        data['quality'] = quality(data)
        window.update_frame(data)
        self.assertTrue(window.peak_fits[0]['valid'])
        center = window.peak_fit_table.item(0,0).text()
        raw = [r['second_code'] for r in data['records']]
        window.digital_gain.setValue(2)
        self.assertEqual(window.peak_fit_table.item(0,0).text(),center)
        np.testing.assert_array_equal(window.curves[0].getData()[1],np.array(raw[:5])*2)
        window.show_peak_fits.setChecked(False)
        self.assertEqual(len(window.fitted_curves[0].getData()[0] or []),0)
        window.show_peak_fits.setChecked(True)
        for rec in data['records']:
            rec['second_code'] = 3
        data['cycle_start_us'] += 70000
        data['sequence'] += 1
        window.update_frame(data)
        self.assertIn('⚠', window.peak_fit_table.item(0,0).text())
        self.assertIn('可能有问题', window.peak_fit_table.item(2,0).text())
        self.assertTrue(window.fit_labels[0].isVisible())
        window.invalidate_plan()
        self.assertEqual(window.peak_fits,[])
        window.close()

    def test_failed_dense_selection_preserves_previous_plan(self):
        from types import SimpleNamespace
        window=TemporaryTestWindow(None)
        previous=window.plan
        window.worker=SimpleNamespace(output='incomplete-reference.json')
        window.reference_ready(dict(rows=[]))
        self.assertIs(window.plan,previous)
        window.worker=None
        window.close()

    def test_dense_scan_enters_manual_45_point_placement(self):
        from types import SimpleNamespace
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        previous = window.plan
        report = {'rows': [
            {'measured_wavelength_nm': float(wavelength),
             'ch1_adc_code': float(value)}
            for wavelength, value in zip(x, y)
        ]}
        window.worker = SimpleNamespace(output='fresh-dense-reference.json')
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.reference_ready(report)
        self.assertIs(window.plan, previous)
        self.assertTrue(window.manual_selection_active)
        self.assertFalse(window.plan_ready)
        self.assertFalse(window.start_button.isEnabled())
        self.assertFalse(window.reference_line_button.isEnabled())
        window.worker = None
        automatic_route = select_points(table, x, y)
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            self.assertTrue(window.add_manual_selection_at(
                automatic_route['rows'][-1]['measured_nm']
            ))
            self.assertTrue(window.add_manual_selection_at(
                automatic_route['rows'][-2]['measured_nm']
            ))
            self.assertTrue(window.undo_manual_selection())
            self.assertEqual(
                window.manual_selection_indices,
                [automatic_route['rows'][-1]['index']],
            )
            self.assertTrue(window.clear_manual_selection())
            for row in reversed(automatic_route['rows']):
                self.assertTrue(window.add_manual_selection_at(row['measured_nm']))
        self.assertFalse(window.manual_selection_active)
        self.assertEqual(window.point_count(), 45)
        self.assertTrue(window.plan['manual_selection'])
        self.assertEqual(
            [row['index'] for row in window.plan['rows']],
            sorted(row['index'] for row in automatic_route['rows']),
        )
        self.assertTrue(window.reference_line_button.isEnabled())
        self.assertFalse(window.start_button.isEnabled())
        window.close()

    def test_dense_points_draw_live_and_failed_points_are_marked(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        window._dense_capture_active = True
        window._live_dense_rows = []
        for index, (value, passed) in enumerate(((100, True), (125, False), (160, True))):
            window.update_dense_reference_point({
                'row': {
                    'index': index,
                    'measured_wavelength_nm': 1525.0 + .02 * index,
                    'ch1_adc_code': value,
                    'point_passed': passed,
                    'failure_reason': '' if passed else '波动/漂移未通过',
                },
                'completed': index + 1,
                'total': 2001,
            })
        np.testing.assert_allclose(window.dense_curve.getData()[1], [100, 125, 160])
        self.assertEqual(len(window.dense_failed_curve.getData()[0]), 1)
        self.assertIn('已标红1个未通过点', window.status.text())
        window.show_dense_failures.setChecked(False)
        self.assertIsNone(window.dense_failed_curve.getData()[0])
        np.testing.assert_allclose(window.dense_curve.getData()[1], [100, 125, 160])
        window.show_dense_failures.setChecked(True)
        self.assertEqual(len(window.dense_failed_curve.getData()[0]), 1)
        window.close()

    def test_live_dbm_dense_curve_rescales_and_hides_previous_markers(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('dbm')
        window._dense_capture_active = True
        window._live_dense_rows = []
        for completed, code in ((1, 1.0), (20, 2000.0)):
            window.update_dense_reference_point({
                'row': {
                    'index': completed - 1,
                    'measured_wavelength_nm': 1525.0 + .02 * (completed - 1),
                    'signal_adc_code': code,
                    'point_passed': True,
                },
                'completed': completed,
                'total': 2001,
            })
        plotted = np.asarray(window.dense_curve.getData()[1], dtype=float)
        expected = np.asarray(adc_code_to_dbm([1.0, 2000.0], 2), dtype=float)
        np.testing.assert_allclose(plotted, expected)
        lower, upper = window.plot.viewRange()[1]
        self.assertLess(lower, float(np.min(expected)))
        self.assertGreater(upper, float(np.max(expected)))
        self.assertEqual(window.selection_markers, [])
        window.close()

    def test_early_stopped_dense_scan_keeps_curve_for_manual_selection(self):
        from types import SimpleNamespace
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        # Enough of the forward spectrum to include all nine peaks, while
        # deliberately stopping before the physical 2001st row.
        last = int(np.searchsorted(x, 1564.0))
        rows = []
        for index, (wavelength, value) in enumerate(zip(x[:last], y[:last])):
            rows.append({
                'index': index,
                'measured_wavelength_nm': float(wavelength),
                'ch1_adc_code': float(value),
                'stable': index != 150,
                'point_passed': index != 150,
                'failure_reason': '' if index != 150 else '波动/漂移未通过',
            })
        window.worker = SimpleNamespace(output='partial-dense-reference.json')
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.reference_ready({
                'rows': rows,
                'complete': False,
                'stopped_early': True,
                'ch1_feedback_selector': 2,
            })
        self.assertTrue(window.manual_selection_active)
        self.assertFalse(window.reference_line_button.isEnabled())
        self.assertEqual(len(window.dense_failed_points), 1)
        self.assertEqual(len(window.dense_failed_curve.getData()[0]), 1)
        self.assertIn('提前停止', window.status.text())
        window.worker = None
        window.close()

    def test_failed_dense_spike_is_drawn_raw_without_automatic_selection(self):
        from types import SimpleNamespace
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        table, x, y = reference()
        previous = window.plan
        failed_index = int(np.argmin(abs(x - 1529.2)))
        rows = []
        for index, (wavelength, value) in enumerate(zip(x, y)):
            failed = index == failed_index
            rows.append({
                'index': index,
                'measured_wavelength_nm': float(wavelength),
                'ch1_adc_code': 25.0 if failed else float(value),
                'stable': not failed,
                'point_passed': not failed,
                'failure_reason': '波动/漂移未通过' if failed else '',
            })
        window.worker = SimpleNamespace(output='failed-spike.json')
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.reference_ready({
                'rows': rows,
                'complete': True,
                'ch1_feedback_selector': 2,
            })
        self.assertIs(window.plan, previous)
        self.assertTrue(window.manual_selection_active)
        self.assertEqual(float(window.dense_values[failed_index]), 25.0)
        self.assertEqual(len(window.dense_failed_curve.getData()[0]), 1)
        window.worker = None
        window.close()

    def test_switch_90pct_result_populates_point_waits_without_reference_gate(self):
        from types import SimpleNamespace
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.prepare(x, y, 'dense.json')
        self.assertFalse(window.plan_ready)
        transitions = []
        for target in range(1, window.point_count()):
            t90 = 800.0 + target * 25.0
            transitions.append({
                'route_source_order': target - 1,
                'route_target_order': target,
                'target_fullband_index': window.plan['rows'][target]['index'],
                'channels': {'CH1': {
                    'meaningful_step': True,
                    't90_us': t90,
                    'classification': 'main_edge_complete',
                }},
            })
        window.worker = SimpleNamespace(output='switch-timing.json')
        window.switch_timing_ready({
            'complete': True,
            'repeats': 3,
            'direction': 'strictly forward, small wavelength to large wavelength',
            'analysis': {'transitions': transitions},
        })
        self.assertEqual(len(window.point_waits), 45)
        self.assertEqual(window.point_waits[0], 650)
        self.assertEqual(window.point_waits[1], 825)
        self.assertTrue(window.use_point_waits.isChecked())
        self.assertEqual(window.point_wait_route, window.point_route_key())
        self.assertFalse(window.plan_ready)
        self.assertIn('44个正向跳转', window.status.text())
        window.worker = None
        window.close()

    def test_current_dense_selection_uses_fixed_half_height_policy(self):
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table), \
             patch('temporary_test_widget.select_points', wraps=select_points) as choose:
            window.prepare(x, y, 'fresh-dense-reference.json')
        self.assertEqual(choose.call_args.kwargs, {'feedback_selector': 2})
        self.assertEqual(window.plan['selection_height_fraction'], .5)
        self.assertEqual(
            window.plan['selection_rule'],
            'five_equal_interval_points_at_or_above_half_height',
        )
        self.assertFalse(window.spacing.isEnabled())
        self.assertEqual(window.plan['reference_source'], 'fresh-dense-reference.json')
        window.close()

    def test_dense_spectrum_points_are_selected_then_moved_with_arrow_keys(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        table, x, y = reference()
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.prepare(x, y, 'fresh-dense-reference.json')
        dense_x, dense_y = window.dense_curve.getData()
        self.assertEqual(len(dense_x), len(x))
        np.testing.assert_allclose(dense_y, y)
        self.assertEqual(len(window.selection_markers), 45)
        self.assertTrue(all(not marker.movable for marker in window.selection_markers))
        before = [[row['index'] for row in window.plan['rows'][group*5:group*5+5]]
                  for group in range(9)]
        window.reference_values = [float(y[row['index']]) for row in window.plan['rows']]
        window.reference_wavelengths = [row['measured_nm'] for row in window.plan['rows']]
        old_reference_x = list(window.reference_wavelengths)
        window.plan_ready = True
        window.start_button.setEnabled(True)
        window.redraw_values()
        moved_index = before[3][1] + 1
        click = Mock()
        click.button.return_value = QtCore.Qt.LeftButton
        window.selection_markers[16].mouseClickEvent(click)
        click.accept.assert_called_once_with()
        self.assertEqual(window.selected_selection_point, (3, 1))
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            event = QtGui.QKeyEvent(
                QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier
            )
            window.keyPressEvent(event)
        after = [[row['index'] for row in window.plan['rows'][group*5:group*5+5]]
                 for group in range(9)]
        self.assertEqual(after[3][1], moved_index)
        self.assertEqual(after[3][0], before[3][0])
        self.assertEqual(after[3][2:], before[3][2:])
        self.assertNotEqual(len(set(np.diff(after[3]))), 1)
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertTrue(window.reference_line_button.isEnabled())
        self.assertIsNone(window.point_waits)
        self.assertEqual(window.reference_wavelengths, old_reference_x)
        self.assertIsNotNone(window.reference_curves[3].getData()[0])
        self.assertIn('手动修改峰4第2点', window.status.text())
        self.assertIn('可直接开始临时测试', window.status.text())
        self.assertEqual(window.selected_selection_point, (3, 1))
        self.assertEqual(window.selection_markers[16].pen.width(), 3)
        self.assertTrue(window.plan['manual_adjustment_reference_reused'])
        window.close()

    def test_last_nine_of_nine_session_restores_ready_without_new_reference(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        table, x, y = reference()
        with tempfile.TemporaryDirectory() as folder, \
             patch('temporary_test_widget.LAST_SESSION_PATH', Path(folder) / 'last.json'):
            window = TemporaryTestWindow(SimpleNamespace())
            window.channel_combo.setCurrentIndex(window.channel_combo.findData(2))
            window.equal_interval_nm.setValue(.06)
            window.equal_interval_settle.setValue(.125)
            window.dense_acquisition_method = 'equal_interval_single'
            with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
                window.prepare(x, y, 'fresh-dense-reference.json')
            reference_values = []
            for group in range(9):
                rows = window.plan['rows'][group*5:group*5+5]
                xx = np.asarray([row['measured_nm'] for row in rows])
                yy = 80 + 400*np.exp(-.5*((xx-xx[2])/(xx[1]-xx[0]))**2)
                reference_values.extend(int(round(value)) for value in yy)
            report = dict(
                complete=True, disarm_ack=True, shutter_ack=True,
                signal_channel=2, feedback_selector=0,
                rows=[dict(index=row['index'], dac_codes=row['codes'], stable=True,
                           signal_saturated=False,
                           signal_adc_code=reference_values[index])
                      for index, row in enumerate(window.plan['rows'])],
            )
            window.reference_line_ready(report)
            self.assertTrue(window.plan_ready)
            expected_indices = [row['index'] for row in window.plan['rows']]
            window.show_dense_selection.setChecked(False)
            window.show_reference.setChecked(False)
            window._set_display_mode('dbm')
            window.close()

            restored = TemporaryTestWindow(SimpleNamespace())
            self.assertTrue(restored.plan_ready)
            self.assertTrue(restored.start_button.isEnabled())
            self.assertEqual(expected_indices, [row['index'] for row in restored.plan['rows']])
            self.assertEqual(restored.reference_values, [float(v) for v in reference_values])
            self.assertEqual(len(restored.dense_wavelengths), len(x))
            self.assertEqual(restored.selected_channel(), 2)
            self.assertEqual(restored.dense_channel, 2)
            self.assertEqual(restored.reference_channel, 2)
            self.assertEqual(restored.dense_acquisition_method, 'equal_interval_single')
            self.assertAlmostEqual(restored.equal_interval_nm.value(), .06)
            self.assertAlmostEqual(restored.equal_interval_settle.value(), .125)
            self.assertIn('CH2', restored.plot.getAxis('left').labelText)
            # A verified route is shown again on startup even if both layers
            # were temporarily hidden during the preceding run.
            self.assertTrue(restored.show_dense_selection.isChecked())
            self.assertTrue(restored.show_reference.isChecked())
            self.assertEqual(restored.display_mode, 'dbm')
            self.assertEqual(restored.display_unit_button.text(), '显示：光功率 dBm')
            self.assertTrue(restored.show_dense_selection.isCheckable())
            self.assertTrue(restored.show_reference.isCheckable())
            self.assertIn('可直接开始临时测试', restored.status.text())
            with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
                self.assertTrue(restored.select_selection_point(3, 1))
                self.assertTrue(restored.move_selected_point(1))
            adjusted_indices = [row['index'] for row in restored.plan['rows']]
            self.assertNotEqual(adjusted_indices, expected_indices)
            self.assertTrue(restored.plan_ready)
            self.assertTrue(restored.start_button.isEnabled())
            restored.close()

            adjusted_restart = TemporaryTestWindow(SimpleNamespace())
            self.assertTrue(adjusted_restart.plan_ready)
            self.assertTrue(adjusted_restart.start_button.isEnabled())
            self.assertEqual(
                adjusted_indices,
                [row['index'] for row in adjusted_restart.plan['rows']],
            )
            self.assertIn('参考虚线仍是调整前数据', adjusted_restart.status.text())
            adjusted_restart.close()

    def test_unsaved_equal_interval_spectrum_survives_page_switch(self):
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        window.dense_acquisition_method = 'equal_interval_single'
        window.equal_interval_nm.setValue(.06)
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.prepare(x[::3], y[::3], 'equal-interval-live.json')
        expected_x = window.dense_wavelengths.copy()
        expected_y = window.dense_values.copy()
        stack = QtWidgets.QStackedWidget()
        other = QtWidgets.QWidget()
        stack.addWidget(window)
        stack.addWidget(other)
        stack.show()
        stack.setCurrentWidget(other)
        self.app.processEvents()
        stack.setCurrentWidget(window)
        self.app.processEvents()
        np.testing.assert_array_equal(window.dense_wavelengths, expected_x)
        np.testing.assert_array_equal(window.dense_values, expected_y)
        np.testing.assert_array_equal(window.dense_curve.getData()[0], expected_x)
        self.assertTrue(window.has_unsaved_spectrum())
        stack.close()

    def test_close_prompt_saves_spectrum_even_before_reference_line(self):
        import tempfile
        from pathlib import Path

        table, x, y = reference()
        with tempfile.TemporaryDirectory() as folder, \
             patch('temporary_test_widget.FINGER_CAPTURE_DIR', Path(folder)), \
             patch('temporary_test_widget.LAST_FINGER_CAPTURE_PATH',
                   Path(folder) / 'last.json'), \
             patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window = TemporaryTestWindow(None)
            window.prepare(x[::2], y[::2], 'unsaved-equal-interval.json')
            self.assertTrue(window.has_unsaved_spectrum())
            with patch.object(
                    QtWidgets.QMessageBox, 'question',
                    return_value=QtWidgets.QMessageBox.Save), \
                 patch.object(
                    QtWidgets.QInputDialog, 'getText',
                    return_value=('关闭前保存', True)):
                self.assertTrue(window.confirm_save_before_close())
            record = Path(folder) / '关闭前保存.json'
            self.assertTrue(record.exists())
            payload = json.loads(record.read_text(encoding='utf-8'))
            self.assertIsNone(payload['reference_values'])
            self.assertFalse(payload['ready'])
            self.assertFalse(window.has_unsaved_spectrum())

            restored = TemporaryTestWindow(None)
            self.assertTrue(restored._load_saved_finger_capture(record))
            self.assertEqual(len(restored.dense_wavelengths), len(x[::2]))
            self.assertFalse(restored.plan_ready)
            self.assertTrue(restored.reference_line_button.isEnabled())
            restored.close()
            window.close()

    def test_close_prompt_cancel_and_discard(self):
        table, x, y = reference()
        window = TemporaryTestWindow(None)
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.prepare(x, y, 'unsaved.json')
        with patch.object(
                QtWidgets.QMessageBox, 'question',
                return_value=QtWidgets.QMessageBox.Cancel):
            self.assertFalse(window.confirm_save_before_close())
        self.assertTrue(window.has_unsaved_spectrum())
        with patch.object(
                QtWidgets.QMessageBox, 'question',
                return_value=QtWidgets.QMessageBox.Discard):
            self.assertTrue(window.confirm_save_before_close())
        self.assertFalse(window.has_unsaved_spectrum())
        window.close()

    def test_startup_defaults_to_nine_peaks_and_three_peak_record_loads_explicitly(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        table, x, _ = reference()
        dense = sum(np.exp(-.5 * ((x - center) / .12) ** 2)
                    for center in (1530., 1542., 1554.))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session_path = root / 'last_session.json'
            record_dir = root / 'finger_captures'
            latest_path = root / 'last_finger_capture.json'
            patches = (
                patch('temporary_test_widget.LAST_SESSION_PATH', session_path),
                patch('temporary_test_widget.FINGER_CAPTURE_DIR', record_dir),
                patch('temporary_test_widget.LAST_FINGER_CAPTURE_PATH', latest_path),
                patch('app_JDSU.load_fullband_accuracy_table', return_value=table),
            )
            for item in patches:
                item.start()
            try:
                window = TemporaryTestWindow(SimpleNamespace())
                window.prepare(x, reference()[2], 'nine-peak-dense.json')
                window.reference_values = [
                    float(np.interp(row['measured_nm'], x, reference()[2]) * 1000)
                    for row in window.plan['rows']
                ]
                window.reference_wavelengths = [
                    float(row['measured_nm']) for row in window.plan['rows']
                ]
                window.plan_ready = True
                with patch.object(
                        QtWidgets.QInputDialog, 'getText',
                        return_value=('9峰默认', True)):
                    self.assertTrue(window.save_finger_capture())

                window.prepare(x, dense, 'three-peak-dense.json')
                self.assertEqual(window.point_count(), 15)
                window.reference_values = [
                    float(np.interp(row['measured_nm'], x, dense) * 1000)
                    for row in window.plan['rows']
                ]
                window.reference_wavelengths = [
                    float(row['measured_nm']) for row in window.plan['rows']
                ]
                window.dense_feedback_selector = 2
                window.reference_feedback_selector = 1
                window._set_display_mode('dbm')
                window.plan_ready = True
                window.start_button.setEnabled(True)
                with patch.object(
                        QtWidgets.QInputDialog, 'getText',
                        return_value=('3号手指', True)):
                    self.assertTrue(window.save_finger_capture())
                self.assertTrue((record_dir / '3号手指.json').exists())
                self.assertTrue(latest_path.exists())
                latest_payload = json.loads(latest_path.read_text(encoding='utf-8'))
                self.assertEqual(latest_payload['peak_count'], 9)
                self.assertIn('3号手指', window.finger_record_label.text())
                self.assertIn('下次仍默认载入9峰', window.status.text())
                window.close()

                restored = TemporaryTestWindow(SimpleNamespace())
                self.assertEqual(restored.point_count(), 45)
                self.assertEqual(restored.peak_count(), 9)
                self.assertIn('9峰默认', restored.finger_record_label.text())

                self.assertTrue(restored._load_saved_finger_capture(
                    record_dir / '3号手指.json'
                ))
                self.assertEqual(restored.point_count(), 15)
                self.assertEqual(restored.peak_count(), 3)
                self.assertEqual(len(restored.dense_wavelengths), 2001)
                self.assertEqual(len(restored.reference_values), 15)
                self.assertEqual(restored.dense_feedback_selector, 2)
                self.assertEqual(restored.reference_feedback_selector, 1)
                self.assertEqual(restored.display_mode, 'dbm')
                self.assertTrue(restored.plan_ready)
                self.assertTrue(restored.start_button.isEnabled())
                self.assertIn('3号手指', restored.finger_record_label.text())
                self.assertIn('可直接开始临时测试', restored.status.text())
                restored.close()
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_live_fit_failures_do_not_revoke_verified_route(self):
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        window.plan = select_points(table, x, y)
        window.render_plan(ready=True)
        data = decode(frame(), 123, 456)
        for record in data['records']:
            record['second_code'] = 3
        data['quality'] = quality(data)
        for sequence in range(16, 19):
            data['sequence'] = sequence
            data['cycle_start_us'] += 70000
            window.update_frame(data)
        self.assertTrue(any(streak >= 3 for streak in window.fit_failure_streaks))
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertIn('连续出现拟合质量警告', window.status.text())
        window.close()

    def test_stop_and_finished_keep_verified_route_restartable(self):
        window = TemporaryTestWindow(None)
        window.plan_ready = True
        window.start_button.setEnabled(False)
        worker = Mock()
        window.worker = worker
        window.stop()
        worker.stop.assert_called_once_with()
        window.finished()
        worker.deleteLater.assert_called_once_with()
        self.assertIsNone(window.worker)
        self.assertTrue(app_JDSU.switch_mode_enable)
        self.assertTrue(window.start_button.isEnabled())
        window.close()

    def test_pre_persistence_reference_line_is_migrated_on_first_restart(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        table, x, y = reference()
        plan = select_points(table, x, y)
        reference_values = []
        for group in range(9):
            rows = plan['rows'][group*5:group*5+5]
            xx = np.asarray([row['measured_nm'] for row in rows])
            yy = 80 + 400*np.exp(-.5*((xx-xx[2])/(xx[1]-xx[0]))**2)
            reference_values.extend(int(round(value)) for value in yy)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dense_path = root / 'reference_1.json'
            dense_path.write_text(json.dumps(dict(
                complete=True,
                rows=[dict(measured_wavelength_nm=float(xx), ch1_adc_code=float(yy))
                      for xx, yy in zip(x, y)],
            )), encoding='utf-8')
            line_path = root / 'reference_line_2.json'
            line_path.write_text(json.dumps(dict(
                complete=True, disarm_ack=True, shutter_ack=True,
                ch1_feedback_selector=2,
                rows=[dict(index=row['index'], dac_codes=row['codes'], stable=True,
                           ch1_saturated=False, ch1_adc_code=reference_values[index])
                      for index, row in enumerate(plan['rows'])],
            )), encoding='utf-8')
            session_path = root / 'last_45_session.json'
            with patch('temporary_test_widget.LAST_SESSION_PATH', session_path), \
                 patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
                restored = TemporaryTestWindow(SimpleNamespace())
            self.assertTrue(restored.plan_ready)
            self.assertTrue(restored.start_button.isEnabled())
            self.assertEqual(reference_values, restored.reference_values)
            self.assertEqual(len(restored.dense_wavelengths), len(x))
            self.assertTrue(session_path.exists())
            self.assertIn('转换为新的自动保存会话', restored.status.text())
            restored.close()

    def test_failed_pre_persistence_reference_line_is_drawn_but_not_ready(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        table, x, y = reference()
        plan = select_points(table, x, y)
        reference_values = [80, 250, 600, 250, 80] * 9
        # Deliberately make peak 7 non-Gaussian.  The old UI still wrote a
        # complete reference_line record for this case, and the new UI must
        # preserve and display it so the operator can adjust the five points.
        reference_values[30:35] = [68, 314, 211, 54, 180]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'reference_1.json').write_text(json.dumps(dict(
                complete=True,
                rows=[dict(measured_wavelength_nm=float(xx), ch1_adc_code=float(yy))
                      for xx, yy in zip(x, y)],
            )), encoding='utf-8')
            (root / 'reference_line_2.json').write_text(json.dumps(dict(
                complete=True, disarm_ack=True, shutter_ack=True,
                ch1_feedback_selector=2,
                rows=[dict(index=row['index'], dac_codes=row['codes'], stable=True,
                           ch1_saturated=False, ch1_adc_code=reference_values[index])
                      for index, row in enumerate(plan['rows'])],
            )), encoding='utf-8')
            session_path = root / 'last_45_session.json'
            with patch('temporary_test_widget.LAST_SESSION_PATH', session_path), \
                 patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
                restored = TemporaryTestWindow(SimpleNamespace())
            self.assertFalse(restored.plan_ready)
            self.assertFalse(restored.start_button.isEnabled())
            self.assertEqual(reference_values, restored.reference_values)
            self.assertEqual(
                [row['index'] for row in plan['rows']],
                [row['index'] for row in restored.plan['rows']],
            )
            self.assertIsNotNone(restored.reference_curves[6].getData()[0])
            self.assertTrue(restored.show_reference.isChecked())
            self.assertTrue(session_path.exists())
            self.assertIn('峰7', restored.status.text())
            self.assertIn('未通过的参考虚线', restored.status.text())
            restored.close()

    def test_reference_line_independent_gain_and_invalidation(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        table,x,y = reference()
        window.plan = select_points(table,x,y)
        window.render_plan(ready=False)
        reference_values = []
        for group in range(9):
            rows = window.plan['rows'][group*5:group*5+5]
            xx = np.asarray([row['measured_nm'] for row in rows])
            yy = 80 + 400*np.exp(-.5*((xx-xx[2])/(xx[1]-xx[0]))**2)
            reference_values.extend(int(round(value)) for value in yy)
        report = dict(complete=True,disarm_ack=True,shutter_ack=True,ch1_feedback_selector=2,
            rows=[dict(index=r['index'],dac_codes=r['codes'],stable=True,ch1_saturated=False,
                       ch1_adc_code=reference_values[i]) for i,r in enumerate(window.plan['rows'])])
        window.reference_line_ready(report)
        self.assertTrue(window.plan_ready)
        data = decode(frame(),123,456)
        data['quality'] = quality(data)
        window.update_frame(data)
        window.reference_gain.setValue(3)
        window.digital_gain.setValue(2)
        np.testing.assert_array_equal(window.reference_curves[0].getData()[1],np.array(reference_values[:5])*3)
        np.testing.assert_array_equal(window.curves[0].getData()[1],[204]*5)
        self.assertEqual(window.table.item(0,5).text(),str(reference_values[0]))
        self.assertEqual(window.reference_values,reference_values)
        window.show_reference.setChecked(False)
        self.assertIsNone(window.reference_curves[0].getData()[0])
        window.show_reference.setChecked(True)
        np.testing.assert_array_equal(window.reference_curves[0].getData()[1],np.array(reference_values[:5])*3)
        window.analog_gain.setCurrentIndex(3)
        self.assertIsNone(window.reference_values)
        self.assertFalse(window.plan_ready)
        window.reference_line_ready(report)
        self.assertIsNone(window.reference_values)
        window.analog_gain.setCurrentIndex(1)
        report['rows'][0]['stable']=False
        window.reference_line_ready(report)
        self.assertEqual(window.reference_values, reference_values)
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertEqual(len(window.reference_failed_curve.getData()[0]), 1)
        self.assertEqual(
            window.table.item(0, 5).background().color().name(), '#fee2e2'
        )
        self.assertIn('仍可直接开始临时测试', window.status.text())
        window.close()

    def test_reference_line_rejects_flat_or_stale_peak_window(self):
        window = TemporaryTestWindow(None)
        table,x,y = reference()
        window.plan = select_points(table,x,y)
        window.render_plan(ready=False)
        report = dict(complete=True,disarm_ack=True,shutter_ack=True,ch1_feedback_selector=2,
            rows=[dict(index=row['index'],dac_codes=row['codes'],stable=True,ch1_saturated=False,
                       ch1_adc_code=80) for row in window.plan['rows']])
        window.reference_line_ready(report)
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertEqual(window.reference_values, [80] * 45)
        self.assertIsNotNone(window.reference_curves[0].getData()[0])
        self.assertIn('参考线已绘制', window.status.text())

        good = [21, 216, 531, 74, 44]
        for group in range(9):
            for point, value in enumerate(good):
                report['rows'][group*5+point]['ch1_adc_code'] = value
        report['rows'][30:35] = [dict(report['rows'][30+i], ch1_adc_code=value)
                                 for i,value in enumerate([68, 314, 211, 54, 180])]
        window.reference_line_ready(report)
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertIn('峰7', window.status.text())
        window.close()

    def test_failed_reference_attempt_keeps_existing_route_startable(self):
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        window.plan = select_points(table, x, y)
        window.render_plan(ready=True)
        old_values = [100] * 45
        window.reference_values = old_values
        window.reference_wavelengths = [row['measured_nm'] for row in window.plan['rows']]
        window.start_button.setEnabled(True)
        window.reference_line_ready(dict(
            complete=False, disarm_ack=True, shutter_ack=True,
            ch1_feedback_selector=2, rows=[],
        ))
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertEqual(window.reference_values, old_values)
        self.assertIn('仍可直接开始临时测试', window.status.text())
        window.close()

    def test_failed_reference_shape_can_restart_as_startable_route(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        table, x, y = reference()
        with tempfile.TemporaryDirectory() as folder, \
             patch('temporary_test_widget.LAST_SESSION_PATH', Path(folder) / 'last.json'), \
             patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window = TemporaryTestWindow(SimpleNamespace())
            window.prepare(x, y, 'fresh-dense-reference.json')
            report = dict(
                complete=True, disarm_ack=True, shutter_ack=True,
                ch1_feedback_selector=2,
                rows=[dict(index=row['index'], dac_codes=row['codes'], stable=True,
                           ch1_saturated=False, ch1_adc_code=80)
                      for row in window.plan['rows']],
            )
            window.reference_line_ready(report)
            self.assertTrue(window.plan_ready)
            self.assertTrue(window.plan['reference_shape_gate_bypassed'])
            window.close()
            restored = TemporaryTestWindow(SimpleNamespace())
            self.assertTrue(restored.plan_ready)
            self.assertTrue(restored.start_button.isEnabled())
            self.assertIn('未通过', restored.status.text())
            restored.close()

    def test_failed_sparse_peak_keeps_line_and_route_for_manual_adjustment(self):
        window = TemporaryTestWindow(None)
        table, x, y = reference()
        window.plan = select_points(table, x, y)
        window.plan['reference_source'] = 'fresh-dense-reference.json'
        window.render_plan(ready=False)
        original_groups = [
            [row['index'] for row in window.plan['rows'][group*5:group*5+5]]
            for group in range(9)
        ]
        fallback_groups = [list(group) for group in original_groups]
        fallback_groups[3] = [index + 1 for index in fallback_groups[3]]
        good = [80, 250, 600, 250, 80]
        window.validated_route_indices = fallback_groups
        window.validated_route_values = good * 9
        window.validated_route_source = 'last-real-nine-of-nine.json'

        first_values = good * 9
        first_values[15:20] = [420, 120, 600, 450, 200]
        first = dict(
            complete=True, disarm_ack=True, shutter_ack=True,
            ch1_feedback_selector=2,
            rows=[
                dict(index=row['index'], dac_codes=row['codes'], stable=True,
                     ch1_saturated=False, ch1_adc_code=first_values[index])
                for index, row in enumerate(window.plan['rows'])
            ],
        )
        with patch('app_JDSU.load_fullband_accuracy_table', return_value=table):
            window.reference_line_ready(first)
        retained_groups = [
            [row['index'] for row in window.plan['rows'][group*5:group*5+5]]
            for group in range(9)
        ]
        self.assertEqual(retained_groups, original_groups)
        self.assertEqual(window.reference_values, first_values)
        self.assertIsNotNone(window.reference_curves[3].getData()[0])
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertIn('峰4', window.status.text())
        self.assertIn('已绘制', window.status.text())

        second = dict(
            complete=True, disarm_ack=True, shutter_ack=True,
            ch1_feedback_selector=2,
            rows=[
                dict(index=row['index'], dac_codes=row['codes'], stable=True,
                     ch1_saturated=False, ch1_adc_code=good[index % 5])
                for index, row in enumerate(window.plan['rows'])
            ],
        )
        window.reference_line_ready(second)
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        window.close()

    def test_three_consecutive_fit_warnings_only_report_live_quality(self):
        window = TemporaryTestWindow(None)
        table,x,y = reference()
        window.plan = select_points(table,x,y)
        window.render_plan(ready=True)
        data = decode(frame(),123,456)
        for record in data['records']:
            record['second_code'] = 3
        data['quality'] = quality(data)
        for sequence in (16, 17, 18):
            data['sequence'] = sequence
            data['cycle_start_us'] += 70000
            window.update_frame(data)
        self.assertTrue(window.plan_ready)
        self.assertTrue(window.start_button.isEnabled())
        self.assertIn('连续出现拟合质量警告', window.status.text())
        window.close()

    def test_reference_worker_uses_same_45_points_and_analog_gain(self):
        table,x,y = reference()
        plan=select_points(table,x,y)
        worker=ReferenceLineWorker(None,plan,'unused.json',0,feedback_selector=3)
        with patch('temporary_reference.acquire_reference',return_value={}) as acquire:
            worker.run()
        self.assertEqual(acquire.call_args.kwargs['selected_rows'],plan['rows'])
        self.assertEqual(acquire.call_args.kwargs['feedback_selector'],3)
        self.assertTrue(acquire.call_args.kwargs['continue_on_unstable'])
        self.assertTrue(acquire.call_args.kwargs['allow_partial_stop'])
        self.assertIn('on_row', acquire.call_args.kwargs)

    def test_one_time_fit_does_not_hide_later_height_changes(self):
        window = TemporaryTestWindow(None)
        window._set_display_mode('adc')
        table, x, y = reference()
        window.plan = select_points(table, x, y)
        window.render_plan()
        data = decode(frame(), 123, 456)
        data['quality'] = quality(data)
        window.update_frame(data)
        initial_range = window.plot.getViewBox().viewRange()[1]
        self.assertAlmostEqual(initial_range[1], 102*1.3)
        for row in data['records']:
            row['second_code'] = 120
        data['sequence'] += 1
        data['cycle_start_us'] += 60000
        window.update_frame(data)
        self.assertEqual(window.plot.getViewBox().viewRange()[1], initial_range)
        self.assertEqual(window.table.item(0,4).text(), '120')
        window.fit_values()
        self.assertAlmostEqual(window.plot.getViewBox().viewRange()[1][1], 120*1.3)
        window.close()


if __name__ == '__main__':
    unittest.main()
