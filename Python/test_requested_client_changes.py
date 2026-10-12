"""Offline checks for startup preferences, automatic points and USB catalog."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import app_JDSU
from PyQt5 import QtWidgets
from temporary_test_widget import TemporaryTestWindow
from temporary_test_mode import auto_select_dense_plan
from test_temporary_test_mode import reference
from usb_firmware_catalog import resolve_usb_package


def spectrum(step=.04, start=1525, stop=1555):
    table, _, _ = reference()
    indices = np.arange(int(round((start - 1525) / .02)),
                        int(round((stop - 1525) / .02)) + 1, int(round(step / .02)))
    x = np.asarray([table[index].measured_nm for index in indices])
    y = np.ones_like(x)
    for center in np.linspace(1528.5, 1551, 9):
        y += 1000 * np.exp(-.5 * ((x - center) / .30) ** 2)
    return table, indices, x, y


class RequestedChangesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_fresh_page_defaults_match_screenshot(self):
        window = TemporaryTestWindow(None)
        try:
            self.assertEqual(window.selected_channel(), 1)
            self.assertEqual(window.scan_start_nm.value(), 1525)
            self.assertEqual(window.scan_stop_nm.value(), 1555)
            self.assertEqual(window.equal_interval_nm.value(), .04)
            self.assertEqual(window.equal_interval_settle.value(), .001)
        finally:
            window.close()

    def test_saved_future_parameters_override_old_capture_without_rewriting_raw_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'preferences.json'
            payload = dict(schema='finger_stress_scan_preferences_v1', signal_channel=1,
                           scan_start_nm=1525, scan_stop_nm=1555,
                           equal_interval_nm=.04, equal_interval_settle_s=.001)
            path.write_text(json.dumps(payload), encoding='utf-8')
            window = TemporaryTestWindow(None)
            try:
                _, _, x, y = spectrum()
                window.dense_wavelengths, window.dense_values = x.copy(), y.copy()
                window.scan_stop_nm.setValue(1565)
                window.equal_interval_nm.setValue(.02)
                with patch('temporary_test_widget.SCAN_PREFERENCES_PATH', path):
                    self.assertTrue(window._load_scan_preferences())
                self.assertEqual(window.scan_stop_nm.value(), 1555)
                self.assertEqual(window.equal_interval_nm.value(), .04)
                np.testing.assert_array_equal(window.dense_values, y)
            finally:
                window.close()

    def test_stride_two_subrange_selects_physical_calibrated_rows(self):
        for start in (1525, 1526):
            table, sampled_indices, x, y = spectrum(start=start)
            plan = auto_select_dense_plan(table, x, y)
            self.assertEqual(len(plan['rows']), 63)
            self.assertEqual(plan['points_per_peak'], 7)
            self.assertEqual(plan['symmetry_basis'], 'fitted_dbm_height')
            for row in plan['rows']:
                self.assertIn(row['index'], sampled_indices)
                self.assertEqual(row['codes'], list(table[row['index']].codes))
            self.assertTrue(all(center > 1528 for center in plan['peaks_nm']))
            for group in range(9):
                rows = plan['rows'][group * 7:group * 7 + 7]
                self.assertEqual(len(rows), 7)
                # Current selection pairs fitted heights, not a fixed 1 nm span.
                self.assertTrue(all(a['measured_nm'] < b['measured_nm']
                                    for a, b in zip(rows, rows[1:])))
            np.testing.assert_array_equal(plan['reference_values'], y)

    def test_failure_preserves_existing_route_and_raw_data(self):
        window = TemporaryTestWindow(None)
        try:
            table, _, x, _ = spectrum()
            window.dense_wavelengths, window.dense_values = x, np.ones_like(x)
            original = window.plan
            with patch.object(app_JDSU, 'load_fullband_accuracy_table', return_value=table):
                self.assertFalse(window.auto_select_points())
            self.assertIs(window.plan, original)
            self.assertIn('手动', window.status.text())
        finally:
            window.close()

    def test_automatic_route_can_be_undone_then_completed_manually(self):
        window = TemporaryTestWindow(None)
        try:
            table, _, x, y = spectrum()
            window.dense_wavelengths, window.dense_values = x, y
            with patch.object(app_JDSU, 'load_fullband_accuracy_table', return_value=table):
                self.assertTrue(window.auto_select_points())
                self.assertFalse(window.plan_ready)
                self.assertFalse(window.start_button.isEnabled())
                self.assertTrue(window.reference_line_button.isEnabled())
                last = window.plan['rows'][-1]['measured_nm']
                self.assertTrue(window.undo_manual_selection())
                self.assertEqual(len(window.manual_selection_indices), 62)
                self.assertFalse(window.reference_line_button.isEnabled())
                self.assertTrue(window.add_manual_selection_at(last))
                self.assertEqual(window.point_count(), 63)
            np.testing.assert_array_equal(window.dense_values, y)
        finally:
            window.close()

    @staticmethod
    def dense_report(x, y, *, complete=True, channel=1):
        return dict(
            rows=[dict(index=index, measured_wavelength_nm=float(wavelength),
                       signal_adc_code=float(value), point_passed=True)
                  for index, (wavelength, value) in enumerate(zip(x, y))],
            complete=complete, requested_point_count=len(x),
            signal_channel=channel, feedback_selector=2,
            acquisition_method='equal_interval_single', settle_s=.001,
        )

    def test_complete_dense_scan_selects_63_points_after_worker_cleanup(self):
        for step, stop in ((.04, 1555), (.02, 1565)):
            with self.subTest(step=step, stop=stop):
                window = TemporaryTestWindow(None)
                try:
                    table, sampled_indices, x, y = spectrum(step=step, stop=stop)
                    worker = SimpleNamespace(output='completed-dense.json',
                                             deleteLater=Mock())
                    window.worker = worker
                    with patch.object(app_JDSU, 'load_fullband_accuracy_table', return_value=table), \
                            patch.object(window, 'auto_select_points', wraps=window.auto_select_points) as select:
                        window.reference_ready(self.dense_report(x, y))
                        select.assert_not_called()
                        self.assertTrue(window._auto_select_after_dense)
                        window.finished()
                        select.assert_called_once_with()
                    worker.deleteLater.assert_called_once_with()
                    self.assertIsNone(window.worker)
                    self.assertFalse(window._auto_select_after_dense)
                    self.assertTrue(window.plan['automatic_selection'])
                    self.assertEqual(len(window.plan['rows']), 63)
                    self.assertTrue(all(row['index'] in sampled_indices
                                        for row in window.plan['rows']))
                    self.assertFalse(window.manual_selection_active)
                    self.assertFalse(window.plan_ready)
                    self.assertFalse(window.start_button.isEnabled())
                    self.assertTrue(window.reference_line_button.isEnabled())
                    self.assertIn('自动选点完成', window.status.text())
                    np.testing.assert_array_equal(window.dense_values, y)
                finally:
                    window.worker = None
                    window.close()

    def test_partial_or_missing_dense_rows_do_not_trigger_automatic_selection(self):
        table, _, x, y = spectrum()
        for kind in ('partial', 'stopped', 'missing', 'duplicate', 'empty'):
            with self.subTest(kind=kind):
                window = TemporaryTestWindow(None)
                try:
                    window.worker = SimpleNamespace(output='incomplete-dense.json',
                                                    deleteLater=Mock())
                    report = self.dense_report(x, y)
                    if kind == 'partial':
                        report['complete'] = False
                    elif kind == 'stopped':
                        report['stopped_early'] = True
                    elif kind == 'missing':
                        report['rows'].pop()
                    elif kind == 'duplicate':
                        report['rows'][-1] = dict(report['rows'][-2])
                    else:
                        report['rows'] = []
                    with patch.object(window, 'auto_select_points') as select:
                        window.reference_ready(report)
                        window.finished()
                        select.assert_not_called()
                    self.assertFalse(window._auto_select_after_dense)
                    self.assertFalse(window.start_button.isEnabled())
                    if kind != 'empty':
                        self.assertTrue(window.manual_selection_active)
                finally:
                    window.worker = None
                    window.close()

    def test_completed_unrecognizable_spectrum_preserves_route_and_allows_manual_selection(self):
        window = TemporaryTestWindow(None)
        try:
            table, _, x, y = spectrum()
            window.dense_wavelengths, window.dense_values = x, y
            with patch.object(app_JDSU, 'load_fullband_accuracy_table', return_value=table):
                self.assertTrue(window.auto_select_points())
                previous = window.plan
                window.worker = SimpleNamespace(output='flat-dense.json', deleteLater=Mock())
                flat = np.ones_like(x)
                window.reference_ready(self.dense_report(x, flat))
                window.finished()
            self.assertIs(window.plan, previous)
            self.assertTrue(window.manual_selection_active)
            self.assertFalse(window.plan_ready)
            self.assertFalse(window.start_button.isEnabled())
            self.assertIn('自动选点失败', window.status.text())
            np.testing.assert_array_equal(window.dense_values, flat)
        finally:
            window.worker = None
            window.close()

    def test_completed_scan_keeps_failed_raw_points_and_requires_new_reference(self):
        window = TemporaryTestWindow(None)
        try:
            table, _, x, y = spectrum()
            report = self.dense_report(x, y)
            report['rows'][25].update(point_passed=False, failure_reason='稳定性未通过')
            window.worker = SimpleNamespace(output='failed-point-dense.json', deleteLater=Mock())
            with patch.object(app_JDSU, 'load_fullband_accuracy_table', return_value=table):
                window.reference_ready(report)
                window.finished()
            self.assertTrue(window.plan['automatic_selection'])
            self.assertEqual(len(window.dense_failed_points), 1)
            self.assertEqual(window.plan['dense_failed_points'], window.dense_failed_points)
            self.assertFalse(window.plan_ready)
            self.assertFalse(window.start_button.isEnabled())
            np.testing.assert_array_equal(window.dense_values, y)
        finally:
            window.worker = None
            window.close()

    def test_wrong_channel_cannot_automatically_replace_existing_route(self):
        window = TemporaryTestWindow(None)
        try:
            _, _, x, y = spectrum()
            previous = window.plan
            window.worker = SimpleNamespace(output='other-channel-dense.json', deleteLater=Mock())
            window.reference_ready(self.dense_report(x, y, channel=2))
            with patch.object(app_JDSU, 'load_fullband_accuracy_table') as load:
                window.finished()
                load.assert_not_called()
            self.assertIs(window.plan, previous)
            self.assertFalse(window.start_button.isEnabled())
            self.assertIn('当前通道', window.status.text())
        finally:
            window.worker = None
            window.close()

    def test_verified_machine_two_catalog_and_wrong_board(self):
        package = resolve_usb_package('machine_2', '395736523033')
        self.assertEqual(package.version_text, '1.0.106')
        with self.assertRaisesRegex(Exception, '不匹配'):
            resolve_usb_package('machine_2', 'different-board')


if __name__ == '__main__':
    unittest.main()
