import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets

from fbg_network_transport import NETWORK_LAN, NETWORK_WAN
from fbg_remote_protocol import MODE_STRESS, TelemetryFrame
from fbg_unified_app import SOURCE_LOCAL, Mach3EmergencyService, UnifiedFbgStudio
from finger_dataset import FingerSessionWriter
from ota_machine_profiles import OtaMachineProfileStore
from machine_identity import MachineIdentityStore
from types import SimpleNamespace


class UnifiedStudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self._ota_tempdir = tempfile.TemporaryDirectory()
        # Unit tests must neither load nor overwrite the operator's real named
        # spectra/rolling routes. A saved machine-three route is valid in the
        # app but cannot be allowed to change an empty-fixture assertion.
        folder = Path(self._ota_tempdir.name)
        override = mock.patch.multiple(
            'temporary_test_widget',
            SCAN_PREFERENCES_PATH=folder / 'scan_preferences.json',
            LAST_SESSION_PATH=folder / 'last_session.json',
            FINGER_CAPTURE_DIR=folder / 'finger_captures',
            LAST_FINGER_CAPTURE_PATH=folder / 'last_finger_capture.json',
        )
        override.start()
        self.addCleanup(override.stop)
        ota_store = OtaMachineProfileStore(
            Path(self._ota_tempdir.name) / "ota_profiles.json",
            protector=lambda value: "test:" + value,
            unprotector=lambda value: value.removeprefix("test:"),
        )
        self.window = UnifiedFbgStudio(
            initial_source=NETWORK_LAN,
            demo=True,
            ota_settings_store=ota_store,
            machine_identity_store=MachineIdentityStore(folder / 'machine_identity.json'),
        )
        self.window.show()
        for _ in range(4):
            self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self._ota_tempdir.cleanup()

    def test_all_operator_pages_are_embedded_in_one_window(self):
        self.assertEqual(self.window.workspace_tabs.count(), 5)
        self.assertIs(
            self.window.local_controller.page_peak.finger_3d_window,
            self.window.finger_controller,
        )
        self.assertIs(
            self.window.remote_controller.finger_3d_window,
            self.window.finger_controller,
        )
        self.assertIs(self.window.source_stack.widget(0), self.window.local_page)
        self.assertIs(self.window.source_stack.widget(1), self.window.remote_page)
        self.assertIsNot(self.window.software_update_page, self.window.ota_page)

    def test_sidebar_developer_mode_is_default_off_and_only_changes_visibility(self):
        page = self.window.local_controller.page_temporary_test
        self.assertFalse(self.window.developer_mode_button.isChecked())
        self.assertTrue(page.point_diagnostics_row.isHidden())
        self.assertTrue(page.developer_budget_row.isHidden())
        delays = (page.first_delay.value(), page.boundary_delay.value())
        self.window.developer_mode_button.click()
        self.assertTrue(page.developer_mode)
        self.assertFalse(page.point_diagnostics_row.isHidden())
        self.assertFalse(page.developer_budget_row.isHidden())
        self.window.resize(1000, 700)
        self.app.processEvents()
        self.assertFalse(self.window.developer_mode_button.isHidden())
        self.window.developer_mode_button.click()
        self.assertFalse(page.developer_mode)
        self.assertTrue(page.point_diagnostics_row.isHidden())
        self.assertEqual((page.first_delay.value(), page.boundary_delay.value()), delays)

    def test_developer_mode_does_not_clear_active_per_point_waits(self):
        page = self.window.local_controller.page_temporary_test
        waits = [300, 850] * 22 + [850]
        page.point_waits = list(waits)
        page.use_point_waits.blockSignals(True)
        page.use_point_waits.setChecked(True)
        page.use_point_waits.blockSignals(False)
        self.window.developer_mode_button.click()
        self.window.developer_mode_button.click()
        self.assertTrue(page.use_point_waits.isChecked())
        self.assertEqual(page.point_waits, waits)

    def test_port_menu_identifies_and_loads_owner_before_usb_open(self):
        import app_JDSU
        fake = SimpleNamespace(port=None, is_open=False, dtr=False,
                               close=mock.Mock(), reset_input_buffer=mock.Mock())
        opened_owners = []
        def open_usb():
            opened_owners.append(self.window.machine_selector.currentData())
            fake.is_open = True
        fake.open = open_usb
        self.window.usb_serial = fake
        context, info = self.usb_identity_context()
        action = QtWidgets.QAction('COM99', self.window.local_controller)
        action.setCheckable(True)
        action.setData('COM99')
        action.triggered.connect(self.window.local_controller.set_com_port)
        with context, mock.patch('app_JDSU.local_control_session', True), \
                mock.patch('app_JDSU.list_ports.comports', return_value=[info]):
            action.trigger()
        self.assertEqual(opened_owners, ['machine_3'])
        self.assertEqual(self.window.usb_port, 'COM99')
        self.assertEqual(self.window.local_controller.port, 'COM99')
        self.assertEqual(fake.port, 'COM99')

    def usb_identity_context(self, serial='207933893232', device='COM99'):
        import app_JDSU
        self.window.demo = False
        self.window.current_source = SOURCE_LOCAL
        self.addCleanup(setattr, self.window, 'demo', True)
        info = SimpleNamespace(device=device, vid=0x0483, pid=0x5740,
                               serial_number=serial)
        return mock.patch.multiple(app_JDSU, ser=self.window.usb_serial,
                                   ser_open=False, ap_open=False,
                                   switch_mode_enable=True), info

    def test_usb_identity_automatically_loads_three_not_com_number(self):
        from machine_profile import get_runtime_machine_id
        context, info = self.usb_identity_context()
        page = self.window.local_controller.page_temporary_test
        page.latest_values = [999] * 45
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]):
            self.assertTrue(self.window._prepare_local_identity('COM99'))
        self.assertEqual(get_runtime_machine_id(), 'machine_3')
        self.assertEqual(self.window.machine_selector.currentData(), 'machine_3')
        self.assertEqual(self.window.local_controller.page_extra.fullband_points[0].codes,
                         (63351, 57089, 16283, 19171, 7806))
        self.assertIsNone(page.latest_values)
        self.assertEqual(page.storage_machine_id, 'machine_3')

    def test_usb_identity_does_not_guess_unknown_or_take_another_machine_table(self):
        context, info = self.usb_identity_context(serial='AABBCCDDEEFF')
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]), \
                mock.patch.object(self.window, '_bind_unknown_usb', return_value=None):
            self.assertFalse(self.window._prepare_local_identity('COM99'))
        self.assertEqual(self.window.machine_selector.currentData(), 'machine_1')
        self.assertIsNone(self.window._confirmed_usb_identity)

    def test_unknown_usb_first_confirmation_is_remembered_as_machine_one(self):
        context, info = self.usb_identity_context(serial='AABBCCDDEEFF')
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]), \
                mock.patch.object(QtWidgets.QInputDialog, 'getItem', return_value=('一号机', True)), \
                mock.patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Yes):
            self.assertTrue(self.window._prepare_local_identity('COM99'))
        self.assertEqual(self.window.machine_identity_store.resolve('AABBCCDDEEFF'), 'machine_1')
        self.assertTrue(self.window.machine_identity_store.path.is_file())

    def test_developer_budget_is_readonly_and_labels_temperature_as_estimate(self):
        page = self.window.local_controller.page_temporary_test
        with mock.patch.object(QtWidgets.QDialog, 'exec_') as execute:
            page.show_developer_budget()
        execute.assert_not_called()
        self.window.developer_mode_button.click()
        captured = []
        def inspect(dialog):
            captured.append(' '.join(label.text() for label in dialog.findChildren(QtWidgets.QLabel)))
            return QtWidgets.QDialog.Rejected
        with mock.patch.object(QtWidgets.QDialog, 'exec_', inspect), \
                mock.patch.object(self.window.usb_serial, 'write') as write:
            page.show_developer_budget()
        write.assert_not_called()
        self.assertIn('不是实时CPU负载', captured[0])
        self.assertIn('38.25 ms', captured[0])
        self.assertIn('暂无有效相邻板端帧', captured[0])

    def test_usb_identity_never_changes_calibration_during_acquisition(self):
        context, info = self.usb_identity_context()
        with context, mock.patch('app_JDSU.switch_mode_enable', False), \
                mock.patch('app_JDSU.list_ports.comports', return_value=[info]):
            self.assertFalse(self.window._prepare_local_identity('COM99'))
        self.assertEqual(self.window.machine_selector.currentData(), 'machine_1')

    def test_usb_identity_worker_guard_rejects_same_com_reused_by_different_board(self):
        context, info = self.usb_identity_context()
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]):
            self.assertTrue(self.window._prepare_local_identity('COM99'))
            self.window.usb_serial.port = 'COM99'
            self.window._guard_usb_identity(self.window.usb_serial)
            info.serial_number = '395736523033'
            with self.assertRaises(ValueError):
                self.window._guard_usb_identity(self.window.usb_serial)

    def test_identified_usb_cannot_manually_load_other_machine_data(self):
        context, info = self.usb_identity_context()
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]):
            self.assertTrue(self.window._prepare_local_identity('COM99'))
            self.window.machine_selector.setCurrentIndex(
                self.window.machine_selector.findData('machine_2'))
        self.assertEqual(self.window.machine_selector.currentData(), 'machine_3')

    def test_multiple_boards_require_explicit_usb_port_choice(self):
        context, info = self.usb_identity_context()
        other = SimpleNamespace(device='COM88', vid=0x0483, pid=0x5740,
                                serial_number='395736523033')
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info, other]):
            self.assertFalse(self.window._prepare_local_identity('COM99'))
            self.assertTrue(self.window._prepare_local_identity('COM88', explicit=True))
        self.assertEqual(self.window.machine_selector.currentData(), 'machine_2')

    def test_usb_autodetection_loads_data_before_acquiring_port(self):
        context, info = self.usb_identity_context()
        order = []
        reload_original = self.window.local_controller.reload_machine_calibration
        def reload(machine_id):
            order.append(('reload', machine_id))
            return reload_original(machine_id)
        def acquire():
            order.append(('acquire', self.window.machine_selector.currentData()))
            return True
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]), \
                mock.patch.object(self.window.local_controller, 'reload_machine_calibration', side_effect=reload), \
                mock.patch.object(self.window.local_controller, 'acquire_local_control', side_effect=acquire):
            self.window._poll_machine_identity()
        self.assertEqual(order, [('reload', 'machine_3'), ('acquire', 'machine_3')])

    def test_usb_autodetection_bad_calibration_never_acquires_port(self):
        context, info = self.usb_identity_context()
        with context, mock.patch('app_JDSU.list_ports.comports', return_value=[info]), \
                mock.patch.object(self.window.local_controller, 'reload_machine_calibration', return_value=['bad table']), \
                mock.patch.object(self.window.local_controller, 'acquire_local_control') as acquire:
            self.window._poll_machine_identity()
        acquire.assert_not_called()
        self.assertIsNone(self.window._confirmed_usb_identity)

    def test_usb_update_blocks_acquisition_reclaim(self):
        self.window.ota_page._usb_dialog = mock.Mock(busy=True)
        try:
            with mock.patch.object(self.window.local_controller, 'acquire_local_control') as acquire:
                self.window.switch_source(SOURCE_LOCAL)
            acquire.assert_not_called()
            self.assertEqual(self.window.current_source, NETWORK_LAN)
        finally:
            self.window.ota_page._usb_dialog = None

    def test_usb_update_blocks_machine_selector_and_reverts_visual_selection(self):
        self.window.ota_page._usb_dialog = mock.Mock(busy=True)
        try:
            self.window.machine_selector.setCurrentIndex(
                self.window.machine_selector.findData('machine_2'))
            self.app.processEvents()
            self.assertEqual(self.window.machine_selector.currentData(), 'machine_1')
            self.assertEqual(self.window.ota_page.current_machine_id, 'machine_1')
        finally:
            self.window.ota_page._usb_dialog = None

    def test_rare_modes_and_mqtt_are_hidden_but_retain_their_handlers(self):
        for index in (0, 1, 4, 5):
            self.assertTrue(self.window.mode_buttons[index].isHidden())
            self.assertIsNotNone(self.window.mode_buttons[index].parent())
        for index in (2, 3, 6):
            self.assertFalse(self.window.mode_buttons[index].isHidden())
        self.assertTrue(self.window.source_buttons[NETWORK_WAN].isHidden())
        self.assertFalse(self.window.source_buttons[NETWORK_LAN].isHidden())

    def test_simplified_navigation_and_device_note_stay_hidden_after_resize(self):
        for width in (1000, 1600):
            self.window.resize(width, 900)
            for page in (0, 3, 4):
                self.window.workspace_tabs.setCurrentIndex(page)
                self.app.processEvents()
                for index in (1, 2, 3, 4):
                    self.assertTrue(self.window.nav_buttons[index].isHidden())
                for index in (0,):
                    self.assertFalse(self.window.nav_buttons[index].isHidden())
                self.assertTrue(self.window.open_finger_button.isHidden())
                self.assertTrue(self.window.device_version.isHidden())
                self.assertTrue(self.window.device_name.isHidden())
                self.assertTrue(self.window.cnc_status_label.isHidden())
                self.assertTrue(self.window.cnc_release_button.isHidden())
                self.assertTrue(self.window.cnc_estop_button.isHidden())
                self.assertTrue(self.window.page_subtitle.isHidden())
                self.assertEqual(self.window.page_subtitle.text(), '')
                self.assertTrue(self.window.mode_buttons[5].isHidden())
        self.assertEqual(self.window.mode_buttons[6].text(), '手指应力测试模式')
        self.assertFalse(self.window.local_controller.menu_page.actions()[5].isVisible())
        self.assertEqual(
            self.window.local_controller.menu_page.actions()[6].text(),
            '手指应力测试模式',
        )

    def test_offline_assistant_policy_hides_update_pages_and_never_checks_github(self):
        for index in (3, 4):
            self.assertFalse(self.window.workspace_tabs.isTabVisible(index))
            self.window.workspace_tabs.setCurrentIndex(index)
            self.assertEqual(self.window.workspace_tabs.currentIndex(), 0)
        self.assertFalse(self.window._ota_version_timer.isActive())
        self.assertFalse(hasattr(self.window, '_release_timer'))
        with mock.patch('fbg_unified_app.ReleaseCheckWorker') as check:
            self.window.demo = False
            try:
                self.window._check_for_updates()
                check.assert_not_called()
                self.window._set_update_bubble(True, 'should never show')
                self.window._release_offers_ready(mock.Mock())
                self.assertTrue(self.window.update_bubble.isHidden())
                self.assertTrue(self.window.software_update_bubble.isHidden())
            finally:
                self.window.demo = True

    def test_removed_static_status_note_does_not_hide_actionable_messages(self):
        self.assertNotIn('同一时刻只启用', self.window.statusBar().currentMessage())
        self.window.statusBar().showMessage('关光尚未确认')
        self.assertEqual(self.window.statusBar().currentMessage(), '关光尚未确认')
        self.assertFalse(self.window.statusBar().isHidden())

    def test_sidebar_thermal_readings_start_unknown_and_stay_visible(self):
        self.assertEqual(self.window.board_temperature_label.text(), '板温：— °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：— RPM')
        for width in (1000, 1600):
            self.window.resize(width, 900)
            self.app.processEvents()
            self.assertFalse(self.window.board_temperature_label.isHidden())
            self.assertFalse(self.window.fan_speed_label.isHidden())

    def test_sidebar_thermal_uses_real_temperature_and_rpm_without_io(self):
        import app_JDSU
        with mock.patch.object(app_JDSU, 'serial_write') as write:
            self.window._display_thermal({
                'temperature_c': 33.42, 'fan_rpm': 1836,
                'fan_duty_permille': 1000, 'thermal_flags': 0x11})
        write.assert_not_called()
        self.assertEqual(self.window.board_temperature_label.text(), '板温：33.4 °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：1836 RPM')

    def test_sidebar_thermal_sensor_and_tach_faults_do_not_show_fake_values(self):
        self.window._display_thermal({
            'temperature_c': 0, 'fan_rpm': 0, 'thermal_flags': 0x0c})
        self.assertEqual(self.window.board_temperature_label.text(), '板温：— °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：反馈异常')
        self.window._display_thermal({
            'temperature_c': 33, 'fan_rpm': 0, 'thermal_flags': 0})
        self.assertEqual(self.window.board_temperature_label.text(), '板温：— °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：— RPM')

    def test_sidebar_thermal_expires_and_rejects_old_or_out_of_order_frames(self):
        stamp = time.monotonic()
        values = {'temperature_c': 33, 'fan_rpm': 900, 'thermal_flags': 0x11,
                  'received_monotonic': stamp}
        self.window._display_thermal(values)
        self.window._display_thermal(dict(values, temperature_c=40,
                                          received_monotonic=stamp - 1))
        self.assertEqual(self.window.board_temperature_label.text(), '板温：33.0 °C')
        with mock.patch('fbg_unified_app.time.monotonic', return_value=stamp + 11):
            self.window._expire_thermal_display()
        self.assertEqual(self.window.board_temperature_label.text(), '板温：— °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：— RPM')
        self.window._display_thermal(values)
        self.assertEqual(self.window.board_temperature_label.text(), '板温：— °C')

    def test_sidebar_thermal_clears_when_changing_machine_or_mode(self):
        values = {'temperature_c': 33, 'fan_rpm': 900, 'thermal_flags': 0x11}
        self.window._display_thermal(values)
        self.window.machine_selector.setCurrentIndex(
            self.window.machine_selector.findData('machine_2'))
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：— RPM')
        self.window._display_thermal(values)
        self.window.current_source = SOURCE_LOCAL
        self.window.local_controller.MCU_mode = 6
        self.window._sync_local_mode()
        self.assertEqual(self.window.board_temperature_label.text(), '板温：— °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：— RPM')

    def test_sidebar_thermal_is_connected_to_native_scan_status(self):
        self.window.current_source = SOURCE_LOCAL
        page = self.window.local_controller.page_peak
        self.window.local_controller.MCU_mode = 0
        self.window._sync_local_mode()
        page.update_temp(34.25)
        page._update_board_status(0, 1, 0, 0, 0, 1520, 320, 0x11)
        self.app.processEvents()
        self.assertEqual(self.window.board_temperature_label.text(), '板温：34.2 °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：1520 RPM')

    def test_single_value_temperature_does_not_invent_fan_rpm(self):
        self.window.current_source = SOURCE_LOCAL
        self.window.local_controller.MCU_mode = 3
        self.window._sync_local_mode()
        self.window.local_controller.page_extra.temp_signal.emit(33.4)
        self.app.processEvents()
        self.assertEqual(self.window.board_temperature_label.text(), '板温：33.4 °C')
        self.assertEqual(self.window.fan_speed_label.text(), '风扇：— RPM')

    def test_machine_one_identity_and_parameter_owner_are_visible(self):
        self.assertIn("一号机", self.window.windowTitle())
        self.assertIn("一号机", self.window.device_name.text())
        self.assertIn("9峰", self.window.device_version.text())
        self.assertIn("2001点", self.window.device_version.text())
        self.assertIn("3峰", self.window.device_version.text())

    def test_left_sidebar_machine_selector_loads_machine_two_calibration(self):
        index = self.window.machine_selector.findData("machine_2")
        self.assertGreaterEqual(index, 0)
        self.window.machine_selector.setCurrentIndex(index)
        self.app.processEvents()
        self.assertEqual(self.window.ota_page.current_machine_id, "machine_2")
        self.assertIn("二号机", self.window.device_name.text())
        self.assertIn("2号解调仪第一次校准", self.window.device_version.text())
        self.assertIn("1716/2001", self.window.device_version.text())
        points = self.window.local_controller.page_unlimited_accuracy.points
        self.assertEqual(len(points), 2001)
        self.assertEqual(points[0].codes, (63351, 52414, 15887, 11423, 4811))
        self.assertEqual(
            self.window.local_controller.page_equal_interval.points[0].codes,
            points[0].codes,
        )
        self.assertEqual(
            self.window.local_controller.page_extra.fullband_points[0].codes,
            points[0].codes,
        )

    def test_left_sidebar_machine_selector_loads_machine_three_provisional_table(self):
        index = self.window.machine_selector.findData("machine_3")
        self.assertGreaterEqual(index, 0)
        self.window.machine_selector.setCurrentIndex(index)
        self.app.processEvents()
        self.assertEqual(self.window.ota_page.current_machine_id, "machine_3")
        self.assertIn("三号机", self.window.device_name.text())
        self.assertIn("临时表", self.window.device_version.text())
        points = self.window.local_controller.page_unlimited_accuracy.points
        self.assertEqual(len(points), 2001)
        self.assertEqual(points[0].codes, (63351, 57089, 16283, 19171, 7806))
        self.assertEqual(
            self.window.local_controller.page_equal_interval.points[0].codes,
            points[0].codes,
        )
        self.assertEqual(
            self.window.local_controller.page_extra.fullband_points[0].codes,
            points[0].codes,
        )
        self.assertIsNone(self.window.local_controller.page_temporary_test.plan)

    def test_real_lan_target_is_pinned_to_selected_machine_not_discovery(self):
        import app_JDSU

        previous_serial = app_JDSU.ser
        self.window.ota_settings_store.save_profile(
            "machine_3", board_ip="192.168.4.62", password="",
            package_path="",
        )
        self.window.ota_page.set_machine("machine_3")
        self.window.demo = False
        try:
            with mock.patch.dict(os.environ, {"FBG_LAN_BOARD_IP": "192.168.4.23"}):
                self.window._select_local_transport(NETWORK_LAN)
            self.assertEqual(self.window.lan_serial.board_host, "192.168.4.62")
            self.assertIs(app_JDSU.ser, self.window.lan_serial)
            self.window._refresh_machine_identity()
            self.assertIn("局域网目标：192.168.4.62", self.window.device_version.text())
        finally:
            self.window.demo = True
            app_JDSU.ser = previous_serial

    def test_real_lan_refuses_machine_without_saved_board_ip(self):
        import app_JDSU

        previous_serial = app_JDSU.ser
        self.window.ota_page.set_machine("machine_3")
        self.window.demo = False
        try:
            with self.assertRaisesRegex(ValueError, "板卡 IP"):
                self.window._select_local_transport(NETWORK_LAN)
            self.assertIs(app_JDSU.ser, previous_serial)
        finally:
            self.window.demo = True

    def test_ota_target_cannot_be_overwritten_by_unrelated_lan_peer(self):
        self.window.ota_settings_store.save_profile(
            "machine_3", board_ip="192.168.4.62", password="",
            package_path="",
        )
        self.window.ota_page.set_machine("machine_3")
        self.window.lan_serial._board_ip = "192.168.4.23"
        self.assertTrue(self.window._prepare_ota())
        self.assertEqual(
            self.window.ota_settings_store.lan_board_ip("machine_3"),
            "192.168.4.62",
        )

    def test_switching_machine_releases_old_lan_before_loading_new_target(self):
        self.window.ota_settings_store.save_profile(
            "machine_3", board_ip="192.168.4.62", password="",
            package_path="",
        )
        self.window.demo = False
        self.window.current_source = NETWORK_LAN
        try:
            with mock.patch.object(self.window, "_release_local_control") as release, \
                 mock.patch.object(
                     self.window.local_controller, "reload_machine_calibration",
                     return_value=[],
                 ) as reload_table, \
                 mock.patch.object(
                     self.window.local_controller, "acquire_local_control",
                     return_value=True,
                 ) as acquire:
                index = self.window.machine_selector.findData("machine_3")
                self.window.machine_selector.setCurrentIndex(index)
            release.assert_called_once_with()
            reload_table.assert_called_once_with("machine_3")
            acquire.assert_called_once_with()
            self.assertEqual(self.window.lan_serial.board_host, "192.168.4.62")
        finally:
            self.window.demo = True

    def test_machine_switch_with_bad_calibration_does_not_reconnect_lan(self):
        self.window.ota_settings_store.save_profile(
            "machine_3", board_ip="192.168.4.62", password="",
            package_path="",
        )
        self.window.demo = False
        self.window.current_source = NETWORK_LAN
        try:
            with mock.patch.object(self.window, "_release_local_control") as release, \
                 mock.patch.object(
                     self.window.local_controller, "reload_machine_calibration",
                     return_value=["点表错误"],
                 ), \
                 mock.patch.object(
                     self.window.local_controller, "acquire_local_control",
                     return_value=True,
                 ) as acquire:
                self.window.machine_selector.setCurrentIndex(
                    self.window.machine_selector.findData("machine_3")
                )
            release.assert_called_once_with()
            acquire.assert_not_called()
            self.assertIsNone(self.window.current_source)
        finally:
            self.window.demo = True

    def test_selected_screen_uses_framed_maximized_window(self):
        screen = QtWidgets.QApplication.primaryScreen()
        self.assertIsNotNone(screen)
        self.assertEqual(
            self.window.windowFlags() & QtCore.Qt.FramelessWindowHint,
            QtCore.Qt.WindowFlags(),
        )
        label = self.window._display_label(screen, 0, primary=True, current=True)
        self.assertIn("屏幕 1", label)
        self.assertIn("主屏", label)
        self.assertIn("当前", label)
        with mock.patch.object(self.window, "showNormal") as normal, \
             mock.patch.object(self.window, "showMaximized") as maximized:
            self.window._maximize_on_screen(screen)
        normal.assert_called_once_with()
        maximized.assert_called_once_with()
        self.assertIn("标题栏", self.window.statusBar().currentMessage())

    def test_initial_window_fits_high_dpi_work_area(self):
        screen=mock.Mock()
        available=QtCore.QRect(-1463,-823,1463,823)
        screen.availableGeometry.return_value=available
        self.window._fit_initial_window(screen)
        self.assertLessEqual(self.window.width(),available.width()-24)
        self.assertLessEqual(self.window.height(),available.height()-48)
        self.assertGreaterEqual(self.window.x(),available.x())
        self.assertGreaterEqual(self.window.y(),available.y())
        self.assertLessEqual(self.window.x()+self.window.width(),available.right()+1)
        self.assertLessEqual(self.window.y()+self.window.height()+32,available.bottom()+1)

    def test_fast_flank_button_is_integrated_and_demo_cannot_drive_laser(self):
        finger = self.window.finger_controller
        self.assertGreater(
            finger.receivers(finger.fast_flank_start_requested),
            0,
        )
        self.assertGreater(
            finger.receivers(finger.fast_flank_stop_requested),
            0,
        )
        finger.fast_flank_start_requested.emit()
        self.app.processEvents()
        self.assertIn("演示模式", finger.fast_flank_status_label.text())
        self.assertIsNone(self.window._fast_flank_worker)

    def test_fast_flank_3d_render_coalesces_to_latest_result(self):
        first = object()
        latest = object()
        self.window._fast_flank_worker = object()
        self.window._fast_flank_ui_timer.stop()
        try:
            with mock.patch.object(
                self.window.finger_controller, "update_from_fast_flank_result"
            ) as render:
                self.window._fast_flank_result_ready(first)
                self.window._fast_flank_result_ready(latest)
                render.assert_not_called()
                self.assertEqual(self.window._fast_flank_results_received, 2)
                self.assertEqual(self.window._fast_flank_render_overwrites, 1)
                self.window._render_latest_fast_flank_result()
                render.assert_called_once_with(latest)
                self.assertEqual(self.window._fast_flank_results_rendered, 1)
                self.assertIsNone(self.window._latest_fast_flank_result)
        finally:
            self.window._fast_flank_worker = None

    def test_hidden_mach3_emergency_control_retains_global_handler(self):
        self.assertTrue(self.window.cnc_estop_button.isHidden())
        self.assertEqual(self.window.cnc_estop_button.property("role"), "emergency")
        self.assertGreater(
            self.window.cnc_estop_button.receivers(
                self.window.cnc_estop_button.clicked
            ),
            0,
        )

    def test_recovery_button_is_enabled_only_for_latched_stop_state(self):
        self.window._mach3_state_changed("ready", "Mach3 在线 · 轴已停止")
        self.assertFalse(self.window.cnc_release_button.isEnabled())
        self.window._mach3_state_changed("stopped", "Mach3 在线 · 急停已锁定")
        self.assertTrue(self.window.cnc_release_button.isEnabled())
        self.window._mach3_state_changed("busy", "正在恢复")
        self.assertFalse(self.window.cnc_release_button.isEnabled())

    def test_recovery_confirmation_accepts_mach3_estop_moving_quirk(self):
        self.window._last_mach3_status = {
            "stopped": True,
            "moving": True,
            "estop": True,
            "estop_led": True,
        }
        with (
            mock.patch.object(
                QtWidgets.QMessageBox,
                "warning",
                return_value=QtWidgets.QMessageBox.Yes,
            ),
            mock.patch.object(
                self.window.mach3_emergency, "release_control"
            ) as release,
        ):
            self.window._confirm_release_estop()
        release.assert_called_once_with()

    def test_recovery_cancel_never_calls_machine(self):
        self.window._last_mach3_status = {
            "stopped": True,
            "moving": False,
            "estop": True,
            "estop_led": True,
        }
        with (
            mock.patch.object(
                QtWidgets.QMessageBox,
                "warning",
                return_value=QtWidgets.QMessageBox.Cancel,
            ),
            mock.patch.object(
                self.window.mach3_emergency, "release_control"
            ) as release,
        ):
            self.window._confirm_release_estop()
        release.assert_not_called()

    def test_cnc_position_is_shown_and_snapshotted_with_adc_frame(self):
        status = {
            "work_x": -9.79375,
            "work_y": 145.565625,
            "work_z": 54.2765625,
            "machine_x": 12.20625,
            "machine_y": 35.565625,
            "machine_z": -14.7234375,
            "moving": False,
            "stopped": True,
            "estop": False,
            "estop_led": False,
        }
        self.window._mach3_status_received(status)
        self.window._mach3_state_changed("ready", "Mach3 在线 · 轴已停止")
        peaks = [[1525.0 + index for index in range(9)] for _ in range(4)]
        self.window.finger_controller.update_from_spectrum(peaks, 42)

        self.assertIn("X -9.794", self.window.cnc_status_label.text())
        self.assertIn(
            "第42帧最近位置", self.window.finger_controller.machine_frame_label.text()
        )
        self.assertEqual(
            self.window.finger_controller.frame_machine_status["work_y"],
            status["work_y"],
        )
        self.assertTrue(self.window.finger_controller.press_tool_loaded)

    def test_out_of_order_machine_status_is_ignored(self):
        newest = {
            "_telemetry_seq": 12,
            "work_x": -5.0,
            "work_y": 12.0,
            "work_z": -22.0,
            "machine_x": -14.8828125,
            "machine_y": -42.2359375,
            "machine_z": -60.759375,
        }
        stale = dict(newest, _telemetry_seq=11, work_x=-9.0)
        self.window._mach3_status_received(newest)
        self.window._mach3_status_received(stale)
        self.assertEqual(self.window._last_mach3_status["work_x"], -5.0)
        self.assertEqual(self.window._last_mach3_sequence, 12)

    def test_press_tool_remains_drawn_when_mach3_position_is_unavailable(self):
        finger = self.window.finger_controller
        finger.set_machine_status_error("测试：坐标暂不可用")
        self.app.processEvents()

        self.assertTrue(finger.press_tool_loaded)
        self.assertEqual(finger.press_tool_pose_kind, "calibration_preview")
        self.assertTrue(finger.press_tool_item.visible())
        self.assertIn("硅胶下边界基准", finger.machine_mapping_label.text())
        self.assertIn("非实时位置", finger.machine_mapping_label.text())

    def test_mach3_status_failure_rebuilds_bridge_and_recovers(self):
        class RecoveringController:
            def __init__(self):
                self.status_calls = 0
                self.close_calls = 0

            def status(self):
                self.status_calls += 1
                if self.status_calls == 1:
                    raise RuntimeError("COM 尚未注册")
                return {
                    "work_x": -6.446875,
                    "work_y": 13.45,
                    "work_z": -22.1375,
                    "moving": False,
                    "estop": False,
                    "estop_led": False,
                }

            def close(self):
                self.close_calls += 1

        controller = RecoveringController()
        service = Mach3EmergencyService(controller=controller)
        states = QtTest.QSignalSpy(service.state_changed)
        statuses = QtTest.QSignalSpy(service.status_received)

        service.query_status()
        if not states:
            self.assertTrue(states.wait(1000))
        self.assertEqual(states[-1][0], "error")
        self.assertEqual(controller.close_calls, 1)

        service.query_status()
        if not statuses:
            self.assertTrue(statuses.wait(1000))
        self.assertEqual(statuses[-1][0]["work_y"], 13.45)
        service.close()

    def test_mach3_telemetry_continues_while_press_command_is_busy(self):
        motion_controller = mock.Mock()
        status_controller = mock.Mock()
        status_controller.status.return_value = {
            "work_x": -6.0,
            "work_y": 12.0,
            "work_z": -23.0,
            "moving": True,
            "estop": False,
            "estop_led": False,
        }
        service = Mach3EmergencyService(
            controller=motion_controller,
            status_controller=status_controller,
        )
        service._busy = True
        statuses = QtTest.QSignalSpy(service.status_received)

        service.query_status()
        if not statuses:
            self.assertTrue(statuses.wait(1000))

        self.assertEqual(statuses[-1][0]["work_z"], -23.0)
        status_controller.status.assert_called_once_with()
        motion_controller.status.assert_not_called()
        service.close()

    def test_release_control_uses_confirmed_bridge_operation(self):
        motion_controller = mock.Mock()
        status_controller = mock.Mock()
        motion_controller.release_emergency_stop.return_value = {
            "spindle_command": "M5",
            "after": {
                "stopped": True,
                "moving": False,
                "estop": False,
                "estop_led": False,
            },
        }
        service = Mach3EmergencyService(
            controller=motion_controller,
            status_controller=status_controller,
        )
        states = QtTest.QSignalSpy(service.state_changed)

        service.release_control()
        while not any(entry[0] == "ready" for entry in states):
            self.assertTrue(states.wait(1000))

        motion_controller.release_emergency_stop.assert_called_once_with(
            operator_confirmed=True
        )
        self.assertIn("主轴保持关闭", states[-1][1])
        service.close()

    def test_click_press_uses_coordinated_xy_and_continuous_z(self):
        class MotionController:
            def __init__(self):
                self.current = {
                    "work_x": -6.446875,
                    "work_y": 13.45,
                    "work_z": -22.1375,
                    "moving": False,
                    "stopped": True,
                    "estop": False,
                    "estop_led": False,
                    "x_homed": True,
                    "y_homed": True,
                    "z_homed": True,
                }
                self.xy_calls = []
                self.axis_calls = []

            def status(self):
                return dict(self.current)

            def move_absolute_xy_continuous(
                self, target_x, target_y, *, feed_mm_min, safe_zone_confirmed
            ):
                self.xy_calls.append(
                    (target_x, target_y, feed_mm_min, safe_zone_confirmed)
                )
                before = dict(self.current)
                self.current["work_x"] = target_x
                self.current["work_y"] = target_y
                return {"before": before, "after": dict(self.current)}

            def move_relative_continuous_axis(
                self, axis, delta, *, feed_mm_min, safe_zone_confirmed
            ):
                self.axis_calls.append((axis, delta, feed_mm_min, safe_zone_confirmed))
                before = dict(self.current)
                self.current[f"work_{axis.lower()}"] += delta
                return {"before": before, "after": dict(self.current)}

            def close(self):
                return None

        controller = MotionController()
        service = Mach3EmergencyService(
            controller=controller, status_controller=controller
        )
        finished = QtTest.QSignalSpy(service.press_finished)
        contact = QtTest.QSignalSpy(service.contact_reached)
        release_started = QtTest.QSignalSpy(service.release_started)
        release_reached = QtTest.QSignalSpy(service.release_reached)
        target = {
            "work_x": -5.0,
            "work_y": 12.0,
            "contact_z": -26.1359375,
            "depth_mm": 0.2,
            "xy_feed_mm_min": 80.0,
            "z_travel_feed_mm_min": 60.0,
            "press_feed_mm_min": 15.0,
        }

        with mock.patch("fbg_unified_app.time.sleep", return_value=None):
            service.press_target(target)
            if not finished:
                self.assertTrue(finished.wait(1000))

        self.assertTrue(finished[-1][0])
        self.assertEqual(controller.xy_calls, [(-5.0, 12.0, 80.0, True)])
        self.assertEqual(len(controller.axis_calls), 3)
        self.assertEqual(controller.axis_calls[0][0], "Z")
        self.assertAlmostEqual(controller.axis_calls[0][2], 60.0)
        self.assertEqual(controller.axis_calls[1][0], "Z")
        self.assertAlmostEqual(controller.axis_calls[1][2], 15.0)
        self.assertEqual(controller.axis_calls[2][0], "Z")
        self.assertAlmostEqual(controller.axis_calls[2][2], 60.0)
        self.assertAlmostEqual(controller.current["work_z"], -22.1359375)
        self.assertEqual(len(contact), 1)
        self.assertEqual(len(release_started), 1)
        self.assertEqual(len(release_reached), 1)
        service.close()

    def test_local_snapshot_keeps_exact_ch1_codes_and_runtime_identity(self):
        page = self.window.local_controller.page_peak
        page.wave_const = [1528.40, 1528.42]
        page.data = [[1, 2], [101, 202], [3, 4], [5, 6]]
        page.adc = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        page.ori_filts = [[0.0, 0.0] for _ in range(4)]
        page.frame_generation = 73
        page.feedback_selectors = [0, 3]
        page.feedback_gain_masks = [0x11, 0x24]
        page.voltage_scalars = [1.0, 8.0, 1.0, 1.0]
        page.board_frame_sequence = 72
        page.board_boot_id = 0x10203040
        page.board_uptime_ms = 12_345
        page.board_table_crc32 = 0xA1B2C3D4
        self.window.current_source = SOURCE_LOCAL

        snapshot = self.window._active_spectrum_snapshot()

        self.assertEqual(snapshot["raw_ch1_adc_codes"], [101, 202])
        self.assertEqual(snapshot["frame_key"], ("local", 0x10203040, 72))
        self.assertEqual(snapshot["ch1_transimpedance_ohm"], 20_000.0)
        self.assertEqual(snapshot["ch1_feedback_selector"], 3)
        self.assertEqual(snapshot["ch1_analogue_gain_mask"], 0x24)
        self.assertEqual(snapshot["ch1_digital_gain_audit"], 8.0)
        self.assertEqual(snapshot["device_sequence"], 72)
        self.assertEqual(snapshot["device_boot_id"], 0x10203040)
        self.assertEqual(snapshot["device_uptime_ms"], 12_345)
        self.assertEqual(snapshot["table_crc32"], 0xA1B2C3D4)
        self.assertIsNone(snapshot["frame_received_monotonic_ns"])
        self.assertGreater(snapshot["captured_monotonic_ns"], 0)

    def test_remote_snapshot_uses_crc_validated_telemetry_identity_and_raw_ch1(self):
        frame = TelemetryFrame(
            mode=MODE_STRESS,
            flags=0,
            boot_id=0x10203040,
            seq=89,
            uptime_ms=12_345,
            temperature_mC=25_000,
            table_crc32=0xA1B2C3D4,
            gainmask0=0x11,
            gainmask1=0x24,
            channel_mask=0x03,
            samples=((11, 101), (22, 202)),
        )
        self.window.current_source = NETWORK_WAN
        self.window.remote_controller.waves = [1528.40, 1528.42]
        self.window.remote_controller.latest_raw_channels = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        self.window.remote_controller.latest_analysis_channels = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        self.window.remote_controller.last_boot_id = frame.boot_id
        self.window.remote_controller.latest_frame_number = frame.seq
        self.window._remember_remote_telemetry(frame)

        snapshot = self.window._active_spectrum_snapshot()

        self.assertEqual(snapshot["raw_ch1_adc_codes"], [101, 202])
        self.assertEqual(snapshot["device_sequence"], 89)
        self.assertEqual(snapshot["device_boot_id"], 0x10203040)
        self.assertEqual(snapshot["device_uptime_ms"], 12_345)
        self.assertEqual(snapshot["table_crc32"], 0xA1B2C3D4)
        self.assertTrue(snapshot["packet_crc_ok"])
        self.assertEqual(snapshot["ch1_analogue_gain_mask"], 0x24)
        self.assertIsNone(snapshot["ch1_transimpedance_ohm"])
        self.assertIsNone(snapshot["ch1_digital_gain_audit"])
        self.assertGreater(snapshot["frame_received_monotonic_ns"], 0)

    def test_local_receipt_survives_plot_and_never_becomes_render_time(self):
        import app_JDSU as local
        from test_stress_multirate import _extension, _native_frame
        from frame_telemetry import MULTIRATE_PROFILE_MAP

        page = self.window.local_controller.page_peak
        self.window.current_source = SOURCE_LOCAL
        page._reset_realtime_frame_stats(clear_buffer=True)
        native = _native_frame(_extension(MULTIRATE_PROFILE_MAP, tuple(range(45))))
        received = time.monotonic_ns() - 2_000_000_000
        local.frames_queue.put(native, received_ns=received)
        previous_generation = page.frame_generation
        page.process_frame()
        self.assertEqual(page.frame_generation, previous_generation + 1)
        self.assertEqual(page.last_frame_received_monotonic_ns, received)
        page.update_plot()
        self.assertIsNone(page.pending_frame_received_ns)
        snapshot = self.window._active_spectrum_snapshot()
        self.assertEqual(snapshot["frame_received_monotonic_ns"], received)
        self.assertGreaterEqual(snapshot["captured_monotonic_ns"] - received, 2_000_000_000)
        # Malformed input partially replaces buffers; it cannot inherit the
        # previous good frame's freshness or advance the published generation.
        local.frames_queue.put(b"bad", received_ns=time.monotonic_ns())
        page.process_frame()
        self.assertIsNone(page.last_frame_received_monotonic_ns)
        self.assertEqual(page.frame_generation, previous_generation + 1)
        page.last_frame_received_monotonic_ns = received
        page._reset_realtime_frame_stats(clear_buffer=True)
        self.assertIsNone(page.last_frame_received_monotonic_ns)

    def test_coverage_records_multiple_new_frames_for_every_contact_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "coverage"
            writer = FingerSessionWriter(session, queue_capacity=64)
            target = {
                "point_index": 5,
                "coverage_repetition": 2,
                "model_x": 0.5,
                "model_y": -0.25,
                "work_x": -6.0,
                "work_y": 13.0,
                "contact_z": -26.0,
                "depth_mm": 0.0,
            }
            frame_counter = 0

            def snapshot():
                nonlocal frame_counter
                frame_counter += 1
                return {
                    "source": SOURCE_LOCAL,
                    "frame_number": frame_counter,
                    "frame_key": ("local", frame_counter),
                    "captured_monotonic_ns": time.monotonic_ns(),
                    "frame_received_monotonic_ns": None,
                    "device_sequence": None,
                    "device_boot_id": None,
                    "device_uptime_ms": None,
                    "device_id": None,
                    "table_crc32": None,
                    "packet_crc_ok": None,
                    "wavelengths_nm": [1528.40, 1528.42],
                    "raw_ch1_adc_codes": [100 + frame_counter, 200 + frame_counter],
                    "ch1_transimpedance_ohm": 20_000.0,
                    "ch1_feedback_selector": 3,
                    "ch1_analogue_gain_mask": 0x24,
                    "ch1_digital_gain_audit": 16.0,
                    "quality_flags": [],
                    "peak_centers_nm": [[], []],
                }

            self.window._coverage_writer = writer
            self.window._coverage_active = True
            self.window._coverage_motion_test = False
            self.window._coverage_current = target
            self.window._coverage_total = 1
            self.window._coverage_completed = 0
            self.window._coverage_records = 0
            self.window._coverage_last_frame_key = None
            self.window._coverage_phase_frame_counts = {}
            self.window.current_source = SOURCE_LOCAL
            try:
                with mock.patch.object(
                    self.window, "_active_spectrum_snapshot", side_effect=snapshot
                ):
                    for phase in ("pre_contact", "contact", "hold", "release"):
                        self.window._begin_coverage_phase(phase, target, {})
                        self.window._capture_coverage_phase_frame()
                        self.window._capture_coverage_phase_frame()
                    self.window._finger_release_reached(target, {})
                writer.close()
            finally:
                self.window._coverage_writer = None
                self.window._coverage_active = False

            frames = [
                json.loads(line)
                for line in (session / "frames.jsonl").read_text("utf-8").splitlines()
            ]
            events = [
                json.loads(line)
                for line in (session / "events.jsonl").read_text("utf-8").splitlines()
            ]
            phases = [frame["quality"]["coverage_phase"] for frame in frames]
            self.assertEqual(len(frames), 13)
            self.assertEqual(
                {phase: phases.count(phase) for phase in set(phases)},
                {"pre_contact": 3, "contact": 3, "hold": 3, "release": 3, "release_complete": 1},
            )
            self.assertEqual(
                [event["kind"] for event in events],
                ["pre_contact", "contact", "hold", "release", "release_complete"],
            )
            self.assertEqual(frames[0]["raw_adc_codes"]["1"], [101, 201])
            self.assertEqual(frames[0]["digital_gain_audit"]["1"], 16.0)
            self.assertIsNone(frames[0]["device_sequence"])
            self.assertIsNone(frames[0]["table_crc32"])
            groups = {event["metadata"]["press_event_id"] for event in events}
            self.assertEqual(len(groups), 1)
            self.assertEqual({frame["quality"]["press_event_id"] for frame in frames}, groups)
            self.assertEqual([e["metadata"]["contact"] for e in events],
                             [None, True, True, None, False])
            self.assertIsNone(events[3]["indentation_mm"])
            self.assertFalse(events[2]["metadata"]["adc_event_synchronization_verified"])
            self.assertFalse(frames[3]["quality"]["adc_event_synchronization_verified"])

    def test_native_coverage_capture_survives_ui_decimation_and_closes(self):
        import app_JDSU as local
        for source in (SOURCE_LOCAL, NETWORK_LAN):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                self.window.current_source = source
                self.window._coverage_writer = FingerSessionWriter(directory)
                self.window._coverage_active = True
                with mock.patch.object(self.window, "_uses_local_pages", return_value=True):
                    self.window._start_coverage_native_capture()
                for n in range(40):
                    local.frames_queue.put(n.to_bytes(2, "big"))
                local.frames_queue.take(latest_only=True)
                self.window._finish_coverage_run(True, "offline capture")
                self.assertIsNone(self.window._coverage_native_recorder)
                self.assertEqual(self.window._coverage_native_stats["written"], 40)
                self.assertEqual(self.window._coverage_native_stats["dropped"], 0)
                records = [json.loads(line) for line in (Path(directory)/"native_frames.jsonl").read_text("utf-8").splitlines()]
                self.assertEqual([bytes.fromhex(r["payload_hex"]) for r in records], [n.to_bytes(2, "big") for n in range(40)])
                events = [json.loads(line) for line in (Path(directory)/"events.jsonl").read_text("utf-8").splitlines()]
                self.assertEqual(events[-1]["metadata"]["native_recording"]["written"], 40)

    def test_native_close_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            self.window._coverage_writer = FingerSessionWriter(directory)
            self.window._coverage_native_recorder = mock.Mock()
            self.window._coverage_native_recorder.close.side_effect = RuntimeError("disk full")
            self.window._coverage_native_recorder.stats.return_value = {"written": 2, "offered": 3, "dropped": 0, "error": "disk full"}
            self.window._finish_coverage_run(True, "test")
            self.assertIsNone(self.window._coverage_native_recorder)
            events = [json.loads(line) for line in (Path(directory)/"events.jsonl").read_text("utf-8").splitlines()]
            self.assertFalse(events[-1]["metadata"]["success"])
            self.assertIn("disk full", self.window.finger_controller.coverage_summary_label.text())

    def test_demo_coverage_plan_never_controls_machine(self):
        finger = self.window.finger_controller
        self.assertGreater(len(finger.coverage_plan), 0)
        finger._request_coverage_run()
        self.app.processEvents()
        self.assertIn("不会控制机床", finger.coverage_summary_label.text())
        self.assertFalse(self.window._coverage_active)

    def test_motion_test_payload_is_one_round_and_never_requests_spectrum(self):
        finger = self.window.finger_controller
        requests = QtTest.QSignalSpy(finger.coverage_run_requested)

        finger._request_coverage_motion_test()
        self.app.processEvents()

        self.assertEqual(len(requests), 1)
        payload = requests[0][0]
        self.assertTrue(payload["motion_test"])
        self.assertEqual(payload["repeats"], 1)
        self.assertEqual(len(payload["points"]), len(finger.coverage_plan))
        self.assertEqual(payload["depth_mm"], 0.0)
        self.assertAlmostEqual(payload["dwell_seconds"], 0.3)
        self.assertIn("不会控制机床", finger.coverage_summary_label.text())

    def test_real_coverage_payload_uses_guarded_depth_control(self):
        finger = self.window.finger_controller
        finger.press_depth_spin.setValue(0.35)
        payload = finger._coverage_request_payload(
            repeats=2, dwell_seconds=1.5, motion_test=False
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["depth_mm"], 0.35)
        self.assertEqual(payload["repeats"], 2)
        self.assertIn("压入量0.35 mm", finger.coverage_summary_label.text())

    def test_coverage_gate_can_synchronize_nine_peaks_from_live_raw_ch1(self):
        finger = self.window.finger_controller
        finger.frame_number = 0
        for channel in finger.current_peaks_by_channel:
            channel[:] = float("nan")
        finger.current_sensor_peaks[:] = float("nan")
        waves = [1525.0 + 0.02 * index for index in range(45)]
        codes = [value for _ in range(9) for value in (1, 4, 30, 5, 2)]
        snapshot = {
            "raw_ch1_adc_codes": codes,
            "wavelengths_nm": waves,
            "frame_number": 27,
        }

        with mock.patch.object(
            self.window, "_active_spectrum_snapshot", return_value=snapshot
        ):
            self.assertTrue(self.window._ensure_finger_spectrum_ready_for_coverage())

        self.assertEqual(finger.selected_channel, 1)
        self.assertEqual(finger.frame_number, 27)
        self.assertEqual(finger.current_sensor_peaks.tolist(), waves[2::5])

    def test_motion_test_contact_does_not_schedule_spectrum_capture(self):
        self.window._coverage_active = True
        self.window._coverage_motion_test = True
        self.window._coverage_total = 31
        self.window._coverage_current = {
            "point_index": 1,
            "coverage_repetition": 1,
        }
        with mock.patch.object(QtCore.QTimer, "singleShot") as schedule:
            self.window._finger_contact_reached(dict(self.window._coverage_current), {})
        schedule.assert_not_called()
        self.assertIn(
            "不采集光谱",
            self.window.finger_controller.coverage_summary_label.text(),
        )

    def test_click_press_remaps_changed_work_offset_via_machine_coordinates(self):
        class OffsetController:
            def __init__(self):
                self.current = {
                    "work_x": -24.0,
                    "work_y": -48.0,
                    "work_z": -11.0,
                    "machine_x": -17.0,
                    "machine_y": -37.0,
                    "machine_z": -8.0,
                    "moving": False,
                    "stopped": True,
                    "estop": False,
                    "estop_led": False,
                    "x_homed": True,
                    "y_homed": True,
                    "z_homed": True,
                }
                self.xy_calls = []
                self.axis_calls = []

            def status(self):
                return dict(self.current)

            def move_absolute_xy_continuous(
                self, target_x, target_y, *, feed_mm_min, safe_zone_confirmed
            ):
                self.xy_calls.append(
                    (target_x, target_y, feed_mm_min, safe_zone_confirmed)
                )
                before = dict(self.current)
                delta_x = target_x - self.current["work_x"]
                delta_y = target_y - self.current["work_y"]
                self.current["work_x"] = target_x
                self.current["work_y"] = target_y
                self.current["machine_x"] += delta_x
                self.current["machine_y"] += delta_y
                return {"before": before, "after": dict(self.current)}

            def move_relative_continuous_axis(
                self, axis, delta, *, feed_mm_min, safe_zone_confirmed
            ):
                self.axis_calls.append(
                    (axis, delta, feed_mm_min, safe_zone_confirmed)
                )
                before = dict(self.current)
                key = axis.lower()
                self.current[f"work_{key}"] += delta
                self.current[f"machine_{key}"] += delta
                return {"before": before, "after": dict(self.current)}

            def close(self):
                return None

        controller = OffsetController()
        service = Mach3EmergencyService(
            controller=controller, status_controller=controller
        )
        finished = QtTest.QSignalSpy(service.press_finished)
        contact = QtTest.QSignalSpy(service.contact_reached)
        with mock.patch("fbg_unified_app.time.sleep", return_value=None):
            service.press_target(
                {
                    "work_x": -5.0,
                    "work_y": 12.0,
                    "contact_z": -26.0,
                    "depth_mm": 0.2,
                    "calibration_machine_minus_work": (-13.0, -50.0, 14.0),
                }
            )
        if not finished:
            self.assertTrue(finished.wait(1000))

        self.assertTrue(finished[-1][0])
        self.assertEqual(controller.xy_calls, [(-25.0, -49.0, 60.0, True)])
        self.assertEqual(len(contact), 1)
        resolved = contact[-1][0]
        self.assertEqual(resolved["coordinate_resolution"], "machine_absolute")
        self.assertAlmostEqual(resolved["target_machine_x"], -18.0)
        self.assertAlmostEqual(resolved["target_machine_y"], -38.0)
        self.assertAlmostEqual(resolved["contact_machine_z"], -12.0)
        self.assertAlmostEqual(resolved["work_x"], -25.0)
        self.assertAlmostEqual(resolved["work_y"], -49.0)
        self.assertAlmostEqual(resolved["contact_z"], -15.0)
        self.assertAlmostEqual(controller.current["machine_z"], -8.0)
        service.close()

    def test_one_shot_click_press_accepts_silicone_in_demo_without_motion(self):
        finger = self.window.finger_controller
        self.window.workspace_tabs.setCurrentIndex(1)
        finger.click_press_arm.setChecked(True)
        self.app.processEvents()

        QtTest.QTest.mouseClick(
            finger.view,
            QtCore.Qt.LeftButton,
            pos=finger.view.rect().center(),
        )
        for _ in range(4):
            self.app.processEvents()

        self.assertFalse(finger.click_press_arm.isChecked())
        self.assertTrue(finger.click_target_item.visible())
        self.assertIn("演示模式不会控制机床", finger.click_press_status_label.text())

    def test_hyperos_settings_navigation_tracks_the_embedded_pages(self):
        self.assertEqual(len(self.window.nav_buttons), 5)
        self.assertTrue(self.window.workspace_tabs.tabBar().isHidden())
        self.window.workspace_tabs.setCurrentIndex(2)
        self.app.processEvents()
        self.assertTrue(self.window.nav_buttons[2].isChecked())
        self.assertEqual(self.window.page_title.text(), "连接与设备配置")
        self.assertFalse(self.window.workflow_strip.isVisible())

    def test_demo_network_source_drives_shared_view(self):
        self.assertEqual(self.window.current_source, NETWORK_LAN)
        self.assertTrue(self.window.source_buttons[NETWORK_LAN].isChecked())
        self.window.open_finger_tab()
        self.assertEqual(self.window.workspace_tabs.currentIndex(), 1)

    def test_real_lan_reuses_all_seven_modes_including_temporary_mode(self):
        class MemoryLanSerial:
            def __init__(self):
                self.port = "LAN"
                self.baudrate = 2_000_000
                self.timeout = 0.2
                self.dtr = False
                self.is_open = False
                self.board_ip = "192.168.3.46"

            def open(self):
                self.is_open = True

            def close(self):
                self.is_open = False

            def reset_input_buffer(self):
                return None

        original_global_serial = __import__("app_JDSU").ser
        original_lan_serial = self.window.lan_serial
        memory_lan = MemoryLanSerial()
        try:
            self.window.demo = False
            self.window.lan_serial = memory_lan
            self.window.switch_source(NETWORK_LAN, force=True)
            self.app.processEvents()

            self.assertIs(
                self.window.source_stack.currentWidget(), self.window.local_page
            )
            self.assertTrue(
                all(button.isEnabled() for button in self.window.mode_buttons)
            )
            self.assertIs(__import__("app_JDSU").ser, memory_lan)
            self.window._mode_requested(3)
            self.assertEqual(self.window.local_controller.MCU_mode, 3)
            self.assertIs(
                self.window.local_controller.stack.currentWidget(),
                self.window.local_controller.page_extra,
            )
            self.window._mode_requested(6)
            self.assertEqual(self.window.local_controller.MCU_mode, 6)
            self.assertIs(
                self.window.local_controller.stack.currentWidget(),
                self.window.local_controller.page_temporary_test,
            )
        finally:
            self.window._release_local_control()
            self.window.current_source = None
            self.window.demo = True
            self.window.lan_serial = original_lan_serial
            __import__("app_JDSU").ser = original_global_serial

    def test_provisioning_preflight_pauses_all_sources(self):
        self.assertTrue(self.window._prepare_provisioning())
        self.assertIsNone(self.window.current_source)
        self.assertFalse(
            any(button.isChecked() for button in self.window.source_buttons.values())
        )

    def test_ota_page_is_embedded_and_preflight_pauses_sources(self):
        self.assertIsNotNone(self.window.ota_page)
        self.assertTrue(self.window._prepare_ota())
        self.assertIsNone(self.window.current_source)
        self.assertFalse(
            any(button.isChecked() for button in self.window.source_buttons.values())
        )

    def test_failed_local_usb_claim_keeps_the_previous_network_source(self):
        remote = self.window.remote_controller
        with (
            mock.patch.object(
                remote, "deactivate_network", wraps=remote.deactivate_network
            ),
            mock.patch.object(remote, "activate_network") as reactivate,
            mock.patch.object(
                self.window.local_controller,
                "acquire_local_control",
                return_value=False,
            ),
            mock.patch.object(
                self.window,
                "_update_source_status",
                wraps=self.window._update_source_status,
            ) as update_status,
        ):
            self.window.switch_source(SOURCE_LOCAL)

        self.assertEqual(self.window.current_source, NETWORK_LAN)
        self.assertTrue(self.window.source_buttons[NETWORK_LAN].isChecked())
        self.assertIs(self.window.source_stack.currentWidget(), self.window.remote_page)
        reactivate.assert_called_once_with(NETWORK_LAN)
        self.assertIn(
            mock.call("本地控制：USB 端口打开失败", "error"),
            update_status.call_args_list,
        )
        self.assertFalse(
            any(
                len(call.args) >= 2
                and call.args[1] == "ok"
                and "本地控制" in call.args[0]
                for call in update_status.call_args_list
            )
        )

    def test_close_is_cancelled_when_unsaved_temporary_spectrum_is_not_resolved(self):
        event = mock.Mock()
        page = self.window.local_controller.page_temporary_test
        with mock.patch.object(
                page, 'confirm_save_before_close', return_value=False) as confirm:
            self.window.closeEvent(event)
        confirm.assert_called_once_with(self.window)
        event.ignore.assert_called_once_with()
        event.accept.assert_not_called()
        self.assertFalse(self.window._closing)


if __name__ == "__main__":
    unittest.main()
