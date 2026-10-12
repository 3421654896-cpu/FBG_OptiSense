import os
import sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
_plugins = Path(sys.prefix) / "Lib/site-packages/PyQt5/Qt5/plugins/platforms"
if _plugins.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_plugins)
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from PyQt5 import QtWidgets
from fbg_usb_ota import UsbFirmwareInfo
from fbg_usb_ota_widget import UsbFirmwareUpdateDialog, UsbUpdateWorker


class UsbWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def dialog(self, ports=()):
        dialog = UsbFirmwareUpdateDialog(machine_label="四号机", ports_provider=lambda: ports)
        self.addCleanup(dialog.close)
        return dialog

    def test_single_native_port_selected_and_debugger_ignored(self):
        ports = [SimpleNamespace(device="COM20", serial_number="207A33813232", vid=0x0483, pid=0x5740),
                 SimpleNamespace(device="COM17", serial_number="probe", vid=0x1A86, pid=0x8012)]
        dialog = self.dialog(ports)
        self.assertEqual(dialog.ports.count(), 1)
        self.assertEqual(dialog.ports.currentData(), ("COM20", "207A33813232"))

    def test_multiple_boards_require_explicit_selection(self):
        ports = [SimpleNamespace(device=f"COM{i}", serial_number=str(i), vid=0x0483, pid=0x5740) for i in (20, 21)]
        dialog = self.dialog(ports)
        self.assertIsNone(dialog.ports.currentData())

    def test_version_text_and_latest_label(self):
        dialog = self.dialog()
        dialog.set_package(SimpleNamespace(version_text="1.0.105", image_size=192000, firmware_version=0x10069))
        dialog._info(UsbFirmwareInfo("207A33813232", 0x10069, 1, 0x38000, True))
        self.assertIn("已是最新版本", dialog.message.text())
        self.assertIn("207A33813232", dialog.device_info.text())

    def test_commit_disables_cancel(self):
        dialog = self.dialog()
        dialog.cancel_button.setEnabled(True)
        dialog._progress(99, "整包校验和重启中，不能取消或断电")
        self.assertFalse(dialog.cancel_button.isEnabled())

    def test_preflight_denial_does_not_start_worker(self):
        dialog = self.dialog([SimpleNamespace(device="COM20", serial_number="207A33813232", vid=0x0483, pid=0x5740)])
        dialog.before_update = lambda: False
        with patch("fbg_usb_ota_widget.UsbUpdateWorker") as worker:
            dialog.read_version()
            worker.assert_not_called()

    def test_customer_controls_are_hidden_and_single_port_is_automatic(self):
        dialog = self.dialog([SimpleNamespace(
            device='COM16', serial_number='395736523033', vid=0x0483, pid=0x5740)])
        for control in (dialog.ports, dialog.refresh_button, dialog.read_button,
                        dialog.browse_button, dialog.log):
            self.assertTrue(control.isHidden())
        self.assertEqual(dialog.start_button.text(), '一键更新')

    def test_one_click_has_no_second_confirmation_and_pauses_before_worker(self):
        dialog = self.dialog([SimpleNamespace(
            device='COM16', serial_number='395736523033', vid=0x0483, pid=0x5740)])
        dialog.set_package(SimpleNamespace(version_text='1.0.106', image_size=195808))
        dialog.before_update = lambda: False
        with patch('fbg_usb_ota_widget.UsbUpdateWorker') as worker, \
             patch.object(QtWidgets.QMessageBox, 'question') as question:
            dialog.start_update()
        question.assert_not_called()
        worker.assert_not_called()

    def test_one_click_worker_skips_same_or_older_package(self):
        package = SimpleNamespace(firmware_version=0x1006a)
        for version in (0x1006a, 0x1006b):
            worker = UsbUpdateWorker('COM16', '395736523033', package, newer_only=True)
            with patch('fbg_usb_ota_widget.UsbOtaClient') as client_type:
                client = client_type.return_value.__enter__.return_value
                client.query_info.return_value = UsbFirmwareInfo(
                    '395736523033', version, 1, 0x38000, True)
                worker.run()
                client.install.assert_not_called()

    def test_one_click_worker_resolves_using_live_serial_then_installs_newer(self):
        package = SimpleNamespace(firmware_version=0x1006a)
        resolver = unittest.mock.Mock(return_value=package)
        worker = UsbUpdateWorker('COM16', '395736523033', newer_only=True,
                                 package_resolver=resolver)
        with patch('fbg_usb_ota_widget.UsbOtaClient') as client_type:
            client = client_type.return_value.__enter__.return_value
            client.query_info.return_value = UsbFirmwareInfo(
                '395736523033', 0x10069, 1, 0x38000, True)
            client.install.return_value = UsbFirmwareInfo(
                '395736523033', 0x1006a, 2, 0x38000, True)
            worker.run()
            resolver.assert_called_once_with('395736523033')
            client.install.assert_called_once()

    def test_missing_bootloader_or_wrong_identity_never_installs(self):
        worker = UsbUpdateWorker('COM16', 'wrong', newer_only=True,
                                 package_resolver=lambda _serial: (_ for _ in ()).throw(ValueError('identity')))
        with patch('fbg_usb_ota_widget.UsbOtaClient') as client_type:
            client = client_type.return_value.__enter__.return_value
            client.query_info.return_value = UsbFirmwareInfo('wrong', 0x10069, 1, 0x38000, True)
            worker.run()
            client.install.assert_not_called()


if __name__ == "__main__": unittest.main()
