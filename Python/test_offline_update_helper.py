import builtins
import os
from pathlib import Path
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
plugin_path = Path(sys.prefix)/'Lib'/'site-packages'/'PyQt5'/'Qt5'/'plugins'/'platforms'
if plugin_path.exists():
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugin_path)
from PyQt5 import QtWidgets
import offline_update_helper as helper
from fbg_usb_ota import UsbFirmwareInfo


class OfflineHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_updater_theme_does_not_import_plot_dependencies(self):
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'pyqtgraph'):
                raise ImportError('Updater must not need plot libraries')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', blocked):
            helper.install_theme(self.app, configure_plots=False)

    def worker(self, serial='395736523033'):
        manifest = {'firmware': [{'usb_serials': ['395736523033'], 'version': '1.0.109'}]}
        worker = helper.UpdateWorker(Path('.'), manifest, 'firmware', ('COM16', serial))
        results, failures = [], []
        worker.result.connect(results.append)
        worker.failed.connect(failures.append)
        return worker, manifest, results, failures

    def test_latest_firmware_reads_version_without_reflashing(self):
        worker, manifest, results, failures = self.worker()
        package = SimpleNamespace(version_text='1.0.109', firmware_version=0x1006D)
        with patch.object(helper, 'load_manifest', return_value=manifest), \
             patch.object(helper, 'checked_payload', return_value=Path('firmware.fbgfw')), \
             patch.object(helper, 'load_package', return_value=package), \
             patch.object(helper, 'UsbOtaClient') as client:
            runtime = client.return_value.__enter__.return_value
            runtime.query_info.return_value = UsbFirmwareInfo('395736523033',0x1006D,99,229376,True)
            worker.run()
            runtime.install.assert_not_called()
        self.assertFalse(failures)
        self.assertEqual(results[0]['status'], 'latest')

    def test_other_machine_rejected_before_opening_usb(self):
        worker, manifest, results, failures = self.worker('OTHER_MACHINE')
        with patch.object(helper, 'load_manifest', return_value=manifest), \
             patch.object(helper, 'UsbOtaClient') as client:
            worker.run()
            client.assert_not_called()
        self.assertFalse(results)
        self.assertIn('未授权', failures[0])

    def test_firmware_update_uses_verified_usb_install_result(self):
        worker, manifest, results, failures = self.worker()
        package = SimpleNamespace(version_text='1.0.109', firmware_version=0x1006D)
        with patch.object(helper, 'load_manifest', return_value=manifest), \
             patch.object(helper, 'checked_payload', return_value=Path('firmware.fbgfw')), \
             patch.object(helper, 'load_package', return_value=package), \
             patch.object(helper, 'UsbOtaClient') as client:
            runtime = client.return_value.__enter__.return_value
            runtime.query_info.return_value = UsbFirmwareInfo('395736523033',0x1006C,99,229376,True)
            runtime.install.return_value = UsbFirmwareInfo('395736523033',0x1006D,100,229376,True)
            runtime.reboot_acknowledged = True
            worker.run()
            runtime.install.assert_called_once()
        self.assertFalse(failures)
        self.assertEqual(results[0]['runtime']['boot_id'], 100)
        self.assertTrue(results[0]['reboot_acknowledged'])

    def test_invalid_signature_disables_both_update_buttons(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.list_ports, 'comports', return_value=[]):
            window = helper.OfflineUpdater(folder)
            self.assertFalse(window.client_button.isEnabled())
            self.assertFalse(window.firmware_button.isEnabled())
            window.close()
