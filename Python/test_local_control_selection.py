import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_JDSU


class NativeCdcSelectionTests(unittest.TestCase):
    @staticmethod
    def port(device, vid, pid, description=""):
        return SimpleNamespace(
            device=device,
            vid=vid,
            pid=pid,
            description=description,
        )

    def test_com7_programmer_is_not_used_when_native_com6_is_missing(self):
        ports = [self.port("COM7", 0x1A86, 0x55D4, "USB programmer UART")]
        self.assertIsNone(app_JDSU.find_native_stm32_cdc_port(ports))

    def test_exact_stm32_vid_pid_is_selected(self):
        ports = [
            self.port("COM7", 0x1A86, 0x55D4),
            self.port("COM9", 0x0483, 0x5740),
        ]
        self.assertEqual(app_JDSU.find_native_stm32_cdc_port(ports), "COM9")


class LocalControlResultTests(unittest.TestCase):
    def tearDown(self):
        app_JDSU.local_control_session = False

    def test_success_is_reported_explicitly(self):
        owner = SimpleNamespace(port="COM9")
        with mock.patch("app_JDSU.ensure_serial_open"), mock.patch.object(
            app_JDSU.ser, "reset_input_buffer"
        ):
            self.assertTrue(app_JDSU.MainWindow.acquire_local_control(owner))
        self.assertTrue(app_JDSU.local_control_session)

    def test_failure_is_reported_and_releases_partial_serial_claim(self):
        owner = SimpleNamespace(port="COM9")
        with mock.patch(
            "app_JDSU.ensure_serial_open", side_effect=RuntimeError("port busy")
        ), mock.patch("app_JDSU.release_serial_if_allowed") as release, mock.patch(
            "app_JDSU.QMessageBox.warning"
        ):
            self.assertFalse(app_JDSU.MainWindow.acquire_local_control(owner))
        self.assertFalse(app_JDSU.local_control_session)
        release.assert_called_once_with()

    def test_first_manual_port_selection_handles_no_previous_action(self):
        action = mock.Mock()
        action.data.return_value = "COM7"
        owner = SimpleNamespace(
            sender=lambda: action,
            port=None,
            port_act=None,
        )
        serial_port = SimpleNamespace(is_open=False, port=None)
        with mock.patch("app_JDSU.ser", serial_port), mock.patch.object(
            app_JDSU, "switch_mode_enable", True
        ), mock.patch.object(app_JDSU, "local_control_session", False):
            app_JDSU.MainWindow.set_com_port(owner)
        self.assertEqual(owner.port, "COM7")
        self.assertIs(owner.port_act, action)
        action.setChecked.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
