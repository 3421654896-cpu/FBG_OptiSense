import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import peak_detection_JDSU as legacy_app


class LegacyPeakDetectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_cancelled_ack_wait_does_not_touch_serial_port(self):
        with mock.patch.object(legacy_app, "serial_write") as write:
            result = legacy_app.wait_ack_0x21_forever(
                b"command", should_continue=lambda: False
            )
        self.assertFalse(result)
        write.assert_not_called()

    def test_wave_table_load_is_independent_of_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            legacy_app.peakWorker, "start"
        ):
            try:
                os.chdir(temporary)
                page = legacy_app.GraphWindow()
                expected = legacy_app.yaml.safe_load(
                    Path(legacy_app.__file__)
                    .resolve()
                    .with_name("wave_const.yaml")
                    .read_text(encoding="utf-8")
                )["Wave_DATA"]
                self.assertEqual(len(expected), 54)
                self.assertEqual(page.yaml, expected)
                page.close()
                page.deleteLater()
                self.app.processEvents()
            finally:
                os.chdir(original)

    def test_legacy_excel_codes_are_exact_uint16_values(self):
        command, _info = legacy_app.excel_operate(
            iter([(None, None, None, None, 1, 2.0, 65535)])
        )
        self.assertEqual(command[3:9], b"\x00\x01\x00\x02\xff\xff")
        with self.assertRaises(ValueError):
            legacy_app.excel_operate(
                iter([(None, None, None, None, 1, 2.5, 3)])
            )
        self.assertEqual(legacy_app.excel_operate(iter(())), (None, None))

    def test_legacy_port_picker_does_not_guess_programmer_uart(self):
        programmer = SimpleNamespace(
            device="COM7", description="USB programmer UART", vid=0x1A86, pid=0x55D4
        )
        native = SimpleNamespace(
            device="COM9", description="STM32 Virtual COM Port", vid=0x0483, pid=0x5740
        )
        with mock.patch(
            "peak_detection_JDSU.list_ports.comports", return_value=[programmer]
        ):
            combo = legacy_app.MQComboBox()
        self.assertEqual(combo.currentIndex(), -1)
        combo.close()

        with mock.patch(
            "peak_detection_JDSU.list_ports.comports",
            return_value=[programmer, native],
        ):
            combo = legacy_app.MQComboBox()
        self.assertEqual(combo.currentData(), "COM9")
        combo.close()

    def test_idle_receiver_thread_can_be_stopped_cleanly(self):
        worker = legacy_app.peakWorker()
        legacy_app.ser_open = False
        worker.start()
        worker.running = False
        with legacy_app.ser_cond:
            legacy_app.ser_cond.notify_all()
        self.assertTrue(worker.wait(1000))
        self.assertFalse(worker.isRunning())
        self.assertEqual(legacy_app.ser.timeout, 0.2)


if __name__ == "__main__":
    unittest.main()
