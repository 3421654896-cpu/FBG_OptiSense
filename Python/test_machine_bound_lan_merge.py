"""Regression checks for per-machine LAN ownership after source merge."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import app_JDSU
from fbg_network_transport import NETWORK_LAN
from fbg_unified_app import UnifiedFbgStudio
from ota_machine_profiles import OtaMachineProfileStore


class MachineBoundLanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.data_directory = tempfile.TemporaryDirectory()
        self.data_override = mock.patch.dict(
            os.environ, {"FBG_STUDIO_DATA_DIR": self.data_directory.name}
        )
        self.data_override.start()
        self.store = OtaMachineProfileStore(
            Path(self.directory.name) / "ota_profiles.json",
            protector=lambda value: "test:" + value,
            unprotector=lambda _value: (_ for _ in ()).throw(
                AssertionError("LAN lookup must not decrypt OTA credentials")
            ),
        )
        self.window = UnifiedFbgStudio(
            initial_source=NETWORK_LAN,
            demo=True,
            ota_settings_store=self.store,
        )
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.data_override.stop()
        self.data_directory.cleanup()
        self.directory.cleanup()

    def test_provisional_machine_three_table_is_not_certified(self):
        points = app_JDSU.load_fullband_accuracy_table(machine_id="machine_3")
        self.assertEqual(len(points), 2001)
        self.assertEqual(points[0].codes, (63351, 57089, 16283, 19171, 7806))
        index = self.window.machine_selector.findData("machine_3")
        self.assertGreaterEqual(index, 0)
        self.window.machine_selector.setCurrentIndex(index)
        self.app.processEvents()
        self.assertEqual(self.window.ota_page.current_machine_id, "machine_3")
        self.assertIn("临时表", self.window.device_version.text())
        self.assertEqual(
            self.window.local_controller.page_unlimited_accuracy.points[0].codes,
            points[0].codes,
        )

    def test_lan_target_is_selected_machine_ip_not_discovered_peer(self):
        self.store.save_profile(
            "machine_3", board_ip="192.168.4.62", password="",
            package_path="",
        )
        self.window.ota_page.set_machine("machine_3")
        self.window.demo = False
        previous_serial = app_JDSU.ser
        try:
            self.window._select_local_transport(NETWORK_LAN)
            self.assertEqual(self.window.lan_serial.board_host, "192.168.4.62")
            self.assertIs(app_JDSU.ser, self.window.lan_serial)
            self.window.lan_serial._board_ip = "192.168.4.23"
            self.assertTrue(self.window._prepare_ota())
            self.assertEqual(self.store.lan_board_ip("machine_3"), "192.168.4.62")
        finally:
            self.window.demo = True
            app_JDSU.ser = previous_serial

    def test_missing_or_invalid_ip_fails_closed(self):
        self.window.ota_page.set_machine("machine_3")
        self.window.demo = False
        previous_serial = app_JDSU.ser
        try:
            with self.assertRaisesRegex(ValueError, "板卡 IP"):
                self.window._select_local_transport(NETWORK_LAN)
            self.assertIs(app_JDSU.ser, previous_serial)
            self.store.save_profile(
                "machine_3", board_ip="not-an-ip", password="",
                package_path="",
            )
            with self.assertRaisesRegex(ValueError, "板卡 IP"):
                self.store.lan_board_ip("machine_3")
        finally:
            self.window.demo = True

    def test_switch_releases_old_lan_before_new_calibration(self):
        self.store.save_profile(
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

    def test_failed_calibration_does_not_reconnect_old_lan(self):
        self.store.save_profile(
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
                 ) as acquire:
                self.window.machine_selector.setCurrentIndex(
                    self.window.machine_selector.findData("machine_3")
                )
            release.assert_called_once_with()
            acquire.assert_not_called()
            self.assertIsNone(self.window.current_source)
        finally:
            self.window.demo = True
