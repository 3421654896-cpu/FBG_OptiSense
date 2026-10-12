"""Desktop update UI never mixes client installers with board firmware."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_JDSU  # Match the application's Qt import order on Windows.
from PyQt5 import QtWidgets

from fbg_update_widget import SoftwareUpdatePage
from github_update_feed import ReleaseAsset


class SoftwareUpdateWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_client_page_only_offers_strictly_newer_installer(self):
        page = SoftwareUpdatePage()
        self.assertFalse(page.update_button.isEnabled())
        old = ReleaseAsset(
            "desktop", "1.0.4", "FBG_OptiSense_Studio_Universal_v1.0.4_Setup.exe",
            "https://github.com/3421654896-cpu/JDSU_Laser/releases/download/v1/test.exe",
            "0" * 64, 100,
        )
        page.set_offer(old)
        self.assertFalse(page.update_button.isEnabled())
        self.assertEqual(page.status.text(), "已是最新版本。")
        newer = ReleaseAsset(
            "desktop", "1.0.5", "FBG_OptiSense_Studio_Universal_v1.0.5_Setup.exe",
            old.url, old.sha256, old.size,
        )
        page.set_offer(newer)
        self.assertTrue(page.update_button.isEnabled())
        self.assertIn("客户端", page.status.text())
        page.close()


if __name__ == "__main__":
    unittest.main()
