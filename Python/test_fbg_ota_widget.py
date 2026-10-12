"""Regression tests for OTA page confirmation and package metadata."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
_qt_platform_plugins = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "PyQt5"
    / "Qt5"
    / "plugins"
    / "platforms"
)
if _qt_platform_plugins.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_platform_plugins)

from PyQt5 import QtWidgets

from fbg_ota_protocol import FirmwarePackage, OtaError
from fbg_ota_widget import (
    OtaUpdatePage,
    OtaWorker,
    WanOtaWorker,
    _format_build_timestamp,
    _manual_power_cycle_required,
)
from ota_machine_profiles import OtaMachineProfileStore
from fbg_mqtt_transport import RemoteMqttConfig


class OtaWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_ready_page(self) -> OtaUpdatePage:
        page = self.make_page()
        page.package = FirmwarePackage(
            path=Path("test.fbgfw"),
            target_id=0xF2050052,
            link_address=0x08008000,
            firmware_version=0x010203,
            build_timestamp=1,
            image_crc32=0,
            image=b"12345678",
        )
        page.board_ip.setText("192.168.3.46")
        page.password.setText("secret")
        return page

    def make_page(self) -> OtaUpdatePage:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        store = OtaMachineProfileStore(
            Path(folder.name) / "ota_profiles.json",
            protector=lambda value: "test:" + value,
            unprotector=lambda value: value.removeprefix("test:"),
        )
        return OtaUpdatePage(settings_store=store)

    def test_switching_machine_saves_and_restores_ota_fields(self) -> None:
        page = self.make_page()
        page.board_ip.setText("192.168.3.46")
        page.password.setText("machine-one-password")
        page.save_current_settings()
        self.assertTrue(page.set_machine("machine_2"))
        self.assertEqual(page.board_ip.text(), "")
        self.assertEqual(page.password.text(), "")
        page.board_ip.setText("192.168.3.47")
        page.password.setText("machine-two-password")
        page.save_current_settings()
        self.assertTrue(page.set_machine("machine_1"))
        self.assertEqual(page.board_ip.text(), "192.168.3.46")
        self.assertEqual(page.password.text(), "machine-one-password")
        self.assertIn("一号机", page.machine_target.text())
        self.assertIsNone(page._thread)

    def test_declining_confirmation_has_no_transport_side_effect(self) -> None:
        page = self.make_ready_page()
        callbacks: list[str] = []
        page.before_update = lambda: callbacks.append("before") or True
        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.No,
        ):
            page._start()
        self.assertEqual(callbacks, [])
        self.assertIsNone(page._thread)

    def test_usb_update_window_entry_is_integrated(self):
        page = self.make_page()
        page.open_usb_update()
        self.addCleanup(page._usb_dialog.close)
        self.assertIn("USB", page.usb_update_button.text())
        self.assertIn("一号机", page._usb_dialog.windowTitle())
        self.assertTrue(page._usb_dialog.isVisible())
        self.assertTrue(page._usb_dialog.auto_update)
        self.assertIsNotNone(page._usb_dialog.package_resolver)

    def test_wireless_update_controls_stay_hidden(self):
        page = self.make_page()
        page.show()
        self.app.processEvents()
        self.assertTrue(page.start_button.isHidden())
        self.assertTrue(page.update_offer.isHidden())
        self.assertTrue(page.developer_panel.isHidden())
        self.assertTrue(page.usb_update_button.isVisible())
        self.assertEqual(page.usb_update_button.text(), '一键更新（USB）')
        page.set_developer_mode(True)
        self.assertTrue(page.developer_panel.isHidden())
        page.close()

    def test_usb_update_busy_blocks_machine_switch(self):
        page = self.make_page()
        page._usb_dialog = mock.Mock(busy=True)
        self.assertTrue(page.busy)
        self.assertFalse(page.set_machine("machine_3"))
        self.assertEqual(page.current_machine_id, "machine_1")
        page._usb_dialog = None

    def test_idle_usb_window_closes_on_machine_switch(self):
        page = self.make_page()
        dialog = mock.Mock(busy=False)
        page._usb_dialog = dialog
        self.assertTrue(page.set_machine("machine_3"))
        dialog.close.assert_called_once()
        self.assertIsNone(page._usb_dialog)

    def test_confirmation_precedes_transport_pause(self) -> None:
        page = self.make_ready_page()
        events: list[str] = []

        def question(*_args, **_kwargs):
            events.append("question")
            return QtWidgets.QMessageBox.Yes

        page.before_update = lambda: events.append("before") or False
        with mock.patch.object(QtWidgets.QMessageBox, "question", side_effect=question):
            page._start()
        self.assertEqual(events, ["question", "before"])
        self.assertIsNone(page._thread)

    def test_customer_one_click_requires_newer_live_version_and_saved_settings(self):
        page = self.make_ready_page()
        page._customer_package_published = True
        self.assertTrue(page.developer_panel.isHidden())
        self.assertTrue(page.log.isHidden())
        self.assertFalse(page.start_button.isEnabled())
        notices = []
        page.update_available_changed.connect(
            lambda available, _message: notices.append(available)
        )
        page.refresh_update_offer()
        self.assertEqual(notices[-1], True)
        self.assertFalse(page.start_button.isEnabled())
        page.set_observed_firmware_version("machine_1", 0x010202)
        self.assertTrue(page.start_button.isEnabled())
        page.before_update = lambda: False
        with mock.patch.object(QtWidgets.QMessageBox, "question") as question:
            page.start_one_click()
        question.assert_not_called()
        self.assertIsNone(page._thread)
        page.set_observed_firmware_version("machine_1", 0x010203)
        self.assertFalse(page.start_button.isEnabled())

    def test_wan_target_requires_explicit_device_binding_and_not_lan_ip(self):
        page = self.make_ready_page()
        page.board_ip.clear()
        page.set_wan_config(RemoteMqttConfig(
            device_id="fbg-BROKERDEFAULT", host="example.invalid"
        ))
        page._customer_package_published = True
        page.set_observed_firmware_version("machine_1", 0x010202)
        self.assertFalse(page.start_button.isEnabled())
        page.mqtt_device_id.setText("fbg-MACHINEONE")
        page.refresh_update_offer()
        self.assertTrue(page.start_button.isEnabled())
        self.assertEqual(page._wan_target().device_id, "fbg-MACHINEONE")
        page.before_update = lambda: False
        page.start_one_click()
        self.assertIsNone(page._thread)

    def test_wan_worker_fails_closed_when_signed_reboot_verification_fails(self):
        page = self.make_ready_page()
        config = RemoteMqttConfig(device_id="fbg-MACHINEONE", host="example.invalid")
        worker = WanOtaWorker(config, "secret", page.package)
        events = []
        worker.completed.connect(lambda: events.append("completed"))
        worker.failed.connect(lambda message: events.append(f"failed:{message}"))
        with mock.patch("fbg_ota_widget.MqttOtaClient") as client_type:
            client_type.return_value.__enter__.return_value.install.side_effect = OtaError(
                "未验证到新启动标识"
            )
            worker.run()
        self.assertEqual(events, ["failed:未验证到新启动标识"])

    def test_bound_wan_one_click_selects_mqtt_worker_without_lan_ip(self):
        page = self.make_ready_page()
        page.board_ip.clear()
        page.mqtt_device_id.setText("fbg-MACHINEONE")
        page.set_wan_config(RemoteMqttConfig(
            device_id="fbg-BROKERDEFAULT", host="example.invalid"
        ))
        page._customer_package_published = True
        page.set_observed_firmware_version("machine_1", 0x010202)
        page.before_update = lambda: True
        with mock.patch.object(WanOtaWorker, "start") as start, \
             mock.patch.object(QtWidgets.QMessageBox, "question") as question:
            page.start_one_click()
        question.assert_not_called()
        start.assert_called_once()
        self.assertIsInstance(page._thread, WanOtaWorker)
        self.assertEqual(page._thread.config.device_id, "fbg-MACHINEONE")
        self.assertEqual(page._active_transport, "wan")

    def test_checked_empty_release_displays_latest_but_failed_check_does_not(self):
        page = self.make_page()
        page.set_available_release_version(None)
        self.assertIn("已是最新版本", page.update_offer.text())
        self.assertFalse(page.start_button.isEnabled())
        page.set_release_check_failed("HTTP 403")
        self.assertIn("无法检查", page.update_offer.text())
        self.assertNotIn("已是最新版本", page.update_offer.text())

    def test_customer_layout_is_compact_on_tall_page(self):
        page = self.make_page()
        page.resize(1080, 695)
        page.show()
        self.app.processEvents()
        self.assertLess(page.update_offer.y(), 210)
        self.assertLess(page.start_button.mapTo(page, page.start_button.rect().topLeft()).y(), 350)
        self.assertTrue(page.progress.isHidden())
        page.set_developer_mode(True)
        self.app.processEvents()
        self.assertTrue(page.customer_spacer.isHidden())
        self.assertTrue(page.log.isVisible())
        page.close()

    def test_extreme_uint64_timestamp_is_display_safe(self) -> None:
        text = _format_build_timestamp(0xFFFFFFFFFFFFFFFF)
        self.assertIn("未知", text)

    def test_worker_does_not_complete_when_reboot_verification_fails(self) -> None:
        page = self.make_ready_page()
        worker = OtaWorker("192.168.3.46", "secret", page.package)
        events: list[str] = []
        worker.completed.connect(lambda: events.append("completed"))
        worker.failed.connect(lambda message: events.append(f"failed:{message}"))
        with mock.patch("fbg_ota_widget.OtaClient") as client_type:
            client_type.return_value.upload.side_effect = OtaError(
                "旧 OTA 连接未断开"
            )
            worker.run()
        self.assertEqual(events, ["failed:旧 OTA 连接未断开"])

    def test_machine_two_worker_uses_measured_robust_recovery_budget(self) -> None:
        page = self.make_ready_page()
        worker = OtaWorker(
            "192.168.3.26",
            "secret",
            page.package,
            robust_recovery=True,
        )
        with mock.patch("fbg_ota_widget.OtaClient") as client_type:
            worker.run()
        client_type.assert_called_once_with(
            "192.168.3.26",
            "secret",
            retries=160,
            response_timeout_s=8.0,
            recovery_quiet_s=30.0,
            retry_owner_timeout_s=20.0,
            chunk_delay_s=0.2,
            adaptive_chunk_pacing=True,
            chunk_delay_max_s=0.45,
            chunk_delay_backoff=1.5,
            chunk_delay_stable_blocks=384,
            post_reboot_recovery=mock.ANY,
            post_reboot_recovery_interval_s=10.0,
        )
        client_type.return_value.upload.assert_called_once()

    def test_after_update_runs_only_from_verified_completion(self) -> None:
        page = self.make_ready_page()
        callbacks: list[FirmwarePackage] = []
        page.after_update = callbacks.append
        with mock.patch.object(QtWidgets.QMessageBox, "warning"):
            page._failed("RAW READY 超时")
        self.assertEqual(callbacks, [])
        with mock.patch.object(QtWidgets.QMessageBox, "information"):
            page._completed()
        self.assertEqual(callbacks, [page.package])

    def test_power_cycle_is_requested_only_after_reboot_confirmation(self) -> None:
        self.assertTrue(
            _manual_power_cycle_required(
                "已收到板卡重启确认，但未能在 600 秒内恢复全功能 LAN：timed out"
            )
        )
        self.assertTrue(
            _manual_power_cycle_required(
                "板卡已确认重启，但旧 OTA 连接未按期断开；无法证明 MCU 已进入自动重启流程"
            )
        )
        self.assertFalse(_manual_power_cycle_required("等待板卡 OTA 回执超时"))
        self.assertFalse(_manual_power_cycle_required("用户取消了 OTA 升级"))

    def test_reboot_recovery_timeout_shows_explicit_power_cycle_prompt(self) -> None:
        page = self.make_ready_page()
        page.set_machine("machine_2")
        message = "已收到板卡重启确认，但未能在 600 秒内恢复全功能 LAN：timed out"
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            page._failed(message)
        self.assertEqual(page.progress.format(), "请断电重上电")
        self.assertIn("二号机", page.progress_message.text())
        self.assertIn("完全断电", page.progress_message.text())
        self.assertIn("10 秒", page.progress_message.text())
        self.assertIn("现在请断电重上电", page.power_cycle_hint.text())
        self.assertEqual(warning.call_args.args[1], "现在需要断电重上电")

    def test_transfer_failure_does_not_request_power_cycle(self) -> None:
        page = self.make_ready_page()
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            page._failed("等待板卡 OTA 回执超时")
        self.assertEqual(page.progress.format(), "升级未完成")
        self.assertIn("不要", page.power_cycle_hint.text())
        self.assertNotIn("现在请断电重上电", page.power_cycle_hint.text())
        self.assertEqual(warning.call_args.args[1], "OTA 升级未完成")


if __name__ == "__main__":
    unittest.main()
