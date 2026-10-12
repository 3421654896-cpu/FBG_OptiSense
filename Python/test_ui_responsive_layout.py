import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from fbg_network_transport import NETWORK_LAN
from fbg_unified_app import UnifiedFbgStudio
from hyperos_theme import install_theme
from responsive_layout import FlowLayout
from ota_machine_profiles import OtaMachineProfileStore
from machine_profile import get_runtime_machine_id, set_runtime_machine_id


WINDOW_SIZES = (
    (960, 620),
    (1024, 640),
    (1280, 720),
    (1366, 768),
    (1600, 900),
)


class ResponsiveLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        install_theme(cls.app)
        cls.base_stylesheet = cls.app.styleSheet()

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        base = Path(folder.name)
        previous_machine = get_runtime_machine_id()
        self.addCleanup(set_runtime_machine_id, previous_machine)
        override = mock.patch.multiple(
            'temporary_test_widget',
            SCAN_PREFERENCES_PATH=base / 'scan_preferences.json',
            LAST_SESSION_PATH=base / 'last_session.json',
            FINGER_CAPTURE_DIR=base / 'finger_captures',
            LAST_FINGER_CAPTURE_PATH=base / 'last_finger_capture.json',
        )
        override.start()
        self.addCleanup(override.stop)
        self.app.setStyleSheet(self.base_stylesheet)
        self.window = UnifiedFbgStudio(initial_source=NETWORK_LAN, demo=True,
                                      ota_settings_store=OtaMachineProfileStore(base / 'ota.json'))
        self.window.show()
        self._drain_events()
        self._install_long_operator_text()

    def tearDown(self):
        self.app.setStyleSheet(self.base_stylesheet)
        self.window.close()
        self._drain_events()

    def _drain_events(self):
        for _ in range(8):
            self.app.processEvents(QtCore.QEventLoop.AllEvents, 20)

    def _install_long_operator_text(self):
        long_status = (
            "数据链路：局域网连接正在重新协商，当前帧仍保留用于显示，"
            "请等待设备身份与波长表校验完成"
        )
        self.window._update_source_status(long_status, "warning")
        self.window.context_hint.setText(
            "本地调试支持全部六种模式；等间隔模式按设定等待时间逐点显示原始ADC；"
            "窗口缩小时这些说明应自动换行，不能遮住模式按钮。"
        )
        remote = self.window.remote_controller
        remote.connection_label.setText(
            "网络：局域网链路正在重新连接并等待板卡握手确认"
        )
        remote.sequence_label.setText(
            "序号：123456789  丢帧：12  乱序：3  最近一次数据仍可查看"
        )
        remote.command_label.setText(
            "命令：等待板卡完成当前采集周期后再确认模式切换"
        )
        self.window.finger_controller.status_label.setText(
            "第123456帧：最大响应位于G9指根右侧，公共漂移已去除，"
            "请复核无应力基准是否仍然有效"
        )
        self.window.ota_page.progress_message.setText(
            "正在校验固件包、目标芯片、向量表和整包CRC，完成后才允许传输"
        )
        self.window.provision_page.status.setText(
            "密码只通过USB写入板卡，不保存到电脑；配置完成后将重新建立局域网链路。"
        )

    def _set_size_and_page(self, size, page_index):
        self.window.resize(*size)
        self.window.workspace_tabs.setCurrentIndex(page_index)
        self._drain_events()
        self.assertEqual((self.window.width(), self.window.height()), size)

    def _assert_flow_geometry(self, root, context):
        for layout in root.findChildren(FlowLayout):
            container = layout.parentWidget()
            if container is None or not container.isVisible():
                continue
            widgets = []
            for index in range(layout.count()):
                item = layout.itemAt(index)
                widget = item.widget() if item is not None else None
                if widget is None or not widget.isVisible():
                    continue
                widgets.append(widget)
                geometry = widget.geometry()
                self.assertGreater(
                    geometry.width(), 0, f"{context}: {widget.objectName()} width"
                )
                self.assertGreater(
                    geometry.height(), 0, f"{context}: {widget.objectName()} height"
                )
                self.assertGreaterEqual(geometry.left(), 0, context)
                self.assertGreaterEqual(geometry.top(), 0, context)
                self.assertLessEqual(
                    geometry.right(), container.rect().right(),
                    f"{context}: flow item exceeds right edge: {type(widget).__name__}",
                )
                self.assertLessEqual(
                    geometry.bottom(), container.rect().bottom(),
                    f"{context}: flow item exceeds bottom edge: {type(widget).__name__}",
                )
                if isinstance(widget, QtWidgets.QPushButton) and widget.text():
                    option = QtWidgets.QStyleOptionButton()
                    option.initFrom(widget)
                    content = widget.style().subElementRect(
                        QtWidgets.QStyle.SE_PushButtonContents, option, widget
                    )
                    required = widget.fontMetrics().horizontalAdvance(widget.text())
                    if not widget.icon().isNull():
                        required += widget.iconSize().width() + 5
                    self.assertLessEqual(
                        required,
                        content.width() + 4,
                        f"{context}: button text clipped: {widget.text()}",
                    )
            for index, first in enumerate(widgets):
                for second in widgets[index + 1 :]:
                    overlap = first.geometry().intersected(second.geometry())
                    self.assertTrue(
                        overlap.isEmpty(),
                        f"{context}: flow items overlap: "
                        f"{type(first).__name__}/{type(second).__name__}",
                    )

    def _assert_wrapped_label_fits(self, label, context):
        self.assertTrue(label.wordWrap(), f"{context}: label must wrap")
        content = label.contentsRect().adjusted(8, 3, -8, -3)
        bounds = label.fontMetrics().boundingRect(
            QtCore.QRect(0, 0, max(1, content.width()), 10000),
            QtCore.Qt.TextWordWrap | QtCore.Qt.AlignLeft,
            label.text(),
        )
        self.assertLessEqual(
            bounds.height(),
            content.height() + label.fontMetrics().height(),
            f"{context}: wrapped text height is clipped",
        )

    def _assert_direct_layout_geometry(self, container, context):
        layout = container.layout()
        self.assertIsNotNone(layout, context)
        widgets = []
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None and widget.isVisible():
                widgets.append(widget)
                self.assertTrue(
                    container.rect().contains(widget.geometry()),
                    f"{context}: direct child is clipped: {type(widget).__name__} "
                    f"{widget.geometry().getRect()} in {container.rect().getRect()}",
                )
        for index, first in enumerate(widgets):
            for second in widgets[index + 1 :]:
                self.assertTrue(
                    first.geometry().intersected(second.geometry()).isEmpty(),
                    f"{context}: direct layout children overlap",
                )

    def test_unified_pages_reflow_at_supported_window_sizes(self):
        for size in WINDOW_SIZES:
            for page_index in range(4):
                with self.subTest(size=size, page=page_index):
                    self._set_size_and_page(size, page_index)
                    self._assert_flow_geometry(
                        self.window, f"unified {size} page {page_index}"
                    )
                    self._assert_direct_layout_geometry(
                        self.window.findChild(QtWidgets.QFrame, "topBar"),
                        f"top bar {size} page {page_index}",
                    )
                    if self.window.page_subtitle.isVisible():
                        self._assert_wrapped_label_fits(
                            self.window.page_subtitle,
                            f"subtitle {size} page {page_index}",
                        )
                    else:
                        self.assertEqual(
                            self.window.page_subtitle.toolTip(),
                            self.window.page_subtitle.text(),
                        )
                    if page_index < 2 and self.window.context_hint.isVisible():
                        self._assert_wrapped_label_fits(
                            self.window.context_hint,
                            f"workflow hint {size} page {page_index}",
                        )
                    if page_index == 0:
                        remote_content = self.window.remote_page.findChild(
                            QtWidgets.QWidget, "remotePageContent"
                        )
                        self._assert_direct_layout_geometry(
                            remote_content, f"remote content {size}"
                        )
                    elif page_index == 1:
                        self._assert_direct_layout_geometry(
                            self.window.finger_page, f"finger content {size}"
                        )

            if size[0] < 1180:
                self.assertEqual(self.window.side_rail.width(), 88)
                self.assertTrue(all(not button.text() for button in self.window.nav_buttons))
            else:
                self.assertGreaterEqual(self.window.side_rail.width(), 246)
                self.assertTrue(all(button.text() for button in self.window.nav_buttons))

    def test_all_local_modes_are_responsive_in_unified_shell(self):
        self.window.source_stack.setCurrentWidget(self.window.local_page)
        local = self.window.local_controller
        for size in WINDOW_SIZES:
            self.window.resize(*size)
            self.window.workspace_tabs.setCurrentIndex(0)
            for mode_index in range(local.stack.count()):
                with self.subTest(size=size, local_mode=mode_index):
                    local.stack.setCurrentIndex(mode_index)
                    self._drain_events()
                    self._assert_flow_geometry(
                        self.window,
                        f"local mode {mode_index} at {size}",
                    )

                    page = local.stack.currentWidget()
                    self.assertTrue(page.isVisible())
                    if mode_index in (0, 1):
                        content = page.findChild(
                            QtWidgets.QWidget, "graphPageContent"
                        )
                        self._assert_direct_layout_geometry(
                            content, f"graph content {size} mode {mode_index}"
                        )
                        self.assertTrue(page.num_scroll.isVisible())
                        self.assertGreaterEqual(page.num_scroll.minimumHeight(), 120)
                    elif mode_index == 3:
                        content = page.findChild(
                            QtWidgets.QWidget, "singleValueContent"
                        )
                        self._assert_direct_layout_geometry(
                            content, f"single value content {size}"
                        )
                        expected = (
                            QtCore.Qt.Horizontal
                            if page.width() >= 980
                            else QtCore.Qt.Vertical
                        )
                        self.assertEqual(page.body_splitter.orientation(), expected)
                    elif mode_index == 2:
                        self._assert_direct_layout_geometry(
                            page, f"auto-calibration page content {size}"
                        )
                    elif mode_index == 4:
                        content = page.findChild(
                            QtWidgets.QWidget, "unlimitedPageContent"
                        )
                        self._assert_direct_layout_geometry(
                            content, f"unlimited content {size}"
                        )

    def test_large_font_and_long_chinese_statuses_remain_readable(self):
        self.app.setStyleSheet(
            self.base_stylesheet
            + "\nQWidget { font-size: 15pt; }"
            + "\nQPushButton { padding: 8px 14px; }"
        )
        self.window.resize(1024, 640)
        for page_index in range(4):
            with self.subTest(page=page_index):
                self.window.workspace_tabs.setCurrentIndex(page_index)
                self._drain_events()
                self._assert_flow_geometry(
                    self.window, f"large-font page {page_index}"
                )

        for label, name in (
            (self.window.source_status, "source status"),
            (self.window.context_hint, "workflow hint"),
            (self.window.remote_controller.connection_label, "remote connection"),
            (self.window.remote_controller.sequence_label, "remote sequence"),
            (self.window.remote_controller.command_label, "remote command"),
            (self.window.finger_controller.status_label, "finger status"),
            (self.window.ota_page.progress_message, "OTA progress"),
            (self.window.provision_page.status, "provision status"),
        ):
            if label.isVisible():
                self._assert_wrapped_label_fits(label, name)


if __name__ == "__main__":
    unittest.main()
