"""Unified desktop studio for local USB, LAN, WAN, fitting and 3-D stress.

The existing acquisition pages remain the single implementation of the local
laser/ADC workflows.  Their widgets are embedded here together with the
network viewer, the mechanical-finger view, and wireless provisioning so the
operator never needs to launch a second application.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
qt_platform_plugins = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "PyQt5"
    / "Qt5"
    / "plugins"
    / "platforms"
)
if qt_platform_plugins.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_platform_plugins)

from PyQt5 import QtCore, QtGui, QtWidgets

import app_JDSU as local_app
from app_version import APP_VERSION
from displacement_pressure_reference import from_machine as displacement_reference
from fbg_lan_serial import FbgLanSerial
from fbg_mqtt_transport import load_remote_config, validate_device_id
from fbg_network_transport import (
    NETWORK_LAN,
    NETWORK_WAN,
    FbgSelectableTransport,
)
from fbg_ota_widget import OtaUpdatePage
from fbg_remote_viewer import (
    MODE_STRESS,
    MODE_TEMPERATURE,
    DemoSource,
    RemoteViewer,
    TransportBridge,
    load_wave_tables,
)
from fbg_wifi_setup import ProvisionWindow
from fast_fullmap_qt import FastFullMapStreamWorker
from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from native_frame_recorder import NativeFrameRecorder
from hyperos_theme import (
    COLORS,
    install_theme,
    make_logo,
    make_nav_icon,
    polish_plots,
    set_status,
)
from hyperos_theme import (
    build_stylesheet as build_hyperos_stylesheet,
)
from mach3_controller import Mach3CommandError, Mach3Controller
from machine_profile import (
    MACHINE_LABEL,
    get_runtime_machine_id,
    runtime_machine_label,
    runtime_parameter_note,
    set_runtime_machine_id,
)
from mechanical_finger_3d import (
    MAX_PRESS_DEPTH_MM,
    MODEL_X_HALF_SPAN_MM,
    MODEL_Y_HALF_SPAN_MM,
    MechanicalFinger3DWindow,
)
from mode_table_manager import activate_pending_runtime_tables
from ota_machine_profiles import (
    MACHINE_LABELS,
    MACHINE_OPTIONS,
    OtaMachineProfileStore,
)
from responsive_layout import FlowLayout, ResponsiveScrollArea
from runtime_paths import output_path

SOURCE_LOCAL = "local"
SOURCE_NAMES = {
    SOURCE_LOCAL: "本地 USB",
    NETWORK_LAN: "局域网",
    NETWORK_WAN: "广域网",
}
LOCAL_MODE_NAMES = (
    "应力寻峰",
    "温度寻峰",
    "自动校准",
    "单值",
    "无限准确模式",
    "等间隔模式",
    "临时测试模式",
)

# Coverage snapshots are UI-decimated. Native USB/LAN frames are also recorded
# independently for offline time-based reconstruction; never train on UI cadence.
# These timers do not influence any CNC command or its safety sequencing.
COVERAGE_CAPTURE_POLL_MS = 10
COVERAGE_CONTACT_PHASE_MS = 70
COVERAGE_POST_RELEASE_MS = 260
COVERAGE_WRITER_QUEUE_CAPACITY = 512


_LEGACY_UNIFIED_STYLE = """
QMainWindow, QWidget {
    background-color: #08111f;
    color: #dce8f6;
    font-family: "Microsoft YaHei UI", "Microsoft YaHei";
    font-size: 13px;
}
QFrame#topBar {
    background-color: #0d192b;
    border-bottom: 1px solid #203653;
}
QLabel#brandTitle {
    color: #f2f8ff;
    font-size: 22px;
    font-weight: 600;
}
QLabel#brandSubtitle, QLabel#sectionCaption {
    color: #7790ad;
    font-size: 11px;
}
QLabel#sourceStatus {
    color: #9fd0ff;
    background-color: #10263d;
    border: 1px solid #234b6e;
    border-radius: 12px;
    padding: 5px 12px;
}
QFrame#controlGroup {
    background-color: #0a1525;
    border: 1px solid #1c304a;
    border-radius: 9px;
}
QPushButton {
    background-color: #12233a;
    color: #cbd9e8;
    border: 1px solid #29415f;
    border-radius: 6px;
    padding: 7px 13px;
}
QPushButton:hover {
    background-color: #193250;
    border-color: #3f73a5;
}
QPushButton:pressed {
    background-color: #0c75d8;
}
QPushButton:checked {
    color: white;
    background-color: #0878df;
    border-color: #50a9ff;
}
QPushButton:disabled {
    color: #536579;
    background-color: #0c1726;
    border-color: #17283b;
}
QTabWidget#workspaceTabs::pane {
    border: 1px solid #1c3048;
    background-color: #08111f;
    top: -1px;
}
QTabBar::tab {
    color: #8ea4bb;
    background-color: #0a1525;
    border: 1px solid #1c3048;
    border-bottom: none;
    min-width: 150px;
    padding: 10px 18px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    color: #eef7ff;
    background-color: #10233a;
    border-top: 2px solid #168cff;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    color: #e4edf7;
    background-color: #0c192a;
    border: 1px solid #2a415e;
    border-radius: 5px;
    padding: 5px 7px;
    selection-background-color: #0878df;
}
QPlainTextEdit {
    color: #c9d9e9;
    background-color: #07121f;
    border: 1px solid #223b58;
    border-radius: 6px;
    padding: 7px;
    font-family: Consolas, "Microsoft YaHei UI";
}
QProgressBar {
    color: #edf7ff;
    background-color: #0b1727;
    border: 1px solid #29435f;
    border-radius: 7px;
    min-height: 24px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #087de5;
    border-radius: 6px;
}
QComboBox QAbstractItemView {
    color: #e4edf7;
    background-color: #0d1a2b;
    selection-background-color: #145a91;
}
QCheckBox { spacing: 7px; }
QTableWidget, QTableView {
    color: #dbe8f5;
    background-color: #091524;
    alternate-background-color: #0d1c2e;
    border: 1px solid #1c314b;
    gridline-color: #1b3048;
}
QHeaderView::section {
    color: #a9c1d8;
    background-color: #102137;
    border: none;
    border-right: 1px solid #213a55;
    border-bottom: 1px solid #213a55;
    padding: 6px;
}
QMenuBar {
    color: #c8d8e8;
    background-color: #0b1626;
    border-bottom: 1px solid #1b2d44;
}
QMenuBar::item:selected, QMenu::item:selected { background-color: #17466f; }
QMenu { color: #dce8f5; background-color: #0e1c2e; border: 1px solid #29415d; }
QStatusBar { color: #7892ad; background-color: #08111f; }
QSplitter::handle { background-color: #1a2c43; }
QScrollBar:vertical { width: 10px; background: #08111f; }
QScrollBar::handle:vertical { min-height: 28px; background: #29445f; border-radius: 5px; }
QToolTip { color: #edf6ff; background-color: #13263c; border: 1px solid #39658c; }
"""

# Kept as a public module constant for compatibility with older launch scripts.
# The actual stylesheet now lives in one dedicated theme module.
UNIFIED_STYLE = build_hyperos_stylesheet()


PAGE_INFO = (
    (
        "光谱与寻峰",
        "实时查看 ADC 反射光谱、拟合曲线与中心波长",
        "spectrum",
    ),
    (
        "机械手指 3D 应力",
        "把当前光谱峰位映射为机械手指表面的空间应力",
        "finger",
    ),
    (
        "连接与设备配置",
        "通过 USB 配置板载 Wi-Fi、MQTT 与设备身份",
        "network",
    ),
    (
        "局域网 OTA 升级",
        "安全校验固件包并在同一局域网内更新板卡程序",
        "ota",
    ),
)


def _load_network_config(config_path: Path, device_override: str | None):
    """Load MQTT settings without making LAN or the main UI depend on them."""
    config = None
    error = None
    if config_path.is_file():
        try:
            config = load_remote_config(config_path)
            if device_override:
                config = replace(config, device_id=validate_device_id(device_override))
        except Exception as exc:
            error = str(exc)
    else:
        error = f"未找到 {config_path.name}"
    device_id = (
        config.device_id
        if config is not None
        else validate_device_id(device_override or "fbg-local")
    )
    return config, device_id, error


class Mach3EmergencyService(QtCore.QObject):
    """Run the command bridge off the GUI thread and publish verified state."""

    state_changed = QtCore.pyqtSignal(str, str)
    status_received = QtCore.pyqtSignal(dict)
    press_finished = QtCore.pyqtSignal(bool, str, dict)
    contact_reached = QtCore.pyqtSignal(dict, dict)
    release_started = QtCore.pyqtSignal(dict, dict)
    release_reached = QtCore.pyqtSignal(dict, dict)

    def __init__(self, parent=None, controller=None, status_controller=None):
        super().__init__(parent)
        self.controller = controller or Mach3Controller()
        # Keep coordinate telemetry independent from the command channel.
        # Otherwise a guarded press sequence holds ``_busy`` for the whole
        # Z-up/XY/Z-down/Z-up cycle and the rendered tool appears to teleport
        # only after the motion has finished.
        self.status_controller = status_controller or (
            self.controller if controller is not None else Mach3Controller()
        )
        self._busy = False
        self._status_busy = False
        self._closed = False
        self._stop_requested = threading.Event()
        self._status_error_count = 0
        self._status_sequence = 0
        self._status_sequence_lock = threading.Lock()

    def _reserve_status_sequence(self) -> int:
        with self._status_sequence_lock:
            self._status_sequence += 1
            return self._status_sequence

    def _publish_status(self, status: dict, *, sequence: int | None = None):
        """Publish one ordered telemetry frame and let the GUI reject stale data."""

        if sequence is None:
            sequence = self._reserve_status_sequence()
        snapshot = dict(status)
        snapshot["_telemetry_seq"] = sequence
        snapshot["_telemetry_monotonic_ns"] = time.monotonic_ns()
        self.status_received.emit(snapshot)

    def query_status(self):
        if self._status_busy or self._closed:
            return
        self._status_busy = True
        # Reserve ordering before the read starts.  If this read is delayed by
        # COM while a later command publishes its final coordinate first, the
        # GUI can identify and discard this older snapshot.
        sequence = self._reserve_status_sequence()

        def work():
            status = None
            try:
                status = self.status_controller.status()
                self._status_error_count = 0
                if status.get("estop") or status.get("estop_led"):
                    state, text = "stopped", "Mach3 在线 · 急停已锁定"
                elif status.get("moving"):
                    state, text = "warning", "Mach3 在线 · 轴正在运动"
                else:
                    state, text = "ready", "Mach3 在线 · 轴已停止"
            except Exception as exc:
                self._status_error_count += 1
                self.status_controller.close()
                state = "error"
                text = f"Mach3 暂未连接，正在自动重连：{exc}"
            finally:
                self._status_busy = False
            if not self._closed:
                if status is not None:
                    self._publish_status(status, sequence=sequence)
                self.state_changed.emit(state, text)

        threading.Thread(target=work, name="mach3-status", daemon=True).start()

    def emergency_stop(self):
        # Emergency stop deliberately has no confirmation dialog and uses an
        # independent bridge so it remains available during a point press.
        if self._closed:
            return
        self._stop_requested.set()
        self.state_changed.emit("busy", "正在下发停止并锁定急停…")

        def work():
            status = None
            controller = Mach3Controller()
            try:
                status = controller.emergency_stop()
                stopped = bool(status.get("stopped"))
                latched = bool(status.get("estop") or status.get("estop_led"))
                if not (stopped and latched):
                    raise Mach3CommandError("Mach3 未确认急停锁定")
                state, text = "stopped", "机床已停止 · 急停已锁定"
            except Exception as exc:
                state, text = "error", f"Mach3 急停链路不可用：{exc}"
            finally:
                controller.close()
            if not self._closed:
                if status is not None:
                    self._publish_status(status)
                self.state_changed.emit(state, text)

        threading.Thread(target=work, name="mach3-emergency", daemon=True).start()

    def release_control(self):
        """Release Reset/E-stop after the UI obtained an explicit safety confirmation."""

        if self._busy or self._closed:
            return
        self._busy = True
        self.state_changed.emit("busy", "正在确认停止状态并恢复机床控制…")

        def work():
            status = None
            try:
                result = self.controller.release_emergency_stop(operator_confirmed=True)
                status = dict(result.get("after") or self.controller.status())
                if status.get("estop") or status.get("estop_led"):
                    raise Mach3CommandError("Mach3仍处于Reset/急停状态")
                state, text = "ready", "Mach3 控制已恢复 · 主轴保持关闭"
            except Exception as exc:
                state, text = "error", f"恢复控制失败：{exc}"
            finally:
                self._busy = False
            if not self._closed:
                if status is not None:
                    self._publish_status(status)
                self.state_changed.emit(state, text)

        threading.Thread(target=work, name="mach3-release", daemon=True).start()

    def press_target(self, target: dict):
        """Run one guarded Z-up/XY/press/Z-up sequence off the GUI thread.

        Calibrated targets use machine-absolute XYZ as the physical source of
        truth. Mach3 work offsets may be re-zeroed between sessions; relative
        moves are therefore derived from the live machine DRO instead of
        rejecting an otherwise unchanged fixture.
        """

        if self._busy or self._closed:
            self.press_finished.emit(False, "机床控制正忙，未执行点选按压", {})
            return
        try:
            target_x = float(target["work_x"])
            target_y = float(target["work_y"])
            contact_z = float(target["contact_z"])
            depth_mm = float(target.get("depth_mm", 0.0))
            xy_feed = float(target.get("xy_feed_mm_min", 60.0))
            z_travel_feed = float(target.get("z_travel_feed_mm_min", 60.0))
            press_feed = float(target.get("press_feed_mm_min", 15.0))
            approach_clearance = float(target.get("approach_clearance_mm", 0.5))
            dwell_seconds = float(target.get("dwell_seconds", 0.35))
            expected_offset = target.get("calibration_machine_minus_work")
            if expected_offset is not None:
                expected_offset = tuple(float(value) for value in expected_offset)
                if len(expected_offset) != 3:
                    raise ValueError
            target_machine_x = target.get("target_machine_x")
            target_machine_y = target.get("target_machine_y")
            contact_machine_z = target.get("contact_machine_z")
            supplied_machine = (
                target_machine_x is not None
                or target_machine_y is not None
                or contact_machine_z is not None
            )
            if supplied_machine:
                if None in (target_machine_x, target_machine_y, contact_machine_z):
                    raise ValueError
                target_machine_x = float(target_machine_x)
                target_machine_y = float(target_machine_y)
                contact_machine_z = float(contact_machine_z)
            elif expected_offset is not None:
                target_machine_x = target_x + expected_offset[0]
                target_machine_y = target_y + expected_offset[1]
                contact_machine_z = contact_z + expected_offset[2]
        except (KeyError, TypeError, ValueError):
            self.press_finished.emit(False, "点选按压目标坐标无效", {})
            return
        if not all(
            math.isfinite(value)
            for value in (
                target_x,
                target_y,
                contact_z,
                depth_mm,
                xy_feed,
                z_travel_feed,
                press_feed,
                approach_clearance,
                dwell_seconds,
                *(expected_offset or ()),
                *(
                    (target_machine_x, target_machine_y, contact_machine_z)
                    if target_machine_x is not None
                    else ()
                ),
            )
        ):
            self.press_finished.emit(False, "点选按压目标坐标无效", {})
            return
        if not 0.0 <= depth_mm <= MAX_PRESS_DEPTH_MM:
            self.press_finished.emit(
                False,
                f"压入量必须位于0～{MAX_PRESS_DEPTH_MM:.2f} mm",
                {},
            )
            return
        if not 10.0 <= xy_feed <= 120.0:
            self.press_finished.emit(False, "XY速度必须位于10～120 mm/min", {})
            return
        if not 10.0 <= z_travel_feed <= 60.0:
            self.press_finished.emit(False, "Z定位速度必须位于10～60 mm/min", {})
            return
        if not 2.0 <= press_feed <= 30.0:
            self.press_finished.emit(False, "按压速度必须位于2～30 mm/min", {})
            return
        if not 0.1 <= approach_clearance <= 2.0:
            self.press_finished.emit(False, "接触前低速段必须位于0.1～2 mm", {})
            return
        if not 0.2 <= dwell_seconds <= 10.0:
            self.press_finished.emit(False, "接触保持时间必须位于0.2～10秒", {})
            return

        self._busy = True
        self._stop_requested.clear()
        self.state_changed.emit("busy", "3D点选按压执行中 · 连续轨迹 · 主轴禁用")

        def move_axis_to(axis: str, target_value: float, *, feed: float):
            status = self.controller.status()
            key = f"work_{axis.lower()}"
            delta = float(target_value) - float(status[key])
            if abs(delta) > 10.0:
                raise Mach3CommandError(f"{axis}轴目标超出单次点选按压安全范围")
            if abs(delta) <= 0.003:
                return status
            if self._stop_requested.is_set():
                raise Mach3CommandError("操作员已触发紧急停止")
            result = self.controller.move_relative_continuous_axis(
                axis,
                delta,
                feed_mm_min=feed,
                safe_zone_confirmed=True,
            )
            status = dict(result.get("after") or self.controller.status())
            return status

        def move_xy_to(target_x_value: float, target_y_value: float):
            status = self.controller.status()
            delta_x = target_x_value - float(status["work_x"])
            delta_y = target_y_value - float(status["work_y"])
            if max(abs(delta_x), abs(delta_y)) > 30.0:
                raise Mach3CommandError("XY目标超出单次点选按压安全范围")
            if max(abs(delta_x), abs(delta_y)) <= 0.003:
                return status
            if self._stop_requested.is_set():
                raise Mach3CommandError("操作员已触发紧急停止")
            result = self.controller.move_absolute_xy_continuous(
                target_x_value,
                target_y_value,
                feed_mm_min=xy_feed,
                safe_zone_confirmed=True,
            )
            return dict(result.get("after") or self.controller.status())

        def move_machine_axis_to(axis: str, target_value: float, *, feed: float):
            status = self.controller.status()
            key = f"machine_{axis.lower()}"
            if key not in status:
                raise Mach3CommandError("机床绝对坐标不可用，未执行按压")
            delta = float(target_value) - float(status[key])
            if abs(delta) > 10.0:
                raise Mach3CommandError(f"{axis}轴目标超出单次点选按压安全范围")
            if abs(delta) <= 0.003:
                return status
            if self._stop_requested.is_set():
                raise Mach3CommandError("操作员已触发紧急停止")
            result = self.controller.move_relative_continuous_axis(
                axis,
                delta,
                feed_mm_min=feed,
                safe_zone_confirmed=True,
            )
            return dict(result.get("after") or self.controller.status())

        def move_machine_xy_to(target_x_value: float, target_y_value: float):
            status = self.controller.status()
            try:
                delta_x = float(target_x_value) - float(status["machine_x"])
                delta_y = float(target_y_value) - float(status["machine_y"])
                work_target_x = float(status["work_x"]) + delta_x
                work_target_y = float(status["work_y"]) + delta_y
            except (KeyError, TypeError, ValueError) as exc:
                raise Mach3CommandError(
                    "机床绝对坐标不可用，未执行按压"
                ) from exc
            if max(abs(delta_x), abs(delta_y)) > 30.0:
                raise Mach3CommandError("XY目标超出单次点选按压安全范围")
            if max(abs(delta_x), abs(delta_y)) <= 0.003:
                return status
            if self._stop_requested.is_set():
                raise Mach3CommandError("操作员已触发紧急停止")
            result = self.controller.move_absolute_xy_continuous(
                work_target_x,
                work_target_y,
                feed_mm_min=xy_feed,
                safe_zone_confirmed=True,
            )
            return dict(result.get("after") or self.controller.status())

        def work():
            status = {}
            success = False
            try:
                status = self.controller.status()
                if status.get("estop") or status.get("estop_led"):
                    raise Mach3CommandError("Mach3处于Reset/急停状态")
                if status.get("moving") and not status.get("stopped"):
                    raise Mach3CommandError("机床仍在运动")
                if not all(
                    status.get(key) for key in ("x_homed", "y_homed", "z_homed")
                ):
                    raise Mach3CommandError("X/Y/Z尚未全部回零")
                resolved_target = dict(target)
                if target_machine_x is not None:
                    safe_machine_z = contact_machine_z + 4.0
                    if float(status["machine_z"]) < safe_machine_z - 0.003:
                        status = move_machine_axis_to(
                            "Z", safe_machine_z, feed=z_travel_feed
                        )
                    status = move_machine_xy_to(
                        target_machine_x, target_machine_y
                    )
                    approach_machine_z = contact_machine_z + approach_clearance
                    status = move_machine_axis_to(
                        "Z", approach_machine_z, feed=z_travel_feed
                    )
                    status = move_machine_axis_to(
                        "Z", contact_machine_z - depth_mm, feed=press_feed
                    )
                    resolved_target.update(
                        {
                            "target_machine_x": target_machine_x,
                            "target_machine_y": target_machine_y,
                            "contact_machine_z": contact_machine_z,
                            "work_x": float(status["work_x"]),
                            "work_y": float(status["work_y"]),
                            "contact_z": float(status["work_z"]) + depth_mm,
                            "coordinate_resolution": "machine_absolute",
                        }
                    )
                else:
                    safe_z = contact_z + 4.0
                    if float(status["work_z"]) < safe_z - 0.003:
                        status = move_axis_to("Z", safe_z, feed=z_travel_feed)
                    status = move_xy_to(target_x, target_y)
                    approach_z = contact_z + approach_clearance
                    status = move_axis_to("Z", approach_z, feed=z_travel_feed)
                    status = move_axis_to(
                        "Z", contact_z - depth_mm, feed=press_feed
                    )
                if not self._closed:
                    self.contact_reached.emit(dict(resolved_target), dict(status))
                time.sleep(dwell_seconds)
                if not self._closed:
                    self.release_started.emit(dict(resolved_target), dict(status))
                if target_machine_x is not None:
                    status = move_machine_axis_to(
                        "Z", safe_machine_z, feed=z_travel_feed
                    )
                else:
                    status = move_axis_to("Z", safe_z, feed=z_travel_feed)
                if not self._closed:
                    self.release_reached.emit(dict(resolved_target), dict(status))
                success = True
                target_text = (
                    f"机床X {target_machine_x:+.3f} Y {target_machine_y:+.3f}"
                    if target_machine_x is not None
                    else f"X {target_x:+.3f} Y {target_y:+.3f}"
                )
                text = (
                    f"点选按压完成：{target_text}，压入 {depth_mm:.2f} mm，"
                    f"XY {xy_feed:.0f} mm/min，已自动抬升"
                )
            except Exception as exc:
                text = f"点选按压中止：{exc}"
                try:
                    status = self.controller.status()
                except Exception:
                    status = {}
            finally:
                self._busy = False
            if not self._closed:
                if status:
                    self._publish_status(status)
                self.state_changed.emit("ready" if success else "error", text)
                self.press_finished.emit(success, text, dict(status))

        threading.Thread(target=work, name="mach3-click-press", daemon=True).start()

    def close(self):
        self._closed = True
        self.controller.close()
        if self.status_controller is not self.controller:
            self.status_controller.close()


class UnifiedFbgStudio(QtWidgets.QMainWindow):
    """One-window host for every operator-facing workflow."""

    lan_connection_changed = QtCore.pyqtSignal(bool, str)

    def __init__(
        self,
        *,
        initial_source: str = SOURCE_LOCAL,
        config_path: Path = APP_DIR / "fbg_remote_config.yaml",
        device_id: str | None = None,
        demo: bool = False,
        ota_settings_store=None,
    ):
        super().__init__()
        self.demo = bool(demo)
        self.current_source: str | None = None
        self.network_config_error = None
        self._closing = False
        self._sweep_close_pending = False
        self._fast_flank_close_pending = False
        self._fast_flank_worker: FastFullMapStreamWorker | None = None
        self._fast_flank_output_path: Path | None = None
        self._latest_fast_flank_result = None
        self._fast_flank_results_received = 0
        self._fast_flank_results_rendered = 0
        self._fast_flank_render_overwrites = 0
        self._fast_flank_ui_timer = QtCore.QTimer(self)
        self._fast_flank_ui_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._fast_flank_ui_timer.setInterval(33)
        self._fast_flank_ui_timer.timeout.connect(
            self._render_latest_fast_flank_result
        )
        self._last_mach3_status = {}
        self._last_mach3_status_received_monotonic_ns = None
        self._last_mach3_sequence = -1
        self._coverage_active = False
        self._coverage_cancel_after_current = False
        self._coverage_stop_reason = None
        self._coverage_queue: list[dict] = []
        self._coverage_total = 0
        self._coverage_completed = 0
        self._coverage_current: dict | None = None
        self._coverage_records = 0
        self._coverage_dataset_path: Path | None = None
        self._coverage_motion_test = False
        self._coverage_writer: FingerSessionWriter | None = None
        self._coverage_native_recorder = None
        self._coverage_native_stats = None
        self._coverage_phase: str | None = None
        self._coverage_phase_event_id: str | None = None
        self._coverage_writer_sequence = 0
        self._coverage_last_frame_key: tuple | None = None
        self._coverage_phase_frame_counts: dict[str, int] = {}
        self._coverage_completion_pending = False
        self._remote_telemetry_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self.ota_settings_store = ota_settings_store or OtaMachineProfileStore()
        initial_machine_id = self.ota_settings_store.selected_machine_id
        set_runtime_machine_id(initial_machine_id)
        self.mach3_emergency = Mach3EmergencyService(self)
        self._coverage_capture_timer = QtCore.QTimer(self)
        self._coverage_capture_timer.setInterval(COVERAGE_CAPTURE_POLL_MS)
        self._coverage_capture_timer.timeout.connect(self._capture_coverage_phase_frame)

        self.setWindowTitle(
            f"FBG OptiSense Studio {APP_VERSION} — "
            f"{runtime_machine_label(initial_machine_id)} · 光栅解调与机械手指应力定位"
        )
        self.setWindowIcon(QtGui.QIcon(make_logo(128)))
        self.resize(1640, 980)
        # Every embedded page now reflows or scrolls, so the studio remains
        # usable on 1366×768 screens and with Windows display scaling enabled.
        self.setMinimumSize(960, 620)

        self.local_controller = local_app.MainWindow(auto_acquire_local=False)
        self.local_controller.page_temporary_test.demo = self.demo
        self.local_page = self.local_controller.takeCentralWidget()
        self.setMenuBar(self.local_controller.menuBar())
        # Keep the native CDC object and the LAN byte-stream adapter side by
        # side.  app_JDSU always talks to its module-level ``ser`` object, so
        # selecting a source only swaps that one endpoint; every page, button,
        # acquisition worker and plot remains literally the USB implementation.
        self.usb_serial = local_app.ser
        self.usb_port = self.local_controller.port
        self.lan_serial = FbgLanSerial(on_connection=self.lan_connection_changed.emit)

        self.bridge = TransportBridge()
        if self.demo:
            wave_tables = load_wave_tables(APP_DIR / "wave_const.yaml")
            self.transport = DemoSource(wave_tables, self.bridge)
            network_device_id = device_id or "demo-jdsu"
            self.network_config = None
        else:
            (
                self.network_config,
                network_device_id,
                self.network_config_error,
            ) = _load_network_config(Path(config_path), device_id)
            self.transport = FbgSelectableTransport(
                self.network_config,
                initial_mode=NETWORK_LAN,
                on_ack=self.bridge.ack.emit,
                on_status=self.bridge.status.emit,
                on_connection=self.bridge.connection.emit,
                on_metadata=self.bridge.metadata.emit,
            )
            wave_tables = {}

        self.remote_controller = RemoteViewer(
            wave_tables,
            self.bridge,
            self.transport,
            network_device_id,
        )
        self.bridge.telemetry.connect(self._remember_remote_telemetry)
        self.remote_controller.network_panel.hide()
        self.remote_page = self.remote_controller.takeCentralWidget()

        self.finger_controller = MechanicalFinger3DWindow()
        self.finger_page = self.finger_controller.takeCentralWidget()
        self._connect_shared_finger_view()
        self.mach3_position_timer = QtCore.QTimer(self)
        self.mach3_position_timer.setInterval(100)
        self.mach3_position_timer.timeout.connect(self.mach3_emergency.query_status)

        self.provision_page = ProvisionWindow()
        self.provision_page.before_provisioning = self._prepare_provisioning
        self._prefill_provisioning_fields()

        self.ota_page = OtaUpdatePage(settings_store=self.ota_settings_store)
        self.ota_page.before_update = self._prepare_ota
        self.ota_page.after_update = self._after_ota_update
        self._generated_mode_table_package_path = None
        self.local_controller.page_equal_interval.ota_package_ready = (
            self._open_generated_ota_package
        )

        self._build_shell()
        self._connect_signals()
        if self.demo:
            self.finger_controller.update_machine_status(
                {
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
                    "simulated": True,
                }
            )
        self._sync_local_mode()
        QtCore.QTimer.singleShot(
            0, lambda: self.switch_source(initial_source, force=True)
        )

    def _connect_shared_finger_view(self):
        stress_page = self.local_controller.page_peak
        stress_page.finger_3d_window = self.finger_controller
        stress_page.finger_3d_open_handler = self.open_finger_tab
        self.remote_controller.finger_3d_window = self.finger_controller
        self.remote_controller.finger_3d_open_handler = self.open_finger_tab
        self.finger_controller.press_target_requested.connect(
            self._finger_press_requested
        )
        self.finger_controller.coverage_run_requested.connect(
            self._finger_coverage_requested
        )
        self.finger_controller.coverage_stop_requested.connect(
            self._finger_coverage_stop_requested
        )
        self.finger_controller.fast_flank_start_requested.connect(
            self._start_fast_flank_stream
        )
        self.finger_controller.fast_flank_stop_requested.connect(
            self._stop_fast_flank_stream
        )
        self.mach3_emergency.contact_reached.connect(self._finger_contact_reached)
        self.mach3_emergency.release_started.connect(self._finger_release_started)
        self.mach3_emergency.release_reached.connect(self._finger_release_reached)
        self.mach3_emergency.press_finished.connect(self._finger_press_finished)

    @QtCore.pyqtSlot()
    def _start_fast_flank_stream(self):
        """Start one bounded, all-fresh F45 optical test; never command the CNC."""

        if self._fast_flank_worker is not None:
            return
        if self.demo:
            self.finger_controller.finish_fast_flank(
                "演示模式不会启动真实激光器", success=False
            )
            return
        if self.current_source != SOURCE_LOCAL:
            self.finger_controller.finish_fast_flank(
                "当前测试版仅允许本地USB直采；请切换到USB本地", success=False
            )
            return
        if self._coverage_active:
            self.finger_controller.finish_fast_flank(
                "硅胶覆盖采集正在运行，未占用光谱链路", success=False
            )
            return
        if not local_app.switch_mode_enable or local_app.ser_open:
            self.finger_controller.finish_fast_flank(
                "请先停止当前光谱模式，再启动45点快速定位测试", success=False
            )
            return
        if local_app.ser is not self.usb_serial or not self.usb_serial.is_open:
            self.finger_controller.finish_fast_flank(
                "本地USB没有处于可用状态，未启动激光器", success=False
            )
            return

        output_dir = output_path()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = output_dir / f"ch1_fast_fullmap_ui_{stamp}.json"
        suffix = 1
        while output.exists() or output.with_suffix(".png").exists():
            output = output_dir / f"ch1_fast_fullmap_ui_{stamp}_{suffix}.json"
            suffix += 1

        worker = FastFullMapStreamWorker(
            self.usb_serial,
            output,
            cycles=60_000,
            spacing_us=50,
            parent=self,
        )
        worker.result_ready.connect(self._fast_flank_result_ready)
        worker.progress.connect(self.finger_controller.set_fast_flank_progress)
        worker.completed.connect(self._fast_flank_completed)
        worker.cancelled.connect(self._fast_flank_cancelled)
        worker.failed.connect(self._fast_flank_failed)
        worker.finished.connect(self._fast_flank_thread_finished)
        self._fast_flank_worker = worker
        self._fast_flank_output_path = output
        self._latest_fast_flank_result = None
        self._fast_flank_results_received = 0
        self._fast_flank_results_rendered = 0
        self._fast_flank_render_overwrites = 0
        self._fast_flank_ui_timer.start()
        local_app.switch_mode_enable = False
        self.finger_controller.set_fast_flank_running(True)
        self.statusBar().showMessage(
            "CH1全新45点定位已启动；前64个有效帧请保持无按压"
        )
        worker.start()

    @QtCore.pyqtSlot()
    def _stop_fast_flank_stream(self):
        worker = self._fast_flank_worker
        if worker is None:
            return
        self.finger_controller.fast_flank_stop_button.setEnabled(False)
        self.finger_controller.fast_flank_status_label.setText(
            "正在中止板卡流并核验DAC停光、SOA关光…"
        )
        worker.stop()

    @QtCore.pyqtSlot(object)
    def _fast_flank_result_ready(self, result):
        if self._fast_flank_worker is None:
            return
        self._fast_flank_results_received += 1
        if self._latest_fast_flank_result is not None:
            self._fast_flank_render_overwrites += 1
        self._latest_fast_flank_result = result

    @QtCore.pyqtSlot()
    def _render_latest_fast_flank_result(self):
        """Render only the newest inference result; never queue stale 3D work."""

        result = self._latest_fast_flank_result
        if result is None:
            return
        self._latest_fast_flank_result = None
        self.finger_controller.update_from_fast_flank_result(result)
        self._fast_flank_results_rendered += 1

    @QtCore.pyqtSlot(object)
    def _fast_flank_completed(self, report):
        analysis = dict(report.get("analysis", {})) if isinstance(report, dict) else {}
        rate = analysis.get("board_rate_hz")
        gate = bool(analysis.get("meets_sampling_rate_for_15hz", False))
        rate_text = "—" if rate is None else f"{float(rate):.2f} Hz"
        self._finish_fast_flank_ui(
            f"采集完成：{rate_text}；15 Hz采样门槛{'通过' if gate else '未通过'}；已确认关光",
            success=gate,
        )

    @QtCore.pyqtSlot(str)
    def _fast_flank_cancelled(self, _output: str):
        self._finish_fast_flank_ui("已安全停止，DAC停光与SOA关光已确认", success=True)

    @QtCore.pyqtSlot(str)
    def _fast_flank_failed(self, error: str):
        self._finish_fast_flank_ui(
            f"快速采集失败：{error}",
            success=False,
        )

    def _finish_fast_flank_ui(self, message: str, *, success: bool):
        self._render_latest_fast_flank_result()
        self._fast_flank_ui_timer.stop()
        if self._fast_flank_results_received:
            message += (
                f"；3D显示{self._fast_flank_results_rendered}/"
                f"{self._fast_flank_results_received}帧（最新帧优先）"
            )
        local_app.switch_mode_enable = True
        self.finger_controller.finish_fast_flank(message, success=success)
        self.statusBar().showMessage(message)

    @QtCore.pyqtSlot()
    def _fast_flank_thread_finished(self):
        worker = self._fast_flank_worker
        self._fast_flank_worker = None
        if worker is not None:
            worker.deleteLater()

    @QtCore.pyqtSlot(dict)
    def _finger_press_requested(self, target: dict):
        if self._coverage_active:
            self.finger_controller.click_press_status_label.setText(
                "3D点选按压：覆盖采集正在运行，单点操作已锁定"
            )
            return
        if self.demo:
            self.finger_controller.click_press_status_label.setText(
                "3D点选按压：演示模式不会控制机床"
            )
            return
        status = self._last_mach3_status
        if not status:
            self.finger_controller.click_press_status_label.setText(
                "3D点选按压：尚未取得Mach3实时坐标，未执行"
            )
            return
        if status.get("estop") or status.get("estop_led"):
            self.finger_controller.click_press_status_label.setText(
                "3D点选按压：Mach3处于Reset/急停状态，未执行"
            )
            return
        self.finger_controller.click_press_arm.setEnabled(False)
        self.finger_controller.click_press_status_label.setText(
            "3D点选按压：正在安全抬升并定位…"
        )
        self.mach3_emergency.press_target(dict(target))

    @QtCore.pyqtSlot(dict)
    def _finger_coverage_requested(self, request: dict):
        if self._coverage_active:
            return
        points = [dict(point) for point in request.get("points", ())]
        motion_test = bool(request.get("motion_test", False))
        repeats = 1 if motion_test else int(request.get("repeats", 0))
        try:
            requested_depth = (
                0.0 if motion_test else float(request.get("depth_mm", 0.0))
            )
        except (TypeError, ValueError):
            requested_depth = math.nan
        if not points or repeats < 1:
            self.finger_controller.coverage_summary_label.setText(
                "覆盖方案为空，未执行"
            )
            return
        if (
            not math.isfinite(requested_depth)
            or not 0.0 <= requested_depth <= MAX_PRESS_DEPTH_MM
        ):
            self.finger_controller.coverage_summary_label.setText(
                f"覆盖采集压入量必须位于0.00～{MAX_PRESS_DEPTH_MM:.2f} mm，未执行"
            )
            return
        if self.demo:
            self.finger_controller.coverage_summary_label.setText(
                f"演示模式：已预览{len(points)}个位置，不会控制机床"
            )
            return
        status = dict(self._last_mach3_status)
        if not status:
            self.finger_controller.coverage_summary_label.setText(
                "等待Mach3实时坐标，未启动覆盖采集"
            )
            return
        if status.get("estop") or status.get("estop_led"):
            self.finger_controller.coverage_summary_label.setText(
                "Mach3处于Reset/急停状态，未启动覆盖采集"
            )
            return
        if not status.get("stopped"):
            self.finger_controller.coverage_summary_label.setText(
                "机床尚未确认停止，未启动覆盖采集"
            )
            return
        if not all(status.get(key) for key in ("x_homed", "y_homed", "z_homed")):
            self.finger_controller.coverage_summary_label.setText(
                "X/Y/Z尚未全部回零，未启动覆盖采集"
            )
            return
        if not motion_test:
            spectrum_ready = self._ensure_finger_spectrum_ready_for_coverage()
            if not spectrum_ready:
                self.finger_controller.coverage_summary_label.setText(
                    "覆盖方案已准备；等待九个光栅峰全部在线后才能开始"
                )
                return

        total = len(points) * repeats
        dialog_title = "确认31点动作测试" if motion_test else "确认硅胶全覆盖采集"
        run_description = (
            f"将依次测试 {len(points)} 个接触位置，共1轮。\n"
            "本次只验证运动与3D同步，不采集或保存光谱。\n"
            if motion_test
            else f"将执行 {len(points)} 个不同位置 × {repeats} 轮 = {total} 次接触。\n"
        )
        answer = QtWidgets.QMessageBox.warning(
            self,
            dialog_title,
            run_description
            + (
                "动作测试压入量固定为 0.00 mm，每次只到标定接触面。\n\n"
                if motion_test
                else (
                    f"本轮压入量为 {requested_depth:.2f} mm"
                    f"（安全上限{MAX_PRESS_DEPTH_MM:.2f} mm）。\n\n"
                )
            )
            + "请确认：\n"
            "• 手指、线缆和整个XY/Z路径无障碍；\n"
            "• 8 mm压面与手指顶面保持平行；\n"
            "• 光谱正在连续更新，物理急停随手可按；\n"
            "• 主轴保持关闭（程序每次定位前强制M5）。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        use_machine_route = all(
            key in status
            and points[0].get(target_key) is not None
            and points[-1].get(target_key) is not None
            for key, target_key in (
                ("machine_x", "target_machine_x"),
                ("machine_y", "target_machine_y"),
            )
        )
        if use_machine_route:
            current_xy = (
                float(status["machine_x"]),
                float(status["machine_y"]),
            )
            route_keys = ("target_machine_x", "target_machine_y")
        else:
            current_xy = (
                float(status.get("work_x", 0.0)),
                float(status.get("work_y", 0.0)),
            )
            route_keys = ("work_x", "work_y")
        first_distance = math.hypot(
            float(points[0][route_keys[0]]) - current_xy[0],
            float(points[0][route_keys[1]]) - current_xy[1],
        )
        last_distance = math.hypot(
            float(points[-1][route_keys[0]]) - current_xy[0],
            float(points[-1][route_keys[1]]) - current_xy[1],
        )
        first_route = (
            points if first_distance <= last_distance else list(reversed(points))
        )
        queue = []
        for repetition in range(1, repeats + 1):
            route = first_route if repetition % 2 else list(reversed(first_route))
            for point in route:
                target = dict(point)
                target.update(
                    {
                        "depth_mm": requested_depth,
                        "xy_feed_mm_min": float(request["xy_feed_mm_min"]),
                        "z_travel_feed_mm_min": float(request["z_travel_feed_mm_min"]),
                        "press_feed_mm_min": float(request["press_feed_mm_min"]),
                        "approach_clearance_mm": 0.5,
                        "dwell_seconds": float(request["dwell_seconds"]),
                        "calibration_machine_minus_work": request.get(
                            "calibration_machine_minus_work"
                        ),
                        "coverage_repetition": repetition,
                    }
                )
                queue.append(target)

        self._coverage_queue = queue
        self._coverage_total = len(queue)
        self._coverage_completed = 0
        self._coverage_current = None
        self._coverage_cancel_after_current = False
        self._coverage_records = 0
        self._coverage_motion_test = motion_test
        self._coverage_dataset_path = None
        self._coverage_writer = None
        self._coverage_phase = None
        self._coverage_phase_event_id = None
        self._coverage_writer_sequence = 0
        self._coverage_last_frame_key = None
        self._coverage_phase_frame_counts.clear()
        self._coverage_completion_pending = False
        if not motion_test:
            output_dir = output_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            session_stem = "finger_coverage_" + time.strftime("%Y%m%d_%H%M%S")
            session_dir = output_dir / session_stem
            suffix = 1
            while session_dir.exists():
                session_dir = output_dir / f"{session_stem}_{suffix:02d}"
                suffix += 1
            try:
                metadata = {
                    "training_eligible": False,
                    "training_exclusion_reason": "Unreviewed geometric-contact acquisition; native timing/table identity and physical labels need qualification before training.",
                    "native_capture": bool(self._uses_local_pages()),
                    "ui_frames_are_decimated": True,
                    "unique_positions": len(points),
                    "repeats": repeats,
                    "total_contacts": len(queue),
                    "spacing_mm": float(request.get("spacing_mm", 2.0)),
                    "press_face_diameter_mm": 8.0,
                    "depth_mm": requested_depth,
                    "dwell_seconds": float(request["dwell_seconds"]),
                    "source_at_start": self.current_source,
                    "capture_phases": (
                        "pre_contact",
                        "contact",
                        "hold",
                        "release",
                        "release_complete",
                    ),
                    "contact_label_basis": "calibrated_z_plane_and_motion_phase_not_force_sensor",
                    "adc_event_synchronization_verified": False,
                }
                self._coverage_writer = FingerSessionWriter(
                    session_dir,
                    queue_capacity=COVERAGE_WRITER_QUEUE_CAPACITY,
                    manifest_extra=metadata,
                )
                self._coverage_dataset_path = session_dir
                self._start_coverage_native_capture()
                self._coverage_writer.write_event(
                    ContactEvent(kind="session_start", metadata=metadata)
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if self._coverage_native_recorder is not None:
                    try:
                        self._coverage_native_recorder.close()
                    except Exception:
                        pass
                    self._coverage_native_recorder = None
                if self._coverage_writer is not None:
                    try:
                        self._coverage_writer.close()
                    except Exception as close_exc:
                        exc = RuntimeError(f"{exc}; 数据关闭失败：{close_exc}")
                self._coverage_dataset_path = None
                self._coverage_writer = None
                self.finger_controller.coverage_summary_label.setText(
                    f"无法建立覆盖采集数据文件，未启动：{exc}"
                )
                return
        self._coverage_active = True
        self.finger_controller.set_coverage_running(True)
        self.finger_controller.set_coverage_progress(
            0,
            self._coverage_total,
            detail=(
                "动作测试：准备移动到第1个接触点（不采光谱）"
                if motion_test
                else "准备移动到第1个接触点"
            ),
        )
        if not motion_test:
            self._coverage_capture_timer.start()
        QtCore.QTimer.singleShot(0, self._run_next_coverage_press)

    def _ensure_finger_spectrum_ready_for_coverage(self) -> bool:
        """Synchronize the 3-D nine-peak gate from a valid live CH1 frame.

        The stress plot may receive a valid sparse 45-slot frame before its
        slower fitted-peak callback updates the 3-D page.  Coverage records the
        raw ADC codes, so a conservative per-segment discrete maximum is enough
        to establish that all nine FBG windows are present.  This never replaces
        the saved raw values or the later fitting algorithm.
        """

        finger = self.finger_controller
        if finger.frame_number > 0 and all(
            math.isfinite(float(value)) for value in finger.current_sensor_peaks
        ):
            return True
        snapshot = self._active_spectrum_snapshot()
        try:
            codes = np.asarray(snapshot["raw_ch1_adc_codes"], dtype=float).reshape(9, 5)
            waves = np.asarray(snapshot["wavelengths_nm"], dtype=float).reshape(9, 5)
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not np.isfinite(codes).all()
            or not np.isfinite(waves).all()
            or np.any(np.max(codes, axis=1) < 12)
            or np.any(codes >= 4090)
        ):
            return False
        centres = waves[np.arange(9), np.argmax(codes, axis=1)]
        peaks_by_channel = [
            finger.current_peaks_by_channel[channel].copy() for channel in range(4)
        ]
        peaks_by_channel[1] = centres
        finger.channel_combo.setCurrentIndex(1)
        finger.update_from_spectrum(
            peaks_by_channel,
            int(snapshot.get("frame_number") or 1),
        )
        return all(math.isfinite(float(value)) for value in finger.current_sensor_peaks)

    def _run_next_coverage_press(self):
        if not self._coverage_active:
            return
        if self._coverage_cancel_after_current:
            self._finish_coverage_run(False, "操作员要求停止，未开始下一点")
            return
        if not self._coverage_queue:
            self._finish_coverage_run(True, "全部接触点已完成")
            return
        self._coverage_current = self._coverage_queue.pop(0)
        point_index = int(self._coverage_current["point_index"]) - 1
        repetition = int(self._coverage_current["coverage_repetition"])
        self.finger_controller.set_coverage_progress(
            self._coverage_completed,
            self._coverage_total,
            active_point_index=point_index,
            detail=f"第{repetition}轮 · 正在定位位置{point_index + 1}",
        )
        if not self._coverage_motion_test:
            self._begin_coverage_phase(
                "pre_contact",
                self._coverage_current,
                dict(self._last_mach3_status),
            )
        self.mach3_emergency.press_target(dict(self._coverage_current))

    @QtCore.pyqtSlot()
    def _finger_coverage_stop_requested(self):
        if not self._coverage_active:
            return
        self._coverage_cancel_after_current = True
        if not getattr(self, "_coverage_stop_reason", None):
            self._coverage_stop_reason = "operator_stop_button"
        self.finger_controller.coverage_summary_label.setText(
            "已请求停止：完成当前接触并抬升后不再前往下一点；需要立即停止请按机床急停"
        )

    @QtCore.pyqtSlot(dict, dict)
    def _finger_contact_reached(self, target: dict, status: dict):
        if not self._coverage_active or self._coverage_current is None:
            return
        point_index = int(target.get("point_index", 1)) - 1
        repetition = int(target.get("coverage_repetition", 1))
        if self._coverage_motion_test:
            self.finger_controller.set_coverage_progress(
                self._coverage_completed,
                self._coverage_total,
                active_point_index=point_index,
                detail=(f"动作测试 · 位置{point_index + 1}已接触，不采集光谱"),
            )
            return
        self.finger_controller.set_coverage_progress(
            self._coverage_completed,
            self._coverage_total,
            active_point_index=point_index,
            detail=f"第{repetition}轮 · 位置{point_index + 1}已接触，正在等待光谱帧",
        )
        self._begin_coverage_phase("contact", target, status)
        token = (
            repetition,
            int(target.get("point_index", 1)),
            self._coverage_completed,
        )
        QtCore.QTimer.singleShot(
            COVERAGE_CONTACT_PHASE_MS,
            lambda value=token, point=dict(target), snapshot=dict(status): (
                self._begin_coverage_hold(value, point, snapshot)
            ),
        )

    def _begin_coverage_hold(self, token: tuple, target: dict, status: dict):
        if self._coverage_token() != token or self._coverage_phase != "contact":
            return
        self._begin_coverage_phase("hold", target, status)

    @QtCore.pyqtSlot(dict, dict)
    def _finger_release_started(self, target: dict, status: dict):
        if (
            not self._coverage_active
            or self._coverage_motion_test
            or self._coverage_current is None
        ):
            return
        self._begin_coverage_phase("release", target, status)
        self.finger_controller.set_coverage_progress(
            self._coverage_completed,
            self._coverage_total,
            active_point_index=int(target.get("point_index", 1)) - 1,
            detail="接触保持完成，正在安全释放并继续记录多帧光谱",
        )

    @QtCore.pyqtSlot(dict, dict)
    def _finger_release_reached(self, target: dict, status: dict):
        if (
            not self._coverage_active
            or self._coverage_motion_test
            or self._coverage_current is None
        ):
            return
        self._begin_coverage_phase("release_complete", target, status)

    def _coverage_token(self) -> tuple | None:
        if self._coverage_current is None:
            return None
        return (
            int(self._coverage_current.get("coverage_repetition", 1)),
            int(self._coverage_current.get("point_index", 1)),
            self._coverage_completed,
        )

    @QtCore.pyqtSlot(object)
    def _remember_remote_telemetry(self, frame):
        """Retain the decoded packet so coverage uses codes, not re-quantized volts."""

        try:
            key = (int(frame.boot_id), int(frame.seq))
        except (AttributeError, TypeError, ValueError):
            return
        self._remote_telemetry_cache[key] = (frame, time.monotonic_ns())
        self._remote_telemetry_cache.move_to_end(key)
        while len(self._remote_telemetry_cache) > 16:
            self._remote_telemetry_cache.popitem(last=False)

    @staticmethod
    def _finite_list(values) -> list[float | None]:
        result = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            result.append(numeric if math.isfinite(numeric) else None)
        return result

    def _active_spectrum_snapshot(self) -> dict:
        captured_monotonic_ns = time.monotonic_ns()
        quality_flags = []
        if self._uses_local_pages():
            page = self.local_controller.page_peak
            raw_channels = [
                self._finite_list(channel) for channel in getattr(page, "adc", ())
            ]
            analysis_channels = [
                self._finite_list(channel) for channel in getattr(page, "ori_filts", ())
            ]
            wavelengths = self._finite_list(getattr(page, "wave_const", ()))
            frame_number = int(getattr(page, "frame_generation", 0))
            raw_code_channels = getattr(page, "data", ())
            try:
                raw_ch1_codes = [int(value) for value in raw_code_channels[1]]
            except (IndexError, TypeError, ValueError):
                raw_ch1_codes = []
                quality_flags.append("invalid_raw_ch1")
            selectors = list(getattr(page, "feedback_selectors", ()))
            selector = selectors[1] if len(selectors) > 1 else None
            try:
                selector = int(selector)
                transimpedance_ohm = float(
                    local_app.PD_FEEDBACK_KOHM_BY_SELECTOR[selector] * 1000.0
                )
            except (IndexError, TypeError, ValueError):
                selector = None
                transimpedance_ohm = None
            gain_masks = list(getattr(page, "feedback_gain_masks", ()))
            gain_mask = int(gain_masks[1]) if len(gain_masks) > 1 else None
            digital_gains = list(getattr(page, "voltage_scalars", ()))
            digital_gain = float(digital_gains[1]) if len(digital_gains) > 1 else None
            device_sequence = getattr(page, "board_frame_sequence", None)
            device_boot_id = getattr(page, "board_boot_id", None)
            device_uptime_ms = getattr(page, "board_uptime_ms", None)
            table_crc32 = getattr(page, "board_table_crc32", None)
            packet_crc_ok = getattr(page, "packet_crc_ok", None)
            frame_received_monotonic_ns = getattr(
                page, "last_frame_received_monotonic_ns", None
            )
            acquisition_profile = getattr(page, "acquisition_profile", None)
            schedule_version = getattr(page, "multirate_schedule_version", None)
            board_schedule = getattr(page, "board_frame_schedule", None)
            acquisition_profile_code = (
                None if board_schedule is None else getattr(board_schedule, "profile", None)
            )
            fresh_point_mask = getattr(page, "fresh_point_mask", None)
            point_age_frames = getattr(page, "point_age_frames", None)
            sample_offset_us = getattr(page, "sample_offset_us", None)
            map_age_frames = getattr(page, "map_age_frames", None)
            bandwidth_discontinuity = bool(
                getattr(page, "bandwidth_discontinuity", False)
            )
            frame_start_device_ms = getattr(page, "frame_start_device_ms", None)
            frame_key = (
                "local",
                device_boot_id,
                device_sequence if device_sequence is not None else frame_number,
            )
            device_id = None
        else:
            raw_channels = [
                self._finite_list(channel)
                for channel in getattr(
                    self.remote_controller, "latest_raw_channels", ()
                )
            ]
            analysis_channels = [
                self._finite_list(channel)
                for channel in getattr(
                    self.remote_controller, "latest_analysis_channels", ()
                )
            ]
            wavelengths = self._finite_list(
                getattr(self.remote_controller, "waves", ())
            )
            frame_number = int(
                getattr(self.remote_controller, "latest_frame_number", 0)
            )
            boot_id = getattr(self.remote_controller, "last_boot_id", None)
            cached = self._remote_telemetry_cache.get((boot_id, frame_number))
            raw_ch1_codes = []
            selector = None
            transimpedance_ohm = None
            digital_gain = None
            gain_mask = None
            device_sequence = None
            device_boot_id = None
            device_uptime_ms = None
            table_crc32 = None
            packet_crc_ok = None
            frame_received_monotonic_ns = None
            acquisition_profile = None
            schedule_version = None
            acquisition_profile_code = None
            fresh_point_mask = None
            point_age_frames = None
            sample_offset_us = None
            map_age_frames = None
            bandwidth_discontinuity = False
            frame_start_device_ms = None
            if cached is not None:
                frame, frame_received_monotonic_ns = cached
                channel_ids = tuple(frame.selected_channels)
                if 1 in channel_ids:
                    ch1_offset = channel_ids.index(1)
                    raw_ch1_codes = [int(point[ch1_offset]) for point in frame.samples]
                gain_mask = int(frame.gainmask1)
                device_sequence = int(frame.seq)
                device_boot_id = int(frame.boot_id)
                device_uptime_ms = int(frame.uptime_ms)
                table_crc32 = int(frame.table_crc32)
                # Telemetry reaches this point only after protocol CRC decoding.
                packet_crc_ok = True
            frame_key = (
                "remote",
                device_boot_id,
                device_sequence if device_sequence is not None else frame_number,
            )
            device_id = getattr(self.remote_controller, "device_id", None)
        if not raw_ch1_codes:
            quality_flags.append("missing_raw_ch1")
        if len(raw_ch1_codes) != len(wavelengths):
            quality_flags.append("point_count_mismatch")
        runtime_machine_id = get_runtime_machine_id()
        runtime_label = runtime_machine_label(runtime_machine_id)
        return {
            "source": self.current_source,
            "machine_id": runtime_machine_id,
            "machine_label": runtime_label,
            "parameter_owner": runtime_label,
            "frame_number": frame_number,
            "frame_key": frame_key,
            "captured_monotonic_ns": captured_monotonic_ns,
            "frame_received_monotonic_ns": frame_received_monotonic_ns,
            "device_sequence": device_sequence,
            "device_boot_id": device_boot_id,
            "device_uptime_ms": device_uptime_ms,
            "device_id": device_id,
            "table_crc32": table_crc32,
            "packet_crc_ok": packet_crc_ok,
            "acquisition_profile": acquisition_profile,
            "schedule_version": schedule_version,
            "acquisition_profile_code": acquisition_profile_code,
            "fresh_point_mask": fresh_point_mask,
            "point_age_frames": point_age_frames,
            "sample_offset_us": sample_offset_us,
            "map_age_frames": map_age_frames,
            "bandwidth_discontinuity": bandwidth_discontinuity,
            "frame_start_device_ms": frame_start_device_ms,
            "wavelengths_nm": wavelengths,
            "raw_ch1_adc_codes": raw_ch1_codes,
            "raw_adc_channels_v": raw_channels,
            "analysis_adc_channels_v": analysis_channels,
            "ch1_transimpedance_ohm": transimpedance_ohm,
            "ch1_feedback_selector": selector,
            "ch1_analogue_gain_mask": gain_mask,
            "ch1_digital_gain_audit": digital_gain,
            "quality_flags": quality_flags,
            "peak_centers_nm": [
                self._finite_list(channel)
                for channel in self.finger_controller.current_peaks_by_channel
            ],
        }

    @staticmethod
    def _status_xyz(status: dict) -> tuple[float, float, float] | None:
        try:
            result = tuple(float(status[f"work_{axis}"]) for axis in "xyz")
        except (KeyError, TypeError, ValueError):
            return None
        return result if all(math.isfinite(value) for value in result) else None

    def _write_coverage_event(
        self, kind: str, target: dict, status: dict
    ) -> str | None:
        writer = self._coverage_writer
        if writer is None:
            return None
        depth = float(target.get("depth_mm", 0.0))
        target_xyz = (
            float(target["work_x"]),
            float(target["work_y"]),
            float(target["contact_z"]) - depth,
        )
        model_x_norm = float(target.get("model_x", math.nan))
        model_y_norm = float(target.get("model_y", math.nan))
        model_xy_mm = (
            model_x_norm * MODEL_X_HALF_SPAN_MM,
            model_y_norm * MODEL_Y_HALF_SPAN_MM,
        )
        # Release starts while the tool may still be touching. Do not label it
        # negative until the motion service confirms the retracted position.
        contact = {"contact": True, "hold": True, "release_complete": False}.get(kind)
        if kind == "pre_contact":
            xyz = self._status_xyz(status)
            if xyz is not None and xyz[2] > float(target["contact_z"]) + 0.1:
                contact = False
        event = ContactEvent(
            kind=kind,
            target_xyz_mm=target_xyz,
            actual_xyz_mm=self._status_xyz(status),
            indentation_mm=depth if kind in {"contact", "hold"} else None,
            indenter_diameter_mm=8.0,
            nominal_contact_area_mm2=(
                math.pi * 4.0**2 if contact is True else (0.0 if contact is False else None)
            ),
            metadata={
                "press_event_id": self._coverage_press_event_id(target),
                "contact": contact,
                "commanded_indentation_mm": depth,
                "nominal_area_basis": "fixed_8mm_diameter_tool_face",
                "pressure_reference": displacement_reference(target, status),
                "contact_label_basis": "calibrated_z_plane_and_motion_phase_not_force_sensor",
                "adc_event_synchronization_verified": False,
                "point_index": int(target.get("point_index", 0)),
                "repetition": int(target.get("coverage_repetition", 0)),
                "model_x_norm": model_x_norm,
                "model_y_norm": model_y_norm,
                "model_xy_mm": model_xy_mm,
                "model_coordinate_origin": "CAD_FRONT_CENTRE",
                "source": self.current_source,
                "machine_status": dict(status),
            },
        )
        try:
            writer.write_event(event)
        except (BufferError, RuntimeError, TypeError, ValueError) as exc:
            self._coverage_cancel_after_current = True
            if not getattr(self, "_coverage_stop_reason", None):
                self._coverage_stop_reason = f"event_write_failed: {type(exc).__name__}: {exc}"
            self.finger_controller.coverage_summary_label.setText(
                f"覆盖事件写入失败；完成当前点后停止：{exc}"
            )
            return None
        return event.event_id

    def _coverage_press_event_id(self, target: dict) -> str:
        session_id = self._coverage_writer.session_id if self._coverage_writer is not None else "inactive"
        return (f"{session_id}:r{int(target.get('coverage_repetition', 0))}"
                f":p{int(target.get('point_index', 0))}:n{self._coverage_completed}")

    def _start_coverage_native_capture(self):
        """Subscribe to the shared native USB/LAN producer, without UI decimation."""
        if self._coverage_native_recorder is not None:
            raise RuntimeError("A native coverage capture is already active")
        self._coverage_native_stats = None
        if self._uses_local_pages() and self._coverage_writer is not None:
            self._coverage_native_recorder = NativeFrameRecorder(
                local_app.frames_queue, self._coverage_writer.session_dir / "native_frames.jsonl",
                capacity=2048,
            )

    def _begin_coverage_phase(self, phase: str, target: dict, status: dict):
        self._coverage_phase = str(phase)
        self._coverage_phase_event_id = self._write_coverage_event(
            self._coverage_phase, target, status
        )
        self._coverage_phase_frame_counts.setdefault(self._coverage_phase, 0)
        self._capture_coverage_phase_frame()

    def _capture_coverage_phase_frame(self):
        if (
            not self._coverage_active
            or self._coverage_motion_test
            or self._coverage_current is None
            or self._coverage_writer is None
            or self._coverage_phase is None
        ):
            return
        if self._coverage_native_recorder is not None:
            native_stats = self._coverage_native_recorder.stats()
            if native_stats["error"] or native_stats["dropped"]:
                self._coverage_cancel_after_current = True
                if not getattr(self, "_coverage_stop_reason", None):
                    self._coverage_stop_reason = f"native_recording_failed: {native_stats}"
                self.finger_controller.coverage_summary_label.setText(
                    "逐帧记录失败；本次数据不可训练，完成当前点后停止"
                )
                return
        spectrum = self._active_spectrum_snapshot()
        frame_key = spectrum.get("frame_key")
        if frame_key is None or frame_key == self._coverage_last_frame_key:
            return
        wavelengths = spectrum.get("wavelengths_nm") or []
        raw_codes = spectrum.get("raw_ch1_adc_codes") or []
        if not wavelengths or len(raw_codes) != len(wavelengths):
            return
        target = self._coverage_current
        try:
            model_x_norm = float(target.get("model_x", math.nan))
            model_y_norm = float(target.get("model_y", math.nan))
            self._coverage_writer_sequence += 1
            frame = SpectrumFrame(
                sequence=self._coverage_writer_sequence,
                monotonic_ns=int(spectrum["captured_monotonic_ns"]),
                wavelengths_nm=wavelengths,
                raw_adc_codes={1: raw_codes},
                transimpedance_ohm={1: spectrum.get("ch1_transimpedance_ohm")},
                digital_gain={1: spectrum.get("ch1_digital_gain_audit")},
                analogue_gain_mask={1: spectrum.get("ch1_analogue_gain_mask")},
                feedback_selector={1: spectrum.get("ch1_feedback_selector")},
                table_crc32=spectrum.get("table_crc32"),
                packet_crc_ok=spectrum.get("packet_crc_ok"),
                device_sequence=spectrum.get("device_sequence"),
                device_boot_id=spectrum.get("device_boot_id"),
                device_uptime_ms=spectrum.get("device_uptime_ms"),
                frame_received_monotonic_ns=spectrum.get("frame_received_monotonic_ns"),
                acquisition_profile=spectrum.get("acquisition_profile"),
                schedule_version=spectrum.get("schedule_version"),
                acquisition_profile_code=spectrum.get("acquisition_profile_code"),
                # TABLE_STATE currently publishes the two-read RC estimate,
                # not either pre-prediction ADC conversion. Preserve provenance.
                adc_value_kind="firmware_rc_estimate",
                fresh_point_mask=spectrum.get("fresh_point_mask"),
                point_age_frames=spectrum.get("point_age_frames"),
                sample_offset_us=spectrum.get("sample_offset_us"),
                map_age_frames=spectrum.get("map_age_frames"),
                bandwidth_discontinuity=bool(
                    spectrum.get("bandwidth_discontinuity", False)
                ),
                frame_start_device_ms=spectrum.get("frame_start_device_ms"),
                quality_flags=tuple(spectrum.get("quality_flags") or ()),
                quality={
                    "coverage_phase": self._coverage_phase,
                    "phase_event_id": self._coverage_phase_event_id,
                    "press_event_id": self._coverage_press_event_id(target),
                    "label_timebase": "host_frame_receive_not_ui_snapshot",
                    "adc_event_synchronization_verified": False,
                    "point_index": int(target.get("point_index", 0)),
                    "repetition": int(target.get("coverage_repetition", 0)),
                    "model_x_norm": model_x_norm,
                    "model_y_norm": model_y_norm,
                    "model_xy_mm": (
                        model_x_norm * MODEL_X_HALF_SPAN_MM,
                        model_y_norm * MODEL_Y_HALF_SPAN_MM,
                    ),
                    "model_coordinate_origin": "CAD_FRONT_CENTRE",
                    "machine_status": dict(self._last_mach3_status),
                    "pressure_reference": displacement_reference(target, self._last_mach3_status),
                    "machine_status_received_monotonic_ns": self._last_mach3_status_received_monotonic_ns,
                    "machine_status_role": "latest_host_status_not_synchronized_with_adc",
                    "host_frame_number": spectrum.get("frame_number"),
                    "device_id": spectrum.get("device_id"),
                    "peak_centers_nm": spectrum.get("peak_centers_nm"),
                },
                mode="stress",
                source=str(spectrum.get("source") or "unknown"),
            )
            self._coverage_writer.write_frame(frame)
            self._coverage_last_frame_key = frame_key
            self._coverage_records += 1
            self._coverage_phase_frame_counts[self._coverage_phase] = (
                self._coverage_phase_frame_counts.get(self._coverage_phase, 0) + 1
            )
            self.finger_controller.set_coverage_progress(
                self._coverage_completed,
                self._coverage_total,
                active_point_index=int(target.get("point_index", 1)) - 1,
                detail=(
                    f"{self._coverage_phase}：已保存{self._coverage_records}帧"
                    " CH1原始光谱"
                ),
            )
        except (BufferError, RuntimeError, TypeError, ValueError) as exc:
            self._coverage_cancel_after_current = True
            if not getattr(self, "_coverage_stop_reason", None):
                self._coverage_stop_reason = f"frame_write_failed: {type(exc).__name__}: {exc}"
            self.finger_controller.coverage_summary_label.setText(
                f"数据文件写入失败；完成当前点后停止：{exc}"
            )

    # Compatibility hook. This timer stores distinct UI snapshots only;
    # the native subscriber, not this timer, preserves all received frames.
    def _capture_coverage_spectrum(self, token: tuple, _target: dict):
        if token == self._coverage_token():
            self._capture_coverage_phase_frame()

    def _finish_coverage_run(self, success: bool, detail: str):
        completed = self._coverage_completed
        total = self._coverage_total
        motion_test = self._coverage_motion_test
        self._coverage_capture_timer.stop()
        writer = self._coverage_writer
        writer_error = None
        if self._coverage_native_recorder is not None:
            try:
                self._coverage_native_stats = self._coverage_native_recorder.close()
            except Exception as exc:
                self._coverage_native_stats = self._coverage_native_recorder.stats()
                writer_error = f"逐帧记录不完整：{exc}"
                success = False
            finally:
                self._coverage_native_recorder = None
        if writer is not None:
            try:
                writer.write_event(
                    ContactEvent(
                        kind="session_end",
                        metadata={
                            "success": bool(success),
                            "detail": str(detail),
                            "stop_reason": getattr(self, "_coverage_stop_reason", None),
                            "completed_contacts": completed,
                            "total_contacts": total,
                            "native_recording": self._coverage_native_stats,
                            "phase_frame_counts": dict(
                                self._coverage_phase_frame_counts
                            ),
                        },
                    )
                )
            except (BufferError, OSError, RuntimeError, TypeError, ValueError) as exc:
                writer_error = f"{writer_error}; {exc}" if writer_error else str(exc)
                success = False
            try:
                writer.close()
            except (BufferError, OSError, RuntimeError, TypeError, ValueError) as exc:
                close_error = str(exc)
                writer_error = (
                    f"{writer_error}；{close_error}" if writer_error else close_error
                )
                success = False
            finally:
                self._coverage_writer = None
        self._coverage_active = False
        self._coverage_queue.clear()
        self._coverage_current = None
        self._coverage_cancel_after_current = False
        self._coverage_motion_test = False
        self._coverage_stop_reason = None
        self._coverage_completion_pending = False
        self._coverage_phase = None
        self._coverage_phase_event_id = None
        self.finger_controller.set_coverage_running(False)
        state = "完成" if success else "停止"
        if writer_error:
            detail = f"{detail}；数据会话关闭失败：{writer_error}"
        result_detail = (
            f"{state}：{detail}；动作测试未采集或保存光谱"
            if motion_test
            else (
                f"{state}：{detail}；已保存{self._coverage_records}条数据"
                + (
                    f" → {self._coverage_dataset_path.name}"
                    if self._coverage_dataset_path is not None
                    else ""
                )
            )
        )
        self.finger_controller.set_coverage_progress(
            completed,
            total,
            detail=result_detail,
        )

    @QtCore.pyqtSlot(bool, str, dict)
    def _finger_press_finished(self, success: bool, text: str, status: dict):
        if self._coverage_active:
            if not success:
                self._finish_coverage_run(False, text)
                return
            if self._coverage_motion_test:
                self._complete_coverage_point(self._coverage_token(), text)
                return
            if self._coverage_completion_pending:
                return
            self._coverage_completion_pending = True
            completed_point = self._coverage_current or {}
            point_index = int(completed_point.get("point_index", 1)) - 1
            self.finger_controller.set_coverage_progress(
                self._coverage_completed,
                self._coverage_total,
                active_point_index=point_index,
                detail="当前点已安全抬升，正在收集释放后多帧光谱",
            )
            token = self._coverage_token()
            QtCore.QTimer.singleShot(
                COVERAGE_POST_RELEASE_MS,
                lambda value=token, result=text: self._complete_coverage_point(
                    value, result
                ),
            )
            return
        self.finger_controller.click_press_arm.setEnabled(True)
        self.finger_controller.click_press_status_label.setText(text)

    def _complete_coverage_point(self, token: tuple | None, _text: str):
        if not self._coverage_active or token != self._coverage_token():
            return
        self._coverage_completion_pending = False
        self._coverage_completed += 1
        completed_point = self._coverage_current or {}
        point_index = int(completed_point.get("point_index", 1)) - 1
        self.finger_controller.set_coverage_progress(
            self._coverage_completed,
            self._coverage_total,
            active_point_index=point_index,
            detail="当前点已接触、保持、释放并完成多帧记录",
        )
        self._coverage_current = None
        self._coverage_phase = None
        self._coverage_phase_event_id = None
        if self._coverage_cancel_after_current:
            self._finish_coverage_run(False, "已在安全抬升后停止")
        elif self._coverage_completed >= self._coverage_total:
            self._finish_coverage_run(True, "全部接触点已完成")
        else:
            QtCore.QTimer.singleShot(250, self._run_next_coverage_press)

    def _prefill_provisioning_fields(self):
        config = getattr(self, "network_config", None)
        if config is None:
            return
        self.provision_page.host.setText(config.host)
        self.provision_page.broker_port.setValue(config.port)
        self.provision_page.tls.setChecked(True)
        self.provision_page.device_id.setText(config.device_id)
        self.provision_page.client_id.setText(config.device_id)
        self.provision_page.topic_prefix.setText(config.topic_root)
        self.provision_page.username.setText(config.username or "")

    def _build_shell(self):
        # The legacy menus are still the canonical port/baud/mode actions, but
        # they are presented through a compact settings button instead of an
        # old-style menu strip across the whole application.
        self.menuBar().hide()
        central = QtWidgets.QWidget(self)
        central.setObjectName("appRoot")
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(16)

        # HyperOS Settings-style left rail.  The underlying QTabWidget remains
        # intact so all existing page indexes and integrations keep working.
        self.side_rail = QtWidgets.QFrame()
        self.side_rail.setObjectName("sideRail")
        self.side_rail.setFixedWidth(246)
        self.rail_layout = QtWidgets.QVBoxLayout(self.side_rail)
        self.rail_layout.setContentsMargins(16, 18, 16, 16)
        self.rail_layout.setSpacing(8)
        rail_layout = self.rail_layout

        brand_row = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel()
        logo.setPixmap(make_logo(44))
        logo.setFixedSize(44, 44)
        brand_row.addWidget(logo)
        brand_text = QtWidgets.QVBoxLayout()
        brand_text.setSpacing(1)
        self.side_brand_name = QtWidgets.QLabel("OptiSense Studio")
        self.side_brand_name.setObjectName("sideBrand")
        self.side_brand_caption = QtWidgets.QLabel("光栅智能解调工作站")
        self.side_brand_caption.setObjectName("sideBrandCaption")
        self.side_brand_name.setWordWrap(True)
        self.side_brand_caption.setWordWrap(True)
        brand_text.addWidget(self.side_brand_name)
        brand_text.addWidget(self.side_brand_caption)
        brand_row.addLayout(brand_text, 1)
        rail_layout.addLayout(brand_row)
        rail_layout.addSpacing(18)

        self.nav_caption = QtWidgets.QLabel("工作台")
        self.nav_caption.setObjectName("navSection")
        self.nav_caption.setContentsMargins(12, 0, 0, 3)
        rail_layout.addWidget(self.nav_caption)
        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        for index, (title, _subtitle, icon_kind) in enumerate(PAGE_INFO):
            button = QtWidgets.QPushButton(title)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setIcon(make_nav_icon(icon_kind, COLORS["text_2"]))
            button.setIconSize(QtCore.QSize(22, 22))
            button.setToolTip(title)
            button.clicked.connect(
                lambda checked=False, value=index: (
                    checked and self.workspace_tabs.setCurrentIndex(value)
                )
            )
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            rail_layout.addWidget(button)

        rail_layout.addStretch(1)
        self.side_device_card = QtWidgets.QFrame()
        self.side_device_card.setObjectName("sideStatusCard")
        self.side_device_layout = QtWidgets.QVBoxLayout(self.side_device_card)
        self.side_device_layout.setContentsMargins(14, 13, 14, 13)
        self.side_device_layout.setSpacing(5)
        device_layout = self.side_device_layout
        self.device_caption = QtWidgets.QLabel("当前机号")
        self.device_caption.setObjectName("navSection")
        device_layout.addWidget(self.device_caption)
        self.machine_selector = QtWidgets.QComboBox()
        self.machine_selector.setObjectName("machineSelector")
        self.machine_selector.setToolTip(
            "选择当前物理机号；同步切换该机的2001点激光表及独立保存的OTA参数"
        )
        for machine_id, machine_label in MACHINE_OPTIONS:
            self.machine_selector.addItem(machine_label, machine_id)
        selected_index = self.machine_selector.findData(
            self.ota_page.current_machine_id
        )
        self.machine_selector.setCurrentIndex(max(selected_index, 0))
        self.machine_selector.currentIndexChanged.connect(
            self._machine_selection_changed
        )
        device_layout.addWidget(self.machine_selector)
        selected_label = MACHINE_LABELS.get(
            self.ota_page.current_machine_id, MACHINE_LABEL
        )
        self.device_name = QtWidgets.QLabel(
            f"{selected_label} · STM32 · PI11210"
        )
        self.device_name.setStyleSheet("font-weight: 700; font-size: 10.5pt;")
        self.device_name.setWordWrap(True)
        device_layout.addWidget(self.device_name)
        self.sidebar_link_value = QtWidgets.QLabel("● 等待数据链路")
        self.sidebar_link_value.setObjectName("sideBrandCaption")
        self.sidebar_link_value.setWordWrap(True)
        device_layout.addWidget(self.sidebar_link_value)
        self.device_version = QtWidgets.QLabel()
        self.device_version.setObjectName("sideBrandCaption")
        self.device_version.setWordWrap(True)
        device_layout.addWidget(self.device_version)
        self.device_options_button = QtWidgets.QPushButton("端口与设备选项")
        self.device_options_button.setIcon(
            make_nav_icon("device", COLORS["text_2"], 20)
        )
        self.device_options_button.setIconSize(QtCore.QSize(20, 20))
        device_layout.addWidget(self.device_options_button)
        self.device_options_menu = QtWidgets.QMenu(self)
        self.device_options_menu.addMenu(self.local_controller.menu_port)
        self.device_options_menu.addMenu(self.local_controller.menu_baud)
        self.device_options_menu.addSeparator()
        self.device_options_menu.addMenu(self.local_controller.menu_page)
        self.device_options_button.clicked.connect(self._show_device_options)
        self.display_maximize_button = QtWidgets.QPushButton("选择屏幕并最大化")
        self.display_maximize_button.setObjectName("displayMaximizeButton")
        self.display_maximize_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
        )
        self.display_maximize_button.setIconSize(QtCore.QSize(20, 20))
        self.display_maximize_button.setToolTip(
            "选择一台显示器并在该屏幕最大化；保留Windows标题栏及右上角三个窗口按钮"
        )
        self.display_maximize_button.clicked.connect(
            self._show_display_maximize_menu
        )
        rail_layout.addWidget(self.side_device_card)
        device_layout.addWidget(self.display_maximize_button)
        self._refresh_machine_identity()
        root.addWidget(self.side_rail)

        main = QtWidgets.QWidget()
        main.setObjectName("mainArea")
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        top = QtWidgets.QFrame()
        top.setObjectName("topBar")
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(22, 15, 22, 14)
        top_layout.setSpacing(11)

        heading_row = QtWidgets.QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)
        heading_row.setSpacing(12)
        heading_widget = QtWidgets.QWidget()
        page_heading = QtWidgets.QVBoxLayout(heading_widget)
        page_heading.setContentsMargins(0, 0, 0, 0)
        page_heading.setSpacing(1)
        self.page_title = QtWidgets.QLabel(PAGE_INFO[0][0])
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QtWidgets.QLabel(PAGE_INFO[0][1])
        self.page_subtitle.setObjectName("pageSubtitle")
        self.page_subtitle.setWordWrap(True)
        page_heading.addWidget(self.page_title)
        page_heading.addWidget(self.page_subtitle)
        heading_row.addWidget(heading_widget, 1)
        self.cnc_status_label = QtWidgets.QLabel("Mach3 指令链路：待检测")
        self.cnc_status_label.setObjectName("cncStatusPill")
        self.cnc_status_label.setProperty("statusKind", "info")
        self.cnc_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cnc_status_label.setToolTip(self.cnc_status_label.text())
        heading_row.addWidget(self.cnc_status_label)
        self.cnc_release_button = QtWidgets.QPushButton("恢复控制")
        self.cnc_release_button.setEnabled(False)
        self.cnc_release_button.setMinimumHeight(44)
        self.cnc_release_button.setToolTip(
            "检测到急停锁定后启用；确认安全后只解除Reset并强制关闭主轴"
        )
        self.cnc_release_button.clicked.connect(self._confirm_release_estop)
        heading_row.addWidget(self.cnc_release_button)
        self.cnc_estop_button = QtWidgets.QPushButton("机床紧急停止")
        self.cnc_estop_button.setObjectName("cncEmergencyButton")
        self.cnc_estop_button.setProperty("role", "emergency")
        self.cnc_estop_button.setMinimumHeight(44)
        self.cnc_estop_button.setToolTip(
            "立即停止 Mach3 并锁定急停；不会自动复位，不会启动主轴"
        )
        self.cnc_estop_button.clicked.connect(self.mach3_emergency.emergency_stop)
        heading_row.addWidget(self.cnc_estop_button)
        top_layout.addLayout(heading_row)

        primary_widget = QtWidgets.QWidget()
        primary = FlowLayout(primary_widget, horizontal_spacing=8, vertical_spacing=7)

        source_caption = QtWidgets.QLabel("数据来源")
        source_caption.setObjectName("sectionCaption")
        primary.addWidget(source_caption)
        source_shell = QtWidgets.QFrame()
        source_shell.setObjectName("segmentedControl")
        source_layout = QtWidgets.QHBoxLayout(source_shell)
        source_layout.setContentsMargins(3, 3, 3, 3)
        source_layout.setSpacing(2)
        self.source_group = QtWidgets.QButtonGroup(self)
        self.source_group.setExclusive(True)
        self.source_buttons = {}
        source_specs = (
            (SOURCE_LOCAL, "USB 本地"),
            (NETWORK_LAN, "LAN 局域网"),
            (NETWORK_WAN, "MQTT 广域网"),
        )
        for source, text in source_specs:
            button = QtWidgets.QPushButton(text)
            button.setCheckable(True)
            button.setProperty("segment", True)
            button.setMinimumWidth(92)
            self.source_group.addButton(button)
            self.source_buttons[source] = button
            source_layout.addWidget(button)
        if not self.demo and self.network_config is None:
            self.source_buttons[NETWORK_WAN].setEnabled(False)
            self.source_buttons[NETWORK_WAN].setToolTip(
                self.network_config_error or "请先配置 MQTT 参数"
            )
        if self.demo:
            self.source_buttons[NETWORK_LAN].setText("演示数据")
            self.source_buttons[NETWORK_WAN].setEnabled(False)
        primary.addWidget(source_shell)

        self.source_status = QtWidgets.QLabel("● 数据链路：准备中")
        self.source_status.setObjectName("sourceStatus")
        self.source_status.setMinimumWidth(0)
        self.source_status.setWordWrap(True)
        self.source_status.setAlignment(QtCore.Qt.AlignCenter)
        primary.addWidget(self.source_status)
        top_layout.addWidget(primary_widget)

        self.workflow_strip = QtWidgets.QFrame()
        self.workflow_strip.setObjectName("workflowStrip")
        workflow_layout = QtWidgets.QVBoxLayout(self.workflow_strip)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(6)
        mode_row = QtWidgets.QWidget()
        secondary = FlowLayout(mode_row, horizontal_spacing=7, vertical_spacing=7)
        mode_caption = QtWidgets.QLabel("工作模式")
        mode_caption.setObjectName("sectionCaption")
        secondary.addWidget(mode_caption)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = []
        for index, text in enumerate(LOCAL_MODE_NAMES):
            button = QtWidgets.QPushButton(text)
            button.setCheckable(True)
            button.setProperty("modeChip", True)
            button.setMinimumWidth(88 if index < 4 else 112)
            button.clicked.connect(
                lambda checked=False, value=index: self._mode_requested(value)
            )
            self.mode_group.addButton(button, index)
            self.mode_buttons.append(button)
            secondary.addWidget(button)
        self.context_hint = QtWidgets.QLabel(
            "USB与LAN局域网均支持七种完整采集模式，包括临时15/45点测试；"
            "广域网提供应力与温度实时监测"
        )
        self.context_hint.setObjectName("softHint")
        self.context_hint.setWordWrap(True)
        self.context_hint.setToolTip(self.context_hint.text())
        self.open_finger_button = QtWidgets.QPushButton("进入 3D 应力视图")
        self.open_finger_button.setProperty("role", "primary")
        self.open_finger_button.clicked.connect(self.open_finger_tab)
        secondary.addWidget(self.open_finger_button)
        workflow_layout.addWidget(mode_row)
        workflow_layout.addWidget(self.context_hint)
        top_layout.addWidget(self.workflow_strip)
        main_layout.addWidget(top)

        self.workspace_tabs = QtWidgets.QTabWidget()
        self.workspace_tabs.setObjectName("workspaceTabs")
        self.workspace_tabs.setDocumentMode(True)
        self.workspace_tabs.tabBar().hide()

        acquisition = QtWidgets.QWidget()
        acquisition.setObjectName("pageCanvas")
        acquisition_layout = QtWidgets.QVBoxLayout(acquisition)
        acquisition_layout.setContentsMargins(0, 0, 0, 0)
        self.source_stack = QtWidgets.QStackedWidget()
        self.source_stack.setObjectName("sourceStack")
        self.source_stack.addWidget(self.local_page)
        self.source_stack.addWidget(self.remote_page)
        acquisition_layout.addWidget(self.source_stack)
        self.workspace_tabs.addTab(acquisition, PAGE_INFO[0][0])
        self.finger_page.setObjectName("fingerPage")
        # The 3-D page contains two wrapping control groups above a sizeable
        # splitter.  At compact window heights those groups legitimately grow
        # to two or three rows; assigning the whole page directly to the tab
        # forced Qt to compress the splitter and then reposition the already
        # expanded groups on top of one another.  Keep the page at its true
        # layout minimum and scroll the viewport instead, just like the other
        # dense operator pages in the unified shell.
        self.finger_scroll = ResponsiveScrollArea()
        self.finger_scroll.setObjectName("fingerScroll")
        self.finger_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.finger_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.finger_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.finger_scroll.setWidget(self.finger_page)
        self.workspace_tabs.addTab(self.finger_scroll, PAGE_INFO[1][0])

        settings_shell = QtWidgets.QWidget()
        settings_shell.setObjectName("pageCanvas")
        settings_shell_layout = QtWidgets.QVBoxLayout(settings_shell)
        settings_shell_layout.setContentsMargins(0, 0, 0, 0)
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setObjectName("settingsScroll")
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        settings_viewport = QtWidgets.QWidget()
        settings_layout = QtWidgets.QHBoxLayout(settings_viewport)
        settings_layout.setContentsMargins(28, 22, 28, 28)
        settings_layout.addStretch(1)
        settings_card = QtWidgets.QFrame()
        settings_card.setObjectName("contentCard")
        settings_card.setMaximumWidth(920)
        settings_card_layout = QtWidgets.QVBoxLayout(settings_card)
        settings_card_layout.setContentsMargins(30, 24, 30, 26)
        settings_card_layout.addWidget(self.provision_page)
        settings_layout.addWidget(settings_card, 20)
        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_viewport)
        settings_shell_layout.addWidget(settings_scroll)
        self.workspace_tabs.addTab(settings_shell, PAGE_INFO[2][0])

        ota_shell = QtWidgets.QWidget()
        ota_shell.setObjectName("pageCanvas")
        ota_shell_layout = QtWidgets.QVBoxLayout(ota_shell)
        ota_shell_layout.setContentsMargins(0, 0, 0, 0)
        ota_scroll = QtWidgets.QScrollArea()
        ota_scroll.setObjectName("otaScroll")
        ota_scroll.setWidgetResizable(True)
        ota_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        ota_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        ota_viewport = QtWidgets.QWidget()
        ota_layout = QtWidgets.QHBoxLayout(ota_viewport)
        ota_layout.setContentsMargins(28, 22, 28, 28)
        ota_layout.addStretch(1)
        ota_card = QtWidgets.QFrame()
        ota_card.setObjectName("contentCard")
        ota_card.setMaximumWidth(1080)
        ota_card_layout = QtWidgets.QVBoxLayout(ota_card)
        ota_card_layout.setContentsMargins(8, 4, 8, 8)
        ota_card_layout.addWidget(self.ota_page)
        ota_layout.addWidget(ota_card, 20)
        ota_layout.addStretch(1)
        ota_scroll.setWidget(ota_viewport)
        ota_shell_layout.addWidget(ota_scroll)
        self.workspace_tabs.addTab(ota_shell, PAGE_INFO[3][0])
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        main_layout.addWidget(self.workspace_tabs, 1)
        root.addWidget(main, 1)

        self.statusBar().showMessage(
            "同一时刻只启用一条数据链路；所有模式共用同一套波长表和拟合算法"
        )
        self._update_source_status("数据链路：准备中", "info")
        self._workspace_changed(0)
        self._refresh_responsive_shell()

    def _refresh_responsive_shell(self):
        """Collapse the text rail when it would starve the active page."""

        if not hasattr(self, "side_rail"):
            return
        compact = self.width() < 1180
        short_window = self.height() < 700
        metrics = QtGui.QFontMetrics(self.font())
        full_width = max(
            246,
            min(
                320,
                max(metrics.horizontalAdvance(item[0]) for item in PAGE_INFO) + 92,
            ),
        )
        self.side_rail.setFixedWidth(88 if compact else full_width)
        self.rail_layout.setContentsMargins(
            8 if compact else 16,
            12 if compact else 18,
            8 if compact else 16,
            10 if compact else 16,
        )
        self.side_device_layout.setContentsMargins(
            6 if compact else 14,
            7 if compact else 13,
            6 if compact else 14,
            7 if compact else 13,
        )
        self.side_brand_name.setVisible(not compact)
        self.side_brand_caption.setVisible(not compact)
        self.nav_caption.setVisible(not compact)
        for index, button in enumerate(self.nav_buttons):
            button.setText("" if compact else PAGE_INFO[index][0])
            button.setIconSize(
                QtCore.QSize(26, 26) if compact else QtCore.QSize(22, 22)
            )
        for widget in (
            self.device_caption,
            self.device_name,
            self.sidebar_link_value,
            self.device_version,
        ):
            widget.setVisible(not compact)
        self.device_options_button.setText("" if compact else "端口与设备选项")
        self.device_options_button.setToolTip("端口与设备选项")
        self.display_maximize_button.setText(
            "" if compact else "选择屏幕并最大化"
        )
        self.display_maximize_button.setToolTip(
            "选择一台显示器并在该屏幕最大化；保留Windows标题栏及右上角三个窗口按钮"
        )
        self.page_subtitle.setVisible(not short_window)
        self.context_hint.setVisible(
            not short_window and self.workspace_tabs.currentIndex() < 2
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_responsive_shell()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.FontChange,
            QtCore.QEvent.ApplicationFontChange,
            QtCore.QEvent.StyleChange,
        ):
            QtCore.QTimer.singleShot(0, self._refresh_responsive_shell)

    def _machine_selection_changed(self, index: int):
        machine_id = str(self.machine_selector.itemData(index) or "")
        previous_id = get_runtime_machine_id()
        if not machine_id:
            return
        if not local_app.switch_mode_enable:
            previous_index = self.machine_selector.findData(previous_id)
            self.machine_selector.blockSignals(True)
            self.machine_selector.setCurrentIndex(max(previous_index, 0))
            self.machine_selector.blockSignals(False)
            self.statusBar().showMessage("当前正在采集，停止后才能切换机号和2001点表")
            return
        if machine_id == previous_id:
            self._refresh_machine_identity()
            return
        if not self.ota_page.set_machine(machine_id):
            return
        errors = self.local_controller.reload_machine_calibration(machine_id)
        self._refresh_machine_identity()
        if errors:
            self.statusBar().showMessage(
                f"已切换为 {runtime_machine_label(machine_id)}，但2001点表不可用："
                f"{errors[0]}"
            )
        else:
            self.statusBar().showMessage(
                f"当前机号已切换为 {runtime_machine_label(machine_id)}；"
                f"已载入{runtime_parameter_note(machine_id)}，OTA参数也已切换"
            )

    def _refresh_machine_identity(self):
        machine_id = self.ota_page.current_machine_id
        machine_label = MACHINE_LABELS.get(machine_id, MACHINE_LABEL)
        self.device_name.setText(f"{machine_label} · STM32 · PI11210")
        note = runtime_parameter_note(machine_id)
        if machine_id == "machine_2":
            note += "；9峰/3峰路线仍需二号机单独采集"
        self.device_version.setText(
            f"{note}\n统一控制 · ADC 光谱 · 3D 定位"
        )
        self.setWindowTitle(
            f"FBG OptiSense Studio {APP_VERSION} — {machine_label} · "
            "光栅解调与机械手指应力定位"
        )

    def _show_device_options(self):
        origin = self.device_options_button.mapToGlobal(
            QtCore.QPoint(0, self.device_options_button.height() + 5)
        )
        self.device_options_menu.exec_(origin)

    @staticmethod
    def _display_label(screen, index: int, *, primary=False, current=False):
        geometry = screen.geometry()
        name = str(screen.name() or f"显示器 {index + 1}")
        suffix = []
        if primary:
            suffix.append("主屏")
        if current:
            suffix.append("当前")
        flags = f" · {' / '.join(suffix)}" if suffix else ""
        return (
            f"屏幕 {index + 1}：{name} · "
            f"{geometry.width()}×{geometry.height()}{flags}"
        )

    def _show_display_maximize_menu(self):
        screens = list(QtWidgets.QApplication.screens())
        menu = QtWidgets.QMenu(self)
        current = None
        handle = self.windowHandle()
        if handle is not None:
            current = handle.screen()
        primary = QtWidgets.QApplication.primaryScreen()
        if not screens:
            action = menu.addAction("未检测到可用显示器")
            action.setEnabled(False)
        for index, screen in enumerate(screens):
            action = menu.addAction(
                self._display_label(
                    screen,
                    index,
                    primary=screen is primary,
                    current=screen is current,
                )
            )
            action.triggered.connect(
                lambda _checked=False, target=screen: self._maximize_on_screen(
                    target
                )
            )
        origin = self.display_maximize_button.mapToGlobal(
            QtCore.QPoint(0, self.display_maximize_button.height() + 5)
        )
        menu.exec_(origin)

    def _maximize_on_screen(self, screen):
        if screen is None or screen not in QtWidgets.QApplication.screens():
            self.statusBar().showMessage("目标显示器已断开，请重新选择屏幕")
            return
        # Deliberately use the normal maximized window state, not borderless
        # fullscreen, so Windows keeps minimize/maximize/close controls.
        self.showNormal()
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(screen)
        geometry = screen.availableGeometry()
        self.setGeometry(geometry)
        self.move(geometry.topLeft())
        self.showMaximized()
        index = QtWidgets.QApplication.screens().index(screen) + 1
        self.statusBar().showMessage(
            f"已在屏幕 {index}（{screen.name() or '未命名显示器'}）最大化；"
            "Windows标题栏和右上角窗口按钮保留"
        )

    def _workspace_changed(self, index: int):
        index = max(0, min(int(index), len(PAGE_INFO) - 1))
        title, subtitle, _icon_kind = PAGE_INFO[index]
        self.page_title.setText(title)
        self.page_subtitle.setText(subtitle)
        self.page_subtitle.setToolTip(subtitle)
        for button_index, button in enumerate(self.nav_buttons):
            selected = button_index == index
            button.setChecked(selected)
            icon_kind = PAGE_INFO[button_index][2]
            icon_color = COLORS["blue"] if selected else COLORS["text_2"]
            button.setIcon(make_nav_icon(icon_kind, icon_color))
        # Mode controls are meaningful beside live spectra and the 3-D stress
        # view, while configuration/OTA pages stay deliberately calm.
        self.workflow_strip.setVisible(index < 2)
        self.open_finger_button.setVisible(index == 0)
        if hasattr(self, "mach3_position_timer") and not self.demo:
            if index < 2:
                if not self.mach3_position_timer.isActive():
                    self.mach3_position_timer.start()
                self.mach3_emergency.query_status()
            else:
                self.mach3_position_timer.stop()
        self._refresh_responsive_shell()

    def _update_source_status(self, text: str, kind: str = "info"):
        clean = str(text).lstrip("● ")
        set_status(self.source_status, f"● {clean}", kind)
        self.sidebar_link_value.setText(f"● {clean}")
        colors = {
            "ok": COLORS["green"],
            "warning": COLORS["orange"],
            "error": COLORS["red"],
            "info": COLORS["blue"],
        }
        self.sidebar_link_value.setStyleSheet(
            f"color: {colors.get(kind, COLORS['blue'])}; font-weight: 600;"
        )

    def _connect_signals(self):
        for source, button in self.source_buttons.items():
            button.clicked.connect(
                lambda checked=False, value=source: (
                    checked and self.switch_source(value)
                )
            )
        for action in self.local_controller.menu_page.actions():
            action.triggered.connect(
                lambda _checked=False: QtCore.QTimer.singleShot(
                    0, self._sync_local_mode
                )
            )
        self.remote_controller.mode_changed.connect(self._sync_remote_mode)
        self.bridge.connection.connect(self._network_connection_changed)
        self.lan_connection_changed.connect(self._lan_connection_changed)
        self.mach3_emergency.status_received.connect(self._mach3_status_received)
        self.mach3_emergency.state_changed.connect(self._mach3_state_changed)

    @QtCore.pyqtSlot(dict)
    def _mach3_status_received(self, status: dict):
        sequence = status.get("_telemetry_seq")
        if sequence is not None:
            try:
                sequence = int(sequence)
            except (TypeError, ValueError):
                return
            if sequence <= self._last_mach3_sequence:
                return
            self._last_mach3_sequence = sequence
        self._last_mach3_status = dict(status)
        self._last_mach3_status_received_monotonic_ns = time.monotonic_ns()
        self.finger_controller.update_machine_status(status)

    @QtCore.pyqtSlot(str, str)
    def _mach3_state_changed(self, state: str, text: str):
        # Track motion at up to 10 Hz while online.  A failed first attach is
        # retried quickly so a shortcut launched immediately after Mach3 does
        # not remain on the yellow calibration-preview position.
        self.mach3_position_timer.setInterval(800 if state == "error" else 100)
        self.cnc_estop_button.setEnabled(True)
        self.cnc_estop_button.setText("机床紧急停止")
        self.cnc_release_button.setEnabled(state == "stopped")
        if self._coverage_active and state in {"stopped", "error"}:
            self._finish_coverage_run(False, f"机床状态中断：{text}")
        work = self._last_mach3_status
        if all(key in work for key in ("work_x", "work_y", "work_z")):
            coordinates = (
                f"X {float(work['work_x']):+.3f}  "
                f"Y {float(work['work_y']):+.3f}  "
                f"Z {float(work['work_z']):+.3f} mm"
            )
            display_text = f"{text} · {coordinates}"
        else:
            display_text = text
        self.cnc_status_label.setText(display_text)
        self.cnc_status_label.setToolTip(display_text)
        if state == "error":
            self._last_mach3_status = {}
            self._last_mach3_status_received_monotonic_ns = None
            self.finger_controller.set_machine_status_error(text)
        kind = {
            "ready": "ok",
            "stopped": "error",
            "warning": "warning",
            "busy": "warning",
            "error": "error",
        }.get(state, "info")
        set_status(self.cnc_status_label, display_text, kind)

    def _confirm_release_estop(self):
        status = dict(self._last_mach3_status)
        if not status:
            QtWidgets.QMessageBox.warning(
                self, "无法恢复控制", "尚未取得Mach3实时状态，未执行任何操作。"
            )
            return
        if not (status.get("estop") or status.get("estop_led")):
            QtWidgets.QMessageBox.information(
                self, "无需恢复", "Mach3当前未锁定Reset/急停。"
            )
            return
        # Mach3 3.x can leave IsMoving asserted while Reset/E-stop is active.
        # IsStopped is its dedicated planner-stop indication; the bridge also
        # verifies all six hardware limits before and after releasing Reset.
        if not status.get("stopped"):
            QtWidgets.QMessageBox.warning(
                self, "无法恢复控制", "机床尚未确认停止，未解除Reset/急停。"
            )
            return
        answer = QtWidgets.QMessageBox.warning(
            self,
            "确认恢复机床控制",
            "请确认：\n"
            "• 机床周围无人且完整运动区域无遮挡；\n"
            "• 压头没有卡住，硬件限位未触发；\n"
            "• 手可以立即触及物理急停。\n\n"
            "继续后只解除Reset/急停并强制发送M5，不会移动任何轴。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.cnc_release_button.setEnabled(False)
        self.mach3_emergency.release_control()

    def _set_checked_source(self, source: str | None):
        self.source_group.setExclusive(False)
        for value, button in self.source_buttons.items():
            button.setChecked(value == source)
        self.source_group.setExclusive(True)

    def _set_checked_mode(self, mode: int | None):
        self.mode_group.setExclusive(False)
        for index, button in enumerate(self.mode_buttons):
            button.setChecked(index == mode)
        self.mode_group.setExclusive(True)

    def _set_local_menus_enabled(self, enabled: bool):
        for menu in (
            self.local_controller.menu_port,
            self.local_controller.menu_baud,
            self.local_controller.menu_page,
        ):
            menu.menuAction().setEnabled(bool(enabled))
        if hasattr(self, "device_options_button"):
            self.device_options_button.setEnabled(bool(enabled))

    def _uses_local_pages(self, source: str | None = None) -> bool:
        """Return whether the source uses the complete USB acquisition pages."""

        selected = self.current_source if source is None else source
        return selected == SOURCE_LOCAL or (selected == NETWORK_LAN and not self.demo)

    def _select_local_transport(self, source: str):
        """Point the legacy pages at CDC or at the same-LAN byte stream."""

        if source == SOURCE_LOCAL:
            local_app.ser = self.usb_serial
            self.local_controller.port = self.usb_port
            self.usb_serial.port = self.usb_port
            self.usb_serial.baudrate = self.local_controller.baud
            return
        if source == NETWORK_LAN and not self.demo:
            local_app.ser = self.lan_serial
            self.local_controller.port = "LAN"
            return
        raise ValueError(f"{source} 不是完整采集页面的数据链路")

    def _can_leave_local(self) -> bool:
        if local_app.switch_mode_enable:
            return True
        QtWidgets.QMessageBox.warning(
            self,
            "当前流程仍在运行",
            "请先点击当前页面的‘关闭/停止’，再切换数据链路。",
        )
        return False

    def _release_local_control(self):
        local_app.local_control_session = False
        with local_app.ser_cond:
            local_app.ser_open = False
            local_app.ser_cond.notify_all()
        local_app.release_serial_if_allowed()

    def switch_source(self, source: str, force: bool = False):
        if source not in SOURCE_NAMES:
            raise ValueError(f"不支持的数据来源：{source}")
        if source == NETWORK_WAN and not self.source_buttons[source].isEnabled():
            self._set_checked_source(self.current_source)
            return
        if source == self.current_source and not force:
            return
        if self._uses_local_pages(self.current_source) and not force:
            if not self._can_leave_local():
                self._set_checked_source(self.current_source)
                return

        previous = self.current_source
        transport_open_failed = False
        try:
            if self._uses_local_pages(previous):
                if previous == SOURCE_LOCAL:
                    self.usb_port = self.local_controller.port
                self._release_local_control()

            if self._uses_local_pages(source):
                self.remote_controller.deactivate_network()
                self._select_local_transport(source)
                self.source_stack.setCurrentWidget(self.local_page)
                self._set_local_menus_enabled(source == SOURCE_LOCAL)
                if not self.local_controller.acquire_local_control():
                    transport_open_failed = True
                    label = "USB 端口" if source == SOURCE_LOCAL else "局域网通道"
                    raise RuntimeError(f"{label}打开失败")
                if source == SOURCE_LOCAL:
                    detail = self.local_controller.port or "STM32 USB"
                    self._update_source_status(f"本地控制：{detail}", "ok")
                    route = "本地 USB"
                else:
                    self._update_source_status(
                        "局域网全功能通道：正在寻找板卡", "warning"
                    )
                    route = "局域网 TCP（USB 同构）"
                self.finger_controller.source_label.setText(
                    f"数据源：{route} ADC 反射光谱 → 当前帧分波段拟合"
                )
            else:
                self.source_stack.setCurrentWidget(self.remote_page)
                self._set_local_menus_enabled(False)
                self.remote_controller.activate_network(source)
                name = "演示数据" if self.demo else SOURCE_NAMES[source]
                self._update_source_status(f"正在连接：{name}", "warning")
                route = (
                    "演示数据"
                    if self.demo
                    else ("局域网 TCP" if source == NETWORK_LAN else "广域网 MQTT")
                )
                self.finger_controller.source_label.setText(
                    f"数据源：{route} ADC 反射光谱 → 当前帧分波段拟合"
                )
        except Exception as exc:
            if transport_open_failed:
                failed_text = (
                    "本地控制：USB 端口打开失败"
                    if source == SOURCE_LOCAL
                    else "局域网全功能通道：打开失败"
                )
                self._update_source_status(failed_text, "error")
            else:
                self._update_source_status("数据链路：启动失败", "error")
                QtWidgets.QMessageBox.warning(self, "数据链路启动失败", str(exc))
            self._set_checked_source(previous)
            if self._uses_local_pages(previous):
                try:
                    self._select_local_transport(previous)
                    self.source_stack.setCurrentWidget(self.local_page)
                    self._set_local_menus_enabled(previous == SOURCE_LOCAL)
                    recovered = self.local_controller.acquire_local_control()
                except Exception:
                    recovered = False
                if not recovered:
                    self.current_source = None
                    self._set_checked_source(None)
                    self._update_source_status("数据链路：原链路恢复失败", "error")
            elif previous in (NETWORK_LAN, NETWORK_WAN):
                try:
                    self.remote_controller.activate_network(previous)
                    self.source_stack.setCurrentWidget(self.remote_page)
                except Exception:
                    self.current_source = None
                    self._set_checked_source(None)
            return

        self.current_source = source
        self._set_checked_source(source)
        if self._uses_local_pages(source):
            self._sync_local_mode()
        else:
            self._sync_remote_mode(self.remote_controller.current_mode)

    def _mode_requested(self, local_mode: int):
        if self._uses_local_pages():
            for action in self.local_controller.menu_page.actions():
                if int(action.data()) == int(local_mode):
                    action.trigger()
                    QtCore.QTimer.singleShot(0, self._sync_local_mode)
                    return
        elif self.current_source in (NETWORK_LAN, NETWORK_WAN):
            if local_mode == 0:
                self.remote_controller._request_mode(MODE_STRESS)
            elif local_mode == 1:
                self.remote_controller._request_mode(MODE_TEMPERATURE)

    def _sync_mode_availability(self):
        local = self._uses_local_pages()
        for index, button in enumerate(self.mode_buttons):
            button.setEnabled(
                local or (not local and index < 2)
            )
        self.open_finger_button.setEnabled(
            local
            and self.local_controller.MCU_mode == 0
            or (
                not local
                and self.current_source in (NETWORK_LAN, NETWORK_WAN)
                and self.remote_controller.current_mode == MODE_STRESS
            )
        )

    def _sync_local_mode(self):
        if self.current_source is not None and not self._uses_local_pages():
            return
        self._set_checked_mode(int(self.local_controller.MCU_mode))
        self._sync_mode_availability()

    @QtCore.pyqtSlot(int)
    def _sync_remote_mode(self, mode):
        if self._uses_local_pages() or self.current_source not in (
            NETWORK_LAN,
            NETWORK_WAN,
        ):
            return
        local_mode = (
            0 if mode == MODE_STRESS else 1 if mode == MODE_TEMPERATURE else None
        )
        self._set_checked_mode(local_mode)
        self._sync_mode_availability()

    @QtCore.pyqtSlot(bool, str)
    def _network_connection_changed(self, connected: bool, detail: str):
        if self._uses_local_pages() or self.current_source not in (
            NETWORK_LAN,
            NETWORK_WAN,
        ):
            return
        prefix = "已连接" if connected else "连接中"
        self._update_source_status(
            f"{prefix}：{detail}", "ok" if connected else "warning"
        )

    @QtCore.pyqtSlot(bool, str)
    def _lan_connection_changed(self, connected: bool, detail: str):
        if self.current_source != NETWORK_LAN or self.demo:
            return
        self._update_source_status(detail, "ok" if connected else "warning")

    def open_finger_tab(self):
        self.workspace_tabs.setCurrentWidget(self.finger_scroll)
        if self._uses_local_pages():
            self.local_controller.page_peak._update_finger_3d()
        elif self.current_source in (NETWORK_LAN, NETWORK_WAN):
            self.remote_controller._update_finger_3d()

    def _prepare_provisioning(self) -> bool:
        if self._uses_local_pages() and not self._can_leave_local():
            return False
        self.remote_controller.deactivate_network()
        self._release_local_control()
        self.current_source = None
        self._set_checked_source(None)
        self._set_checked_mode(None)
        self._set_local_menus_enabled(False)
        self._update_source_status("设备配置模式：所有数据链路已暂停", "warning")
        self.statusBar().showMessage(
            "正在通过 STM32 USB 写入 Wi-Fi/MQTT 参数；完成后在顶部重新选择数据来源"
        )
        return True

    def _prepare_ota(self) -> bool:
        """Give the OTA socket exclusive ownership of the board's LAN port."""
        if self._uses_local_pages() and not self._can_leave_local():
            return False
        learned_ip = self.lan_serial.board_ip or getattr(
            self.transport, "board_ip", None
        )
        if learned_ip:
            self.ota_page.set_board_ip(learned_ip)
        self.remote_controller.deactivate_network()
        self._release_local_control()
        self.current_source = None
        self._set_checked_source(None)
        self._set_checked_mode(None)
        self._set_local_menus_enabled(False)
        self._update_source_status("OTA 升级模式：激光与数据链路已暂停", "warning")
        self.statusBar().showMessage(
            "OTA 独占局域网端口；升级完成后在顶部重新选择数据来源"
        )
        return True

    def _open_generated_ota_package(self, package_path: str):
        """Load a promoted mode table package and take the operator to OTA."""
        if not self.ota_page.set_package_path(package_path):
            return
        self._generated_mode_table_package_path = str(Path(package_path).resolve())
        self.workspace_tabs.setCurrentIndex(3)
        self._workspace_changed(3)
        learned_ip = self.lan_serial.board_ip or getattr(
            self.transport, "board_ip", None
        )
        if learned_ip:
            self.ota_page.set_board_ip(learned_ip)
        self.statusBar().showMessage(
            "新点表固件包已校验；输入OTA密码后仍需点击“开始安全升级”确认上传"
        )

    def _after_ota_update(self, package):
        expected = self._generated_mode_table_package_path
        actual = str(Path(package.path).resolve()) if package is not None else None
        if expected and actual == expected:
            self._generated_mode_table_package_path = None
        try:
            activated = activate_pending_runtime_tables(actual) if actual else False
            if not activated:
                return
            self.local_controller.reload_mode_tables()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "新点表桌面端载入失败",
                f"板卡OTA已经完成，但桌面波长轴未能刷新：{exc}\n"
                "请关闭并重新打开本程序后再采集。",
            )
            return
        self.statusBar().showMessage(
            "新应力/温度点表已同步到板卡和桌面；板卡重启后可重新选择数据来源"
        )

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        # Some lifecycle harnesses intentionally exercise closeEvent with a
        # minimal owner.  Treat an absent optional worker exactly as idle.
        fast_worker = getattr(self, "_fast_flank_worker", None)
        if fast_worker is not None and fast_worker.isRunning():
            fast_worker.stop()
            if not self._fast_flank_close_pending:
                self._fast_flank_close_pending = True
                fast_worker.finished.connect(self._retry_close_after_fast_flank)
            self.statusBar().showMessage(
                "正在结束18点快速采集并确认DAC停光、SOA安全关光…"
            )
            event.ignore()
            return
        sweep_page = self.local_controller.page_ap6150
        temporary_page = getattr(self.local_controller, "page_temporary_test", None)
        temporary_worker = getattr(temporary_page, "worker", None)
        if temporary_worker is not None and temporary_worker.isRunning():
            temporary_page.stop()
            if not getattr(self, "_temporary_close_pending", False):
                self._temporary_close_pending = True
                temporary_worker.finished.connect(self.close)
            self.statusBar().showMessage("正在结束临时45点采集并确认安全关光…")
            event.ignore()
            return
        sweep_worker = getattr(sweep_page, "worker", None)
        if sweep_worker is not None and sweep_worker.isRunning():
            sweep_worker.stop()
            if not self._sweep_close_pending:
                self._sweep_close_pending = True
                sweep_worker.finished.connect(self._retry_close_after_sweep)
            self.statusBar().showMessage("正在结束自动校准并确认SOA安全关光…")
            event.ignore()
            return
        thread = getattr(self.provision_page, "_thread", None)
        if thread is not None and thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "正在写入配置",
                "请等待 Wi-Fi/MQTT 配置写入完成后再关闭程序。",
            )
            event.ignore()
            return
        if self.ota_page.busy:
            QtWidgets.QMessageBox.information(
                self,
                "正在 OTA 升级",
                "请等待 OTA 完成，或先在升级页点击取消。",
            )
            event.ignore()
            return
        if (temporary_page is not None
                and not temporary_page.confirm_save_before_close(self)):
            event.ignore()
            return
        self._closing = True
        try:
            # Seal any in-progress dataset before Qt destroys the timer and UI.
            # This only drains the background file queue; it never issues a CNC
            # command and therefore does not alter the established safe motion
            # path or the emergency-stop behaviour.
            if self._coverage_active or self._coverage_writer is not None:
                self._finish_coverage_run(False, "软件关闭，采集会话已安全封存")
            self.mach3_position_timer.stop()
            self.remote_controller.deactivate_network()
            self.local_controller.close()
            self.lan_serial.close()
            self.finger_controller.close()
            self.mach3_emergency.close()
        finally:
            event.accept()

    def _retry_close_after_sweep(self):
        self._sweep_close_pending = False
        QtCore.QTimer.singleShot(0, self.close)

    def _retry_close_after_fast_flank(self):
        self._fast_flank_close_pending = False
        QtCore.QTimer.singleShot(0, self.close)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FBG 光栅解调统一工作站")
    parser.add_argument(
        "--source",
        choices=(SOURCE_LOCAL, NETWORK_LAN, NETWORK_WAN),
        default=SOURCE_LOCAL,
        help="启动时的数据来源",
    )
    parser.add_argument(
        "--config",
        default=str(APP_DIR / "fbg_remote_config.yaml"),
        help="MQTT 配置文件",
    )
    parser.add_argument("--device-id", help="临时覆盖设备编号")
    parser.add_argument("--demo", action="store_true", help="使用内置演示光谱")
    parser.add_argument(
        "--demo-seconds", type=float, default=0.0, help="演示自动退出秒数"
    )
    parser.add_argument(
        "--view",
        choices=("spectrum", "finger", "settings", "ota"),
        default="spectrum",
        help="启动时显示的主页",
    )
    parser.add_argument("--screenshot", help="调试：保存主窗口截图")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("FBG OptiSense Studio")
    app.setWindowIcon(QtGui.QIcon(make_logo(128)))
    install_theme(app)
    initial_source = NETWORK_LAN if args.demo else args.source
    window = UnifiedFbgStudio(
        initial_source=initial_source,
        config_path=Path(args.config),
        device_id=args.device_id,
        demo=args.demo,
    )
    polish_plots(window)
    window.show()

    view_indexes = {"spectrum": 0, "finger": 1, "settings": 2, "ota": 3}
    window.workspace_tabs.setCurrentIndex(view_indexes[args.view])
    if args.demo:
        QtCore.QTimer.singleShot(
            650, lambda: window.remote_controller._request_mode(MODE_STRESS)
        )
    if args.screenshot:
        screenshot_path = Path(args.screenshot).resolve()

        def save_screenshot():
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(screenshot_path))

        QtCore.QTimer.singleShot(1800, save_screenshot)
    if args.demo_seconds > 0:
        QtCore.QTimer.singleShot(
            int(args.demo_seconds * 1000), lambda: (window.close(), app.quit())
        )
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
