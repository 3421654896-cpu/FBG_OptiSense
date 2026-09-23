"""LAN/WAN viewer for raw FBG samples, fitted spectra and mode control.

The viewer never opens the STM32 USB CDC port or talks to the laser directly.  It consumes
versioned telemetry from MQTT, reproduces the local segment-aware fitting path,
and can request one of the two scanning modes through acknowledged commands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
import zlib
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Dict

# Match the local GUI: locate the Windows Qt platform plugin before pyqtgraph or
# PyQt5 imports Qt.  This is required by the bundled .venv on this workstation.
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

import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5 import QtCore, QtWidgets

from fbg_mqtt_transport import load_remote_config, validate_device_id
from fbg_network_transport import (
    FbgSelectableTransport,
    NETWORK_LAN,
    NETWORK_WAN,
)
from fbg_peak_fitting import (
    AdaptivePeakTracker,
    PeakDisplayNormalizer,
    TemperaturePeakTracker,
    build_segments,
    detect_wavelength_gaps,
    fit_channel_segments,
    mask_unconnected_fbg_segments,
    rounded_display_curve,
)
from fbg_processing_common import (
    ADC_CODE_COUNT,
    ADC_REFERENCE_V,
    adc_codes_to_voltage,
    precision_median_fuse,
    scan_min_prominence,
)
from fbg_remote_protocol import MetadataFrame, ModeAck, TelemetryFrame
from mode_table_manager import MODE_POINT_COUNT_LIMITS
from responsive_layout import FlowLayout, ResponsiveScrollArea, ScrollContentWidget


MODE_STRESS = 0
MODE_TEMPERATURE = 3
MODE_NAMES = {MODE_STRESS: "应力寻峰模式", MODE_TEMPERATURE: "温度寻峰模式"}
CHANNEL_COLORS = ("#FFD84D", "#FF7373", "#77E36E", "#77B9FF")


def load_wave_tables(yaml_path: Path) -> Dict[int, np.ndarray]:
    def read_axis(path: Path, mode_name: str) -> np.ndarray:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = data.get("Wave_DATA")
        lower, upper = MODE_POINT_COUNT_LIMITS[mode_name]
        if not isinstance(rows, list) or not lower <= len(rows) <= upper:
            raise ValueError(
                f"{path.name} 应包含 {lower}～{upper} 个 Wave_DATA 点"
            )
        axis = np.asarray(
            [float(row[0]) + float(row[1]) * 0.001 for row in rows],
            dtype=float,
        )
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{path.name} Wave_DATA 必须是严格递增的有限波长")
        return axis

    temperature = read_axis(Path(yaml_path), "temperature")
    stress = read_axis(Path(yaml_path).with_name("stress_wave_const.yaml"), "stress")
    return {MODE_STRESS: stress, MODE_TEMPERATURE: temperature}


class TransportBridge(QtCore.QObject):
    telemetry = QtCore.pyqtSignal(object)
    metadata = QtCore.pyqtSignal(object)
    ack = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    connection = QtCore.pyqtSignal(bool, str)


class DemoSource(QtCore.QObject):
    """Local signal generator used to verify the GUI without an MQTT broker."""

    def __init__(self, wave_tables, bridge, parent=None):
        super().__init__(parent)
        self.wave_tables = wave_tables
        self.bridge = bridge
        self.mode = MODE_TEMPERATURE
        self.seq = 0
        self.boot_id = 0x44454D4F  # "DEMO"
        self.started_at = time.monotonic()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._emit_frame)

    @property
    def connected(self):
        return self.timer.isActive()

    def start(self):
        self.timer.start(1000)
        self.bridge.connection.emit(True, "演示数据源已启动")
        self.bridge.status.emit('{"online":true,"source":"demo"}')
        for mode, waves in self.wave_tables.items():
            self.bridge.metadata.emit(
                MetadataFrame(
                    mode=int(mode),
                    table_crc32=self._local_crc(waves),
                    wavelength_pm=tuple(
                        int(value) for value in np.rint(waves * 1000.0)
                    ),
                )
            )
        self._emit_frame()

    def stop(self):
        self.timer.stop()
        self.bridge.connection.emit(False, "演示数据源已停止")

    def publish_mode(self, mode, command_id):
        if mode not in MODE_NAMES:
            raise ValueError("不支持的模式")
        accepted = ModeAck(int(command_id), "ACCEPTED", int(mode), 0)
        QtCore.QTimer.singleShot(60, lambda: self.bridge.ack.emit(accepted))

        def apply_mode():
            self.mode = int(mode)
            self.timer.setInterval(125 if self.mode == MODE_STRESS else 1000)
            applied = ModeAck(int(command_id), "APPLIED", self.mode, self.seq)
            self.bridge.ack.emit(applied)
            self._emit_frame()

        QtCore.QTimer.singleShot(260, apply_mode)

    @staticmethod
    def _local_crc(waves):
        pm = np.rint(np.asarray(waves) * 1000.0).astype(">u4", copy=False)
        return zlib.crc32(pm.tobytes()) & 0xFFFFFFFF

    def _emit_frame(self):
        waves = self.wave_tables[self.mode]
        segments = build_segments(len(waves), detect_wavelength_gaps(waves))
        phase = time.monotonic() - self.started_at
        channels = range(4) if self.mode == MODE_STRESS else range(2)
        channel_mask = 0x0F if self.mode == MODE_STRESS else 0x03
        volts = np.full((4, len(waves)), 0.24, dtype=float)

        if self.mode == MODE_STRESS:
            allowed = {2: range(0, 3), 0: range(3, 6), 1: range(6, 9)}
            shift_nm = 0.025 * math.sin(phase * 2.2)
        else:
            # Demonstrate that either precision channel may contain a valid
            # peak in any wavelength section.
            allowed = {
                0: range(0, len(segments), 2),
                1: range(1, len(segments), 2),
            }
            shift_nm = 0.004 * math.sin(phase * 0.18)

        for channel, segment_ids in allowed.items():
            for segment_id in segment_ids:
                segment = segments[segment_id]
                x = waves[segment]
                center = float((x[0] + x[-1]) * 0.5 + shift_nm)
                sigma = max(float(x[-1] - x[0]) / 5.0, 0.025)
                amplitude = 0.14 if channel == 2 else 1.55
                volts[channel, segment] += amplitude * np.exp(
                    -0.5 * ((x - center) / sigma) ** 2
                )
        if self.mode == MODE_STRESS:
            volts[3] = 1.0 + 0.08 * np.sin(waves * 0.8 + phase)

        rng = np.random.default_rng(self.seq + 1234)
        volts += rng.normal(0.0, 0.0015, size=volts.shape)
        codes = np.clip(np.rint(volts * ADC_CODE_COUNT / ADC_REFERENCE_V), 0, 4095)
        samples = tuple(
            tuple(int(codes[channel, point]) for channel in channels)
            for point in range(len(waves))
        )
        frame = TelemetryFrame(
            mode=self.mode,
            flags=0,
            boot_id=self.boot_id,
            seq=self.seq,
            uptime_ms=int((time.monotonic() - self.started_at) * 1000.0),
            temperature_mC=int(25000 + 800 * math.sin(phase * 0.08)),
            table_crc32=self._local_crc(waves),
            gainmask0=0,
            gainmask1=0,
            channel_mask=channel_mask,
            samples=samples,
        )
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.bridge.telemetry.emit(frame)


class RemoteViewer(QtWidgets.QMainWindow):
    mode_changed = QtCore.pyqtSignal(int)

    def __init__(self, wave_tables, bridge, transport, device_id, parent=None):
        super().__init__(parent)
        self.wave_tables = wave_tables
        self.bridge = bridge
        self.transport = transport
        self.device_id = device_id

        self.current_mode = None
        self.active_table_crc = None
        self.metadata_cache = {}
        self.waves = np.array([], dtype=float)
        self.segments = []
        self.trackers = []
        self.precision_history = deque(maxlen=5)
        self.normalizer = PeakDisplayNormalizer(channel_count=4)
        self.pending_command_id = None
        self.pending_mode = None
        self.pending_accepted = False
        self.pending_retry_count = 0
        self.mqtt_connected = False
        self.device_status_online = False
        self.local_control_active = False
        self.lan_control_active = False
        self.last_telemetry_at = None
        self.arrival_times = deque(maxlen=80)
        self.last_boot_id = None
        self.last_seq = None
        self.last_uptime_ms = None
        self.last_gain_masks = None
        self.restart_candidate = None
        self.missing_sequences = 0
        self.out_of_order_sequences = 0
        self.latest_peak_centers = [np.array([], dtype=float) for _ in range(4)]
        self.latest_frame_number = 0
        self.latest_raw_channels = []
        self.latest_analysis_channels = []
        self.finger_3d_window = None
        self.finger_3d_open_handler = None

        self.setWindowTitle("光栅解调网络监控（原始数据 + 拟合）")
        self.resize(1500, 980)
        self._build_ui()

        bridge.telemetry.connect(self._on_telemetry)
        bridge.metadata.connect(self._on_metadata)
        bridge.ack.connect(self._on_ack)
        bridge.status.connect(self._on_device_status)
        bridge.connection.connect(self._on_connection)
        self.stale_timer = QtCore.QTimer(self)
        self.stale_timer.timeout.connect(self._refresh_online_state)
        self.stale_timer.start(500)
        self.telemetry_timer = QtCore.QTimer(self)
        self.telemetry_timer.timeout.connect(self._poll_latest_telemetry)
        self.telemetry_timer.start(20)
        self._show_waiting_for_table(MODE_TEMPERATURE, None)

    def _poll_latest_telemetry(self):
        take_latest = getattr(self.transport, "take_latest_telemetry", None)
        if take_latest is None:
            return
        frame = take_latest()
        if frame is not None:
            self._on_telemetry(frame)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        central.setObjectName("remotePage")
        self.setCentralWidget(central)
        page_layout = QtWidgets.QVBoxLayout(central)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = ResponsiveScrollArea()
        self.page_scroll.setObjectName("remotePageScroll")
        self.page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = ScrollContentWidget()
        content.setObjectName("remotePageContent")
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(9)
        self.page_scroll.setWidget(content)
        page_layout.addWidget(self.page_scroll)

        status_card = QtWidgets.QFrame()
        status_card.setObjectName("remoteCard")
        status_row = FlowLayout(
            status_card,
            margin=9,
            horizontal_spacing=8,
            vertical_spacing=7,
        )
        self.connection_label = QtWidgets.QLabel("网络：未连接")
        self.online_label = QtWidgets.QLabel("设备：离线")
        self.mode_label = QtWidgets.QLabel("模式：等待数据")
        self.temperature_label = QtWidgets.QLabel("板温：-- ℃")
        self.rate_label = QtWidgets.QLabel("刷新：0.00 Hz")
        self.sequence_label = QtWidgets.QLabel("序号：--  丢帧：0")
        self.crc_label = QtWidgets.QLabel("波长表CRC：--------")
        for label in (
            self.connection_label,
            self.online_label,
            self.mode_label,
            self.temperature_label,
            self.rate_label,
            self.sequence_label,
            self.crc_label,
        ):
            label.setMinimumHeight(28)
            label.setObjectName("metricBadge")
            label.setWordWrap(True)
            status_row.addWidget(label)
        outer.addWidget(status_card)

        board_card = QtWidgets.QFrame()
        board_card.setObjectName("statusCard")
        board_row = FlowLayout(
            board_card,
            margin=8,
            horizontal_spacing=10,
            vertical_spacing=7,
        )
        self.fan_status_label = QtWidgets.QLabel("风扇：等待板温控制状态")
        self.fan_status_label.setWordWrap(True)
        board_row.addWidget(self.fan_status_label)
        outer.addWidget(board_card)

        self.network_panel = QtWidgets.QWidget()
        network_row = FlowLayout(
            self.network_panel, horizontal_spacing=8, vertical_spacing=7
        )
        network_row.addWidget(QtWidgets.QLabel("数据链路："))
        self.network_combo = QtWidgets.QComboBox()
        self.network_combo.addItem("局域网直连（同一 Wi-Fi）", NETWORK_LAN)
        self.network_combo.addItem("广域网 MQTT", NETWORK_WAN)
        current_network = getattr(self.transport, "network_mode", None)
        initial_network = current_network or getattr(
            self.transport, "initial_mode", NETWORK_LAN
        )
        initial_index = self.network_combo.findData(initial_network)
        if initial_index >= 0:
            self.network_combo.setCurrentIndex(initial_index)
        selectable_network = hasattr(self.transport, "select_network")
        if not selectable_network:
            self.network_combo.clear()
            self.network_combo.addItem("演示数据", "demo")
        self.network_combo.setEnabled(selectable_network)
        self.network_combo.currentIndexChanged.connect(self._on_network_changed)
        network_row.addWidget(self.network_combo)
        self.network_note = QtWidgets.QLabel(
            "两条网络链路二选一；打开 STM32 USB 本地调试时，板端会同时暂停两者。"
        )
        self.network_note.setObjectName("softHint")
        self.network_note.setWordWrap(True)
        network_row.addWidget(self.network_note)
        outer.addWidget(self.network_panel)

        command_card = QtWidgets.QFrame()
        command_card.setObjectName("controlGroup")
        command_row = FlowLayout(
            command_card,
            margin=9,
            horizontal_spacing=8,
            vertical_spacing=7,
        )
        command_row.addWidget(QtWidgets.QLabel("远程模式："))
        self.stress_button = QtWidgets.QPushButton("切换到应力寻峰模式")
        self.temperature_button = QtWidgets.QPushButton("切换到温度寻峰模式")
        self.stress_button.clicked.connect(lambda: self._request_mode(MODE_STRESS))
        self.temperature_button.clicked.connect(
            lambda: self._request_mode(MODE_TEMPERATURE)
        )
        command_row.addWidget(self.stress_button)
        command_row.addWidget(self.temperature_button)
        self.finger_3d_button = QtWidgets.QPushButton("机械手指3D应力定位")
        self.finger_3d_button.setProperty("role", "primary")
        self.finger_3d_button.clicked.connect(self.open_finger_3d)
        command_row.addWidget(self.finger_3d_button)
        self.command_label = QtWidgets.QLabel("命令：空闲")
        self.command_label.setObjectName("softHint")
        self.command_label.setWordWrap(True)
        command_row.addWidget(self.command_label)
        outer.addWidget(command_card)

        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.raw_plot = pg.PlotWidget(title="原始 ADC 采样（未滤波、未拟合、未数字放大）")
        self.raw_plot.setObjectName("measurementPlot")
        self.raw_plot.setLabel("bottom", "波长", units="nm")
        self.raw_plot.setLabel("left", "原始 ADC 电压", units="V")
        self.raw_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.raw_plot.getAxis("left").enableAutoSIPrefix(False)
        self.raw_plot.setYRange(-0.1, 2.5)
        self.raw_plot.showGrid(x=True, y=True, alpha=0.25)
        self.raw_plot.addLegend()
        self.raw_curves = [
            self.raw_plot.plot(
                pen=pg.mkPen(CHANNEL_COLORS[channel], width=1),
                symbol="o",
                symbolSize=4,
                symbolBrush=CHANNEL_COLORS[channel],
                symbolPen=None,
                name=f"CH{channel} 原始",
            )
            for channel in range(4)
        ]

        self.plot = pg.PlotWidget(title="分段拟合光谱（峰值由此计算）")
        self.plot.setObjectName("measurementPlot")
        self.plot.setLabel("bottom", "波长", units="nm")
        self.plot.setLabel("left", "拟合显示电压", units="V")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.setYRange(-0.1, 2.5)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()
        self.plot.setXLink(self.raw_plot)
        self.curves = [
            self.plot.plot(
                pen=pg.mkPen(CHANNEL_COLORS[channel], width=2),
                name=f"CH{channel} 拟合",
            )
            for channel in range(4)
        ]
        plot_splitter.addWidget(self.raw_plot)
        plot_splitter.addWidget(self.plot)
        plot_splitter.setSizes([330, 390])
        outer.addWidget(plot_splitter, 1)

        self.peak_table = QtWidgets.QTableWidget(4, 0)
        self.peak_table.setVerticalHeaderLabels([f"CH{i}" for i in range(4)])
        self.peak_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peak_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.peak_table.setMaximumHeight(190)
        self.peak_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        outer.addWidget(self.peak_table)

        self.message_label = QtWidgets.QLabel("等待遥测数据")
        self.message_label.setObjectName("softHint")
        self.message_label.setWordWrap(True)
        self.message_label.setContentsMargins(4, 2, 4, 2)
        outer.addWidget(self.message_label)

    def _on_network_changed(self, index):
        select_network = getattr(self.transport, "select_network", None)
        if select_network is None:
            return False
        requested = self.network_combo.itemData(int(index))
        previous = getattr(self.transport, "network_mode", None)
        if requested == previous:
            return True
        try:
            select_network(str(requested))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "网络切换失败", str(exc))
            restore = self.network_combo.findData(previous)
            if restore >= 0:
                self.network_combo.blockSignals(True)
                self.network_combo.setCurrentIndex(restore)
                self.network_combo.blockSignals(False)
            return False

        self.mqtt_connected = False
        self.device_status_online = False
        self.local_control_active = False
        self.lan_control_active = False
        self.last_telemetry_at = None
        self.last_boot_id = None
        self.last_seq = None
        self.last_uptime_ms = None
        self.restart_candidate = None
        self.arrival_times.clear()
        self.pending_command_id = None
        self.pending_mode = None
        self.pending_accepted = False
        self.pending_retry_count = 0
        self.command_label.setText("命令：网络链路已切换")
        self.message_label.setText(
            "已切换到局域网直连"
            if requested == NETWORK_LAN
            else "已切换到广域网 MQTT"
        )
        self._refresh_mode_buttons()
        self._refresh_online_state()
        return True

    def activate_network(self, mode=NETWORK_LAN):
        """Start/select a network source when hosted by the unified studio."""
        select_network = getattr(self.transport, "select_network", None)
        if select_network is None:
            if not getattr(self.transport, "connected", False):
                self.transport.start()
            return
        index = self.network_combo.findData(str(mode))
        if index < 0:
            raise ValueError(f"不支持的数据链路：{mode}")
        self.network_combo.blockSignals(True)
        self.network_combo.setCurrentIndex(index)
        self.network_combo.blockSignals(False)
        if not self._on_network_changed(index):
            raise RuntimeError("数据链路未能启动")

    def deactivate_network(self):
        """Stop all network receivers without destroying the viewer state."""
        self.transport.stop()
        self.mqtt_connected = False
        self.device_status_online = False
        self.local_control_active = False
        self.lan_control_active = False
        self.last_telemetry_at = None
        self.pending_command_id = None
        self.pending_mode = None
        self.pending_accepted = False
        self.pending_retry_count = 0
        self.connection_label.setText("网络：已暂停")
        self.command_label.setText("命令：网络链路已暂停")
        self._refresh_mode_buttons()
        self._refresh_online_state()

    def _set_mode(self, mode, waves, table_crc):
        if mode not in MODE_NAMES:
            return
        self.current_mode = int(mode)
        self.active_table_crc = int(table_crc)
        self.waves = np.asarray(waves, dtype=float)
        self.segments = build_segments(
            len(self.waves), detect_wavelength_gaps(self.waves)
        )
        self._reset_tracking()
        self.raw_plot.setXRange(
            float(self.waves[0]) - 0.5, float(self.waves[-1]) + 0.5
        )
        self.plot.setXRange(float(self.waves[0]) - 0.5, float(self.waves[-1]) + 0.5)
        visible = {0, 1} if self.current_mode == MODE_TEMPERATURE else {0, 1, 2, 3}
        for channel, curve in enumerate(self.curves):
            curve.setVisible(channel in visible)
            curve.setData([], [])
            self.raw_curves[channel].setVisible(channel in visible)
            self.raw_curves[channel].setData([], [])
        for row in range(4):
            self.peak_table.setRowHidden(row, row not in visible)
        self.peak_table.setColumnCount(len(self.segments))
        self.peak_table.setHorizontalHeaderLabels(
            [f"峰{index + 1}" for index in range(len(self.segments))]
        )
        self.peak_table.clearContents()
        self.mode_label.setText(f"模式：{MODE_NAMES[self.current_mode]}")
        self._refresh_mode_buttons()
        self.mode_changed.emit(self.current_mode)

    def _reset_tracking(self):
        if self.current_mode not in MODE_NAMES or not self.segments:
            self.trackers = []
        else:
            if self.current_mode == MODE_TEMPERATURE:
                self.trackers = [
                    TemperaturePeakTracker(len(self.segments)) for _ in range(4)
                ]
            else:
                # Match the local stress path: every valid current-frame
                # wavelength is the force measurement and is never delayed by
                # a second-frame confirmation or temporal smoothing.
                self.trackers = [
                    AdaptivePeakTracker(
                        len(self.segments), immediate_response=True
                    )
                    for _ in range(4)
                ]
        self.precision_history.clear()
        self.last_gain_masks = None
        self.normalizer.reset()

    def _show_waiting_for_table(self, mode, table_crc):
        self.current_mode = int(mode)
        self.active_table_crc = None
        self.waves = np.array([], dtype=float)
        self.segments = []
        self.precision_history.clear()
        for curve in self.raw_curves:
            curve.setData([], [])
        for curve in self.curves:
            curve.setData([], [])
        self.peak_table.setColumnCount(0)
        suffix = f"（CRC {table_crc:08X}）" if table_crc is not None else ""
        self.mode_label.setText(f"模式：{MODE_NAMES[int(mode)]}，等待波长表{suffix}")
        self.message_label.setText("等待与原始帧表版本匹配的波长元数据")
        self._refresh_mode_buttons()

    @QtCore.pyqtSlot(object)
    def _on_metadata(self, metadata):
        if not isinstance(metadata, MetadataFrame) or metadata.mode not in MODE_NAMES:
            return
        waves = np.asarray(metadata.wavelength_pm, dtype=float) * 0.001
        if (
            waves.size < 2
            or not np.all(np.isfinite(waves))
            or np.any(np.diff(waves) <= 0.0)
        ):
            self.message_label.setText("收到无效的波长表元数据")
            return
        self.metadata_cache[(metadata.mode, metadata.table_crc32)] = waves
        self.message_label.setText(
            f"已缓存{MODE_NAMES[metadata.mode]}波长表：{waves.size}点，"
            f"CRC {metadata.table_crc32:08X}"
        )

    def _refresh_mode_buttons(self):
        can_send = self._mode_command_link_ready() and self.pending_command_id is None
        self.stress_button.setEnabled(can_send and self.current_mode != MODE_STRESS)
        self.temperature_button.setEnabled(
            can_send and self.current_mode != MODE_TEMPERATURE
        )
        for button, active in (
            (self.stress_button, self.current_mode == MODE_STRESS),
            (self.temperature_button, self.current_mode == MODE_TEMPERATURE),
        ):
            button.setProperty("activeMode", bool(active))
            button.style().unpolish(button)
            button.style().polish(button)
        self.finger_3d_button.setEnabled(
            self.current_mode == MODE_STRESS and bool(self.segments)
        )

    def _lan_link_connected(self):
        """Return whether the selected transport has a live LAN TCP link.

        LAN is deliberately usable before the first ADC telemetry frame.  The
        board boots in the safe EXTRA state, so requiring a telemetry-derived
        online flag here would prevent the very first scan command that can
        produce telemetry.
        """
        return bool(
            self.mqtt_connected
            and getattr(self.transport, "network_mode", None) == NETWORK_LAN
        )

    def _mode_command_link_ready(self):
        """Separate command-path readiness from ADC-data freshness."""
        if not self.mqtt_connected or self.local_control_active:
            return False
        if self.lan_control_active and not self._lan_link_connected():
            return False
        return bool(self.device_status_online or self._lan_link_connected())

    def _start_selected_lan_mode_if_idle(self):
        """Start the selected scan after LAN connect when the board is idle."""
        if (
            self._lan_link_connected()
            and not self.device_status_online
            and self.pending_command_id is None
            and self.current_mode in MODE_NAMES
        ):
            self._request_mode(self.current_mode)

    def open_finger_3d(self):
        """Show/select the 3D view fed by the current network spectrum."""
        if self.current_mode != MODE_STRESS:
            return
        if callable(self.finger_3d_open_handler):
            self.finger_3d_open_handler()
            self._update_finger_3d()
            return
        if self.finger_3d_window is None:
            try:
                from mechanical_finger_3d import MechanicalFinger3DWindow

                self.finger_3d_window = MechanicalFinger3DWindow(parent=self)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "3D应力界面启动失败",
                    f"无法建立3D机械手指界面：\n{exc}",
                )
                self.finger_3d_window = None
                return
        self.finger_3d_window.show()
        self.finger_3d_window.raise_()
        self.finger_3d_window.activateWindow()
        self._update_finger_3d()

    def _update_finger_3d(self):
        if self.current_mode != MODE_STRESS or self.finger_3d_window is None:
            return
        self.finger_3d_window.update_from_spectrum(
            self.latest_peak_centers, self.latest_frame_number
        )

    def _request_mode(self, mode):
        if self.pending_command_id is not None:
            return
        if self.local_control_active:
            self.command_label.setText("板卡正在由 USB 本地程序控制，远端命令已停用")
            self._refresh_mode_buttons()
            return
        if self.lan_control_active:
            self.command_label.setText("板卡正在向局域网查看器发送，广域网命令已停用")
            self._refresh_mode_buttons()
            return
        if not self._mode_command_link_ready():
            self.command_label.setText("设备尚未在线，未发送模式命令")
            self._refresh_mode_buttons()
            return
        command_id = secrets.randbits(32)
        try:
            self.transport.publish_mode(int(mode), command_id)
        except Exception as exc:
            self.command_label.setText(f"命令发送失败：{exc}")
            return
        self.pending_command_id = command_id
        self.pending_mode = int(mode)
        self.pending_accepted = False
        self.pending_retry_count = 1
        self.command_label.setText(
            f"命令 {command_id:08X}：等待设备接收 {MODE_NAMES[int(mode)]}"
        )
        self._refresh_mode_buttons()
        self._schedule_command_retry(command_id)
        QtCore.QTimer.singleShot(30000, lambda: self._command_timeout(command_id))

    def _schedule_command_retry(self, command_id, delay_ms=3000):
        QtCore.QTimer.singleShot(
            int(delay_ms), lambda: self._retry_pending_command(command_id)
        )

    def _retry_pending_command(self, command_id):
        if self.pending_command_id != command_id or self.pending_accepted:
            return
        if (
            not self._mode_command_link_ready()
        ):
            self._schedule_command_retry(command_id)
            return
        if self.pending_retry_count >= 5:
            return
        try:
            self.transport.publish_mode(int(self.pending_mode), command_id)
        except Exception as exc:
            self.command_label.setText(f"命令重试失败：{exc}")
            self._schedule_command_retry(command_id)
            return
        self.pending_retry_count += 1
        self.command_label.setText(
            f"命令 {command_id:08X}：等待设备接收（第 {self.pending_retry_count} 次发送）"
        )
        self._schedule_command_retry(command_id)

    def _command_timeout(self, command_id):
        if self.pending_command_id != command_id:
            return
        self.command_label.setText(f"命令 {command_id:08X}：30秒内未完成")
        self.pending_command_id = None
        self.pending_mode = None
        self.pending_accepted = False
        self.pending_retry_count = 0
        self._refresh_mode_buttons()

    @QtCore.pyqtSlot(object)
    def _on_ack(self, ack):
        if not isinstance(ack, ModeAck):
            return
        if ack.command_id != self.pending_command_id:
            return
        if ack.mode != self.pending_mode:
            self.command_label.setText(
                f"命令 {ack.command_id:08X}：设备返回了不匹配的模式，结果未知"
            )
            self.pending_command_id = None
            self.pending_mode = None
            self.pending_accepted = False
            self.pending_retry_count = 0
            self._refresh_mode_buttons()
            return
        if ack.status == "ACCEPTED":
            self.pending_accepted = True
            self.command_label.setText(
                f"命令 {ack.command_id:08X}：设备已接收，等待实际切换"
            )
        elif ack.status == "APPLIED":
            self.command_label.setText(
                f"命令 {ack.command_id:08X}：已应用，起始序号 {ack.applied_seq}"
            )
            self.pending_command_id = None
            self.pending_mode = None
            self.pending_accepted = False
            self.pending_retry_count = 0
            self._refresh_mode_buttons()
        elif ack.status == "REJECTED":
            self.command_label.setText(f"命令 {ack.command_id:08X}：设备拒绝")
            self.pending_command_id = None
            self.pending_mode = None
            self.pending_accepted = False
            self.pending_retry_count = 0
            self._refresh_mode_buttons()

    @QtCore.pyqtSlot(bool, str)
    def _on_connection(self, connected, detail):
        self.mqtt_connected = bool(connected)
        self.connection_label.setText(f"网络：{detail}")
        self.connection_label.setProperty(
            "statusKind", "ok" if connected else "error"
        )
        self.connection_label.style().unpolish(self.connection_label)
        self.connection_label.style().polish(self.connection_label)
        if not connected:
            self.restart_candidate = None
        if not connected and self.pending_command_id is not None:
            self.command_label.setText(
                f"命令 {self.pending_command_id:08X}：网络断开，恢复后自动重试"
            )
            self.pending_accepted = False
            self.pending_retry_count = 0
        elif connected and self.pending_command_id is not None and self._mode_command_link_ready():
            self._schedule_command_retry(self.pending_command_id, 1000)
        elif connected and self._lan_link_connected():
            # Give an already-running temperature scan enough time to deliver
            # its first frame.  If no frame arrives, the board is in its safe
            # idle state and the selected mode is started automatically.
            QtCore.QTimer.singleShot(1500, self._start_selected_lan_mode_if_idle)
        self._refresh_mode_buttons()

    @QtCore.pyqtSlot(str)
    def _on_device_status(self, payload):
        was_online = self.device_status_online
        normalized = payload.strip().lower()
        was_local = self.local_control_active
        was_lan = self.lan_control_active
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                self.device_status_online = bool(decoded.get("online", False))
                control = str(decoded.get("control", "")).strip().lower()
                self.local_control_active = control == "local"
                self.lan_control_active = control == "lan"
            else:
                self.device_status_online = bool(decoded)
                self.local_control_active = False
                self.lan_control_active = False
        except Exception:
            self.device_status_online = normalized in {"1", "on", "online", "true"} or normalized.startswith(
                "online|"
            )
            self.local_control_active = normalized.startswith("online|local|")
            self.lan_control_active = normalized.startswith("online|lan|")
        if (self.local_control_active or self.lan_control_active) and self.pending_command_id is not None:
            owner = "USB 本地控制" if self.local_control_active else "局域网控制"
            self.command_label.setText(
                f"命令 {self.pending_command_id:08X}：已取消，板卡切换为{owner}"
            )
            self.pending_command_id = None
            self.pending_mode = None
            self.pending_accepted = False
            self.pending_retry_count = 0
        elif self.local_control_active:
            self.command_label.setText("USB 本地程序正在控制板卡，远端命令已停用")
        elif self.lan_control_active:
            self.command_label.setText("局域网查看器正在控制板卡，广域网命令已停用")
        if not self.device_status_online and self.pending_command_id is not None:
            self.pending_accepted = False
            self.pending_retry_count = 0
        elif (
            self.device_status_online
            and not was_online
            and not self.local_control_active
            and not self.lan_control_active
            and self.pending_command_id is not None
        ):
            self._schedule_command_retry(self.pending_command_id, 1000)
        if was_local and not self.local_control_active:
            self.command_label.setText("USB 本地程序已退出，远端控制已自动恢复")
        elif was_lan and not self.lan_control_active:
            self.command_label.setText("局域网控制已释放，广域网控制已自动恢复")
        self._refresh_online_state()
        self._refresh_mode_buttons()

    def _accept_sequence(self, frame):
        uptime_wrapped = (
            self.last_uptime_ms is not None
            and self.last_uptime_ms > 0xF0000000
            and frame.uptime_ms < 0x10000000
        )
        uptime_restarted = (
            self.last_uptime_ms is not None
            and not uptime_wrapped
            and frame.uptime_ms + 2000 < self.last_uptime_ms
        )

        if self.last_boot_id != frame.boot_id:
            restarted = self.last_boot_id is not None
            self.last_boot_id = frame.boot_id
            self.last_seq = frame.seq
            self.last_uptime_ms = frame.uptime_ms
            self.restart_candidate = None
            self.missing_sequences = 0
            self.out_of_order_sequences = 0
            if restarted:
                self.arrival_times.clear()
                self._reset_tracking()
            return True

        if uptime_restarted:
            candidate = self.restart_candidate
            confirmed = (
                candidate is not None
                and candidate[0] == frame.boot_id
                and frame.seq == ((candidate[1] + 1) & 0xFFFFFFFF)
                and frame.uptime_ms > candidate[2]
                and frame.uptime_ms - candidate[2] <= 5000
            )
            if confirmed:
                self.last_seq = frame.seq
                self.last_uptime_ms = frame.uptime_ms
                self.restart_candidate = None
                self.missing_sequences = 0
                self.out_of_order_sequences = 0
                self.arrival_times.clear()
                self._reset_tracking()
                return True

            # A single delayed old frame must not move the accepted sequence
            # backwards.  Keep it only as a restart candidate; a consecutive
            # frame on the same low-uptime timeline is required to confirm it.
            self.restart_candidate = (frame.boot_id, frame.seq, frame.uptime_ms)
            self.out_of_order_sequences += 1
            return False

        self.restart_candidate = None
        expected = (int(self.last_seq) + 1) & 0xFFFFFFFF
        delta = (int(frame.seq) - expected) & 0xFFFFFFFF
        if delta == 0:
            self.last_seq = frame.seq
            self.last_uptime_ms = frame.uptime_ms
            return True
        elif delta < 0x80000000:
            self.missing_sequences += delta
            self.last_seq = frame.seq
            self.last_uptime_ms = frame.uptime_ms
            return True
        else:
            self.out_of_order_sequences += 1
            return False

    def _update_rate(self, frame):
        """Use device sequence/uptime so MQTT bursts do not inflate scan Hz."""
        sample = (int(frame.seq), int(frame.uptime_ms))
        self.arrival_times.append(sample)
        while (
            len(self.arrival_times) > 1
            and ((sample[1] - self.arrival_times[0][1]) & 0xFFFFFFFF) > 10000
        ):
            self.arrival_times.popleft()
        if len(self.arrival_times) >= 2:
            first_seq, first_ms = self.arrival_times[0]
            last_seq, last_ms = self.arrival_times[-1]
            elapsed_ms = (last_ms - first_ms) & 0xFFFFFFFF
            sequence_delta = (last_seq - first_seq) & 0xFFFFFFFF
            rate = (
                sequence_delta * 1000.0 / elapsed_ms
                if 0 < elapsed_ms < 0x80000000 and sequence_delta < 0x80000000
                else 0.0
            )
        else:
            rate = 0.0
        self.rate_label.setText(f"刷新：{rate:.2f} Hz")

    @QtCore.pyqtSlot(object)
    def _on_telemetry(self, frame):
        if not isinstance(frame, TelemetryFrame):
            return
        if frame.mode not in MODE_NAMES:
            self.message_label.setText(f"忽略不支持的模式 {frame.mode}")
            return
        now = time.monotonic()
        if not self._accept_sequence(frame):
            self.sequence_label.setText(
                f"序号：{self.last_seq}  丢帧：{self.missing_sequences}  "
                f"乱序：{self.out_of_order_sequences}（已丢弃）"
            )
            return
        self.last_telemetry_at = now
        self.device_status_online = True
        self._update_rate(frame)
        self._update_board_status(frame)
        table_key = (frame.mode, frame.table_crc32)
        remote_waves = self.metadata_cache.get(table_key)
        if remote_waves is None:
            self._show_waiting_for_table(frame.mode, frame.table_crc32)
            self.temperature_label.setText(
                f"板温：{frame.temperature_mC / 1000.0:.3f} ℃"
            )
            self.sequence_label.setText(
                f"序号：{frame.seq}  丢帧：{self.missing_sequences}"
            )
            self.crc_label.setText(
                f"波长表CRC：{frame.table_crc32:08X}（等待匹配元数据）"
            )
            self._refresh_online_state()
            return
        if (
            self.current_mode != frame.mode
            or self.active_table_crc != frame.table_crc32
        ):
            self._set_mode(frame.mode, remote_waves, frame.table_crc32)
        if frame.point_count != len(self.waves):
            self.message_label.setText(
                f"点数不匹配：收到 {frame.point_count}，当前模式需要 {len(self.waves)}"
            )
            return

        channel_ids = tuple(frame.selected_channels)
        sample_array = np.asarray(frame.samples, dtype=float)
        expected_shape = (len(self.waves), len(channel_ids))
        if sample_array.shape != expected_shape:
            self.message_label.setText(
                f"通道数据形状错误：{sample_array.shape}，应为 {expected_shape}"
            )
            return
        invalid_codes = int(np.count_nonzero(sample_array > 4095.0))
        sample_array[sample_array > 4095.0] = np.nan
        matrix = np.full((4, len(self.waves)), np.nan, dtype=float)
        for offset, channel in enumerate(channel_ids):
            matrix[channel] = adc_codes_to_voltage(sample_array[:, offset])
        raw_matrix = matrix.copy()

        if self.current_mode == MODE_TEMPERATURE:
            gain_masks = (frame.gainmask0, frame.gainmask1)
            if self.last_gain_masks is not None and gain_masks != self.last_gain_masks:
                self.precision_history.clear()
            self.last_gain_masks = gain_masks
            matrix = precision_median_fuse(
                self.precision_history, matrix, channel_ids
            )

        self.temperature_label.setText(f"板温：{frame.temperature_mC / 1000.0:.3f} ℃")
        self.sequence_label.setText(
            f"序号：{frame.seq}  丢帧：{self.missing_sequences}"
            + (
                f"  乱序：{self.out_of_order_sequences}"
                if self.out_of_order_sequences
                else ""
            )
        )
        self.crc_label.setText(f"波长表CRC：{frame.table_crc32:08X}")
        self._render(
            matrix,
            set(channel_ids),
            raw_channels=raw_matrix,
            frame_number=frame.seq,
        )
        warning = f"；{invalid_codes}个ADC码越界已忽略" if invalid_codes else ""
        self.message_label.setText(
            f"设备 {self.device_id}，启动ID {frame.boot_id:08X}，在线时长 "
            f"{frame.uptime_ms / 1000.0:.1f}s{warning}"
        )
        self._refresh_online_state()

    def _update_board_status(self, frame):
        duty = min(max(frame.fan_duty_permille / 10.0, 0.0), 100.0)
        if frame.thermal_flags & 0x0004:
            control = "温度传感器故障，安全满速"
        elif frame.thermal_flags & 0x0002:
            control = "≥33℃，满速锁定"
        else:
            control = "33℃ PID"
        tach = "，转速反馈异常" if frame.thermal_flags & 0x0008 else ""
        self.fan_status_label.setText(
            f"风扇：{duty:.1f}% / {frame.fan_rpm} RPM，{control}{tach}"
        )

    def _render(
        self, channels, present_channels, raw_channels=None, frame_number=None
    ):
        temperature_mode = self.current_mode == MODE_TEMPERATURE
        raw_channels = channels if raw_channels is None else raw_channels
        self.latest_raw_channels = [
            np.asarray(values, dtype=float).copy() for values in raw_channels
        ]
        self.latest_analysis_channels = [
            np.asarray(values, dtype=float).copy() for values in channels
        ]
        fbg_channels = {0, 1} if temperature_mode else {0, 1, 2}
        fits_by_channel = [[] for _ in range(4)]
        tracked_by_channel = [np.full(len(self.segments), np.nan) for _ in range(4)]

        for channel in range(4):
            if channel not in present_channels:
                self.curves[channel].setData([], [])
                self.raw_curves[channel].setData([], [])
                continue
            values = channels[channel]
            threshold = scan_min_prominence(0.03, channel, temperature_mode)
            fits = fit_channel_segments(
                self.waves,
                values,
                self.segments,
                min_prominence_v=threshold,
                allow_edge_peak=not temperature_mode,
            )
            mask_unconnected_fbg_segments(
                fits,
                channel,
                temperature_mode,
                stress_ch1_all_segments=not temperature_mode,
            )
            fits_by_channel[channel] = fits
            if channel in fbg_channels:
                tracked_by_channel[channel] = self.trackers[channel].update(fits)

        scales = (
            np.ones(4, dtype=float)
            if temperature_mode
            else self.normalizer.update(fits_by_channel)
        )
        visible = {0, 1} if temperature_mode else {0, 1, 2, 3}
        for channel in visible:
            if channel not in present_channels:
                continue
            fits = fits_by_channel[channel]
            raw_x, raw_y = rounded_display_curve(
                self.waves,
                raw_channels[channel],
                self.segments,
                fits,
                display_scale=1.0,
                fitted=False,
            )
            self.raw_curves[channel].setData(raw_x, raw_y, connect="finite")
            display_fits = []
            for index, fit in enumerate(fits):
                center = tracked_by_channel[channel][index]
                display_fits.append(
                    replace(fit, center_nm=float(center))
                    if fit.valid and np.isfinite(center)
                    else fit
                )
            scale = float(scales[channel]) if channel >= 2 else 1.0
            plot_x, plot_y = rounded_display_curve(
                self.waves,
                channels[channel],
                self.segments,
                display_fits,
                display_scale=scale,
                fitted=True,
            )
            self.curves[channel].setData(plot_x, plot_y, connect="finite")

        for row in range(4):
            for column in range(len(self.segments)):
                center = tracked_by_channel[row][column]
                text = f"{center:.4f}" if np.isfinite(center) else "—"
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                fits = fits_by_channel[row]
                if column < len(fits) and fits[column].valid:
                    item.setToolTip(
                        f"R²={fits[column].r_squared:.4f}，"
                        f"估计不确定度={fits[column].center_std_pm:.2f} pm"
                    )
                self.peak_table.setItem(row, column, item)

        self.latest_peak_centers = [
            np.asarray(values, dtype=float).copy() for values in tracked_by_channel
        ]
        if frame_number is not None:
            self.latest_frame_number = int(frame_number)
        self._update_finger_3d()

    def _refresh_online_state(self):
        stale_timeout = 15.0 if self.current_mode == MODE_TEMPERATURE else 5.0
        fresh = (
            self.last_telemetry_at is not None
            and time.monotonic() - self.last_telemetry_at <= stale_timeout
        )
        if self.mqtt_connected and self.local_control_active:
            text, kind = "设备：USB 本地调试中，网络原始数据已暂停", "info"
        elif self.mqtt_connected and self.lan_control_active:
            text, kind = "设备：局域网链路已占用，广域网原始数据已暂停", "info"
        elif self.mqtt_connected and fresh:
            text, kind = "设备：在线，数据正常", "ok"
        elif self.mqtt_connected and self.device_status_online:
            text, kind = "设备：已连接，但数据超时", "warning"
        else:
            text, kind = "设备：离线/数据超时", "error"
        self.online_label.setText(text)
        self.online_label.setProperty("statusKind", kind)
        self.online_label.style().unpolish(self.online_label)
        self.online_label.style().polish(self.online_label)

    def closeEvent(self, event):
        try:
            self.transport.stop()
        finally:
            if self.finger_3d_window is not None:
                self.finger_3d_window.close()
            super().closeEvent(event)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="光栅解调局域网/广域网监控")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("fbg_remote_config.yaml")),
        help="本机MQTT配置文件",
    )
    parser.add_argument("--device-id", help="临时覆盖配置中的device_id")
    parser.add_argument(
        "--network",
        choices=(NETWORK_LAN, NETWORK_WAN),
        default=NETWORK_LAN,
        help="启动时使用局域网直连或广域网 MQTT",
    )
    parser.add_argument("--demo", action="store_true", help="不连接MQTT，使用模拟数据")
    parser.add_argument(
        "--demo-seconds",
        type=float,
        default=0.0,
        help="演示模式自动关闭秒数（0表示不自动关闭）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("光栅解调网络监控")
    wave_tables = (
        load_wave_tables(Path(__file__).with_name("wave_const.yaml"))
        if args.demo
        else {}
    )
    bridge = TransportBridge()

    if args.demo:
        device_id = args.device_id or "demo-jdsu"
        transport = DemoSource(wave_tables, bridge)
    else:
        config_path = Path(args.config)
        config = None
        config_error = None
        if config_path.is_file():
            try:
                config = load_remote_config(config_path)
                if args.device_id:
                    config = replace(
                        config, device_id=validate_device_id(args.device_id)
                    )
            except Exception as exc:
                config_error = str(exc)
        elif args.network == NETWORK_WAN:
            config_error = "未找到 fbg_remote_config.yaml"
        if args.network == NETWORK_WAN and config is None:
            QtWidgets.QMessageBox.critical(
                None,
                "广域网配置错误",
                config_error or "广域网模式需要 MQTT 配置",
            )
            return 2
        device_id = (
            config.device_id
            if config is not None
            else validate_device_id(args.device_id or "fbg-local")
        )
        transport = FbgSelectableTransport(
            config,
            initial_mode=args.network,
            on_ack=bridge.ack.emit,
            on_status=bridge.status.emit,
            on_connection=bridge.connection.emit,
            on_metadata=bridge.metadata.emit,
        )

    viewer = RemoteViewer(wave_tables, bridge, transport, device_id)
    viewer.show()
    try:
        transport.start()
    except Exception as exc:
        QtWidgets.QMessageBox.critical(viewer, "网络启动失败", str(exc))
    if args.demo and args.demo_seconds > 0:
        def finish_demo():
            viewer.close()
            app.quit()

        QtCore.QTimer.singleShot(int(args.demo_seconds * 1000), finish_demo)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
