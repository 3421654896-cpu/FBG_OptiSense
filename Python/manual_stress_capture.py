"""Fixed 5×5 top-view manual press data collection fed by temporary test mode.

The 45-point source is the temporary-test page's live scan stream
(``TemporaryTestWindow.update_frame``), NOT the main spectrum page's native
USB ``frames_queue``. Every decoded temporary frame carries 45 records; this
collector intentionally discards each ``first_code`` and keeps only the stable
post-settle ``second_code`` as the labelled training sample.
"""

from __future__ import annotations

from pathlib import Path
import re
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from mechanical_finger_3d import (
    BODY_BASE_RGBA,
    SILICONE_BASE_RGBA,
    VIEW_BACKGROUND_RGBA,
    _silicone_pad_mesh,
    load_stl_mesh,
    prepare_thumb_mesh,
    silicone_surface_height,
)
from paxini_px6d import PaxiniForceSensor
from temporary_peak_fit import fit_temporary_frame
import temporary_optical_units as units


GRID_ROWS = 5
GRID_COLUMNS = 5
GRID_POINT_COUNT = GRID_ROWS * GRID_COLUMNS
STRESS_POINT_COUNT = 45
LATEST_FRAME_MAX_AGE_MS = 2000.0

# Label, top-view mesh-normalized x/y. The 25 markers form a true 5×5
# rectangle over the same silicone sensing area visible in the original 3-D
# stress model's top view.
_FINGER_X_LEVELS = (0.30, 0.40, 0.50, 0.60, 0.70)
_FINGER_Y_LEVELS = (0.16, 0.255, 0.35, 0.445, 0.54)
FINGER_POINT_LAYOUT = tuple(
    (row * 5 + column + 1, x, y)
    for row, y in enumerate(_FINGER_Y_LEVELS)
    for column, x in enumerate(_FINGER_X_LEVELS)
)


FINGER_POINT_BY_LABEL = {
    label: (x, y) for label, x, y in FINGER_POINT_LAYOUT
}
FINGER_TOP_MESH_PATH = Path(__file__).resolve().parent / "assets" / "thumb_real.STL"


def _surface_region(y):
    if y < 0.18:
        return "指尖"
    if y < 0.47:
        return "上段指腹"
    if y < 0.64:
        return "中段指腹"
    if y < 0.79:
        return "下段指腹"
    return "指根"


class FingerSurfaceMap(QtWidgets.QWidget):
    """Original 3-D finger model with a clickable 5×5 surface annotation grid."""

    def __init__(self, *, on_selected=None, parent=None):
        super().__init__(parent)
        self.on_selected = on_selected
        self.uses_real_3d_model = False
        self.setMinimumSize(520, 560)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = gl.GLViewWidget(self)
        self.view.setBackgroundColor(VIEW_BACKGROUND_RGBA)
        self.view.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.view)

        vertices, faces = load_stl_mesh(FINGER_TOP_MESH_PATH)
        prepared, _normalized = prepare_thumb_mesh(vertices)
        lower = np.min(prepared, axis=0)
        upper = np.max(prepared, axis=0)
        span = upper - lower
        colors = np.tile(np.asarray(BODY_BASE_RGBA, dtype=float), (len(prepared), 1))
        self.body_item = gl.GLMeshItem(
            vertexes=prepared,
            faces=faces,
            vertexColors=colors,
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
            glOptions="opaque",
        )
        self.view.addItem(self.body_item)

        silicone_vertices, silicone_faces, _silicone_norm = _silicone_pad_mesh(
            (lower[:2].copy(), upper[:2].copy()), float(span[2])
        )
        silicone_colors = np.tile(
            np.asarray(SILICONE_BASE_RGBA, dtype=float),
            (len(silicone_vertices), 1),
        )
        self.silicone_item = gl.GLMeshItem(
            vertexes=silicone_vertices,
            faces=silicone_faces,
            vertexColors=silicone_colors,
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
            glOptions="translucent",
        )
        self.view.addItem(self.silicone_item)

        self.model_lower = lower
        self.model_upper = upper
        self.model_span = span
        self.marker_positions = np.asarray(
            [self._model_position(x, y) for _label, x, y in FINGER_POINT_LAYOUT],
            dtype=float,
        )
        self.marker_item = gl.GLScatterPlotItem(
            pos=self.marker_positions,
            size=9.0,
            color=np.tile(np.asarray((0.05, 0.32, 0.62, 0.98)), (25, 1)),
            pxMode=True,
        )
        self.view.addItem(self.marker_item)


        self.text_items = []
        for (label, _x, _y), position in zip(
            FINGER_POINT_LAYOUT, self.marker_positions
        ):
            text = gl.GLTextItem(
                pos=position
                + np.asarray((0.0, 0.0, max(float(span[2]) * 0.028, 0.9))),
                text=str(label),
                color=(0.05, 0.20, 0.45, 1.0),
            )
            self.view.addItem(text)
            self.text_items.append(text)

        center = (lower + upper) * 0.5
        self.view.opts["center"] = QtGui.QVector3D(
            float(center[0]), float(center[1]), float(center[2])
        )
        distance = max(float(np.max(span)) * 1.58, 72.0)
        self.view.setCameraPosition(elevation=90, azimuth=-90, distance=distance)
        self.uses_real_3d_model = True

        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        # Invisible full-surface hit targets keep point selection working while
        # the visible annotation stays minimal (markers + tiny numbers).
        for label, _x, _y in FINGER_POINT_LAYOUT:
            button = QtWidgets.QPushButton("", self)
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setToolTip(f"手指表面位置编号 {label}")
            button.setStyleSheet(
                "QPushButton { background: transparent; border: none; }"
                "QPushButton:checked {"
                " background: rgba(255,159,67,70);"
                " border: 2px solid rgba(255,159,67,230);"
                " border-radius: 12px;"
                "}"
            )
            button.clicked.connect(
                lambda _checked=False, value=label: self._selected(value)
            )
            self.group.addButton(button, label)
            self.buttons.append(button)
        QtCore.QTimer.singleShot(0, self._position_buttons)


    def _model_position(self, x_fraction, y_fraction):
        x_norm = 2.0 * float(x_fraction) - 1.0
        y_norm = 1.0 - 2.0 * float(y_fraction)
        x = self.model_lower[0] + float(x_fraction) * self.model_span[0]
        y = self.model_upper[1] - float(y_fraction) * self.model_span[1]
        z = float(silicone_surface_height(x_norm, y_norm, self.model_span[2]))
        return (x, y, z + max(float(np.max(self.model_span)) * 0.010, 0.32))

    def _selected(self, label):
        if self.on_selected is not None:
            self.on_selected(int(label))

    def _project_to_widget(self, point):
        width = max(1, int(self.view.width()))
        height = max(1, int(self.view.height()))
        viewport = (0, 0, width, height)
        combined = (
            self.view.projectionMatrix(viewport, viewport) * self.view.viewMatrix()
        )
        clip = combined * QtGui.QVector4D(
            float(point[0]), float(point[1]), float(point[2]), 1.0
        )
        if abs(float(clip.w())) <= 1e-12:
            return None
        ndc_x = float(clip.x() / clip.w())
        ndc_y = float(clip.y() / clip.w())
        origin = self.view.mapTo(self, QtCore.QPoint(0, 0))
        return (
            origin.x() + (ndc_x + 1.0) * 0.5 * width,
            origin.y() + (1.0 - ndc_y) * 0.5 * height,
        )

    def _position_buttons(self):
        # Invisible hit targets: compact squares around each projected point.
        projected = [
            self._project_to_widget(point) for point in self.marker_positions
        ]
        if any(point is None for point in projected):
            for button in self.buttons:
                button.hide()
            return
        gaps = [
            projected[row * 5 + column + 1][0] - projected[row * 5 + column][0]
            for row in range(5)
            for column in range(4)
        ]
        vertical = [
            projected[(row + 1) * 5 + column][1] - projected[row * 5 + column][1]
            for row in range(4)
            for column in range(5)
        ]
        gap = max(22.0, min(gaps + vertical))
        marker_size = int(max(18, min(30, gap * 0.58)))
        for button, (x, y) in zip(self.buttons, projected):
            button.setGeometry(
                int(x - marker_size / 2),
                int(y - marker_size / 2),
                marker_size,
                marker_size,
            )
            button.show()
            button.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._position_buttons)

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._position_buttons)



def extract_frame_vector(frame) -> list:
    """Return the 45-point second-sample CH1 vector of one temporary frame."""

    records = list(frame.get("records") or [])
    if len(records) != STRESS_POINT_COUNT:
        raise ValueError(f"临时帧不是45点，收到{len(records)}点")
    return [int(record["second_code"]) for record in records]


def _float_list(values):
    return [float(value) for value in values]


def build_training_views(frame, wavelengths) -> dict:
    """Build stable ADC, physical-unit and Makima training views.

    Only ``second_code`` is retained: it is the post-settle ADC reading used
    by the live peak fitter. dBm is an engineering estimate using the frame's
    feedback selector, not a traceable power-meter calibration.
    """

    x = [float(value) for value in wavelengths]
    if len(x) != STRESS_POINT_COUNT:
        raise ValueError(f"波长坐标不是45点，收到{len(x)}点")
    stable_adc = extract_frame_vector(frame)
    selector = int(frame.get("feedback_selector", 0) or 0)
    stable_voltage = _float_list(units.adc_code_to_voltage(stable_adc))
    stable_dbm = _float_list(
        units.adc_code_to_dbm(stable_adc, selector, floor_nonpositive=True)
    )
    rows = [{"measured_nm": value} for value in x]
    fits = fit_temporary_frame(
        rows, stable_adc, warmed_up=int(frame.get("sequence", 0)) >= 16
    )
    fit_records = []
    for peak_index, fit in enumerate(fits):
        start = peak_index * 5
        record = dict(fit)
        record.update(
            peak_index=peak_index,
            input_wavelengths_nm=x[start:start + 5],
            input_adc=stable_adc[start:start + 5],
            input_voltage_v=stable_voltage[start:start + 5],
            input_dbm=stable_dbm[start:start + 5],
        )
        curve_adc = record.get("curve_adc")
        if curve_adc is not None:
            record["curve_voltage_v"] = _float_list(
                units.adc_code_to_voltage(curve_adc)
            )
            record["curve_dbm"] = _float_list(
                units.adc_code_to_dbm(
                    curve_adc, selector, floor_nonpositive=True
                )
            )
        fit_records.append(record)
    return {
        "wavelengths_nm": x,
        "stable_adc": stable_adc,
        "stable_voltage_v": stable_voltage,
        "stable_optical_power_dbm": stable_dbm,
        "conversion": {
            "adc_reference_v": float(units.ADC_REFERENCE_V),
            "adc_code_count": float(units.ADC_CODE_COUNT),
            "feedback_selector": selector,
            "transimpedance_ohm": float(units.transimpedance_ohm(selector)),
            "pd_responsivity_a_per_w": float(units.PD_RESPONSIVITY_A_PER_W),
            "dbm_is_engineering_estimate": True,
        },
        "makima_peak_fits": fit_records,
    }


class ForceVectorWidget(QtWidgets.QWidget):
    """Independent live-force display with colored vectors and value cards.

    This widget deliberately does not share the finger-model GL scene.  It
    combines the two visual elements requested by the operator: an intuitive
    RGB force-direction graphic and clearly readable Fx/Fy/Fz values.
    """

    AXIS_COLORS = {
        "Fx": (232, 72, 85),
        "Fy": (35, 177, 111),
        "Fz": (58, 124, 246),
    }
    AUTO_SCALE_N = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connected = False
        self.force_xyz = (0.0, 0.0, 0.0)
        self.moments = (0.0, 0.0, 0.0)
        self.setMinimumSize(300, 330)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        self.zero_button = QtWidgets.QPushButton("标定归零", self)
        self.zero_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.zero_button.setToolTip(
            "确保传感器空载、静止后点击；发送与官方PX6D上位机一致的"
            "重新标定命令（0x10）。"
        )
        self.zero_button.setStyleSheet(
            "QPushButton {"
            " background: #E8F0FF; color: #2F6FED;"
            " border: 1px solid #B9D1FF; border-radius: 7px;"
            " padding: 4px 10px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #DCE9FF; border-color: #8FB4FF; }"
            "QPushButton:pressed { background: #CFE0FF; }"
            "QPushButton:disabled { color: #9AA5B5; background: #EEF1F5;"
            " border-color: #DDE2EA; }"
        )
        self._position_zero_button()

    def _position_zero_button(self):
        self.zero_button.setGeometry(max(188, self.width() - 106), 10, 92, 30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_zero_button()

    def set_force(self, fx, fy, fz, *, connected=True, moments=None):
        if fx is None or fy is None or fz is None or not connected:
            self.connected = False
            self.force_xyz = (0.0, 0.0, 0.0)
        else:
            self.connected = True
            self.force_xyz = (float(fx), float(fy), float(fz))
            self.moments = tuple(
                0.0 if value is None else float(value) for value in (moments or ())
            ) or (0.0, 0.0, 0.0)
        self.update()

    @staticmethod
    def _draw_arrow_head(painter, tip, direction, color, size=12.0):
        import math

        angle = math.atan2(direction.y(), direction.x())
        back = angle + math.pi
        spread = 0.48
        left = QtCore.QPointF(
            tip.x() + size * math.cos(back + spread),
            tip.y() + size * math.sin(back + spread),
        )
        right = QtCore.QPointF(
            tip.x() + size * math.cos(back - spread),
            tip.y() + size * math.sin(back - spread),
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QtGui.QPolygonF([tip, left, right]))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        width = float(self.width())
        height = float(self.height())
        # Soft, borderless panel integrated with the application's light UI.
        outer = QtCore.QRectF(1, 1, width - 2, height - 2)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(247, 249, 252))
        painter.drawRoundedRect(outer, 14, 14)

        title_font = painter.font()
        title_font.setPointSizeF(11)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(35, 43, 58))
        painter.drawText(QtCore.QPointF(18, 27), "实时力")
        status = "已连接" if self.connected else "未连接"
        painter.setPen(
            QtGui.QColor(38, 166, 104)
            if self.connected
            else QtGui.QColor(143, 151, 164)
        )
        painter.drawText(QtCore.QPointF(77, 27), status)
        hint_font = painter.font()
        hint_font.setPointSizeF(8)
        hint_font.setBold(False)
        painter.setFont(hint_font)
        painter.setPen(QtGui.QColor(126, 135, 149))
        painter.drawText(QtCore.QPointF(18, 49), "空载并保持静止后执行归零")

        # Top section: a clean pseudo-3D triad. Arrow length and direction
        # respond to the signed live force while retaining a readable stub.
        fx, fy, fz = self.force_xyz
        origin = QtCore.QPointF(width * 0.47, height * 0.34)
        reach = min(width * 0.29, height * 0.20)
        vectors = (
            ("Fx", fx, QtCore.QPointF(1.0, 0.20)),
            ("Fy", fy, QtCore.QPointF(-0.72, 0.52)),
            ("Fz", fz, QtCore.QPointF(0.0, -1.0)),
        )
        label_font = painter.font()
        label_font.setPointSizeF(9.5)
        label_font.setBold(True)
        painter.setFont(label_font)
        for name, value, base_dir in vectors:
            sign = 1.0 if value >= 0 else -1.0
            direction = QtCore.QPointF(base_dir.x() * sign, base_dir.y() * sign)
            norm = (direction.x() ** 2 + direction.y() ** 2) ** 0.5
            direction = QtCore.QPointF(direction.x() / norm, direction.y() / norm)
            ratio = min(1.0, abs(value) / self.AUTO_SCALE_N)
            length = reach * (0.36 + 0.64 * ratio)
            tip = QtCore.QPointF(
                origin.x() + direction.x() * length,
                origin.y() + direction.y() * length,
            )
            rgb = self.AXIS_COLORS[name]
            color = (
                QtGui.QColor(*rgb)
                if self.connected
                else QtGui.QColor(160, 167, 178)
            )
            painter.setPen(QtGui.QPen(color, 5.0, QtCore.Qt.SolidLine,
                                      QtCore.Qt.RoundCap))
            painter.drawLine(origin, tip)
            self._draw_arrow_head(painter, tip, direction, color, size=13.0)
            painter.setPen(color)
            label_pos = QtCore.QPointF(
                tip.x() + direction.x() * 9 - 8,
                tip.y() + direction.y() * 9 + 4,
            )
            painter.drawText(label_pos, name)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(54, 62, 77))
        painter.drawEllipse(origin, 5, 5)

        # Bottom section: three independent numeric cards. The number is the
        # primary visual element; axis color, sign, decimals and N are explicit.
        card_left = 14.0
        card_width = width - 28.0
        card_height = max(48.0, min(58.0, (height * 0.43 - 16.0) / 3.0))
        start_y = height - (card_height * 3.0 + 22.0)
        for row, (name, value) in enumerate((("Fx", fx), ("Fy", fy), ("Fz", fz))):
            y = start_y + row * (card_height + 7.0)
            rect = QtCore.QRectF(card_left, y, card_width, card_height)
            painter.setBrush(QtGui.QColor(255, 255, 255))
            painter.setPen(QtGui.QPen(QtGui.QColor(231, 235, 242), 1.0))
            painter.drawRoundedRect(rect, 10, 10)
            rgb = self.AXIS_COLORS[name]
            color = QtGui.QColor(*rgb) if self.connected else QtGui.QColor(150, 158, 170)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QtCore.QRectF(card_left + 10, y + 11, 5, card_height - 22), 2, 2)

            axis_font = painter.font()
            axis_font.setPointSizeF(10)
            axis_font.setBold(True)
            painter.setFont(axis_font)
            painter.setPen(color)
            painter.drawText(QtCore.QRectF(card_left + 25, y, 42, card_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, name)

            value_font = painter.font()
            value_font.setPointSizeF(16)
            value_font.setBold(True)
            painter.setFont(value_font)
            state = "—" if not self.connected else f"{value:+.3f}"
            painter.setPen(QtGui.QColor(34, 42, 56))
            painter.drawText(QtCore.QRectF(card_left + 68, y, card_width - 112, card_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, state)

            unit_font = painter.font()
            unit_font.setPointSizeF(9)
            unit_font.setBold(False)
            painter.setFont(unit_font)
            painter.setPen(QtGui.QColor(109, 118, 133))
            painter.drawText(QtCore.QRectF(card_left + card_width - 35, y, 25, card_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, "N")


class ManualStressCaptureWindow(QtWidgets.QMainWindow):
    """Numbered 25-point grid that saves one 45-value vector per button press."""

    force_zero_finished = QtCore.pyqtSignal(bool, str)
    grid_shape = (GRID_ROWS, GRID_COLUMNS)

    def __init__(
        self,
        frame_source,
        *,
        wavelength_provider,
        output_root: Path,
        parent=None,
    ):
        """``frame_source`` is the TemporaryTestWindow providing live frames."""

        super().__init__(parent)
        self.frame_source = frame_source
        self.wavelength_provider = wavelength_provider
        self.output_root = Path(output_root)
        self.writer: FingerSessionWriter | None = None
        self.session_dir: Path | None = None
        self.selected_point: int | None = None
        self.sample_index = 0
        self.sequence = 0
        self.capture_counts = {index: 0 for index in range(1, 26)}
        self.latest_frame = None
        self.latest_values = None
        self.latest_peak_fits = None
        self.rejected_non45_frames = 0
        self.frames_seen = 0
        self.last_frame_monotonic = None
        # Paxini PX6D six-axis force sensor (RS485 over USB CDC, auto-push).
        # Reading is best-effort: when absent/busy, samples simply record a
        # null force plus the reason.
        self.force_sensor = PaxiniForceSensor("COM9")
        self.force_sensor_enabled = True
        self.force_last_error: str | None = None
        self.force_notice = None
        self.force_notice_until = 0.0
        self.force_zero_finished.connect(self._force_zero_done)
        try:
            self.force_sensor.start()
        except Exception as exc:
            self.force_last_error = str(exc)

        self.setWindowTitle("25点矩形点阵应力数据采集（临时测试模式数据源）")
        self._build_ui()

    # ------------------------------------------------------------------ link

    def attach(self):
        """Start receiving frames from the temporary test page."""

        listeners = getattr(self.frame_source, "frame_listeners", None)
        if listeners is None:
            raise TypeError("frame_source 必须是临时测试页（TemporaryTestWindow）")
        if self._on_temporary_frame not in listeners:
            listeners.append(self._on_temporary_frame)

    def detach(self):
        listeners = getattr(self.frame_source, "frame_listeners", None)
        if listeners is not None and self._on_temporary_frame in listeners:
            listeners.remove(self._on_temporary_frame)

    def _on_temporary_frame(self, frame):
        """Called on the GUI thread for every decoded temporary-mode frame."""

        self.frames_seen += 1
        try:
            values = extract_frame_vector(frame)
        except (ValueError, KeyError, TypeError):
            self.rejected_non45_frames += 1
            return
        self.latest_frame = frame
        self.latest_values = values
        self.last_frame_monotonic = time.monotonic()
        # Keep the same Makima five-point fits the temporary page displays, so
        # each saved sample can carry both raw vectors and fitted centres.
        rows = getattr(self.frame_source, "plan", None) or {}
        plan_rows = list(rows.get("rows") or [])
        if len(plan_rows) == STRESS_POINT_COUNT:
            try:
                self.latest_peak_fits = fit_temporary_frame(
                    plan_rows, values, True
                )
            except (ValueError, TypeError):
                self.latest_peak_fits = None
        else:
            self.latest_peak_fits = None

    # ------------------------------------------------------------------- UI

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        central.setObjectName("manualStressGridPage")
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("手动应力采集 · 手指表面25点位置图")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        note = QtWidgets.QLabel(
            "数据来自“临时测试模式”的实时45点扫描。操作：先在临时测试页开始"
            "连续采集 → 回到本页选择点阵编号 → 手动按住该位置 → 点击"
            "“采集当前45点”。每次点击只保留稳定后的第二次ADC值及其"
            "电压、dBm估算值，并保存这三种表示对应的9峰Makima五点拟合"
            "中心和完整曲线，编号作为按压位置标签。"
        )
        note.setWordWrap(True)
        note.setObjectName("softHint")
        outer.addWidget(note)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("数据集："))
        self.dataset_name_edit = QtWidgets.QLineEdit(
            "press_grid_" + time.strftime("%Y%m%d_%H%M%S")
        )
        self.dataset_name_edit.setMinimumWidth(220)
        controls.addWidget(self.dataset_name_edit)
        self.capture_button = QtWidgets.QPushButton("采集当前45点")
        self.capture_button.setProperty("role", "primary")
        self.capture_button.clicked.connect(self.capture_current_press)
        controls.addWidget(self.capture_button)
        self.close_session_button = QtWidgets.QPushButton("保存并关闭数据集")
        self.close_session_button.clicked.connect(self.close_session)
        controls.addWidget(self.close_session_button)
        controls.addStretch(1)
        outer.addLayout(controls)

        surface_card = QtWidgets.QFrame()
        surface_card.setObjectName("contentCard")
        surface_outer = QtWidgets.QVBoxLayout(surface_card)
        surface_outer.setContentsMargins(16, 12, 16, 12)
        top_direction = QtWidgets.QLabel("指尖")
        top_direction.setAlignment(QtCore.Qt.AlignCenter)
        top_direction.setObjectName("sectionCaption")
        surface_outer.addWidget(top_direction)

        # The requested order is force display on the LEFT and finger model on
        # the RIGHT. They are layout siblings, so the force graphic can never
        # be painted over the model or its 25 location markers.
        body_row = QtWidgets.QHBoxLayout()
        self.force_model_layout = body_row
        body_row.setSpacing(16)
        self.force_vector = ForceVectorWidget(parent=surface_card)
        self.force_vector.setMinimumWidth(310)
        self.zero_button = self.force_vector.zero_button
        self.zero_button.clicked.connect(self._request_force_zero)
        body_row.addWidget(self.force_vector, 2)

        self.surface_map = FingerSurfaceMap(
            on_selected=self.select_grid_point,
            parent=surface_card,
        )
        self.grid_group = self.surface_map.group
        self.grid_buttons = self.surface_map.buttons
        body_row.addWidget(self.surface_map, 3)
        surface_outer.addLayout(body_row, 1)

        bottom_direction = QtWidgets.QLabel("指根")
        bottom_direction.setAlignment(QtCore.Qt.AlignCenter)
        bottom_direction.setObjectName("sectionCaption")
        surface_outer.addWidget(bottom_direction)
        outer.addWidget(surface_card, 1)

        self.selected_label = QtWidgets.QLabel("当前未选择编号")
        self.selected_label.setObjectName("statusPill")
        self.selected_label.setProperty("statusKind", "warning")
        outer.addWidget(self.selected_label)

        self.capture_status_label = QtWidgets.QLabel(
            "等待完整45点临时模式扫描；请选择1～25中的一个位置编号"
        )
        self.capture_status_label.setWordWrap(True)
        self.capture_status_label.setObjectName("softHint")
        outer.addWidget(self.capture_status_label)

        self.counts_label = QtWidgets.QLabel("已采集：0条")
        self.counts_label.setWordWrap(True)
        self.counts_label.setObjectName("softHint")
        outer.addWidget(self.counts_label)

        self.link_label = QtWidgets.QLabel()
        self.link_label.setWordWrap(True)
        self.link_label.setObjectName("softHint")
        outer.addWidget(self.link_label)

        self.force_label = QtWidgets.QLabel("帕西尼力传感器：检测中…")
        self.force_label.setWordWrap(True)
        self.force_label.setObjectName("softHint")
        outer.addWidget(self.force_label)


        self.link_timer = QtCore.QTimer(self)
        self.link_timer.setInterval(500)
        self.link_timer.timeout.connect(self._update_link_label)
        self.link_timer.start()

        # Force arrows animate on their own ~20 fps timer, independent of the
        # 0.5 s status refresh.
        self.force_timer = QtCore.QTimer(self)
        self.force_timer.setInterval(50)
        self.force_timer.timeout.connect(self._refresh_force_arrows)
        self.force_timer.start()

    def _refresh_force_arrows(self):
        """Feed the newest live force into the arrow visualizations."""

        sensor = self.force_sensor
        latest = sensor._latest
        connected = (
            self.force_sensor_enabled
            and latest is not None
            and sensor._latest_ns is not None
            and (time.monotonic_ns() - sensor._latest_ns) / 1e6
            <= LATEST_FRAME_MAX_AGE_MS
        )
        if connected:
            fx, fy, fz = latest[0], latest[1], latest[2]
        else:
            fx = fy = fz = None
        # Only the independent panel is updated; the finger model stays a pure
        # 25-point position-label surface.
        self.force_vector.set_force(
            fx, fy, fz, connected=connected, moments=latest[3:] if latest else None
        )

    def _frame_age_ms(self):
        if self.last_frame_monotonic is None:
            return float("inf")
        return (time.monotonic() - self.last_frame_monotonic) * 1000.0

    def _update_link_label(self):
        """Surface why no temporary 45-point frame is arriving, with the fix."""

        if self.frames_seen == 0:
            text = (
                "数据链路：尚未收到临时测试模式帧。请切到“临时测试模式”页，"
                "完成选点后点击“开始”连续采集（45点路线），再回本页采集。"
            )
            kind = "warning"
        elif self.latest_values is None:
            text = (
                f"数据链路：已收到{self.frames_seen}帧，但都不是45点；"
                "请确认临时测试页使用的是45点选点路线。"
            )
            kind = "warning"
        else:
            age = self._frame_age_ms()
            fresh = age <= LATEST_FRAME_MAX_AGE_MS
            text = (
                f"数据链路正常：最新45点帧 {age:.0f} ms 前"
                + ("" if fresh else "（已过期，扫描可能已停止）")
            )
            kind = "ok" if fresh else "warning"
        self.link_label.setText(text)
        self.link_label.setProperty("statusKind", kind)
        self.link_label.style().unpolish(self.link_label)
        self.link_label.style().polish(self.link_label)
        self._update_force_label()

    def _request_force_zero(self, _checked=False, *, confirm=True):
        """Run the official PX6D zero calibration without blocking the UI."""

        if self.force_sensor.frames_received == 0:
            self.force_notice = "帕西尼未连接，不能归零"
            self.force_notice_until = time.monotonic() + 3.0
            self._update_force_label()
            return False
        if confirm:
            answer = QtWidgets.QMessageBox.question(
                self,
                "帕西尼力传感器归零",
                "请确保传感器完全无外力。\n\n"
                "将调用与官方上位机一致的重新标定命令，是否继续？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return False
        self.zero_button.setEnabled(False)
        self.force_notice = "正在归零，请勿触碰传感器…"
        self.force_notice_until = float("inf")
        self._update_force_label()

        def worker():
            try:
                self.force_sensor.calibrate_zero(timeout_s=3.0)
            except Exception as exc:
                self.force_zero_finished.emit(False, str(exc))
            else:
                self.force_zero_finished.emit(True, "")

        threading.Thread(
            target=worker, name="paxini-zero", daemon=True
        ).start()
        return True

    def _force_zero_done(self, success, error):
        self.zero_button.setEnabled(True)
        self.force_notice_until = time.monotonic() + 4.0
        if success:
            self.force_notice = "帕西尼归零成功；当前无外力应接近0 N"
            self.force_last_error = None
        else:
            self.force_notice = f"帕西尼归零失败：{error}"
            self.force_last_error = str(error)
        self._update_force_label()

    def _update_force_label(self):
        """Live Paxini PX6D connection status and current XYZ reading."""

        sensor = self.force_sensor
        if self.force_notice and time.monotonic() < self.force_notice_until:
            self.force_label.setText(self.force_notice)
            self.force_label.setProperty(
                "statusKind",
                "ok" if "成功" in self.force_notice else "warning",
            )
            self.force_label.style().unpolish(self.force_label)
            self.force_label.style().polish(self.force_label)
            return
        if not self.force_sensor_enabled:
            self.force_vector.set_force(None, None, None, connected=False)
            text = "帕西尼力传感器：已禁用（本数据集不记录力值）"
            kind = "warning"
        elif sensor.frames_received == 0:
            self.force_vector.set_force(None, None, None, connected=False)
            reason = sensor.last_error or self.force_last_error
            text = (
                "帕西尼力传感器（COM9）：未收到数据"
                + (f"（{reason}）" if reason else "")
                + "；请确认传感器已上电、串口未被官方上位机占用"
            )
            kind = "warning"
        else:
            age_ms = (time.monotonic_ns() - sensor._latest_ns) / 1e6
            fx, fy, fz, _mx, _my, _mz = sensor._latest
            self.force_vector.set_force(fx, fy, fz, connected=True)
            zero_text = (
                f" · 官方归零{sensor.zero_count}次"
                if sensor.zero_count else ""
            )
            if age_ms > LATEST_FRAME_MAX_AGE_MS:
                text = (
                    f"帕西尼力传感器（COM9）：数据中断（最新帧 {age_ms:.0f} ms 前，"
                    f"累计{sensor.frames_received}帧）{zero_text}"
                )
                kind = "warning"
            else:
                text = (
                    f"帕西尼力传感器（COM9）：已连接 · Fx={fx:+.3f} N  "
                    f"Fy={fy:+.3f} N  Fz={fz:+.3f} N"
                    f"（累计{sensor.frames_received}帧）{zero_text}"
                )
                kind = "ok"
        self.force_label.setText(text)
        self.force_label.setProperty("statusKind", kind)
        self.force_label.style().unpolish(self.force_label)
        self.force_label.style().polish(self.force_label)

    @staticmethod
    def _safe_name(value):
        clean = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '_', str(value).strip())
        return clean.strip('_') or "press_grid"

    def select_grid_point(self, label):
        label = int(label)
        if not 1 <= label <= GRID_POINT_COUNT:
            raise ValueError("grid point must be 1..25")
        self.selected_point = label
        button = self.grid_group.button(label)
        if button is not None:
            button.setChecked(True)
        x, y = FINGER_POINT_BY_LABEL[label]
        region = _surface_region(y)
        count = self.capture_counts[label]
        self.selected_label.setText(
            f"当前标签：编号 {label} · {region} · 已采{count}条"
        )
        self.selected_label.setProperty("statusKind", "ok")
        self.selected_label.style().unpolish(self.selected_label)
        self.selected_label.style().polish(self.selected_label)
        self.capture_status_label.setText(
            f"请手动按住编号{label}对应位置，然后点击“采集当前45点”"
        )
        return True

    # -------------------------------------------------------------- capture

    def _ensure_session(self):
        if self.writer is not None:
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        stem = self._safe_name(self.dataset_name_edit.text())
        session_dir = self.output_root / stem
        suffix = 1
        while session_dir.exists():
            session_dir = self.output_root / f"{stem}_{suffix:02d}"
            suffix += 1
        metadata = {
            "dataset_kind": "fixed_25_point_press_grid",
            "training_eligible": True,
            "input_contract": "stable_adc_with_physical_and_makima_views",
            "frame_source": "temporary_test_mode_live_frames",
            "sample_kinds": [
                "stable_second_code",
                "stable_voltage_v",
                "stable_optical_power_dbm",
                "makima_curves_adc_voltage_dbm",
            ],
            "grid_shape": [GRID_ROWS, GRID_COLUMNS],
            "label_range": [1, GRID_POINT_COUNT],
            "view": "finger_surface_front_25_points",
            "point_layout": [
                {"label": label, "x": x, "y": y}
                for label, x, y in FINGER_POINT_LAYOUT
            ],
            "intended_task": "classify press_label_1_to_25_from_45_point_vector",
        }
        self.writer = FingerSessionWriter(
            session_dir, queue_capacity=512, manifest_extra=metadata
        )
        self.session_dir = session_dir
        self.writer.write_event(ContactEvent(kind="session_start", metadata=metadata))
        self.dataset_name_edit.setEnabled(False)

    def _read_force_sample(self):
        """Best-effort six-axis read from the auto-push stream; never blocks."""

        if not self.force_sensor_enabled:
            self.force_last_error = "force_sensor_disabled"
            return None
        try:
            values = self.force_sensor.read_six_axis(timeout_s=0.2)
        except Exception as exc:
            self.force_last_error = str(exc)
            return None
        self.force_last_error = None
        return values

    def capture_current_press(self):
        if self.selected_point is None:
            self.capture_status_label.setText("请先选择1～25中的位置编号")
            return False
        force = None
        force_error = None
        wavelengths = list(self.wavelength_provider() or ())
        if len(wavelengths) != STRESS_POINT_COUNT:
            self.capture_status_label.setText(
                "临时测试页尚未完成45点参考线，波长表不是45点，无法采集"
            )
            return False
        if self.latest_values is None:
            self.capture_status_label.setText(
                "尚未收到临时模式45点帧；请先在临时测试页开始连续采集"
            )
            return False
        age_ms = self._frame_age_ms()
        if age_ms > LATEST_FRAME_MAX_AGE_MS:
            self.capture_status_label.setText(
                f"最新45点帧已过期（{age_ms:.0f} ms），请在临时测试页保持连续采集"
            )
            return False

        self._ensure_session()
        label = int(self.selected_point)
        surface_x, surface_y = FINGER_POINT_BY_LABEL[label]
        region = _surface_region(surface_y)
        # Retain the historical row/column fields for old readers, but the
        # authoritative annotation is now the hand-surface coordinate below.
        row = (label - 1) // GRID_COLUMNS + 1
        column = (label - 1) % GRID_COLUMNS + 1
        self.sample_index += 1
        self.sequence += 1
        capture_ns = time.monotonic_ns()
        frame = self.latest_frame or {}
        training_views = build_training_views(frame, wavelengths)
        stable_adc = training_views["stable_adc"]
        selector = training_views["conversion"]["feedback_selector"]
        try:
            force = self._read_force_sample()
        except Exception as exc:
            force = None
            force_error = str(exc) or type(exc).__name__
        quality = {
            "dataset_kind": "fixed_25_point_press_grid",
            "press_label": label,
            "grid_row": row,
            "grid_column": column,
            "surface_x_normalized": float(surface_x),
            "surface_y_normalized": float(surface_y),
            "surface_region": region,
            "sample_index": self.sample_index,
            "source_frame_sequence": int(frame.get("sequence", -1)),
            "source_frame_age_ms": float(age_ms),
            "source_frame_quality": dict(frame.get("quality") or {}),
            "training_views": training_views,
            # Paxini PX6D six-axis force at the capture instant.  None means
            # the sensor was absent/busy; force_error says why.
            "paxini_force": None if force is None else {
                "fx_n": force[0],
                "fy_n": force[1],
                "fz_n": force[2],
                "mx_nm": force[3],
                "my_nm": force[4],
                "mz_nm": force[5],
                "monotonic_ns": time.monotonic_ns(),
                "official_zero_count": self.force_sensor.zero_count,
                "last_official_zero_ns": self.force_sensor.last_zero_ns,
            },
            "force_error": force_error if force is None else self.force_last_error,
            # Flat aliases keep old analysis scripts working while new models
            # consume the complete nested training_views schema.
            "feedback_selector": selector,
            "transimpedance_ohm": training_views["conversion"][
                "transimpedance_ohm"
            ],
            "stable_adc": stable_adc,
            "stable_voltage_v": training_views["stable_voltage_v"],
            "stable_dbm": training_views["stable_optical_power_dbm"],
            "peak_fits": training_views["makima_peak_fits"],
        }
        self.writer.write_event(ContactEvent(
            kind="press_sample",
            monotonic_ns=capture_ns,
            metadata={
                "press_label": label,
                "grid_row": row,
                "grid_column": column,
                "surface_x_normalized": float(surface_x),
                "surface_y_normalized": float(surface_y),
                "surface_region": region,
                "sample_index": self.sample_index,
                "source_frame_sequence": int(frame.get("sequence", -1)),
            },
        ))
        self.writer.write_frame(SpectrumFrame(
            sequence=self.sequence,
            monotonic_ns=capture_ns,
            wavelengths_nm=wavelengths,
            raw_adc_codes={1: stable_adc},
            adc_value_kind="direct_adc",
            quality=quality,
            mode="stress",
            source="temporary_test_live_frame",
            transimpedance_ohm={1: units.transimpedance_ohm(selector)},
        ))
        self.capture_counts[label] += 1
        count = self.capture_counts[label]
        total = sum(self.capture_counts.values())
        button = self.grid_group.button(label)
        button.setToolTip(
            f"手指表面位置编号 {label} · {region} · 已采{count}条"
        )
        self.selected_label.setText(
            f"当前标签：编号 {label} · {region} · 已采{count}条"
        )
        self.counts_label.setText(
            f"已采集：{total}条 · 当前编号{label}累计{count}条 · "
            f"丢弃非45点帧{self.rejected_non45_frames}"
        )
        self.capture_status_label.setText(
            f"采集成功：编号{label} → 第{self.sample_index}条数据（45点CH1）"
            + (
                ""
                if force is not None
                else " · 力传感器未读到"
            )
        )
        return True

    def close_session(self):
        if self.writer is None:
            return False
        total = sum(self.capture_counts.values())
        self.writer.write_event(ContactEvent(
            kind="session_end",
            metadata={
                "total_samples": total,
                "counts_by_label": dict(self.capture_counts),
                "rejected_non45_frames": self.rejected_non45_frames,
            },
        ))
        path = self.session_dir
        self.writer.close()
        self.writer = None
        self.dataset_name_edit.setEnabled(True)
        self.capture_status_label.setText(
            f"数据集已保存：{path} · 共{total}条45点数据"
        )
        return True

    def closeEvent(self, event):
        try:
            self.link_timer.stop()
            self.detach()
            self.close_session()
        finally:
            self.force_sensor.close()
            event.accept()
