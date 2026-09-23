"""HyperOS-inspired visual system for the unified FBG desktop studio.

The theme deliberately borrows the calm hierarchy, soft materials and rounded
setting cards of Xiaomi HyperOS while keeping measurement plots dark and high
contrast.  It contains no acquisition logic, so the instrument behaviour stays
independent from appearance changes.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

COLORS = {
    "canvas": "#F5F5F7",
    "surface": "#FFFFFF",
    "surface_soft": "#F0F1F4",
    "surface_blue": "#EAF2FF",
    "line": "#E7E8EC",
    "line_strong": "#D8DAE0",
    "text": "#1B1C20",
    "text_2": "#62656C",
    "text_3": "#93969D",
    "blue": "#3482FF",
    "blue_pressed": "#1F6FE8",
    "green": "#18A66A",
    "orange": "#FF7A33",
    "red": "#E84C4C",
    "plot": "#15171C",
    "plot_grid": "#3A3E47",
    "plot_text": "#B8BEC9",
}


def _palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(COLORS["canvas"]))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(COLORS["text"]))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(COLORS["surface"]))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#FAFAFB"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#292B31"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#FFFFFF"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(COLORS["text"]))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(COLORS["surface"]))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(COLORS["text"]))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#FFFFFF"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(COLORS["blue"]))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#FFFFFF"))
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.Text,
        QtGui.QColor("#B8BBC2"),
    )
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.ButtonText,
        QtGui.QColor("#B8BBC2"),
    )
    return palette


def build_stylesheet() -> str:
    """Return one centralized Qt style sheet for every embedded workflow."""

    return r"""
QWidget {
    color: #1B1C20;
    font-family: "Microsoft YaHei UI", "MiSans", "Segoe UI Variable", "Segoe UI";
    font-size: 9.5pt;
}
QMainWindow, QWidget#appRoot, QWidget#pageCanvas,
QWidget#graphPage, QWidget#remotePage, QWidget#fingerPage,
QStackedWidget#sourceStack, QTabWidget#workspaceTabs {
    background-color: #F5F5F7;
}
QFrame#sideRail {
    background-color: #FFFFFF;
    border: 1px solid #E7E8EC;
    border-radius: 24px;
}
QFrame#topBar, QFrame#contentCard, QFrame#controlGroup,
QFrame#statusCard, QFrame#remoteCard {
    background-color: #FFFFFF;
    border: 1px solid #E7E8EC;
    border-radius: 20px;
}
QFrame#sideStatusCard {
    background-color: #F6F7F9;
    border: 1px solid #ECEDEF;
    border-radius: 16px;
}
QFrame#segmentedControl {
    background-color: #F0F1F4;
    border: none;
    border-radius: 14px;
}
QFrame#workflowStrip {
    background-color: transparent;
    border: none;
}
QLabel#brandEyebrow, QLabel#sectionCaption, QLabel#navSection {
    color: #93969D;
    font-size: 8.5pt;
    font-weight: 600;
}
QLabel#sideBrand {
    color: #1B1C20;
    font-size: 13pt;
    font-weight: 700;
}
QLabel#sideBrandCaption, QLabel#brandSubtitle, QLabel#pageSubtitle,
QLabel#softHint {
    color: #777A82;
    font-size: 9pt;
}
QLabel#pageTitle {
    color: #17181B;
    font-size: 19pt;
    font-weight: 700;
}
QLabel#brandTitle {
    color: #1B1C20;
    font-size: 15pt;
    font-weight: 700;
}
QLabel#metricBadge, QLabel[metric="true"] {
    color: #4D5057;
    background-color: #F1F2F5;
    border: 1px solid #ECEDEF;
    border-radius: 10px;
    padding: 5px 10px;
}
QLabel#sourceStatus, QLabel#statusPill {
    color: #246BD3;
    background-color: #EAF2FF;
    border: 1px solid #D8E7FF;
    border-radius: 13px;
    padding: 6px 12px;
    font-weight: 600;
}
QLabel#sourceStatus[statusKind="ok"], QLabel#statusPill[statusKind="ok"] {
    color: #11784D;
    background-color: #E5F7EF;
    border-color: #CDEFE0;
}
QLabel#sourceStatus[statusKind="warning"], QLabel#statusPill[statusKind="warning"] {
    color: #A95517;
    background-color: #FFF1E6;
    border-color: #FFE0CA;
}
QLabel#sourceStatus[statusKind="error"], QLabel#statusPill[statusKind="error"] {
    color: #B83232;
    background-color: #FDEAEA;
    border-color: #F7D2D2;
}
QLabel[statusKind="ok"] { color: #11784D; font-weight: 700; }
QLabel[statusKind="warning"] { color: #A95517; font-weight: 700; }
QLabel[statusKind="error"] { color: #B83232; font-weight: 700; }
QLabel[statusKind="info"] { color: #246BD3; font-weight: 700; }
QLabel[role="warning"] {
    color: #9B551E;
    background-color: #FFF3E8;
    border: 1px solid #FFE0C7;
    border-radius: 14px;
    padding: 12px 14px;
}
QPushButton {
    min-height: 22px;
    color: #34363B;
    background-color: #F0F1F4;
    border: 1px solid transparent;
    border-radius: 11px;
    padding: 7px 14px;
}
QPushButton:hover {
    color: #1E5FBE;
    background-color: #E8EFFA;
}
QPushButton:pressed {
    color: #FFFFFF;
    background-color: #3482FF;
}
QPushButton:checked {
    color: #FFFFFF;
    background-color: #3482FF;
}
QPushButton:disabled {
    color: #B2B5BC;
    background-color: #F2F3F5;
}
QPushButton[role="emergency"], QPushButton#cncEmergencyButton {
    min-height: 36px;
    color: #FFFFFF;
    background-color: #D92D3A;
    border: 1px solid #BE1F2D;
    border-radius: 13px;
    padding: 8px 18px;
    font-weight: 800;
}
QPushButton[role="emergency"]:hover, QPushButton#cncEmergencyButton:hover {
    color: #FFFFFF;
    background-color: #BE1F2D;
    border-color: #981724;
}
QPushButton[role="emergency"]:pressed, QPushButton#cncEmergencyButton:pressed {
    color: #FFFFFF;
    background-color: #8E1420;
}
QPushButton[role="emergency"]:disabled, QPushButton#cncEmergencyButton:disabled {
    color: #FFE7E9;
    background-color: #B44A52;
}
QLabel#cncStatusPill {
    color: #62656C;
    background-color: #F0F1F4;
    border: 1px solid #E2E3E7;
    border-radius: 12px;
    padding: 7px 11px;
    font-weight: 600;
}
QWidget#cncEmergencyOverlay {
    background-color: #FFFFFF;
    border: 2px solid #D92D3A;
    border-radius: 16px;
}
QLabel#cncOverlayStatus {
    color: #62656C;
    font-size: 8.5pt;
    font-weight: 600;
}
QLabel#cncOverlayStatus[statusKind="stopped"] { color: #B83232; }
QLabel#cncOverlayStatus[statusKind="error"] { color: #B83232; }
QLabel#cncOverlayStatus[statusKind="ready"] { color: #11784D; }
QPushButton#navButton {
    min-height: 28px;
    color: #55585F;
    background-color: transparent;
    border: none;
    border-radius: 14px;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
}
QPushButton#navButton:hover {
    color: #2E64AF;
    background-color: #F1F5FB;
}
QPushButton#navButton:checked {
    color: #236BDA;
    background-color: #EAF2FF;
}
QPushButton[segment="true"] {
    min-height: 26px;
    color: #666970;
    background-color: transparent;
    border: none;
    border-radius: 11px;
    padding: 6px 13px;
}
QPushButton[segment="true"]:hover {
    color: #2A61AF;
    background-color: #E8EBF0;
}
QPushButton[segment="true"]:checked {
    color: #236BDA;
    background-color: #FFFFFF;
}
QPushButton[modeChip="true"] {
    min-height: 24px;
    color: #676A71;
    background-color: #F4F5F7;
    border: 1px solid #EAEBEE;
    border-radius: 12px;
    padding: 6px 13px;
}
QPushButton[modeChip="true"]:checked {
    color: #236BDA;
    background-color: #EAF2FF;
    border-color: #D8E7FF;
}
QPushButton[activeMode="true"] {
    color: #11784D;
    background-color: #E5F7EF;
    border-color: #CDEFE0;
    font-weight: 700;
}
QPushButton[role="primary"] {
    color: #FFFFFF;
    background-color: #3482FF;
    border-color: #3482FF;
    font-weight: 600;
}
QPushButton[role="primary"]:hover { background-color: #2B78EE; }
QPushButton[role="primary"]:pressed { background-color: #1F6FE8; }
QPushButton[role="danger"] {
    color: #C53C3C;
    background-color: #FDEAEA;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    color: #26282D;
    background-color: #F5F6F8;
    border: 1px solid #E2E4E8;
    border-radius: 11px;
    padding: 7px 10px;
    selection-color: #FFFFFF;
    selection-background-color: #3482FF;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #C9CDD5;
    background-color: #F8F9FA;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #75A9FF;
    background-color: #FFFFFF;
}
QLineEdit:read-only {
    color: #65686F;
    background-color: #F1F2F4;
}
QComboBox::drop-down {
    width: 28px;
    border: none;
}
QComboBox QAbstractItemView {
    color: #26282D;
    background-color: #FFFFFF;
    border: 1px solid #E1E3E7;
    border-radius: 10px;
    padding: 4px;
    selection-color: #236BDA;
    selection-background-color: #EAF2FF;
}
QPlainTextEdit {
    color: #DDE3EC;
    background-color: #191B20;
    border: 1px solid #292C33;
    border-radius: 15px;
    padding: 10px;
    font-family: "Cascadia Mono", Consolas, "Microsoft YaHei UI";
}
QCheckBox, QRadioButton { spacing: 7px; color: #55585F; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 17px;
    height: 17px;
}
QGroupBox {
    color: #33353A;
    background-color: #FFFFFF;
    border: 1px solid #E7E8EC;
    border-radius: 18px;
    margin-top: 15px;
    padding: 17px 13px 13px 13px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 0 7px;
    color: #55585F;
    background-color: #FFFFFF;
}
QTableWidget, QTableView {
    color: #303238;
    background-color: #FFFFFF;
    alternate-background-color: #F8F9FA;
    border: 1px solid #E7E8EC;
    border-radius: 15px;
    gridline-color: #ECEDEF;
    selection-color: #236BDA;
    selection-background-color: #EAF2FF;
}
QHeaderView::section {
    color: #686B72;
    background-color: #F5F6F8;
    border: none;
    border-right: 1px solid #E9EAED;
    border-bottom: 1px solid #E4E5E9;
    padding: 7px;
    font-weight: 600;
}
QProgressBar {
    color: #4D5057;
    background-color: #ECEEF2;
    border: none;
    border-radius: 8px;
    min-height: 25px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #3482FF;
    border-radius: 8px;
}
QMenuBar {
    color: #676A71;
    background-color: #FFFFFF;
    border-bottom: 1px solid #E8E9ED;
    padding: 2px 8px;
}
QMenuBar::item { padding: 5px 10px; border-radius: 7px; }
QMenuBar::item:selected { color: #236BDA; background-color: #EAF2FF; }
QMenu {
    color: #303238;
    background-color: #FFFFFF;
    border: 1px solid #DFE1E6;
    padding: 6px;
}
QMenu::item { padding: 7px 28px 7px 12px; border-radius: 7px; }
QMenu::item:selected { color: #236BDA; background-color: #EAF2FF; }
QStatusBar {
    color: #898C93;
    background-color: #F5F5F7;
    border-top: none;
}
QSplitter::handle { background-color: #E5E7EB; }
QSplitter::handle:horizontal { width: 5px; }
QSplitter::handle:vertical { height: 5px; }
QScrollBar:vertical {
    width: 10px;
    margin: 3px;
    background: transparent;
}
QScrollBar::handle:vertical {
    min-height: 28px;
    background: #C7CAD0;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #AEB2BA; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    height: 10px;
    margin: 3px;
    background: transparent;
}
QScrollBar::handle:horizontal {
    min-width: 28px;
    background: #C7CAD0;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QTabWidget#workspaceTabs::pane { border: none; background-color: transparent; }
QScrollArea#settingsScroll, QScrollArea#otaScroll,
QScrollArea#singleValueScroll, QScrollArea#peakValueScroll,
QScrollArea#remotePageScroll, QScrollArea#graphPageScroll,
QScrollArea#unlimitedPageScroll, QScrollArea#apPageScroll {
    border: none;
    background-color: transparent;
}
QGraphicsView {
    background-color: #15171C;
    border: 1px solid #2A2D34;
    border-radius: 17px;
}
QToolTip {
    color: #FFFFFF;
    background-color: #292B31;
    border: 1px solid #3B3E45;
    border-radius: 7px;
    padding: 5px 8px;
}
QMessageBox { background-color: #F5F5F7; }
"""


def install_theme(app: QtWidgets.QApplication) -> None:
    """Install the light HyperOS-style palette before windows are built."""

    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = QtGui.QFont("Microsoft YaHei UI")
    font.setPointSizeF(9.5)
    app.setFont(font)
    app.setStyleSheet(build_stylesheet())
    pg.setConfigOptions(
        antialias=True,
        background=COLORS["plot"],
        foreground=COLORS["plot_text"],
    )


def set_status(label: QtWidgets.QLabel, text: str, kind: str = "info") -> None:
    """Update a pill label and force Qt to re-evaluate its dynamic selector."""

    label.setText(text)
    label.setProperty("statusKind", kind)
    label.style().unpolish(label)
    label.style().polish(label)
    label.update()


def polish_plots(root: QtWidgets.QWidget) -> None:
    """Keep every pyqtgraph view legible inside the soft light shell."""

    for plot in root.findChildren(pg.PlotWidget):
        plot.setBackground(COLORS["plot"])
        plot.setObjectName(plot.objectName() or "measurementPlot")
        item = plot.getPlotItem()
        item.showGrid(x=True, y=True, alpha=0.13)
        for axis_name in ("left", "bottom", "right", "top"):
            axis = item.getAxis(axis_name)
            axis.setPen(pg.mkPen("#555B66", width=1))
            axis.setTextPen(pg.mkPen(COLORS["plot_text"]))
        if getattr(item, "titleLabel", None) is not None:
            item.titleLabel.setText(item.titleLabel.text, color="#DDE3EC", size="11pt")


def make_logo(size: int = 44) -> QtGui.QPixmap:
    """Create the small blue glass-like product mark without external assets."""

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    rect = QtCore.QRectF(1, 1, size - 2, size - 2)
    gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
    gradient.setColorAt(0.0, QtGui.QColor("#66B5FF"))
    gradient.setColorAt(0.52, QtGui.QColor("#3482FF"))
    gradient.setColorAt(1.0, QtGui.QColor("#675CFF"))
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(gradient)
    painter.drawRoundedRect(rect, size * 0.28, size * 0.28)

    pen = QtGui.QPen(QtGui.QColor("#FFFFFF"), max(2.0, size / 17.0))
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    painter.setPen(pen)
    path = QtGui.QPainterPath()
    path.moveTo(size * 0.19, size * 0.57)
    path.cubicTo(
        size * 0.31,
        size * 0.24,
        size * 0.40,
        size * 0.78,
        size * 0.53,
        size * 0.43,
    )
    path.cubicTo(
        size * 0.64,
        size * 0.17,
        size * 0.73,
        size * 0.73,
        size * 0.82,
        size * 0.42,
    )
    painter.drawPath(path)
    painter.end()
    return pixmap


def make_nav_icon(kind: str, color: str, size: int = 22) -> QtGui.QIcon:
    """Draw compact monochrome navigation symbols in the current state color."""

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color), 1.9)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    w = float(size)

    if kind == "spectrum":
        points = QtGui.QPolygonF(
            [
                QtCore.QPointF(w * 0.08, w * 0.60),
                QtCore.QPointF(w * 0.25, w * 0.60),
                QtCore.QPointF(w * 0.38, w * 0.25),
                QtCore.QPointF(w * 0.52, w * 0.78),
                QtCore.QPointF(w * 0.66, w * 0.40),
                QtCore.QPointF(w * 0.80, w * 0.60),
                QtCore.QPointF(w * 0.92, w * 0.60),
            ]
        )
        painter.drawPolyline(points)
    elif kind == "finger":
        painter.drawRoundedRect(QtCore.QRectF(w * 0.18, w * 0.08, w * 0.64, w * 0.84), 7, 7)
        painter.setBrush(QtGui.QColor(color))
        for x, y in ((0.36, 0.33), (0.64, 0.33), (0.36, 0.60), (0.64, 0.60)):
            painter.drawEllipse(QtCore.QPointF(w * x, w * y), 1.35, 1.35)
    elif kind == "network":
        painter.drawArc(QtCore.QRectF(w * 0.10, w * 0.12, w * 0.80, w * 0.74), 35 * 16, 110 * 16)
        painter.drawArc(QtCore.QRectF(w * 0.25, w * 0.30, w * 0.50, w * 0.47), 35 * 16, 110 * 16)
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(QtCore.QPointF(w * 0.5, w * 0.80), 1.7, 1.7)
    elif kind == "ota":
        painter.drawRoundedRect(QtCore.QRectF(w * 0.13, w * 0.54, w * 0.74, w * 0.34), 4, 4)
        painter.drawLine(QtCore.QPointF(w * 0.5, w * 0.10), QtCore.QPointF(w * 0.5, w * 0.64))
        painter.drawLine(QtCore.QPointF(w * 0.30, w * 0.31), QtCore.QPointF(w * 0.5, w * 0.10))
        painter.drawLine(QtCore.QPointF(w * 0.70, w * 0.31), QtCore.QPointF(w * 0.5, w * 0.10))
    elif kind == "device":
        painter.drawRoundedRect(QtCore.QRectF(w * 0.12, w * 0.17, w * 0.76, w * 0.66), 5, 5)
        painter.drawLine(QtCore.QPointF(w * 0.30, w * 0.37), QtCore.QPointF(w * 0.70, w * 0.37))
        painter.drawLine(QtCore.QPointF(w * 0.30, w * 0.58), QtCore.QPointF(w * 0.56, w * 0.58))
    else:
        painter.drawEllipse(QtCore.QRectF(w * 0.18, w * 0.18, w * 0.64, w * 0.64))

    painter.end()
    return QtGui.QIcon(pixmap)
