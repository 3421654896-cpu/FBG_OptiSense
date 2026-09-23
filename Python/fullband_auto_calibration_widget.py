"""Qt page for the resumable 145 mA full-band auto-calibration workflow."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

from runtime_paths import output_path
from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "fullband_auto_calibration.py"
DEFAULT_OUTPUT = output_path("fullband_145_auto")
STAGE_COUNT = 17


class FullbandAutoCalibrationWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str, str)
    stage_signal = QtCore.pyqtSignal(str, int, int)
    result_signal = QtCore.pyqtSignal(bool, str, str)

    def __init__(
        self,
        port: str,
        meter: str,
        output_dir: Path,
        parent=None,
        certification_direction: str = "forward",
        power_relative_tolerance: float = 0.03,
    ):
        super().__init__(parent)
        self.port = str(port)
        self.meter = str(meter)
        self.output_dir = Path(output_dir)
        self.certification_direction = str(certification_direction)
        self.power_relative_tolerance = float(power_relative_tolerance)
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.result_output = ""

    def stop(self) -> None:
        self.cancel_requested = True
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            self.log_signal.emit("已请求安全停止；正在等待当前测量收尾并确认SOA关光…", "WARNING")
        except Exception as exc:
            self.log_signal.emit(f"发送安全停止请求失败：{exc}", "ERROR")

    @staticmethod
    def _event(line: str) -> dict | None:
        marker = "@@AUTO@@ "
        if not line.startswith(marker):
            return None
        try:
            value = json.loads(line[len(marker) :])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def run(self) -> None:
        command = [
            sys.executable,
            str(SCRIPT),
            "--port",
            self.port,
            "--meter",
            self.meter,
            "--output-dir",
            str(self.output_dir),
            "--certification-direction",
            self.certification_direction,
            "--power-relative-tolerance",
            f"{self.power_relative_tolerance:.9g}",
        ]
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        success = False
        message = "自动校准未完成"
        try:
            self.process = subprocess.Popen(
                command,
                cwd=HERE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            assert self.process.stdout is not None
            for raw in self.process.stdout:
                line = raw.rstrip("\r\n")
                event = self._event(line)
                if event is None:
                    if line:
                        self.log_signal.emit(line, "INFO")
                    continue
                kind = str(event.get("kind", ""))
                if kind in {"stage_started", "stage_completed", "stage_skipped"}:
                    self.stage_signal.emit(
                        str(event.get("name", "准备中")),
                        int(event.get("index", 0)),
                        int(event.get("total", STAGE_COUNT)),
                    )
                elif kind == "preflight":
                    if event.get("dry_run"):
                        continue
                    self.log_signal.emit(
                        f"仪器已确认：{event.get('meter')}；板卡145 mA专用配置关光预检通过",
                        "INFO",
                    )
                elif kind == "safety":
                    level = "INFO" if event.get("shutter_confirmed") else "ERROR"
                    text = (
                        "SOA最终硬件关光已确认"
                        if event.get("shutter_confirmed")
                        else f"SOA最终关光未确认：{event.get('error')}"
                    )
                    self.log_signal.emit(text, level)
                elif kind == "complete":
                    success = not bool(event.get("dry_run"))
                    self.result_output = str(event.get("output", ""))
                    target = event.get("target_power_mw")
                    message = (
                        f"2001点自动校准和发布完成，共同目标功率 {float(target):.3f} mW"
                        if target is not None
                        else "2001点自动校准完成"
                    )
                elif kind == "cancelled":
                    message = str(event.get("message", "操作员已取消"))
                elif kind in {"failed", "stage_failed"}:
                    message = str(event.get("error", "自动校准失败"))
                    self.log_signal.emit(message, "ERROR")
            code = self.process.wait()
            if code != 0 and success:
                success = False
                message = f"校准程序异常退出（{code}）"
            elif code == 130 or self.cancel_requested:
                success = False
                message = "校准已安全停止；检查点已保留，可直接继续"
            elif code != 0 and not message:
                message = f"校准程序退出码为{code}"
        except Exception as exc:
            success = False
            message = str(exc)
            self.log_signal.emit(message, "ERROR")
        finally:
            self.process = None
            self.result_signal.emit(success, message, self.result_output)


class FullbandAutoCalibrationWindow(QtWidgets.QWidget):
    calibration_promoted = QtCore.pyqtSignal(str)

    def __init__(
        self,
        prepare_session: Callable[[], str] | None = None,
        finish_session: Callable[[bool], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.prepare_session = prepare_session
        self.finish_session = finish_session
        self.worker: FullbandAutoCalibrationWorker | None = None

        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.content_scroll = QtWidgets.QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setObjectName("autoCalibrationPageContent")
        self.content_scroll.setWidget(content)
        page_layout.addWidget(self.content_scroll)

        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        title = QtWidgets.QLabel("145 mA · 2001点自动波长/功率校准")
        title.setObjectName("pageHeading")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        root.addWidget(title)

        summary = QtWidgets.QLabel(
            "自动测量1525.00～1565.00 nm（0.02 nm间隔），搜索每个波长的高功率单模分支，"
            "以最弱点决定全波段共同最大功率，并按下方选择的单点功率容差闭环验收。"
            "只有2001/2001点、所选运行路径独立复测和PDT/PDR参考全部通过，"
            "才会备份并替换正式点表。"
        )
        summary.setWordWrap(True)
        summary.setObjectName("softHint")
        root.addWidget(summary)

        limits = QtWidgets.QLabel(
            "专用上限：GAIN 145 mA · SOA 145 mA · PHASE 10 mA · WAVE-A/B 30 mA"
            "　|　普通单值、18点、45点仍为135 mA"
        )
        limits.setObjectName("metricBadge")
        limits.setWordWrap(True)
        root.addWidget(limits)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.meter_edit = QtWidgets.QLineEdit("GPIB0::7::INSTR")
        self.meter_edit.setPlaceholderText("例如 GPIB0::7::INSTR")
        form.addRow("AQ6150B", self.meter_edit)

        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItem(
            "固定升序（推荐，与2001点及临时45点一致）", "forward"
        )
        self.direction_combo.addItem("正向＋反向（附加迟滞审计）", "bidirectional")
        form.addRow("扫描路径", self.direction_combo)

        self.power_tolerance_combo = QtWidgets.QComboBox()
        self.power_tolerance_combo.addItem("±0.5%（严格）", 0.005)
        self.power_tolerance_combo.addItem("±1%", 0.01)
        self.power_tolerance_combo.addItem("±2%", 0.02)
        self.power_tolerance_combo.addItem("±3%（当前推荐）", 0.03)
        self.power_tolerance_combo.setCurrentIndex(3)
        form.addRow("单点功率容差", self.power_tolerance_combo)

        output_row = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT))
        self.output_button = QtWidgets.QPushButton("选择目录")
        self.output_button.clicked.connect(self.choose_output_dir)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(self.output_button)
        form.addRow("断点/报告目录", output_row)
        root.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始或继续自动校准")
        self.start_button.setProperty("role", "primary")
        self.stop_button = QtWidgets.QPushButton("安全停止")
        self.stop_button.setEnabled(False)
        self.open_button = QtWidgets.QPushButton("打开结果目录")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.open_button.clicked.connect(self.open_output_dir)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, STAGE_COUNT)
        self.progress.setValue(0)
        self.progress.setFormat("等待开始")
        root.addWidget(self.progress)

        self.status_label = QtWidgets.QLabel(
            "启动时先核验AQ6150B身份，并在SOA硬件关光状态下验证板卡确实支持145 mA。"
        )
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.log_area = QtWidgets.QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumBlockCount(3000)
        self.log_area.setMinimumHeight(220)
        root.addWidget(self.log_area, 1)

    def set_dac_label(self, _dac_type) -> None:
        return

    def choose_output_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择145 mA自动校准目录", self.output_edit.text().strip()
        )
        if selected:
            self.output_edit.setText(selected)

    def open_output_dir(self) -> None:
        path = Path(self.output_edit.text().strip() or DEFAULT_OUTPUT).resolve()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def append_log(self, message: str, level: str = "INFO") -> None:
        self.log_area.appendPlainText(f"[{level}] {message}")
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        meter = self.meter_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not meter or not output_text:
            QtWidgets.QMessageBox.warning(self, "参数不完整", "请填写AQ6150B地址和结果目录。")
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "确认开始145 mA自动校准",
            "该流程会独占板卡和AQ6150B，可能持续较长时间。\n"
            "确认光纤连接、散热和AQ6150B量程正常后继续。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            port = self.prepare_session() if self.prepare_session else "COM6"
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "无法开始自动校准", str(exc))
            return

        self.worker = FullbandAutoCalibrationWorker(
            port,
            meter,
            Path(output_text),
            self,
            certification_direction=str(self.direction_combo.currentData()),
            power_relative_tolerance=float(self.power_tolerance_combo.currentData()),
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.stage_signal.connect(self.on_stage)
        self.worker.result_signal.connect(self.on_result)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.meter_edit.setEnabled(False)
        self.output_edit.setEnabled(False)
        self.output_button.setEnabled(False)
        self.direction_combo.setEnabled(False)
        self.power_tolerance_combo.setEnabled(False)
        self.status_label.setText("自动校准正在运行；关闭软件或点击停止都会先请求安全关光。")

    def stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.stop_button.setEnabled(False)
            self.status_label.setText("正在安全停止并等待SOA硬件关光确认…")
            self.worker.stop()

    def on_stage(self, name: str, index: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(max(0, min(index, total)))
        self.progress.setFormat(f"{index}/{total} · {name}")
        self.status_label.setText(f"当前步骤：{name}")

    def on_result(self, success: bool, message: str, output: str) -> None:
        try:
            if self.finish_session is not None:
                try:
                    self.finish_session(bool(success))
                except Exception as exc:
                    success = False
                    message = f"校准进程已结束，但桌面串口恢复/关光核验失败：{exc}"
                    self.append_log(message, "ERROR")
        finally:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.meter_edit.setEnabled(True)
            self.output_edit.setEnabled(True)
            self.output_button.setEnabled(True)
            self.direction_combo.setEnabled(True)
            self.power_tolerance_combo.setEnabled(True)
        self.status_label.setText(message)
        self.append_log(message, "INFO" if success else "WARNING")
        if success:
            self.progress.setValue(self.progress.maximum())
            self.progress.setFormat("2001/2001校准、复测和发布完成")
            self.calibration_promoted.emit(output)
            QtWidgets.QMessageBox.information(self, "自动校准完成", message)
        elif not self.worker or not self.worker.cancel_requested:
            QtWidgets.QMessageBox.warning(self, "自动校准未完成", message)

    def shutdown(self, wait_ms: int = 20000) -> bool:
        if self.worker is None or not self.worker.isRunning():
            return True
        self.worker.stop()
        return bool(self.worker.wait(int(wait_ms)))


__all__ = ["FullbandAutoCalibrationWindow", "FullbandAutoCalibrationWorker"]
