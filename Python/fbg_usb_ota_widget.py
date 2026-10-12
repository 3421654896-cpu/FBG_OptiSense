"""Dedicated physical USB firmware update window."""
from PyQt5 import QtCore, QtWidgets
from serial.tools import list_ports

from fbg_ota_protocol import load_package
from fbg_usb_ota import UsbOtaClient


class UsbUpdateWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    info = QtCore.pyqtSignal(object)
    installed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, port, serial_number, package=None, parent=None,
                 package_resolver=None, newer_only=False):
        super().__init__(parent)
        self.port, self.serial_number, self.package = port, serial_number, package
        self._cancelled = False
        self.package_resolver = package_resolver
        self.newer_only = newer_only

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            with UsbOtaClient(self.port, self.serial_number) as client:
                if self.newer_only:
                    info = client.query_info()
                    if self.package_resolver is not None:
                        self.package = self.package_resolver(info.serial_number)
                    if self.package is None:
                        raise ValueError('暂无适用的 USB 固件更新包')
                    self.info.emit(info)
                    if info.firmware_version >= self.package.firmware_version:
                        return
                    if not info.bootloader_ready:
                        raise ValueError('USB 更新引导程序尚未就绪，请联系开发者初始化')
                    if self._cancelled:
                        return
                if self.package is None:
                    self.info.emit(client.query_info())
                else:
                    result = client.install(
                        self.package,
                        progress=lambda p: self.progress.emit(
                            min(99, int(p.sent * 100 / max(1, p.total))), p.message),
                        cancelled=lambda: self._cancelled)
                    self.installed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class UsbFirmwareUpdateDialog(QtWidgets.QDialog):
    firmware_installed = QtCore.pyqtSignal(object)
    version_observed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, *, machine_label="解调仪", package=None,
                 before_update=None, ports_provider=list_ports.comports,
                 auto_update=False, package_resolver=None):
        super().__init__(parent)
        self.machine_label, self.package = machine_label, package
        self.before_update, self.ports_provider = before_update, ports_provider
        self.worker = None
        self._already_current = False
        self.auto_update = bool(auto_update)
        self.package_resolver = package_resolver
        self.setWindowTitle(f"USB 固件更新 — {machine_label}")
        self.resize(680, 360)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        title = QtWidgets.QLabel("USB 固件更新")
        title.setObjectName("brandTitle")
        layout.addWidget(title)
        hint = QtWidgets.QLabel(
            "连接解调仪原生 USB 数据口即可更新，不需要 Wi-Fi 或 OTA 密码。"
            "USB 序列号确定实际更新目标，软件机号档案不代表硬件身份。"
            "读取设备或更新前会暂停采集并关光；写入、校验和重启期间请保持供电。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        port_row = QtWidgets.QHBoxLayout()
        port_row.addWidget(QtWidgets.QLabel("USB 设备"))
        self.ports = QtWidgets.QComboBox()
        self.ports.setMinimumContentsLength(24)
        self.ports.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ports.currentIndexChanged.connect(self._selection_changed)
        port_row.addWidget(self.ports, 1)
        self.refresh_button = QtWidgets.QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_ports)
        port_row.addWidget(self.refresh_button)
        self.read_button = QtWidgets.QPushButton("读取版本")
        self.read_button.clicked.connect(self.read_version)
        port_row.addWidget(self.read_button)
        layout.addLayout(port_row)
        self.device_info = QtWidgets.QLabel(f"软件档案：{machine_label}；等待读取所选 USB 设备")
        self.device_info.setWordWrap(True)
        layout.addWidget(self.device_info)
        self.port_row_label = port_row.itemAt(0).widget()
        package_row = QtWidgets.QHBoxLayout()
        self.package_info = QtWidgets.QLabel()
        self.package_info.setWordWrap(True)
        package_row.addWidget(self.package_info, 1)
        self.browse_button = QtWidgets.QPushButton("选择固件包")
        self.browse_button.clicked.connect(self.browse)
        package_row.addWidget(self.browse_button)
        layout.addLayout(package_row)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("等待更新")
        layout.addWidget(self.progress)
        self.message = QtWidgets.QLabel("首次启用需要先安装支持 USB 更新的固件与更新引导程序。")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(250)
        layout.addWidget(self.log, 1)
        layout.addStretch(1)
        buttons = QtWidgets.QHBoxLayout()
        self.cancel_button = QtWidgets.QPushButton("取消更新")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        self.start_button = QtWidgets.QPushButton("USB 一键更新")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start_update)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)
        self.refresh_ports()
        self.set_package(package)
        for control in (self.refresh_button, self.read_button, self.browse_button, self.log):
            control.hide()
        self.start_button.setText('一键更新')
        self.message.setText('自动读取设备版本并检查更新；无需输入 IP 或密码。')
        if self.auto_update:
            QtCore.QTimer.singleShot(0, self._auto_start)

    def _auto_start(self):
        if self.isVisible():
            self.start_update()

    def _selection_changed(self):
        self._already_current = False
        if hasattr(self, 'start_button'):
            self.start_button.setText('一键更新')
        self.device_info.setText(f"软件档案：{self.machine_label}；等待读取所选 USB 设备")
        self.message.setText("等待读取当前 USB 设备")
        self.progress.setValue(0)
        self.progress.setFormat("等待更新")

    @property
    def busy(self):
        return bool(self.worker is not None and self.worker.isRunning())

    def refresh_ports(self):
        if self.busy:
            return
        previous = self.ports.currentData()
        self.ports.clear()
        ports = [p for p in self.ports_provider()
                 if p.vid == 0x0483 and p.pid == 0x5740 and p.serial_number]
        if len(ports) != 1:
            self.ports.addItem("请选择需要更新的解调仪", None)
        for p in ports:
            self.ports.addItem(f"{p.device} · STM32 · {p.serial_number}",
                               (p.device, p.serial_number))
        for i in range(self.ports.count()):
            if previous is not None and self.ports.itemData(i) == previous:
                self.ports.setCurrentIndex(i)
        self.ports.setVisible(len(ports) != 1)
        self.port_row_label.setVisible(len(ports) != 1)
        if len(ports) == 1:
            self.device_info.setText(f'USB 设备：{ports[0].device} · {ports[0].serial_number}')

    def set_package(self, package):
        self.package = package
        self.package_info.setText(
            '连接后自动选择并校验适用的固件更新包'
            if self.package_resolver is not None else
            f"固件 v{package.version_text} · {package.image_size / 1024:.1f} KB"
            if package is not None else "尚未选择 .fbgfw 固件包")
        self.start_button.setEnabled(
            (package is not None or self.package_resolver is not None) and not self.busy)

    def browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 USB 更新固件", "", "FBG 固件 (*.fbgfw)")
        if path:
            try:
                self.set_package(load_package(path))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "固件包不可用", str(exc))

    def _launch(self, package=None, *, one_click=False):
        if self.busy:
            return
        target = self.ports.currentData()
        if target is None:
            self.refresh_ports()
            target = self.ports.currentData()
        if target is None:
            self.message.setText('请连接原生 USB 数据口；多台设备同时接入时，请先选择更新目标。')
            return
        available = {(p.device, p.serial_number) for p in self.ports_provider()
                     if p.vid == 0x0483 and p.pid == 0x5740 and p.serial_number}
        if target not in available:
            self.message.setText('USB 设备已变化，请重新选择目标后更新。')
            self.refresh_ports()
            return
        if package is not None and not one_click:
            answer = QtWidgets.QMessageBox.question(
                self, "确认 USB 固件更新",
                f"软件档案：{self.machine_label}\n实际 USB 目标：{target[0]}\n设备序列号：{target[1]}\n"
                f"将安装 v{package.version_text}。请确认连接的是目标机，并保持供电。",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        if self.before_update is not None and not self.before_update():
            return
        self.worker = UsbUpdateWorker(
            *target, package=package, parent=self,
            package_resolver=self.package_resolver if one_click else None,
            newer_only=one_click)
        self.worker.progress.connect(self._progress)
        self.worker.info.connect(self._info)
        self.worker.installed.connect(self._installed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.progress.setValue(0)
        self.message.setText("正在读取 USB 设备" if package is None else "正在预检并建立 USB 更新连接")
        for control in (self.ports, self.refresh_button, self.read_button,
                        self.browse_button, self.start_button):
            control.setEnabled(False)
        self.cancel_button.setEnabled(package is not None or one_click)
        self.worker.start()

    def read_version(self):
        self._launch()

    def start_update(self):
        if self.package is not None or self.package_resolver is not None:
            self._launch(self.package, one_click=True)

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)

    def _progress(self, percent, message):
        self.progress.setValue(percent)
        self.progress.setFormat(f"{percent}%")
        self.message.setText(message)
        self.log.appendPlainText(message)
        if "不能取消" in message:
            self.cancel_button.setEnabled(False)

    def _info(self, info):
        if self.worker is not None and self.worker.package is not None:
            self.package = self.worker.package
            self.package_info.setText(
                f'可用固件 v{self.package.version_text} · {self.package.image_size / 1024:.1f} KB')
        self.device_info.setText(
            f"实际 USB 设备：{info.serial_number} · 当前固件 v{info.version_text}")
        self.version_observed.emit(info.firmware_version)
        if not info.bootloader_ready:
            self.message.setText("更新引导程序尚未就绪，请先通过 CMSIS 初始化。")
        elif self.package is not None and info.firmware_version >= self.package.firmware_version:
            self._already_current = True
            self.progress.setFormat('已是最新版本')
            self.message.setText("已是最新版本（相对于当前可用更新包）")
        else:
            self.message.setText("USB 更新通道已就绪")

    def _installed(self, info):
        self._info(info)
        self.progress.setValue(100)
        self.progress.setFormat("更新完成")
        self.message.setText("新版本和新的启动标识均已确认，更新完成，无需断电。")
        self.log.appendPlainText(self.message.text())
        self.firmware_installed.emit(self.package)

    def _failed(self, message):
        self.progress.setFormat("未完成")
        self.message.setText(message)
        self.log.appendPlainText(message)

    def _finished(self):
        for control in (self.ports, self.refresh_button, self.read_button, self.browse_button):
            control.setEnabled(True)
        self.start_button.setEnabled(
            not self._already_current and (self.package is not None or self.package_resolver is not None))
        self.start_button.setText('已是最新版本' if self._already_current else '一键更新')
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event):
        if self.busy:
            self.message.setText("更新仍在运行，请等待完成；提交前可点击取消更新。")
            event.ignore()
        else:
            event.accept()

    def reject(self):
        if not self.busy:
            super().reject()
