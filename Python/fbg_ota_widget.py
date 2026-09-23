"""Qt page for safe same-LAN STM32 firmware updates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from fbg_ota_protocol import (
    FirmwarePackage,
    OtaClient,
    OtaError,
    OtaProgress,
    discover_board,
    load_package,
)
from ota_machine_profiles import (
    MACHINE_LABELS,
    OtaCredentialError,
    OtaMachineProfileStore,
)
from ota_debug_recovery import resume_halted_verified_ota
from responsive_layout import FlowLayout


def _format_build_timestamp(value: int) -> str:
    """Render a package timestamp without letting hostile uint64 values crash Qt."""

    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "未知（时间戳超出本机范围）"


def _manual_power_cycle_required(message: str) -> bool:
    """Return true only after the board has accepted the rebooting image."""

    text = str(message)
    reboot_confirmed_but_not_online = (
        "已收到板卡重启确认" in text and "恢复全功能 LAN" in text
    )
    reboot_confirmed_but_socket_stuck = (
        "板卡已确认重启" in text and "旧 OTA 连接未按期断开" in text
    )
    return reboot_confirmed_but_not_online or reboot_confirmed_but_socket_stuck


class OtaWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    completed = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        host: str,
        password: str,
        package: FirmwarePackage,
        *,
        robust_recovery: bool = False,
    ):
        super().__init__()
        self.host = host
        self.password = password
        self.package = package
        self.robust_recovery = bool(robust_recovery)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            options = {}
            if self.robust_recovery:
                # Machine 2's fitted BW20 link was measured dropping TCP ACKs
                # and ICMP under sustained OTA traffic.  Start conservatively,
                # increase pacing after a broken seed, and ease it only after a
                # long stable ACK window.  The guarded pyOCD callback handles a
                # reset intercepted by an attached debugger without SWD flash
                # writes. Authentication and all CRC checks remain unchanged.
                options = {
                    "retries": 160,
                    "response_timeout_s": 8.0,
                    "recovery_quiet_s": 30.0,
                    "retry_owner_timeout_s": 20.0,
                    "chunk_delay_s": 0.2,
                    "adaptive_chunk_pacing": True,
                    "chunk_delay_max_s": 0.45,
                    "chunk_delay_backoff": 1.5,
                    "chunk_delay_stable_blocks": 384,
                    "post_reboot_recovery": lambda: resume_halted_verified_ota(
                        self.package
                    ),
                    "post_reboot_recovery_interval_s": 10.0,
                }
            client = OtaClient(self.host, self.password, **options)

            def on_progress(value: OtaProgress):
                percent = int(value.sent * 100 / max(value.total, 1))
                self.progress.emit(percent, value.message)

            client.upload(
                self.package,
                progress=on_progress,
                cancelled=lambda: self._cancelled,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.progress.emit(100, "新固件已运行，全功能局域网已自动恢复")
            self.completed.emit()


class DiscoveryWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(object)

    def run(self):
        self.completed.emit(discover_board())


class OtaUpdatePage(QtWidgets.QWidget):
    """Operator-facing OTA page with separate encrypted settings per machine."""

    before_update = None
    after_update = None

    def __init__(self, parent=None, *, settings_store=None):
        super().__init__(parent)
        self.setObjectName("otaPage")
        self.settings_store = settings_store or OtaMachineProfileStore()
        self.current_machine_id = self.settings_store.selected_machine_id
        self.current_machine_label = MACHINE_LABELS[self.current_machine_id]
        self._restoring_machine_settings = False
        self.package: FirmwarePackage | None = None
        self._thread: OtaWorker | None = None
        self._discovery_thread: DiscoveryWorker | None = None
        self._build_ui()
        self.board_ip.editingFinished.connect(self.save_current_settings)
        self.password.editingFinished.connect(self.save_current_settings)
        self.set_machine(self.current_machine_id, save_previous=False)

    @property
    def busy(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("局域网 OTA 固件升级")
        title.setObjectName("brandTitle")
        caption = QtWidgets.QLabel(
            "固件先写入独立暂存区，整包校验通过后才重启安装；"
            "接收阶段中断不会破坏当前程序。"
        )
        caption.setWordWrap(True)
        caption.setObjectName("brandSubtitle")
        layout.addWidget(title)
        layout.addWidget(caption)

        self.machine_target = QtWidgets.QLabel()
        self.machine_target.setObjectName("softHint")
        self.machine_target.setWordWrap(True)
        layout.addWidget(self.machine_target)

        warning = QtWidgets.QLabel(
            "首次启用：需要用烧录线写入一次‘OTA引导程序 + 搬移后应用’。"
            "从第二次开始才可只用局域网升级。"
        )
        warning.setWordWrap(True)
        warning.setProperty("role", "warning")
        layout.addWidget(warning)

        card = QtWidgets.QFrame()
        card.setObjectName("controlGroup")
        form = QtWidgets.QGridLayout(card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)

        self.package_path = QtWidgets.QLineEdit()
        self.package_path.setReadOnly(True)
        self.package_path.setPlaceholderText("请选择已校验的 .fbgfw 固件包")
        browse = QtWidgets.QPushButton("选择固件包")
        browse.clicked.connect(self._browse)
        form.addWidget(QtWidgets.QLabel("固件包"), 0, 0)
        form.addWidget(self.package_path, 0, 1)
        form.addWidget(browse, 0, 2)

        self.package_info = QtWidgets.QLabel("尚未选择固件")
        self.package_info.setWordWrap(True)
        self.package_info.setObjectName("brandSubtitle")
        form.addWidget(self.package_info, 1, 1, 1, 2)

        self.board_ip = QtWidgets.QLineEdit()
        self.board_ip.setPlaceholderText("例如 192.168.3.46")
        discover = QtWidgets.QPushButton("自动查找")
        discover.clicked.connect(self._discover)
        form.addWidget(QtWidgets.QLabel("板卡 IP"), 2, 0)
        form.addWidget(self.board_ip, 2, 1)
        form.addWidget(discover, 2, 2)

        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password.setPlaceholderText(
            "默认与板卡 Wi-Fi 密码相同；按机号用 Windows 当前用户加密保存"
        )
        show_password = QtWidgets.QCheckBox("显示")
        show_password.toggled.connect(
            lambda checked: self.password.setEchoMode(
                QtWidgets.QLineEdit.Normal
                if checked
                else QtWidgets.QLineEdit.Password
            )
        )
        form.addWidget(QtWidgets.QLabel("OTA 授权密码"), 3, 0)
        form.addWidget(self.password, 3, 1)
        form.addWidget(show_password, 3, 2)

        layout.addWidget(card)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("等待升级")
        layout.addWidget(self.progress)
        self.progress_message = QtWidgets.QLabel("等待选择并校验固件包")
        self.progress_message.setObjectName("softHint")
        self.progress_message.setWordWrap(True)
        self.progress_message.setMinimumHeight(
            2 * self.progress_message.fontMetrics().lineSpacing() + 8
        )
        layout.addWidget(self.progress_message)

        self.power_cycle_hint = QtWidgets.QLabel(
            "断电规则：写入、校验和自动重启期间都不要断电；"
            "只有此处明确显示“现在请断电重上电”时，才执行完全断电。"
        )
        self.power_cycle_hint.setWordWrap(True)
        self.power_cycle_hint.setProperty("role", "warning")
        layout.addWidget(self.power_cycle_hint)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMinimumHeight(170)
        self.log.setPlaceholderText("升级过程、分块回执和校验结果会显示在这里")
        layout.addWidget(self.log, 1)

        buttons_widget = QtWidgets.QWidget()
        buttons = FlowLayout(
            buttons_widget, horizontal_spacing=8, vertical_spacing=8
        )
        self.cancel_button = QtWidgets.QPushButton("取消")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.start_button = QtWidgets.QPushButton("开始安全升级")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self._start)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.start_button)
        layout.addWidget(buttons_widget)

    def set_machine(self, machine_id: str, *, save_previous=True) -> bool:
        """Switch OTA target without changing the owner of calibration tables."""

        if machine_id not in MACHINE_LABELS:
            return False
        if save_previous and hasattr(self, "board_ip"):
            self.save_current_settings()
        self.current_machine_id = machine_id
        self.current_machine_label = MACHINE_LABELS[machine_id]
        self.machine_target.setText(
            f"当前 OTA 目标：{self.current_machine_label}。"
            "IP、授权密码和最近固件包均按机号独立记忆。"
        )
        self._restoring_machine_settings = True
        credential_error = None
        try:
            try:
                profile = self.settings_store.load_profile(machine_id)
            except OtaCredentialError as exc:
                credential_error = str(exc)
                profile = {
                    "board_ip": "",
                    "package_path": "",
                    "password": "",
                }
            self.board_ip.setText(profile.get("board_ip", ""))
            self.password.setText(profile.get("password", ""))
            self.package = None
            self.package_path.clear()
            self.package_info.setText("尚未选择固件")
            package_path = profile.get("package_path", "")
            if package_path and Path(package_path).is_file():
                self.set_package_path(package_path, quiet=True, persist=False)
            self.settings_store.set_selected_machine(machine_id)
        finally:
            self._restoring_machine_settings = False
        self._append(f"已切换 OTA 目标为 {self.current_machine_label}")
        if credential_error:
            self._append(f"已保存的 OTA 密码未载入：{credential_error}")
            self.progress_message.setText(
                f"{self.current_machine_label}的密码未载入，请重新输入"
            )
        return True

    def save_current_settings(self) -> bool:
        if self._restoring_machine_settings:
            return True
        try:
            self.settings_store.save_profile(
                self.current_machine_id,
                board_ip=self.board_ip.text(),
                password=self.password.text(),
                package_path=self.package_path.text(),
            )
        except (OtaCredentialError, OSError, ValueError) as exc:
            self._append(f"OTA 参数保存失败：{exc}")
            self.progress_message.setText(f"OTA 参数未能保存：{exc}")
            return False
        return True

    def set_board_ip(self, value: str | None):
        if value:
            self.board_ip.setText(str(value))
            self.save_current_settings()

    def set_package_path(
        self,
        value: str | Path,
        *,
        quiet: bool = False,
        persist: bool = True,
    ) -> bool:
        """Load and display a package generated by another page in the studio."""
        try:
            package = load_package(value)
        except Exception as exc:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "固件包不可用", str(exc))
            return False
        self.package = package
        self.package_path.setText(str(package.path))
        timestamp = _format_build_timestamp(package.build_timestamp)
        self.package_info.setText(
            f"目标 STM32F205RE  ·  版本 {package.version_text}  ·  "
            f"{package.image_size / 1024:.1f} KB  ·  CRC32 {package.image_crc32:08X}  ·  "
            f"构建 {timestamp}"
        )
        self._append("固件包的包头、整包 CRC 和 STM32 向量表校验通过")
        if persist:
            self.save_current_settings()
        return True

    def _append(self, message: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {message}")

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 FBG 固件包", str(Path.cwd()), "FBG 固件 (*.fbgfw)"
        )
        if not path:
            return
        self.set_package_path(path)

    def _discover(self):
        if self._discovery_thread is not None and self._discovery_thread.isRunning():
            return
        self._append("正在当前局域网查找 FBG 板卡…")
        self._discovery_thread = DiscoveryWorker(self)
        self._discovery_thread.completed.connect(self._discovered)
        self._discovery_thread.start()

    def _discovered(self, host):
        if host:
            self.board_ip.setText(str(host))
            self.save_current_settings()
            self._append(f"已找到板卡 {host}")
        else:
            self._append("未找到可连接的板卡，请确认电脑与板卡在同一 Wi-Fi")

    def _start(self):
        if self.busy:
            return
        if self.package is None:
            QtWidgets.QMessageBox.warning(self, "缺少固件包", "请先选择 .fbgfw 固件包。")
            return
        if not self.board_ip.text().strip():
            QtWidgets.QMessageBox.warning(self, "缺少板卡 IP", "请输入或自动查找板卡 IP。")
            return
        if not self.password.text():
            QtWidgets.QMessageBox.warning(
                self, "缺少授权密码", "请输入板卡 OTA 授权密码。"
            )
            return
        self.save_current_settings()
        answer = QtWidgets.QMessageBox.question(
            self,
            "确认局域网 OTA",
            f"目标机号：{self.current_machine_label}\n"
            f"将向 {self.board_ip.text().strip()} 写入版本 "
            f"{self.package.version_text}。\n\n"
            "升级期间激光器会关闭，板卡校验后自动重启。是否继续？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        # Pausing USB/LAN/MQTT is an operational side effect, so do it only
        # after the operator has confirmed the upgrade.
        if self.before_update is not None and not self.before_update():
            return

        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self.progress_message.setText("正在建立局域网升级连接")
        self.power_cycle_hint.setText(
            "断电规则：OTA 正在运行，现在不要断电。"
            "软件会先等待板卡自动重启；仅在确认镜像已提交但自动恢复失败后，"
            "才会明确提示手动断电。"
        )
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._thread = OtaWorker(
            self.board_ip.text().strip(),
            self.password.text(),
            self.package,
            robust_recovery=self.current_machine_id == "machine_2",
        )
        self._thread.progress.connect(self._progress)
        self._thread.completed.connect(self._completed)
        self._thread.failed.connect(self._failed)
        self._thread.finished.connect(self._finished)
        self._append("已暂停光谱数据链路，开始 OTA 预检")
        self._thread.start()

    def _progress(self, percent: int, message: str):
        self.progress.setValue(percent)
        self.progress.setFormat(f"{percent}%")
        self.progress.setToolTip(message)
        self.progress_message.setText(message)
        self._append(message)

    def _cancel(self):
        if self._thread is not None:
            self._thread.cancel()
            self.cancel_button.setEnabled(False)
            self._append("已请求取消；当前分块回执后停止")

    def _completed(self):
        self.progress.setValue(100)
        self.progress.setFormat("100%")
        self.progress_message.setText("升级完成：重启版本与 RAW READY 均已确认")
        self._append(
            "板卡已自动重启；REBOOT 版本和全功能局域网 RAW READY 均已确认。"
        )
        self.power_cycle_hint.setText(
            "升级完成：板卡已自动重启并恢复局域网，无需断电。"
        )
        QtWidgets.QMessageBox.information(
            self,
            "OTA 已完成",
            f"板卡已确认安装版本 {self.package.version_text} 并完成自动重启。\n"
            "全功能局域网也已自动恢复，无需断电；现在可重新选择‘LAN 局域网’。",
        )
        if self.after_update is not None:
            self.after_update(self.package)

    def _failed(self, message: str):
        if _manual_power_cycle_required(message):
            instruction = (
                f"现在请将{self.current_machine_label}完全断电，等待 10 秒后重新上电。"
                "不要再次点击 OTA；重新上电后先等待 USB 和局域网恢复，再核对运行版本。"
            )
            self.progress.setFormat("请断电重上电")
            self.progress_message.setText(instruction)
            self.power_cycle_hint.setText(
                "现在请断电重上电：固件镜像已经完整提交并收到板卡重启确认，"
                "但自动重新联机超时；此时可以安全执行上述断电操作。"
            )
            self._append(f"自动重启恢复超时：{message}")
            self._append(instruction)
            QtWidgets.QMessageBox.warning(
                self,
                "现在需要断电重上电",
                f"{message}\n\n{instruction}",
            )
            return

        self.progress.setFormat("升级未完成")
        self.progress_message.setText(message)
        self.power_cycle_hint.setText(
            "当前没有达到可安全断电的确认条件，请不要按本提示断电；"
            "先保留现场并检查错误原因。"
        )
        self._append(f"升级失败：{message}")
        QtWidgets.QMessageBox.warning(
            self,
            "OTA 升级未完成",
            f"{message}\n\n"
            "尚未确认镜像已提交并进入重启流程，现在不要断电重上电，也不要立即重试。",
        )

    def _finished(self):
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)


__all__ = [
    "OtaUpdatePage",
    "OtaWorker",
    "_manual_power_cycle_required",
]
