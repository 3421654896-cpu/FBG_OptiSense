"""Qt page for safe same-LAN STM32 firmware updates."""

from __future__ import annotations

from datetime import datetime
from dataclasses import replace
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
from fbg_mqtt_transport import RemoteMqttConfig, validate_device_id
from fbg_mqtt_ota import MqttOtaClient
from fbg_usb_ota_widget import UsbFirmwareUpdateDialog
from usb_firmware_catalog import resolve_usb_package
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


class WanOtaWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    completed = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config: RemoteMqttConfig, password: str,
                 package: FirmwarePackage):
        super().__init__()
        self.config = config
        self.password = password
        self.package = package
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            with MqttOtaClient(self.config, self.password) as client:
                client.install(
                    self.package,
                    progress=lambda sent, total: self.progress.emit(
                        min(99, int(sent * 100 / max(total, 1))),
                        f"广域网已传输 {sent / 1024:.1f} / {total / 1024:.1f} KB",
                    ),
                    cancelled=lambda: self._cancelled,
                    phase=lambda stage: self.progress.emit(
                        99, "整包校验和重启中，不能取消或断电"
                    ) if stage == "commit" else None,
                )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.progress.emit(100, "已验证新固件版本和新的设备启动标识")
            self.completed.emit()


class WanVersionWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(str, str, int)
    failed = QtCore.pyqtSignal(str, str, str)

    def __init__(self, machine_id: str, config: RemoteMqttConfig, password: str):
        super().__init__()
        self.machine_id = machine_id
        self.config = config
        self.password = password

    def run(self):
        try:
            with MqttOtaClient(self.config, self.password) as client:
                _, version = client.ping()
        except Exception as exc:
            self.failed.emit(self.machine_id, self.config.device_id, str(exc))
        else:
            self.completed.emit(self.machine_id, self.config.device_id, version)


class OtaUpdatePage(QtWidgets.QWidget):
    """Operator-facing OTA page with separate encrypted settings per machine."""

    update_available_changed = QtCore.pyqtSignal(bool, str)

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
        self._customer_package_published = False
        self._release_feed_checked = False
        self._release_check_error = ""
        self._available_release_version: str | None = None
        self._observed_versions: dict[str, int] = {}
        self._thread: OtaWorker | None = None
        self.wan_config: RemoteMqttConfig | None = None
        self._active_transport = "lan"
        self._discovery_thread: DiscoveryWorker | None = None
        self._wan_version_thread: WanVersionWorker | None = None
        self._wan_probe_error = ""
        self._usb_dialog = None
        self._build_ui()
        self.board_ip.editingFinished.connect(self.save_current_settings)
        self.password.editingFinished.connect(self.save_current_settings)
        self.set_machine(self.current_machine_id, save_previous=False)

    @property
    def busy(self) -> bool:
        return bool((self._thread is not None and self._thread.isRunning())
                    or (self._usb_dialog is not None and self._usb_dialog.busy))

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("解调仪固件更新")
        title.setObjectName("brandTitle")
        caption = QtWidgets.QLabel(
            "连接解调仪原生 USB 数据口，点击一次即可自动检查并更新固件。"
            "无需 Wi-Fi、IP 或密码；升级期间请保持供电。"
        )
        caption.setWordWrap(True)
        caption.setObjectName("brandSubtitle")
        caption.setMaximumWidth(780)
        layout.addWidget(title)
        layout.addWidget(caption)

        self.machine_target = QtWidgets.QLabel()
        self.machine_target.setObjectName("softHint")
        self.machine_target.setWordWrap(True)
        layout.addWidget(self.machine_target)

        self.update_offer = QtWidgets.QLabel("正在等待设备版本信息…")
        self.update_offer.setWordWrap(True)
        self.update_offer.setObjectName("brandSubtitle")
        self.update_offer.setMaximumWidth(780)
        layout.addWidget(self.update_offer)
        self.update_offer.hide()  # Wireless OTA entry is temporarily hidden.

        warning = QtWidgets.QLabel(
            "首次启用：需要用烧录线写入一次‘OTA引导程序 + 搬移后应用’。"
            "旧板卡先用已有局域网或烧录线安装广域网版固件；之后可远程升级。"
        )
        warning.setWordWrap(True)
        warning.setProperty("role", "warning")
        self.first_use_warning = warning

        card = QtWidgets.QFrame()
        self.developer_card = card
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

        self.mqtt_device_id = QtWidgets.QLineEdit()
        self.mqtt_device_id.setPlaceholderText("设备专属 ID，例如 fbg-395736563033")
        self.mqtt_device_id.editingFinished.connect(self.save_current_settings)
        self.mqtt_device_id.editingFinished.connect(self._wan_binding_changed)
        form.addWidget(QtWidgets.QLabel("广域网设备 ID"), 4, 0)
        form.addWidget(self.mqtt_device_id, 4, 1, 1, 2)

        self.manual_update_button = QtWidgets.QPushButton("开发者：手动刷写固件包")
        self.manual_update_button.clicked.connect(lambda: self._start())
        form.addWidget(self.manual_update_button, 5, 1, 1, 2)

        self.developer_panel = QtWidgets.QWidget()
        developer_layout = QtWidgets.QVBoxLayout(self.developer_panel)
        developer_layout.setContentsMargins(0, 0, 0, 0)
        developer_layout.addWidget(self.first_use_warning)
        developer_layout.addWidget(card)
        self.developer_panel.hide()
        layout.addWidget(self.developer_panel)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("等待升级")
        self.progress.setMaximumWidth(780)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.progress_message = QtWidgets.QLabel("等待发现可更新版本")
        self.progress_message.setObjectName("softHint")
        self.progress_message.setWordWrap(True)
        self.progress_message.setMinimumHeight(
            2 * self.progress_message.fontMetrics().lineSpacing() + 8
        )
        self.progress_message.setMaximumWidth(780)
        self.progress_message.hide()
        layout.addWidget(self.progress_message)

        self.power_cycle_hint = QtWidgets.QLabel(
            "断电规则：写入、校验和自动重启期间都不要断电；"
            "只有此处明确显示“现在请断电重上电”时，才执行完全断电。"
        )
        self.power_cycle_hint.setWordWrap(True)
        self.power_cycle_hint.setProperty("role", "warning")
        layout.addWidget(self.power_cycle_hint)
        self.power_cycle_hint.hide()

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMinimumHeight(170)
        self.log.setPlaceholderText("升级过程、分块回执和校验结果会显示在这里")
        self.log.hide()
        layout.addWidget(self.log, 1)

        buttons_widget = QtWidgets.QWidget()
        buttons = FlowLayout(
            buttons_widget, horizontal_spacing=8, vertical_spacing=8
        )
        self.cancel_button = QtWidgets.QPushButton("取消")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        self.start_button = QtWidgets.QPushButton("一键更新")
        self.start_button.setProperty("role", "primary")
        self.start_button.setMinimumWidth(150)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_one_click)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.start_button)
        self.start_button.hide()
        self.usb_update_button = QtWidgets.QPushButton("一键更新（USB）")
        self.usb_update_button.setProperty('role', 'primary')
        self.usb_update_button.setMinimumWidth(150)
        self.usb_update_button.clicked.connect(self.open_usb_update)
        buttons.addWidget(self.usb_update_button)
        layout.addWidget(buttons_widget)

        # Keep customer controls together at the top of a tall page. In
        # developer mode the visible log takes the spare height instead.
        self.customer_spacer = QtWidgets.QWidget()
        self.customer_spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        layout.addWidget(self.customer_spacer, 1)

    def set_developer_mode(self, enabled: bool) -> None:
        """Keep commissioning and diagnostics available without crowding customers."""

        self.developer_panel.hide()
        self.log.setVisible(bool(enabled))
        self.customer_spacer.setVisible(not enabled)

    def open_usb_update(self):
        if self.busy:
            return
        if self._usb_dialog is not None:
            self._usb_dialog.close()
            self._usb_dialog.deleteLater()
        machine_id, package = self.current_machine_id, self.package
        self._usb_dialog = UsbFirmwareUpdateDialog(
            self, machine_label=self.current_machine_label, package=self.package,
            before_update=self.before_update, auto_update=True,
            package_resolver=lambda serial_number:
                resolve_usb_package(machine_id, serial_number, package))
        # A USB serial identifies hardware, whereas the selected machine here
        # identifies calibration/remote profiles. Do not cache a USB version
        # as that profile's wireless version without a proven serial binding.
        self._usb_dialog.firmware_installed.connect(self._usb_installed)
        self._usb_dialog.show()

    def _usb_installed(self, package):
        if self.after_update is not None:
            self.after_update(package)
        self.refresh_update_offer()

    def set_wan_config(self, config: RemoteMqttConfig | None) -> None:
        self.wan_config = config
        self._observed_versions.pop(self.current_machine_id, None)
        self.refresh_update_offer()

    def _wan_binding_changed(self) -> None:
        self._observed_versions.pop(self.current_machine_id, None)
        self._wan_probe_error = ""
        self.refresh_update_offer()
        self.check_wan_version()

    def check_wan_version(self) -> None:
        """Read firmware version from a signed online challenge, never retained status."""
        target = self._wan_target()
        if target is None or not self.password.text() or self.busy:
            return
        if self._wan_version_thread is not None and self._wan_version_thread.isRunning():
            return
        self._wan_probe_error = ""
        worker = WanVersionWorker(self.current_machine_id, target, self.password.text())
        self._wan_version_thread = worker
        worker.completed.connect(self._wan_version_observed)
        worker.failed.connect(self._wan_version_failed)
        worker.start()
        self.refresh_update_offer()

    def _wan_version_observed(self, machine_id: str, device_id: str, version: int) -> None:
        target = self._wan_target()
        if machine_id != self.current_machine_id or target is None or target.device_id != device_id:
            return
        self._wan_probe_error = ""
        self.set_observed_firmware_version(machine_id, version)

    def _wan_version_failed(self, machine_id: str, device_id: str, message: str) -> None:
        target = self._wan_target()
        if machine_id != self.current_machine_id or target is None or target.device_id != device_id:
            return
        self._wan_probe_error = message
        self.refresh_update_offer()

    def _wan_target(self) -> RemoteMqttConfig | None:
        device_id = self.mqtt_device_id.text().strip()
        if not device_id or self.wan_config is None:
            return None
        try:
            return replace(self.wan_config, device_id=validate_device_id(device_id))
        except ValueError:
            return None

    def set_available_release_version(self, version: str | None) -> None:
        """Record the result of a successful public Release check."""

        self._release_feed_checked = True
        self._release_check_error = ""
        self._available_release_version = version
        if version is None or (
            self.package is not None and self.package.version_text != version
        ):
            self._customer_package_published = False
        self.refresh_update_offer()

    def set_release_check_failed(self, message: str) -> None:
        """A failed network check is not proof that the firmware is current."""

        self._release_feed_checked = False
        self._release_check_error = message
        self.refresh_update_offer()

    def set_observed_firmware_version(self, machine_id: str, version: int) -> None:
        """Accept only a live version observed for the currently selected unit."""

        if machine_id not in MACHINE_LABELS or not 0 <= int(version) <= 0xFFFFFFFF:
            return
        self._observed_versions[machine_id] = int(version)
        if machine_id == self.current_machine_id:
            self.refresh_update_offer()

    def refresh_update_offer(self) -> None:
        """Offer only a validated, strictly newer package for this machine."""

        observed = self._observed_versions.get(self.current_machine_id)
        ready = bool(
            not self.busy
            and not self._release_check_error
            and observed is not None
            and self.package is not None
            and self._customer_package_published
            and self.package.firmware_version > observed
            and (self._wan_target() is not None or self.board_ip.text().strip())
            and self.password.text()
        )
        notice = bool(
            not self.busy
            and not self._release_check_error
            and self.package is not None
            and self._customer_package_published
            and (observed is None or self.package.firmware_version > observed)
        )
        if self.busy:
            text = "正在更新，请保持设备供电。"
        elif self._release_check_error:
            text = "暂时无法检查固件更新，请稍后重试。"
        elif self._release_feed_checked and self._available_release_version is None:
            text = "已是最新版本（当前没有可用的公开固件更新）。"
        elif self._release_feed_checked and self._available_release_version and (
            self.package is None or not self._customer_package_published
        ):
            text = f"固件 v{self._available_release_version} 已发布，正在准备更新包。"
        elif observed is None:
            if notice:
                if self._wan_target() is not None:
                    text = ("广域网设备未能确认在线版本，请检查设备连接。"
                            if self._wan_probe_error else "正在通过广域网确认设备固件版本…")
                else:
                    text = (
                        f"GitHub 已发布适合{self.current_machine_label}的固件 "
                        f"v{self.package.version_text}；连接并开始采集后确认是否需要更新。"
                    )
            else:
                text = "连接解调仪并开始采集后，软件会读取当前固件版本。"
        elif self.package is None or not self._customer_package_published:
            text = "正在检查这台解调仪的固件更新。"
        elif self.package.firmware_version <= observed:
            text = "已是最新版本。"
        elif (self._wan_target() is None and not self.board_ip.text().strip()) or not self.password.text():
            text = "发现新版本，但本机尚未完成 OTA 配置；请联系开发者。"
        else:
            path = "广域网" if self._wan_target() is not None else "局域网"
            text = f"发现适合{self.current_machine_label}的固件 v{self.package.version_text}，可经{path}一键更新。"
        self.update_offer.setText(text)
        self.start_button.setEnabled(ready)
        self.update_available_changed.emit(notice, text)

    def start_one_click(self) -> None:
        """Customer path: no file, IP, password or confirmation dialog."""

        self.refresh_update_offer()
        if not self.start_button.isEnabled():
            return
        self._start(confirm=False)

    def set_machine(self, machine_id: str, *, save_previous=True) -> bool:
        """Switch OTA target without changing the owner of calibration tables."""

        if machine_id not in MACHINE_LABELS:
            return False
        if self.busy:
            return False
        if self._usb_dialog is not None:
            self._usb_dialog.close()
            self._usb_dialog.deleteLater()
            self._usb_dialog = None
        if save_previous and hasattr(self, "board_ip"):
            self.save_current_settings()
        self.current_machine_id = machine_id
        self.current_machine_label = MACHINE_LABELS[machine_id]
        self._release_feed_checked = False
        self._release_check_error = ""
        self._available_release_version = None
        # A version observed on another physical unit must never authorize an
        # unattended update after the operator changes the selected machine.
        self._observed_versions.pop(machine_id, None)
        self._wan_probe_error = ""
        self.machine_target.setText(
            f"当前设备：{self.current_machine_label}"
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
            self.mqtt_device_id.setText(profile.get("mqtt_device_id", ""))
            self.package = None
            self._customer_package_published = False
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
            self.progress_message.show()
        self.refresh_update_offer()
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
                mqtt_device_id=self.mqtt_device_id.text(),
            )
        except (OtaCredentialError, OSError, ValueError) as exc:
            self._append(f"OTA 参数保存失败：{exc}")
            self.progress_message.setText(f"OTA 参数未能保存：{exc}")
            self.progress_message.show()
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
        published: bool = False,
    ) -> bool:
        """Load and display a package generated by another page in the studio."""
        try:
            package = load_package(value)
        except Exception as exc:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "固件包不可用", str(exc))
            return False
        self.package = package
        self._customer_package_published = bool(published)
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
        self.refresh_update_offer()
        if published:
            self.check_wan_version()
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

    def _start(self, *, confirm: bool = True):
        if self.busy:
            return
        if self.package is None:
            QtWidgets.QMessageBox.warning(self, "缺少固件包", "请先选择 .fbgfw 固件包。")
            return
        wan_target = self._wan_target()
        if wan_target is None and not self.board_ip.text().strip():
            QtWidgets.QMessageBox.warning(self, "缺少板卡 IP", "请输入或自动查找板卡 IP。")
            return
        if not self.password.text():
            QtWidgets.QMessageBox.warning(
                self, "缺少授权密码", "请输入板卡 OTA 授权密码。"
            )
            return
        self.save_current_settings()
        if confirm:
            answer = QtWidgets.QMessageBox.question(
                self,
                "确认固件 OTA",
                f"目标机号：{self.current_machine_label}\n"
                f"将经{'广域网' if wan_target else '局域网'}向 "
                f"{wan_target.device_id if wan_target else self.board_ip.text().strip()} 写入版本 "
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
        self.progress.show()
        self.progress_message.setText("正在建立广域网升级连接" if wan_target else "正在建立局域网升级连接")
        self.progress_message.show()
        self.power_cycle_hint.setText(
            "断电规则：OTA 正在运行，现在不要断电。"
            "软件会先等待板卡自动重启；仅在局域网方式确认镜像已提交但自动恢复失败后，"
            "才会明确提示手动断电。"
        )
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.power_cycle_hint.show()
        self._active_transport = "wan" if wan_target else "lan"
        if wan_target:
            self._thread = WanOtaWorker(wan_target, self.password.text(), self.package)
        else:
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
        self.refresh_update_offer()

    def _progress(self, percent: int, message: str):
        self.progress.setValue(percent)
        self.progress.setFormat(f"{percent}%")
        self.progress.setToolTip(message)
        self.progress_message.setText(message)
        self._append(message)
        if self._active_transport == "wan" and "不能取消" in message:
            self.cancel_button.setEnabled(False)

    def _cancel(self):
        if self._thread is not None:
            self._thread.cancel()
            self.cancel_button.setEnabled(False)
            self._append("已请求取消；当前分块回执后停止")

    def _completed(self):
        self.progress.setValue(100)
        self.progress.setFormat("100%")
        if self._active_transport == "wan":
            self.progress_message.setText("升级完成：新启动标识与固件版本均已确认")
            self._append("板卡已自动重启；广域网身份握手确认新版本正在运行。")
        else:
            self.progress_message.setText("升级完成：重启版本与 RAW READY 均已确认")
            self._append("板卡已自动重启；REBOOT 版本和全功能局域网 RAW READY 均已确认。")
        self.power_cycle_hint.setText(
            "升级完成：板卡已自动重启并通过版本核验，无需断电。"
        )
        if self.package is not None:
            self.set_observed_firmware_version(
                self.current_machine_id, self.package.firmware_version
            )
        QtWidgets.QMessageBox.information(
            self,
            "OTA 已完成",
            f"板卡已确认安装版本 {self.package.version_text} 并完成自动重启。\n"
            + ("广域网在线身份与版本均已验证，无需断电。" if self._active_transport == "wan"
               else "全功能局域网也已自动恢复，无需断电；现在可重新选择‘LAN 局域网’。"),
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
        self.refresh_update_offer()
        self.cancel_button.setEnabled(False)
        self.cancel_button.hide()


__all__ = [
    "OtaUpdatePage",
    "OtaWorker",
    "_manual_power_cycle_required",
]
