"""One-time USB provisioning tool for the onboard BW20 Wi-Fi module.

Credentials are sent directly to the STM32 over its native USB CDC port.  The
tool deliberately does not write passwords to disk or print the packet.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

# PyQt's platform plugin is not always discovered when the tool is started
# through a shortcut or from a copied virtual environment.
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

import serial
from serial.tools import list_ports

from responsive_layout import configure_form_layout
from PyQt5 import QtCore, QtWidgets


PROVISION_FRAME_SIZE = 808
PROVISION_PAYLOAD_END = 804
PROVISION_VERSION = 1
PROVISION_COMMAND = 0x20
STM32_CDC_VID_PID = (0x0483, 0x5740)


@dataclass(frozen=True)
class ProvisioningSettings:
    ssid: str
    wifi_password: str
    broker_host: str
    broker_port: int
    client_id: str
    mqtt_username: str
    mqtt_password: str
    topic_prefix: str
    device_id: str
    tls: bool = True
    enabled: bool = True
    persist: bool = True


_FIELD_LIMITS = (
    ("ssid", 32),
    ("wifi_password", 63),
    ("broker_host", 127),
    ("client_id", 63),
    ("mqtt_username", 63),
    ("mqtt_password", 63),
    ("topic_prefix", 96),
    ("device_id", 32),
)


def _encode_field(settings: ProvisioningSettings, name: str, limit: int) -> bytes:
    value = str(getattr(settings, name)).encode("utf-8")
    if len(value) > limit:
        raise ValueError(f"{name} is longer than {limit} UTF-8 bytes")
    if any(marker in value for marker in (b"\x00", b"\r", b"\n", b'"')):
        raise ValueError(f"{name} contains an unsupported control character")
    if b"," in value:
        raise ValueError(f"{name} 不能包含逗号（板载 AT 指令以逗号分隔）")
    if name in {"topic_prefix", "device_id"} and any(
        marker in value for marker in (b"+", b"#", b">")
    ):
        raise ValueError(f"{name} 不能包含 MQTT 通配符 +/# 或字符 >")
    if name == "device_id" and not re.fullmatch(
        rb"fbg-[A-Za-z0-9]{1,28}", value
    ):
        raise ValueError(
            "device_id 必须为 fbg- 加1..28位 ASCII 字母或数字；"
            "后缀不能包含点、下划线或连字符"
        )
    return value


def build_provisioning_frame(settings: ProvisioningSettings) -> bytes:
    """Build the fixed 808-byte USB command accepted by the STM32 firmware."""

    for name in ("tls", "enabled", "persist"):
        if not isinstance(getattr(settings, name), bool):
            raise ValueError(f"{name} must be true or false")
    if isinstance(settings.broker_port, bool) or not isinstance(
        settings.broker_port, int
    ):
        raise ValueError("broker_port must be an integer")
    if not (1 <= settings.broker_port <= 65535):
        raise ValueError("broker_port must be in the range 1..65535")
    if not settings.ssid:
        raise ValueError("Wi-Fi SSID is required")
    if not settings.broker_host:
        raise ValueError("MQTT broker host is required")
    if not settings.device_id:
        raise ValueError("device_id is required")
    if not settings.topic_prefix.strip().strip("/"):
        raise ValueError("topic_prefix is required")
    if settings.enabled and not settings.tls:
        raise ValueError("广域网 MQTT 配置必须启用 TLS")

    fields = [_encode_field(settings, name, limit) for name, limit in _FIELD_LIMITS]
    total_fields = sum(map(len, fields))
    if 18 + total_fields > PROVISION_PAYLOAD_END:
        raise ValueError("provisioning fields do not fit in the USB frame")

    frame = bytearray(PROVISION_FRAME_SIZE)
    frame[0:4] = b"\xFF\xFF\x01\x20"
    frame[4] = PROVISION_VERSION
    frame[5] = (0x01 if settings.enabled else 0) | (0x02 if settings.persist else 0)
    for index, value in enumerate(fields):
        frame[6 + index] = len(value)
    frame[14:16] = settings.broker_port.to_bytes(2, "big")
    frame[16] = 2 if settings.tls else 1
    frame[17] = 0

    position = 18
    for value in fields:
        frame[position : position + len(value)] = value
        position += len(value)

    checksum = zlib.crc32(frame[:PROVISION_PAYLOAD_END]) & 0xFFFFFFFF
    frame[PROVISION_PAYLOAD_END:] = checksum.to_bytes(4, "big")
    return bytes(frame)


def find_provisioning_ack(
    buffer: bytes, expected_crc32: int | None = None
) -> tuple[int, int, int, int] | None:
    """Find a 20-byte configuration ACK in a mixed USB telemetry stream.

    Returns ``(status, wifi_state, config_valid, config_crc32)``.
    """

    for offset in range(max(0, len(buffer) - 4096), max(0, len(buffer) - 19)):
        if buffer[offset : offset + 2] != b"\xFF\xFF":
            continue
        candidate = buffer[offset : offset + 20]
        if candidate[2] > 3 or candidate[3] != PROVISION_COMMAND:
            continue
        if candidate[18:20] != b"\xFF\xEF":
            continue
        status = candidate[4]
        wifi_state = candidate[5]
        config_valid = candidate[6]
        if status > 4 or wifi_state > 5 or config_valid not in (0, 1):
            continue
        if candidate[7] not in (0, 1) or candidate[15] > 2:
            continue
        config_crc32 = int.from_bytes(candidate[8:12], "big")
        if status == 0 and expected_crc32 is not None and config_crc32 != expected_crc32:
            continue
        return status, wifi_state, config_valid, config_crc32
    return None


class ProvisionWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, port: str, frame: bytes):
        super().__init__()
        self.port = port
        self.frame = frame
        self.expected_crc32 = int.from_bytes(frame[PROVISION_PAYLOAD_END:], "big")

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            with serial.Serial(self.port, 115200, timeout=0.15, write_timeout=2.0) as connection:
                connection.reset_input_buffer()
                connection.write(self.frame)
                connection.flush()
                deadline = QtCore.QDeadlineTimer(6000)
                received = bytearray()
                while not deadline.hasExpired():
                    chunk = connection.read(1024)
                    if chunk:
                        received.extend(chunk)
                        if len(received) > 8192:
                            del received[:-4096]
                        ack = find_provisioning_ack(received, self.expected_crc32)
                        if ack is not None:
                            status, wifi_state, valid, checksum = ack
                            if status == 0 and valid:
                                self.finished.emit(
                                    True,
                                    "配置已写入。板卡正在连接 Wi-Fi 和公网 MQTT。"
                                    f"\n配置校验号：{checksum:08X}，无线状态：{wifi_state}",
                                )
                            else:
                                self.finished.emit(
                                    False,
                                    f"单片机拒绝了配置（状态 {status}，无线状态 {wifi_state}）。",
                                )
                            return
            self.finished.emit(
                False,
                "已发送配置，但 6 秒内没有收到确认。请确认主程序已关闭且选择的是 STM32 原生 USB 端口。",
            )
        except Exception as exc:  # serial backends expose several exception types
            self.finished.emit(False, f"配置失败：{exc}")


class ProvisionWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("settingsPage")
        self.setWindowTitle("板载 Wi-Fi 与远程监控配置")
        self.resize(620, 560)
        self._thread: QtCore.QThread | None = None
        self._worker: ProvisionWorker | None = None
        # The unified application uses this hook to stop LAN/WAN reception and
        # release the selected CDC port before the one-time provisioning transfer.
        self.before_provisioning = None

        title = QtWidgets.QLabel("无线与远程连接")
        title.setObjectName("brandTitle")
        subtitle = QtWidgets.QLabel(
            "像系统设置一样集中管理板卡网络；本地、局域网与广域网始终三选一。"
        )
        subtitle.setObjectName("brandSubtitle")
        subtitle.setWordWrap(True)

        connection_group = QtWidgets.QGroupBox("连接参数")
        form = QtWidgets.QFormLayout(connection_group)
        configure_form_layout(form)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)
        self.port = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("刷新端口")
        port_row = QtWidgets.QHBoxLayout()
        port_row.addWidget(self.port, 1)
        port_row.addWidget(self.refresh_button)
        form.addRow("STM32 USB 端口", port_row)

        self.ssid = QtWidgets.QLineEdit()
        self.wifi_password = QtWidgets.QLineEdit()
        self.wifi_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.host = QtWidgets.QLineEdit()
        self.host.setPlaceholderText("例如：你的 MQTT 服务域名")
        self.broker_port = QtWidgets.QSpinBox()
        self.broker_port.setRange(1, 65535)
        self.broker_port.setValue(8883)
        self.tls = QtWidgets.QCheckBox("启用 TLS（正式使用建议开启）")
        self.tls.setToolTip(
            "正式使用必须开启；证书校验能力取决于 BW20 固件"
        )
        self.tls.setChecked(True)
        self.device_id = QtWidgets.QLineEdit("fbg-395736563033")
        self.client_id = QtWidgets.QLineEdit("fbg-395736563033")
        self.username = QtWidgets.QLineEdit()
        self.mqtt_password = QtWidgets.QLineEdit()
        self.mqtt_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.topic_prefix = QtWidgets.QLineEdit("fbg/v1")

        form.addRow("Wi-Fi 名称", self.ssid)
        form.addRow("Wi-Fi 密码", self.wifi_password)
        form.addRow("MQTT 服务器", self.host)
        form.addRow("MQTT 端口", self.broker_port)
        form.addRow("连接安全", self.tls)

        identity_group = QtWidgets.QGroupBox("设备身份与主题")
        identity_form = QtWidgets.QFormLayout(identity_group)
        configure_form_layout(identity_form)
        identity_form.setHorizontalSpacing(18)
        identity_form.setVerticalSpacing(12)
        identity_form.addRow("设备编号", self.device_id)
        identity_form.addRow("MQTT 客户端编号", self.client_id)
        identity_form.addRow("MQTT 用户名", self.username)
        identity_form.addRow("MQTT 密码", self.mqtt_password)
        identity_form.addRow("主题前缀", self.topic_prefix)

        self.send_button = QtWidgets.QPushButton("写入板卡并连接")
        self.send_button.setMinimumHeight(42)
        self.send_button.setProperty("role", "primary")
        self.status = QtWidgets.QLabel(
            "密码只通过 USB 写入板卡，本工具不会把密码保存到电脑文件中。"
        )
        self.status.setObjectName("softHint")
        self.status.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(connection_group)
        layout.addWidget(identity_group)
        layout.addWidget(self.send_button)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.refresh_button.clicked.connect(self.refresh_ports)
        self.send_button.clicked.connect(self.start_provisioning)
        self.refresh_ports()

    def refresh_ports(self) -> None:
        current = self.port.currentData()
        self.port.clear()
        ports = list(list_ports.comports())
        native_port = None
        for item in ports:
            label = f"{item.device} — {item.description}"
            self.port.addItem(label, item.device)
            if (
                getattr(item, "vid", None),
                getattr(item, "pid", None),
            ) == STM32_CDC_VID_PID and native_port is None:
                native_port = item.device
        if current:
            index = self.port.findData(current)
            if index >= 0:
                self.port.setCurrentIndex(index)
                return
        if native_port is not None:
            self.port.setCurrentIndex(self.port.findData(native_port))
        else:
            # Do not silently send credentials to a programmer/debug UART or
            # some unrelated USB serial adapter.  The operator may still make
            # an explicit selection from the visible list.
            self.port.setCurrentIndex(-1)

    def _settings(self) -> ProvisioningSettings:
        # Keep the operator-supplied identity byte-for-byte.  Trimming here
        # would silently turn a legacy/invalid identity into a different valid
        # one, which can redirect MQTT topics to the wrong device.
        device_id = self.device_id.text()
        client_id = self.client_id.text().strip() or device_id
        return ProvisioningSettings(
            ssid=self.ssid.text(),
            wifi_password=self.wifi_password.text(),
            broker_host=self.host.text().strip(),
            broker_port=self.broker_port.value(),
            client_id=client_id,
            mqtt_username=self.username.text(),
            mqtt_password=self.mqtt_password.text(),
            topic_prefix=self.topic_prefix.text().strip().strip("/"),
            device_id=device_id,
            tls=self.tls.isChecked(),
        )

    def start_provisioning(self) -> None:
        port = self.port.currentData()
        if not port:
            QtWidgets.QMessageBox.warning(self, "没有端口", "没有找到可用串口。")
            return
        try:
            frame = build_provisioning_frame(self._settings())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "配置不完整", str(exc))
            return

        # Validate all fields before pausing another active source.  This hook
        # still runs immediately before USB ownership changes.
        if callable(self.before_provisioning):
            try:
                if not bool(self.before_provisioning()):
                    return
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "无法进入配置模式", str(exc)
                )
                return

        self.send_button.setEnabled(False)
        self.status.setText("正在写入配置并等待单片机确认……")
        self._thread = QtCore.QThread(self)
        self._worker = ProvisionWorker(str(port), frame)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._provisioning_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @QtCore.pyqtSlot(bool, str)
    def _provisioning_finished(self, success: bool, message: str) -> None:
        self.send_button.setEnabled(True)
        self.status.setText(message)
        box = QtWidgets.QMessageBox.information if success else QtWidgets.QMessageBox.warning
        box(self, "配置完成" if success else "配置未完成", message)
        self._thread = None
        self._worker = None

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "正在写入",
                "板卡配置仍在写入，请等待本次操作完成后再关闭窗口。",
            )
            event.ignore()
            return
        super().closeEvent(event)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--port", help="preselect a serial port")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = ProvisionWindow()
    if args.port:
        index = window.port.findData(args.port)
        if index >= 0:
            window.port.setCurrentIndex(index)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
