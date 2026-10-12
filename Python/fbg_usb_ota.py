"""Physical STM32 CDC firmware update; no radio, broker or debugger required.

USB identity is checked both against Windows descriptors and the MCU UID.
USB:<serial> is a public identity binding, not an authorization secret. The
firmware only accepts this transport from its physical CDC receive callback.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Callable

import serial
from serial.tools import list_ports

from fbg_ota_protocol import (
    FirmwarePackage, OtaCancelled, OtaError, OtaProgress, OTA_CHUNK_SIZE,
    OTA_MAX_IMAGE_SIZE, OTA_TARGET_NAME, crc32, load_package,
)

USB_FRAME_SIZE = 808
USB_HEADER = b"\xff\xff\x02\x22"
USB_COMMAND_LIMIT = 500
USB_MIN_FIRMWARE_VERSION = 0x00010069  # v1.0.105 first implements FUSB1.


def build_usb_update_frame(command: str) -> bytes:
    payload = command.encode("ascii")
    if not 0 < len(payload) < USB_COMMAND_LIMIT or any(
        marker in payload for marker in (b"\0", b"\r", b"\n")
    ):
        raise ValueError("USB 更新命令长度或格式无效")
    frame = bytearray(USB_FRAME_SIZE)
    frame[:4] = USB_HEADER
    frame[4] = 1
    frame[6:8] = len(payload).to_bytes(2, "big")
    frame[8:8 + len(payload)] = payload
    frame[804:] = crc32(frame[:804]).to_bytes(4, "big")
    return bytes(frame)


@dataclass(frozen=True)
class UsbFirmwareInfo:
    serial_number: str
    firmware_version: int
    boot_id: int
    max_image_size: int
    bootloader_ready: bool

    @property
    def version_text(self) -> str:
        v = self.firmware_version
        return f"{v >> 16}.{(v >> 8) & 255}.{v & 255}"


def _hex32(value: str) -> int:
    if len(value) != 8 or any(c not in "0123456789ABCDEF" for c in value):
        raise OtaError("USB 更新回执中的数值格式无效")
    return int(value, 16)


class UsbOtaClient:
    def __init__(self, port: str, serial_number: str, *, timeout_s: float = 8,
                 reboot_timeout_s: float = 90, retries: int = 3,
                 serial_factory=serial.Serial, ports_provider=list_ports.comports):
        if len(serial_number) != 12 or any(
            c not in "0123456789ABCDEF" for c in serial_number
        ):
            raise ValueError("必须指定 STM32 的完整 USB 序列号")
        self.port, self.serial_number = port, serial_number
        self.timeout_s = timeout_s
        self.reboot_timeout_s = reboot_timeout_s
        self.retries = max(1, retries)
        self.serial_factory, self.ports_provider = serial_factory, ports_provider
        self.device = None
        self.buffer = bytearray()
        self.commit_started = False
        self.reboot_acknowledged = False

    def _open(self, *, reconnect: bool = False):
        ports = [p for p in self.ports_provider()
                 if p.vid == 0x0483 and p.pid == 0x5740
                 and p.serial_number == self.serial_number
                 and (reconnect or p.device.upper() == self.port.upper())]
        if len(ports) != 1:
            raise OtaError("USB 端口/设备身份不匹配，未向其他设备发送命令")
        self.port = ports[0].device
        self.device = self.serial_factory(self.port, 115200, timeout=.05, write_timeout=3)
        self.device.dtr = True
        self.device.reset_input_buffer()
        self.buffer.clear()

    def __enter__(self):
        self._open()
        return self

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None

    def __exit__(self, *_):
        self.close()

    def _reconnect(self):
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self._open(reconnect=True)
                return
            except (OtaError, serial.SerialException, OSError):
                self.close()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.25)

    def _exchange(self, command: str) -> list[str]:
        if self.device is None:
            raise OtaError("USB 设备尚未打开")
        frame = build_usb_update_frame(command)
        if self.device.write(frame) != len(frame):
            raise OtaError("USB 更新帧未完整发送")
        self.device.flush()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            # Wait for the first byte, then drain only available bytes. Asking
            # CDC for 4096 bytes would add the full serial timeout to every ACK.
            self.buffer.extend(self.device.read(min(max(1, self.device.in_waiting), 4096)))
            while b"\n" in self.buffer:
                end = self.buffer.index(b"\n")
                raw = bytes(self.buffer[:end])
                del self.buffer[:end + 1]
                starts = [i for prefix in (b"FOTA1|", b"FUSB1|")
                          if (i := raw.find(prefix)) >= 0]
                if not starts:
                    continue
                try:
                    fields = raw[min(starts):].decode("ascii").strip().split("|")
                except UnicodeDecodeError:
                    continue
                if len(fields) > 1 and fields[0] == "FOTA1" and fields[1] in ("ERROR", "ERR"):
                    raise OtaError("板卡拒绝 USB 更新：" + "|".join(fields[2:]))
                return fields
            if len(self.buffer) > 16384:
                del self.buffer[:-512]
        raise TimeoutError("等待板卡 USB 固件更新回执超时")

    def query_info(self) -> UsbFirmwareInfo:
        try:
            reply = self._exchange("FUSB1|INFO")
        except TimeoutError as exc:
            raise OtaError(
                "未收到 USB 更新协议回执。旧固件首次启用需通过 CMSIS "
                "或已有无线更新安装 v1.0.105 及以上支持版本。"
            ) from exc
        if len(reply) != 7 or reply[:2] != ["FUSB1", "INFO"]:
            raise OtaError("USB 设备信息回执无效")
        if reply[2] != self.serial_number:
            raise OtaError("MCU UID 与 USB 序列号不一致，拒绝刷写")
        result = UsbFirmwareInfo(reply[2], *(_hex32(x) for x in reply[3:6]),
                                 bool(_hex32(reply[6]) & 1))
        if not 0 < result.max_image_size <= OTA_MAX_IMAGE_SIZE:
            raise OtaError("设备报告的应用区边界无效")
        return result

    def install(self, package: FirmwarePackage, *,
                progress: Callable[[OtaProgress], None] | None = None,
                cancelled: Callable[[], bool] = lambda: False) -> UsbFirmwareInfo:
        # Re-read the on-disk package immediately before any write operation.
        fresh = load_package(package.path)
        if fresh != package:
            raise OtaError("固件包在选择后发生变化，拒绝刷写")
        if package.firmware_version < USB_MIN_FIRMWARE_VERSION:
            raise OtaError("此旧版固件不支持 USB 更新与启动核验，请选择 v1.0.105 或更新的固件包")
        before = self.query_info()
        if not before.bootloader_ready or package.image_size > before.max_image_size:
            raise OtaError("设备未安装兼容更新引导程序，需先通过 CMSIS 初始化")
        nonce = secrets.token_hex(8).upper()
        prefix = (f"FOTA1|BEGIN|{OTA_TARGET_NAME}|{package.firmware_version:08X}|"
                  f"{package.image_size:08X}|{package.image_crc32:08X}|{nonce}")
        mac = hmac.new(f"USB:{self.serial_number}".encode(), prefix.encode(),
                       hashlib.sha256).hexdigest().upper()
        sent = 0

        def notify(message: str):
            if progress:
                progress(OtaProgress(sent, package.image_size, message))

        def check_cancelled():
            if not self.commit_started and cancelled():
                raise OtaCancelled("USB 升级已取消")

        try:
            for attempt in range(self.retries):
                try:
                    check_cancelled()
                    reply = self._exchange(prefix + "|" + mac)
                    if len(reply) != 5 or reply[:2] != ["FOTA1", "READY"]:
                        raise OtaError("USB BEGIN 回执无效")
                    session, sent, boot = map(_hex32, reply[2:])
                    if boot != before.boot_id or not 0 <= sent <= package.image_size:
                        raise OtaError("USB 续传身份或位置无效")
                    notify("已确认设备身份，正在通过 USB 写入暂存区")
                    while sent < package.image_size:
                        check_cancelled()
                        block = package.image[sent:sent + OTA_CHUNK_SIZE]
                        line = (f"FOTA1|DATA|{session:08X}|{sent:08X}|{crc32(block):08X}|"
                                + base64.b64encode(block).decode())
                        reply = self._exchange(line)
                        if (len(reply) != 4 or reply[:2] != ["FOTA1", "ACK"]
                                or _hex32(reply[2]) != session
                                or _hex32(reply[3]) != sent + len(block)):
                            raise OtaError("USB 分块回执或写入位置不匹配")
                        sent += len(block)
                        notify(f"USB 已传输 {sent / 1024:.1f} / {package.image_size / 1024:.1f} KB")
                    check_cancelled()
                    self.commit_started = True
                    notify("整包校验和重启中，不能取消或断电")
                    try:
                        reply = self._exchange(
                            f"FOTA1|END|{session:08X}|{sent:08X}|{package.image_crc32:08X}")
                    except (TimeoutError, serial.SerialException, OSError):
                        # The reset may race Windows delivering the final ACK.
                        # Verify the new runtime; never issue another BEGIN.
                        break
                    if (len(reply) != 3 or reply[:2] != ["FOTA1", "REBOOT"]
                            or _hex32(reply[2]) != package.firmware_version):
                        raise OtaError("USB 重启版本回执不匹配")
                    self.reboot_acknowledged = True
                    break
                except (TimeoutError, serial.SerialException, OSError):
                    if attempt + 1 >= self.retries:
                        raise
                    notify("USB 回执中断，尝试从板卡确认的位置续传")
                    self.close()
                    self._reconnect()
        except OtaCancelled:
            if self._exchange("FUSB1|ABORT") != ["FUSB1", "ABORTED"]:
                raise OtaError("取消请求未确认，请保持供电并检查状态")
            raise OtaCancelled("已确认取消 USB 升级，原固件未修改，保持关光")
        except Exception:
            # Do not leave a failed, uncommitted USB transfer owning the board
            # forever. After END has been sent its outcome may be ambiguous,
            # so never abort/erase there; only verify the runtime or report it.
            if not self.commit_started:
                try:
                    if self.device is None:
                        self._reconnect()
                    if self._exchange("FUSB1|ABORT") == ["FUSB1", "ABORTED"]:
                        notify("已释放未提交的 USB 更新任务，原固件未修改，保持关光")
                except (OtaError, TimeoutError, serial.SerialException, OSError):
                    notify("未确认释放 USB 更新状态；请保持供电并重新尝试更新")
            raise
        self.close()
        deadline = time.monotonic() + self.reboot_timeout_s
        notify("正在等待 USB 重新枚举并核验新版本")
        while time.monotonic() < deadline:
            try:
                self._open(reconnect=True)
                after = self.query_info()
                if (after.firmware_version == package.firmware_version
                        and after.boot_id != before.boot_id):
                    notify("USB 升级完成：新版本与新启动标识均已确认")
                    return after
            except (OtaError, TimeoutError, serial.SerialException, OSError):
                pass
            finally:
                self.close()
            time.sleep(1)
        raise OtaError("USB 更新已进入提交阶段，但未确认新版本启动。请保持供电并检查状态。")
