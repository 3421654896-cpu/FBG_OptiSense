"""Secure same-LAN firmware package and OTA transfer helpers.

The wire protocol intentionally stays ASCII because the BW20 AT firmware
delivers socket payloads through a line-oriented UART indication.  Firmware
bytes are base64 encoded and every block is acknowledged before the next one
is sent, which also gives deterministic backpressure while STM32 flash is
being programmed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import os
import secrets
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


OTA_PORT = 45670
OTA_TARGET_NAME = "F205RE"
OTA_TARGET_ID = 0xF2050052
OTA_LINK_ADDRESS = 0x08008000
OTA_APPLICATION_END = 0x08040000
OTA_RAM_START = 0x20000000
OTA_RAM_END = 0x20020000
OTA_CHUNK_SIZE = 180
OTA_MAX_IMAGE_SIZE = OTA_APPLICATION_END - OTA_LINK_ADDRESS
OTA_RAW_HELLO = b"FBGL1|RAW"
OTA_RAW_RESPONSE_HEADER = struct.Struct(">4sBBIH")
OTA_RAW_RESPONSE_MAGIC = b"FBR1"
OTA_RAW_RESPONSE_VERSION = 1
OTA_RAW_READY_KIND = 2
OTA_RAW_MAX_PAYLOAD = 2048

PACKAGE_MAGIC = b"FBGFW1\0\0"
PACKAGE_FORMAT_VERSION = 1
PACKAGE_HEADER = struct.Struct(">8sHHIIIIQ20sII")
PACKAGE_HEADER_SIZE = PACKAGE_HEADER.size


class OtaError(RuntimeError):
    """An OTA package or transfer did not pass a required safety check."""


class OtaCancelled(OtaError):
    """Raised when the operator cancels an OTA transfer."""


@dataclass(frozen=True)
class FirmwarePackage:
    path: Path
    target_id: int
    link_address: int
    firmware_version: int
    build_timestamp: int
    image_crc32: int
    image: bytes

    @property
    def image_size(self) -> int:
        return len(self.image)

    @property
    def version_text(self) -> str:
        major = (self.firmware_version >> 16) & 0xFFFF
        minor = (self.firmware_version >> 8) & 0xFF
        patch = self.firmware_version & 0xFF
        return f"{major}.{minor}.{patch}"


@dataclass(frozen=True)
class OtaProgress:
    sent: int
    total: int
    message: str


def crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def encode_version(major: int, minor: int = 0, patch: int = 0) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (major, minor, patch)
    ):
        raise ValueError("版本号必须是整数")
    if not 0 <= major <= 0xFFFF or not 0 <= minor <= 0xFF or not 0 <= patch <= 0xFF:
        raise ValueError("版本号超出范围")
    return (major << 16) | (minor << 8) | patch


def parse_version(value: str) -> int:
    parts = [part.strip() for part in str(value).split(".")]
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise ValueError("版本号应为 1、1.2 或 1.2.3")
    numbers = [int(part) for part in parts]
    numbers += [0] * (3 - len(numbers))
    return encode_version(*numbers)


def validate_application_image(image: bytes) -> tuple[int, int]:
    if len(image) < 8:
        raise OtaError("固件太小，缺少 STM32 向量表")
    if len(image) > OTA_MAX_IMAGE_SIZE:
        raise OtaError("固件超过 224 KB 应用区上限")
    initial_sp, reset_vector = struct.unpack_from("<II", image, 0)
    if not OTA_RAM_START <= initial_sp <= OTA_RAM_END or initial_sp & 3:
        raise OtaError(f"无效的初始栈指针 0x{initial_sp:08X}")
    reset_address = reset_vector & ~1
    if not OTA_LINK_ADDRESS <= reset_address < OTA_APPLICATION_END:
        raise OtaError(
            f"复位向量 0x{reset_vector:08X} 不属于 0x{OTA_LINK_ADDRESS:08X} 应用区"
        )
    if reset_address >= OTA_LINK_ADDRESS + len(image):
        raise OtaError("复位向量指向固件映像文件之外")
    if not reset_vector & 1:
        raise OtaError("复位向量缺少 Thumb 位")
    return initial_sp, reset_vector


def build_package(
    image_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    firmware_version: int,
    *,
    build_timestamp: Optional[int] = None,
) -> FirmwarePackage:
    image_source = Path(image_path)
    output = Path(output_path)
    if image_source.resolve() == output.resolve():
        raise ValueError("固件包输出路径不能覆盖原始 BIN 文件")
    if image_source.stat().st_size > OTA_MAX_IMAGE_SIZE:
        raise OtaError("固件超过 224 KB 应用区上限")
    image = image_source.read_bytes()
    validate_application_image(image)
    if isinstance(firmware_version, bool) or not isinstance(firmware_version, int):
        raise ValueError("固件版本必须是 uint32 整数")
    if not 0 <= firmware_version <= 0xFFFFFFFF:
        raise ValueError("固件版本必须是 uint32 整数")
    timestamp = int(time.time()) if build_timestamp is None else build_timestamp
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError("构建时间戳必须是 uint64 整数")
    if not 0 <= timestamp <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("构建时间戳必须是 uint64 整数")
    image_crc = crc32(image)
    fields = (
        PACKAGE_MAGIC,
        PACKAGE_FORMAT_VERSION,
        PACKAGE_HEADER_SIZE,
        OTA_TARGET_ID,
        OTA_LINK_ADDRESS,
        firmware_version,
        len(image),
        timestamp,
        bytes(20),
        image_crc,
        0,
    )
    provisional = PACKAGE_HEADER.pack(*fields)
    header_crc = crc32(provisional[:-4])
    header = PACKAGE_HEADER.pack(*fields[:-1], header_crc)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + image)
    return load_package(output)


def load_package(path: str | os.PathLike[str]) -> FirmwarePackage:
    package_path = Path(path)
    if package_path.stat().st_size > PACKAGE_HEADER_SIZE + OTA_MAX_IMAGE_SIZE:
        raise OtaError("固件包超过应用区最大尺寸")
    raw = package_path.read_bytes()
    if len(raw) < PACKAGE_HEADER_SIZE:
        raise OtaError("不是完整的 FBG 固件包")
    values = PACKAGE_HEADER.unpack_from(raw)
    (
        magic,
        format_version,
        header_size,
        target_id,
        link_address,
        firmware_version,
        image_size,
        build_timestamp,
        _reserved,
        image_crc,
        header_crc,
    ) = values
    if magic != PACKAGE_MAGIC or format_version != PACKAGE_FORMAT_VERSION:
        raise OtaError("固件包格式或版本不受支持")
    if header_size != PACKAGE_HEADER_SIZE:
        raise OtaError("固件包头长度不正确")
    if crc32(raw[: PACKAGE_HEADER_SIZE - 4]) != header_crc:
        raise OtaError("固件包头 CRC 校验失败")
    if target_id != OTA_TARGET_ID or link_address != OTA_LINK_ADDRESS:
        raise OtaError("固件包不适用于当前 STM32F205RE 板卡")
    if _reserved != bytes(len(_reserved)):
        raise OtaError("固件包包含当前版本不支持的保留字段")
    image = raw[PACKAGE_HEADER_SIZE:]
    if len(image) != image_size:
        raise OtaError("固件包长度与包头不一致")
    if crc32(image) != image_crc:
        raise OtaError("固件映像 CRC 校验失败")
    validate_application_image(image)
    return FirmwarePackage(
        path=package_path.resolve(),
        target_id=target_id,
        link_address=link_address,
        firmware_version=firmware_version,
        build_timestamp=build_timestamp,
        image_crc32=image_crc,
        image=image,
    )


def _local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 53))
            addresses.append(str(probe.getsockname()[0]))
        finally:
            probe.close()
    except OSError:
        pass
    try:
        addresses.extend(
            str(item[4][0])
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        )
    except OSError:
        pass
    return list(
        dict.fromkeys(
            address
            for address in addresses
            if not address.startswith("127.") and not address.startswith("169.254.")
        )
    )


def candidate_hosts() -> list[str]:
    candidates: list[str] = []
    for local in _local_ipv4_addresses():
        try:
            network = ipaddress.ip_network(f"{local}/24", strict=False)
        except ValueError:
            continue
        candidates.extend(str(host) for host in network.hosts() if str(host) != local)
    return list(dict.fromkeys(candidates))


def discover_board(
    *, port: int = OTA_PORT, timeout_s: float = 0.15
) -> Optional[str]:
    def try_host(host: str) -> Optional[str]:
        try:
            with socket.create_connection((host, port), timeout=timeout_s) as sock:
                sock.sendall(b"FBGL1|HELLO")
                return host
        except OSError:
            return None

    hosts = candidate_hosts()
    if not hosts:
        return None
    executor = ThreadPoolExecutor(max_workers=min(48, len(hosts)))
    futures = {executor.submit(try_host, host): host for host in hosts}
    found = None
    try:
        for future in as_completed(futures):
            result = future.result()
            if result:
                found = result
                break
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    return found


class OtaClient:
    """Synchronous uploader intended to run in a worker thread."""

    def __init__(
        self,
        host: str,
        password: str,
        *,
        port: int = OTA_PORT,
        connect_timeout_s: float = 3.0,
        response_timeout_s: float = 20.0,
        retries: int = 12,
        reconnect_delay_s: float = 30.0,
        recovery_quiet_s: float = 150.0,
        reboot_timeout_s: float = 600.0,
        reboot_initial_delay_s: float = 5.0,
        reboot_poll_s: float = 1.0,
        reboot_disconnect_timeout_s: float = 8.0,
        retry_owner_timeout_s: float = 12.0,
        chunk_delay_s: float = 0.0,
        adaptive_chunk_pacing: bool = False,
        chunk_delay_max_s: float = 0.45,
        chunk_delay_backoff: float = 1.5,
        chunk_delay_stable_blocks: int = 384,
        raw_connection_hold_s: float = 6.0,
        raw_hello_interval_s: float = 0.5,
        raw_release_cooldown_s: float = 6.0,
        post_reboot_recovery: Optional[Callable[[], Optional[str]]] = None,
        post_reboot_recovery_interval_s: float = 10.0,
    ) -> None:
        self.host = str(ipaddress.ip_address(host.strip()))
        self.port = int(port)
        self.password = str(password)
        self.connect_timeout_s = float(connect_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self.retries = max(1, int(retries))
        self.reconnect_delay_s = max(0.0, float(reconnect_delay_s))
        self.recovery_quiet_s = max(0.0, float(recovery_quiet_s))
        self.reboot_timeout_s = max(0.1, float(reboot_timeout_s))
        self.reboot_initial_delay_s = min(
            max(0.0, float(reboot_initial_delay_s)),
            self.reboot_timeout_s * 0.25,
        )
        self.reboot_poll_s = max(0.05, float(reboot_poll_s))
        self.reboot_disconnect_timeout_s = max(
            0.1, float(reboot_disconnect_timeout_s)
        )
        self.retry_owner_timeout_s = max(0.1, float(retry_owner_timeout_s))
        self.chunk_delay_s = max(0.0, float(chunk_delay_s))
        self.adaptive_chunk_pacing = bool(adaptive_chunk_pacing)
        self.chunk_delay_max_s = max(
            self.chunk_delay_s, float(chunk_delay_max_s)
        )
        self.chunk_delay_backoff = max(1.01, float(chunk_delay_backoff))
        self.chunk_delay_stable_blocks = max(1, int(chunk_delay_stable_blocks))
        # One accepted BW20 seed must remain alive beyond the firmware's
        # 3.5-second stale-owner timeout.  Closing and recreating a socket on
        # every probe can continually lose the race to that old owner.
        self.raw_connection_hold_s = max(0.1, float(raw_connection_hold_s))
        self.raw_hello_interval_s = max(0.05, float(raw_hello_interval_s))
        # A successful RAW probe itself becomes the selected BW20 child.  Its
        # TCP close is asynchronous from the STM32's point of view: if the
        # disconnect URC is delayed/lost, recovery is the 3.5 s owner lease,
        # a 0.75 s stale-delete grace and finally AT+SOCKETDEL.  Do not let the
        # GUI/CLI create the real LAN client inside that cleanup window.
        self.raw_release_cooldown_s = max(0.0, float(raw_release_cooldown_s))
        self.post_reboot_recovery = post_reboot_recovery
        self.post_reboot_recovery_interval_s = max(
            1.0, float(post_reboot_recovery_interval_s)
        )
        self._buffer = bytearray()
        self.source_boot_id: Optional[int] = None
        self.final_boot_id: Optional[int] = None
        self.recovery_count = 0
        self.final_chunk_delay_s = self.chunk_delay_s

    @staticmethod
    def _hex_u32(value: str, field_name: str) -> int:
        text = str(value)
        if len(text) != 8 or any(ch not in "0123456789ABCDEF" for ch in text):
            raise OtaError(f"板卡返回的{field_name}不是8位十六进制数")
        return int(text, 16)

    def _connect(self, *, send_hello: bool = True) -> socket.socket:
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        )
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.response_timeout_s)
        if send_hello:
            sock.sendall(b"FBGL1|HELLO")
            time.sleep(0.08)
        return sock

    def _response(self, sock: socket.socket) -> list[str]:
        deadline = time.monotonic() + self.response_timeout_s
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                raise OtaError("板卡已断开 OTA 连接")
            self._buffer.extend(chunk)
            if len(self._buffer) > 4096 and b"\n" not in self._buffer:
                raise OtaError("板卡 OTA 回执超过单行长度上限")
            while b"\n" in self._buffer:
                line, _, remainder = self._buffer.partition(b"\n")
                self._buffer[:] = remainder
                marker = line.find(b"FOTA1|")
                if marker >= 0:
                    try:
                        text = line[marker:].decode("ascii", errors="strict").strip()
                    except UnicodeDecodeError as exc:
                        raise OtaError("板卡 OTA 回执不是 ASCII 文本") from exc
                    fields = text.split("|")
                    if len(fields) >= 2 and fields[1] == "ERROR":
                        detail = " | ".join(fields[2:]) or "未知错误"
                        raise OtaError(f"板卡拒绝升级：{detail}")
                    return fields
        raise OtaError("等待板卡 OTA 回执超时")

    @staticmethod
    def _send(sock: socket.socket, line: str) -> None:
        encoded = line.encode("ascii")
        if len(encoded) >= 500:
            raise OtaError("OTA 指令超过无线模块单行上限")
        sock.sendall(encoded)

    @staticmethod
    def _close(sock: Optional[socket.socket]) -> None:
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    @staticmethod
    def _version_text(version: int) -> str:
        return (
            f"{(version >> 16) & 0xFFFF}."
            f"{(version >> 8) & 0xFF}.{version & 0xFF}"
        )

    def _wait_for_raw_ready(
        self,
        sock: socket.socket,
        *,
        timeout_s: float,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> int:
        """Keep one socket alive while repeatedly requesting RAW ownership."""

        buffer = bytearray()
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        next_hello = 0.0
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                raise OtaCancelled("用户取消了 OTA 升级")
            now = time.monotonic()
            if now >= next_hello:
                sock.sendall(OTA_RAW_HELLO)
                next_hello = now + self.raw_hello_interval_s
            remaining = deadline - time.monotonic()
            until_hello = next_hello - time.monotonic()
            sock.settimeout(
                min(0.25, max(0.01, remaining), max(0.01, until_hello))
            )
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise OtaError("板卡在 RAW READY 前断开连接")
            buffer.extend(chunk)
            while buffer:
                start = buffer.find(OTA_RAW_RESPONSE_MAGIC)
                if start < 0:
                    if len(buffer) > 3:
                        del buffer[:-3]
                    break
                if start:
                    del buffer[:start]
                if len(buffer) < OTA_RAW_RESPONSE_HEADER.size:
                    break
                magic, version, kind, sequence, payload_length = (
                    OTA_RAW_RESPONSE_HEADER.unpack_from(buffer)
                )
                if (
                    magic != OTA_RAW_RESPONSE_MAGIC
                    or version != OTA_RAW_RESPONSE_VERSION
                    or payload_length > OTA_RAW_MAX_PAYLOAD
                ):
                    del buffer[0]
                    continue
                total = OTA_RAW_RESPONSE_HEADER.size + payload_length + 4
                if len(buffer) < total:
                    break
                packet = bytes(buffer[:total])
                del buffer[:total]
                expected_crc = int.from_bytes(packet[-4:], "big")
                if crc32(packet[:-4]) != expected_crc:
                    continue
                payload = packet[OTA_RAW_RESPONSE_HEADER.size : -4]
                if kind == OTA_RAW_READY_KIND and not payload:
                    return int(sequence)
        raise OtaError("全功能 LAN 未返回有效的 RAW READY")

    def query_raw_ready(
        self,
        *,
        timeout_s: Optional[float] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> int:
        """Validate RAW READY and leave BW20 time to reap the probe socket."""

        sock: Optional[socket.socket] = None
        ready = False
        boot_id = 0
        try:
            # Sending only RAW avoids a HELLO+RAW coalescing ambiguity in older
            # BW20 firmware, whose parser treats one modem indication as one
            # command payload.
            sock = self._connect(send_hello=False)
            boot_id = self._wait_for_raw_ready(
                sock,
                timeout_s=(
                    self.raw_connection_hold_s
                    if timeout_s is None
                    else max(0.05, float(timeout_s))
                ),
                cancelled=cancelled,
            )
            ready = True
        finally:
            # Use exactly the same full-close lifecycle as FbgLanSerial.  A
            # half-close followed by receive draining is not the lifecycle
            # used by the application and produced a different BW20 failure
            # mode during repeated acceptance checks.
            self._close(sock)
        if ready:
            self._cancelable_sleep(self.raw_release_cooldown_s, cancelled)
        return boot_id

    def wait_for_reboot_disconnect(
        self,
        sock: socket.socket,
        *,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Observe the accepted OTA socket disappear after the REBOOT reply."""

        deadline = time.monotonic() + self.reboot_disconnect_timeout_s
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                raise OtaCancelled("用户取消了 OTA 升级")
            remaining = deadline - time.monotonic()
            sock.settimeout(min(0.25, max(0.01, remaining)))
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
        raise OtaError(
            "板卡已确认重启，但旧 OTA 连接未按期断开；"
            "无法证明 MCU 已进入自动重启流程"
        )

    @staticmethod
    def _cancelable_sleep(
        seconds: float, cancelled: Optional[Callable[[], bool]]
    ) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            if cancelled is not None and cancelled():
                raise OtaCancelled("用户取消了 OTA 升级")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(0.1, remaining))

    def wait_for_raw_ready(
        self,
        *,
        progress: Optional[Callable[[str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        previous_boot_id: Optional[int] = None,
        require_boot_identity: bool = False,
    ) -> int:
        """Wait until the rebooted application restores full-function LAN."""

        if progress is not None:
            progress("旧连接已断开，等待板卡安装新固件并恢复全功能局域网")
        wait_started = time.monotonic()
        deadline = wait_started + self.reboot_timeout_s
        next_recovery = wait_started + self.post_reboot_recovery_interval_s
        last_error: Optional[Exception] = None
        last_recovery_message: Optional[str] = None
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                raise OtaCancelled("用户取消了 OTA 升级")
            try:
                # Retain this connection for several seconds and re-send RAW
                # in-place.  This lets the board's 3.5-second stale session
                # expire without throwing away the new accepted TCP seed.
                remaining = deadline - time.monotonic()
                boot_id = self.query_raw_ready(
                    timeout_s=min(self.raw_connection_hold_s, remaining),
                    cancelled=cancelled,
                )
                if require_boot_identity and boot_id == 0:
                    raise OtaError("新固件未返回 OTA 启动标识")
                if previous_boot_id is not None and boot_id == previous_boot_id:
                    raise OtaError("板卡仍返回升级前的启动标识")
                if progress is not None:
                    progress("板卡已自动重新联机，全功能 LAN RAW READY 已确认")
                return boot_id
            except OtaCancelled:
                raise
            except (OSError, OtaError) as exc:
                last_error = exc
            if (
                self.post_reboot_recovery is not None
                and time.monotonic() >= next_recovery
            ):
                next_recovery = (
                    time.monotonic() + self.post_reboot_recovery_interval_s
                )
                try:
                    recovery_message = self.post_reboot_recovery()
                except Exception as exc:
                    recovery_message = f"调试器恢复检查未执行：{exc}"
                if (
                    recovery_message
                    and recovery_message != last_recovery_message
                    and progress is not None
                ):
                    progress(recovery_message)
                    last_recovery_message = recovery_message
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                self._cancelable_sleep(min(self.reboot_poll_s, remaining), cancelled)
        detail = str(last_error or "板卡未重新出现")
        raise OtaError(
            "已收到板卡重启确认，但未能在 "
            f"{self.reboot_timeout_s:.0f} 秒内恢复全功能 LAN：{detail}"
        )

    def upload(
        self,
        package: FirmwarePackage,
        progress: Optional[Callable[[OtaProgress], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        verify_reboot: bool = True,
    ) -> None:
        if not self.password:
            raise OtaError("请输入 OTA 授权密码（默认与板卡 Wi-Fi 密码相同）")
        nonce = secrets.token_hex(8).upper()
        image = package.image
        sent = 0

        def notify(message: str) -> None:
            if progress is not None:
                progress(OtaProgress(sent, len(image), message))

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise OtaCancelled("用户取消了 OTA 升级")

        check_cancelled()
        notify("正在连接板卡")
        last_error: Optional[Exception] = None
        reboot_accepted = False
        reboot_already_confirmed = False
        reboot_socket: Optional[socket.socket] = None
        source_boot_id: Optional[int] = None
        self.source_boot_id = None
        self.final_boot_id = None
        self.recovery_count = 0
        current_chunk_delay = self.chunk_delay_s
        stable_blocks = 0
        for attempt in range(self.retries):
            sock: Optional[socket.socket] = None
            try:
                check_cancelled()
                if attempt == 0:
                    sock = self._connect()
                else:
                    # A freshly reconnected BW20 seed can remain only a
                    # candidate while the disconnected OTA owner completes its
                    # 3.5 s lease and explicit delete.  Repeating RAW on this
                    # same socket yields an unambiguous READY only after the
                    # candidate has actually been promoted; BEGIN is never
                    # sent into the candidate-only window.
                    sock = self._connect(send_hello=False)
                    self._buffer.clear()
                    retry_boot_id = self._wait_for_raw_ready(
                        sock,
                        timeout_s=self.retry_owner_timeout_s,
                        cancelled=cancelled,
                    )
                    if (
                        source_boot_id is not None
                        and retry_boot_id != source_boot_id
                    ):
                        # END and metadata commit may succeed even when BW20
                        # loses the signed REBOOT reply.  The firmware's RAW
                        # token is the committed OTA session id, so it changes
                        # only after the bootloader installs this exact image.
                        # Do not send another BEGIN into the new application:
                        # that would erase staging and start a needless second
                        # upgrade after an already successful install.
                        self.final_boot_id = retry_boot_id
                        notify(
                            "REBOOT 回执虽丢失，但新启动标识已确认升级安装完成"
                        )
                        self._close(sock)
                        sock = None
                        reboot_accepted = True
                        reboot_already_confirmed = True
                        break
                    # RAW READY proves that this exact seed is now the owner,
                    # but it also places the firmware transport in binary RAW
                    # mode.  Switch the same connection back to conventional
                    # text mode before BEGIN so FOTA READY/ACK replies are not
                    # hidden behind the RAW-only transmit selector.
                    sock.sendall(b"FBGL1|HELLO")
                    time.sleep(0.08)
                self._buffer.clear()
                begin_prefix = (
                    f"FOTA1|BEGIN|{OTA_TARGET_NAME}|"
                    f"{package.firmware_version:08X}|{package.image_size:08X}|"
                    f"{package.image_crc32:08X}|{nonce}"
                )
                signature = hmac.new(
                    self.password.encode("utf-8"),
                    begin_prefix.encode("ascii"),
                    hashlib.sha256,
                ).hexdigest().upper()
                self._send(sock, f"{begin_prefix}|{signature}")
                response = self._response(sock)
                # Combo-AT can finish publishing the final ACK from the old
                # TCP seed only after the replacement seed has been promoted.
                # That delayed ACK belongs to the already persisted DATA
                # block, not to this BEGIN. On a recovery connection only,
                # drain a bounded number of stale ACK lines and wait for the
                # authenticated READY carrying the authoritative resume
                # offset. An ACK on the first connection remains an error.
                stale_ack_count = 0
                while (
                    attempt > 0
                    and len(response) == 4
                    and response[1] == "ACK"
                    and stale_ack_count < 8
                ):
                    stale_ack_count += 1
                    if stale_ack_count == 1:
                        notify("重连收到上一连接迟到 ACK，继续等待 READY")
                    response = self._response(sock)
                if attempt > 0 and len(response) == 3 and response[1] == "REBOOT":
                    accepted_version = self._hex_u32(
                        response[2], "迟到的待重启固件版本"
                    )
                    if accepted_version != package.firmware_version:
                        raise OtaError(
                            "板卡迟到的重启版本与升级包不一致："
                            f"{self._version_text(accepted_version)} != "
                            f"{package.version_text}"
                        )
                    notify("重连收到上一连接迟到 REBOOT，已确认整包提交")
                    reboot_accepted = True
                    reboot_socket = sock
                    sock = None
                    break
                if len(response) not in (4, 5) or response[1] != "READY":
                    response_preview = "|".join(response)
                    if len(response_preview) > 160:
                        response_preview = response_preview[:157] + "..."
                    raise OtaError(
                        "板卡返回了无效的 READY 回执："
                        f"{response_preview!r}"
                    )
                session = self._hex_u32(response[2], "会话编号")
                sent = self._hex_u32(response[3], "续传位置")
                source_boot_id = (
                    self._hex_u32(response[4], "升级前启动标识")
                    if len(response) == 5
                    else None
                )
                self.source_boot_id = source_boot_id
                if sent < 0 or sent > len(image):
                    raise OtaError("板卡返回了无效的续传位置")
                notify("已通过鉴权，正在写入 OTA 暂存区")

                # On BW20 recovery the READY response proves ownership but the
                # module may still be draining that text response through its
                # UART/TCP bridge.  Sending the first resumed DATA immediately
                # is disproportionately likely to lose its ACK.  Apply the
                # current pacing once before the first resumed block as well as
                # between subsequent blocks.
                if attempt > 0 and sent < len(image) and current_chunk_delay:
                    self._cancelable_sleep(current_chunk_delay, cancelled)

                while sent < len(image):
                    check_cancelled()
                    block = image[sent : sent + OTA_CHUNK_SIZE]
                    encoded = base64.b64encode(block).decode("ascii")
                    line = (
                        f"FOTA1|DATA|{session:08X}|{sent:08X}|"
                        f"{crc32(block):08X}|{encoded}"
                    )
                    self._send(sock, line)
                    response = self._response(sock)
                    if len(response) != 4 or response[1] != "ACK":
                        raise OtaError("板卡返回了无效的分块回执")
                    if self._hex_u32(response[2], "会话编号") != session:
                        raise OtaError("OTA 会话编号不一致")
                    next_offset = self._hex_u32(response[3], "分块写入位置")
                    if next_offset != sent + len(block):
                        raise OtaError("OTA 分块写入位置不一致")
                    sent = next_offset
                    stable_blocks += 1
                    if (
                        self.adaptive_chunk_pacing
                        and current_chunk_delay > self.chunk_delay_s
                        and stable_blocks >= self.chunk_delay_stable_blocks
                    ):
                        current_chunk_delay = max(
                            self.chunk_delay_s,
                            current_chunk_delay / self.chunk_delay_backoff,
                        )
                        stable_blocks = 0
                        notify(
                            "连接已连续稳定，分块间隔缓降至 "
                            f"{current_chunk_delay * 1000:.0f} ms"
                        )
                    notify(f"已写入 {sent / 1024:.1f} / {len(image) / 1024:.1f} KB")
                    if current_chunk_delay:
                        self._cancelable_sleep(current_chunk_delay, cancelled)

                check_cancelled()
                self._send(
                    sock,
                    f"FOTA1|END|{session:08X}|{len(image):08X}|"
                    f"{package.image_crc32:08X}",
                )
                response = self._response(sock)
                if len(response) != 3 or response[1] != "REBOOT":
                    raise OtaError("板卡未确认重启升级")
                accepted_version = self._hex_u32(response[2], "待重启固件版本")
                if accepted_version != package.firmware_version:
                    raise OtaError(
                        "板卡确认的待重启版本与升级包不一致："
                        f"{self._version_text(accepted_version)} != "
                        f"{package.version_text}"
                    )
                notify("整包校验通过，板卡正在重启并安装新固件")
                reboot_accepted = True
                reboot_socket = sock
                sock = None
                break
            except OtaCancelled:
                raise
            except (OSError, OtaError) as exc:
                last_error = exc
                self.recovery_count += 1
                if (
                    self.adaptive_chunk_pacing
                    and source_boot_id is not None
                    and sent < len(image)
                ):
                    stable_blocks = 0
                    increased_delay = min(
                        self.chunk_delay_max_s,
                        max(
                            self.chunk_delay_s,
                            current_chunk_delay * self.chunk_delay_backoff,
                        ),
                    )
                    if increased_delay > current_chunk_delay:
                        current_chunk_delay = increased_delay
                # Release the failed BW20 seed before the backoff.  Sleeping
                # while the old socket remains open gives the module no cleanup
                # window and the immediate reconnect can inherit a stale ConID.
                if sock is not None:
                    self._close(sock)
                    sock = None
                if attempt + 1 >= self.retries:
                    break
                pacing = (
                    f"；分块间隔自适应为 {current_chunk_delay * 1000:.0f} ms"
                    if self.adaptive_chunk_pacing and sent < len(image)
                    else ""
                )
                notify(
                    f"连接中断，正在第 {attempt + 2} 次恢复续传{pacing}："
                    f"{type(exc).__name__}: {exc}"
                )
                # Once BEGIN has been accepted, a broken BW20 TCPServer needs
                # a genuinely quiet interval to delete/verify/recreate itself.
                # Probing during that recovery manufactures another inherited
                # child and can keep the module in an endless cleanup loop.
                recovery_delay = (
                    self.recovery_quiet_s
                    if source_boot_id is not None
                    else self.reconnect_delay_s
                )
                self._cancelable_sleep(recovery_delay, cancelled)
            finally:
                if sock is not None:
                    self._close(sock)
        self.final_chunk_delay_s = current_chunk_delay
        if not reboot_accepted:
            raise OtaError(str(last_error or "OTA 升级失败"))
        if verify_reboot:
            if reboot_already_confirmed:
                return
            if reboot_socket is None:
                raise OtaError("内部错误：缺少待释放的 OTA 连接")
            # Firmware 1.0.65+ drains the signed REBOOT reply and asks Combo-AT
            # to delete the exact accepted TCP seed before resetting only the
            # STM32.  Close the old socket locally after the receipt, then
            # require the new application's RAW READY boot token.  The token
            # is committed with the OTA metadata and changes only after the
            # bootloader installs that image.
            notify("已核对重启版本，正在释放旧连接并等待新固件启动")
            self._close(reboot_socket)
            reboot_socket = None
            # The firmware intentionally spends one second draining the signed
            # REBOOT reply and deleting the accepted OTA seed.  Connecting a
            # RAW probe inside that window lets the old application accept a
            # new seed which then survives the MCU-only reset.  Stay silent
            # until the new generation has started its bounded socket cleanup.
            notify("旧连接已释放，静默等待板卡完成重启前的 seed 清理")
            self._cancelable_sleep(self.reboot_initial_delay_s, cancelled)
            self.final_boot_id = self.wait_for_raw_ready(
                progress=notify,
                cancelled=cancelled,
                previous_boot_id=source_boot_id,
                require_boot_identity=True,
            )
        else:
            self._close(reboot_socket)


__all__ = [
    "FirmwarePackage",
    "OtaClient",
    "OtaCancelled",
    "OtaError",
    "OtaProgress",
    "build_package",
    "crc32",
    "discover_board",
    "load_package",
    "parse_version",
    "validate_application_image",
]
