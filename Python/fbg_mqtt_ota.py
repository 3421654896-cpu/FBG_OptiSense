"""Authenticated, stop-and-wait STM32 OTA through a device's MQTT command topic.

This path is intentionally independent of the LAN TCP OTA path. It requires
firmware with FOTA1 MQTT support and a broker account allowed to publish only
to the intended device. The board authenticates BEGIN and the complete image.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import queue
import secrets
import ssl
import threading
import time
from pathlib import Path
from typing import Callable

from fbg_mqtt_transport import RemoteMqttConfig, load_remote_config
from fbg_ota_protocol import OTA_CHUNK_SIZE, FirmwarePackage, OtaCancelled, OtaError, crc32, load_package


def signed_ping(nonce: str, secret: str) -> str:
    prefix = f"FOTA1|PING|{nonce}"
    return f"{prefix}|{hmac.new(secret.encode(), prefix.encode('ascii'), hashlib.sha256).hexdigest().upper()}"


def signed_abort(nonce: str, secret: str) -> str:
    prefix = f"FOTA1|ABORT|{nonce}"
    return f"{prefix}|{hmac.new(secret.encode(), prefix.encode('ascii'), hashlib.sha256).hexdigest().upper()}"


def parse_pong(line: str, nonce: str, secret: str) -> tuple[int, int]:
    fields = line.strip().split("|")
    if len(fields) != 6 or fields[:2] != ["FOTA1", "PONG"] or fields[2] != nonce:
        raise OtaError("设备身份握手不匹配")
    expected = hmac.new(secret.encode(), "|".join(fields[:-1]).encode("ascii"), hashlib.sha256).hexdigest().upper()
    if not hmac.compare_digest(fields[-1], expected):
        raise OtaError("设备身份握手签名无效")
    try:
        return int(fields[3], 16), int(fields[4], 16)
    except ValueError as exc:
        raise OtaError("设备身份握手格式无效") from exc


class MqttOtaClient:
    def __init__(self, config: RemoteMqttConfig, secret: str):
        if not secret:
            raise ValueError("OTA 授权密钥不能为空")
        self.config = config
        self.secret = secret
        self._client = None
        self._connected = threading.Event()
        self._subscribed = threading.Event()
        self._responses: queue.Queue[str] = queue.Queue(maxsize=64)
        self.cancelled: Callable[[], bool] = lambda: False

    def __enter__(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise OtaError("缺少 paho-mqtt，请安装 requirements_remote.txt") from exc
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"fbg-ota-{secrets.token_hex(8)}",
            protocol=mqtt.MQTTv311,
        )
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_subscribe = self._on_subscribe
        client.on_message = self._on_message
        if self.config.username is not None:
            client.username_pw_set(self.config.username, self.config.password)
        context = ssl.create_default_context(
            cafile=str(self.config.ca_file) if self.config.ca_file else None
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        if self.config.client_cert is not None:
            context.load_cert_chain(str(self.config.client_cert), str(self.config.client_key))
        client.tls_set_context(context)
        self._client = client
        try:
            client.connect_async(self.config.host, self.config.port, self.config.keepalive_s)
            client.loop_start()
            if not self._subscribed.wait(30):
                raise OtaError("MQTT 未连接或 ACK 订阅未确认")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        client, self._client = self._client, None
        self._connected.clear()
        self._subscribed.clear()
        if client is not None:
            client.disconnect()
            client.loop_stop()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            return
        self._connected.set()
        self._subscribed.clear()
        client.subscribe(self.config.ack_topic, qos=1)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected.clear()
        self._subscribed.clear()

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties):
        if reason_codes and all(not code.is_failure for code in reason_codes):
            self._subscribed.set()

    def _on_message(self, client, userdata, message):
        if message.topic != self.config.ack_topic or message.retain:
            return
        try:
            line = message.payload.decode("ascii").strip()
        except UnicodeDecodeError:
            return
        if not line.startswith("FOTA1|") or len(line) > 160:
            return
        try:
            self._responses.put_nowait(line)
        except queue.Full:
            try:
                self._responses.get_nowait()
                self._responses.put_nowait(line)
            except queue.Empty:
                pass

    def _exchange(self, command: str, accepted: tuple[str, ...], *,
                  matches: Callable[[list[str]], bool] | None = None,
                  timeout: float = 12.0, tries: int = 6) -> list[str]:
        if self._client is None:
            raise OtaError("MQTT 未连接")
        for _ in range(tries):
            if self.cancelled():
                raise OtaCancelled("用户已取消广域网 OTA")
            if not self._subscribed.wait(10):
                continue
            while True:
                try:
                    self._responses.get_nowait()
                except queue.Empty:
                    break
            info = self._client.publish(self.config.command_topic, command.encode("ascii"), qos=1, retain=False)
            if info.rc != 0:
                time.sleep(1)
                continue
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.cancelled():
                    raise OtaCancelled("用户已取消广域网 OTA")
                try:
                    fields = self._responses.get(timeout=min(0.5, deadline - time.monotonic())).split("|")
                except queue.Empty:
                    continue
                if len(fields) < 2 or fields[0] != "FOTA1":
                    continue
                if fields[1] != "PONG":
                    if len(fields) < 4:
                        continue
                    signed_text = "|".join(fields[:-1])
                    expected_mac = hmac.new(
                        self.secret.encode(), signed_text.encode("ascii"), hashlib.sha256
                    ).hexdigest().upper()
                    if not hmac.compare_digest(fields[-1], expected_mac):
                        continue
                    fields = fields[:-1]
                if fields[1] == "ERROR":
                    raise OtaError("设备拒绝 OTA 命令：" + "|".join(fields[2:]))
                if fields[1] in accepted and (matches is None or matches(fields)):
                    return fields
        raise OtaError("设备未在限定重试内回复 OTA 命令")

    def ping(self) -> tuple[int, int]:
        nonce = secrets.token_hex(8).upper()
        fields = self._exchange(signed_ping(nonce, self.secret), ("PONG",),
                                matches=lambda f: len(f) == 6 and f[2] == nonce)
        return parse_pong("|".join(fields), nonce, self.secret)

    def install(self, package: FirmwarePackage,
                progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None,
                phase: Callable[[str], None] | None = None) -> None:
        self.cancelled = cancelled or (lambda: False)
        old_boot, current_version = self.ping()
        if package.firmware_version <= current_version:
            raise OtaError("设备已是相同或更新版本，拒绝重复广域网刷写")
        nonce = secrets.token_hex(8).upper()
        prefix = (
            f"FOTA1|BEGIN|F205RE|{package.firmware_version:08X}|"
            f"{package.image_size:08X}|{package.image_crc32:08X}|{nonce}"
        )
        signature = hmac.new(self.secret.encode(), prefix.encode("ascii"), hashlib.sha256).hexdigest().upper()
        begin = f"{prefix}|{signature}"
        failures = 0
        end_attempted = False
        reboot_ack = False

        def abort_before_commit() -> None:
            self.cancelled = lambda: False
            try:
                self._exchange(
                    signed_abort(nonce, self.secret), ("ABORTED",),
                    matches=lambda f: len(f) == 3 and f[2] == nonce,
                    timeout=5, tries=2,
                )
            except OtaError:
                pass  # Device will release the uncommitted WAN session on idle timeout.

        while True:
            try:
                ready = self._exchange(begin, ("READY",), timeout=30)
            except OtaCancelled:
                abort_before_commit()
                raise
            if len(ready) != 5:
                raise OtaError("READY 回执格式错误")
            try:
                session, offset, boot = (int(item, 16) for item in ready[2:])
            except ValueError as exc:
                raise OtaError("READY 回执数字无效") from exc
            if boot != old_boot or not 0 <= offset <= package.image_size:
                raise OtaError("设备启动标识或续传位置不匹配")
            if offset % 4 and offset != package.image_size:
                raise OtaError("续传位置未按闪存字对齐")
            try:
                while offset < package.image_size:
                    if self.cancelled():
                        raise OtaCancelled("用户已取消广域网 OTA")
                    block = package.image[offset:offset + OTA_CHUNK_SIZE]
                    data = base64.b64encode(block).decode("ascii")
                    command = f"FOTA1|DATA|{session:08X}|{offset:08X}|{crc32(block):08X}|{data}"
                    ack = self._exchange(
                        command, ("ACK",),
                        matches=lambda f: len(f) == 4 and f[2] == f"{session:08X}"
                        and f[3] == f"{offset + len(block):08X}",
                    )
                    if len(ack) != 4:
                        raise OtaError("OTA 分块回执不匹配")
                    offset += len(block)
                    if progress:
                        progress(offset, package.image_size)
                if self.cancelled():
                    raise OtaCancelled("用户已取消广域网 OTA")
                mac = hmac.new(self.secret.encode(), package.image, hashlib.sha256).hexdigest().upper()
                end = f"FOTA1|END|{session:08X}|{package.image_size:08X}|{package.image_crc32:08X}|{mac}"
                # Once END is sent, cancelling is unsafe: the device may
                # already have committed metadata and be entering reboot.
                if phase:
                    phase("commit")
                self.cancelled = lambda: False
                end_attempted = True
                reboot = self._exchange(
                    end, ("REBOOT",), timeout=30,
                    matches=lambda f: len(f) == 3 and f[2] == f"{package.firmware_version:08X}",
                )
                if len(reboot) != 3:
                    raise OtaError("设备确认的固件版本不匹配")
                reboot_ack = True
                break
            except OtaCancelled:
                abort_before_commit()
                raise
            except OtaError as exc:
                if end_attempted:
                    if "未在限定重试" in str(exc):
                        break  # Verify a new boot; never restart a possibly committed image.
                    raise
                failures += 1
                if failures >= 8 or "未在限定重试" not in str(exc):
                    raise
                time.sleep(min(failures * 2, 10))

        # A REBOOT acknowledgement only confirms staged/committed bytes. A
        # fresh signed PONG with a new boot ID confirms actual installation.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                new_boot, new_version = self.ping()
            except OtaError:
                continue
            if new_boot != old_boot and new_version == package.firmware_version:
                return
        if reboot_ack:
            raise OtaError("固件已提交，但未验证到新启动标识和新版本；请现场确认，不要重复刷写")
        raise OtaError("END 回执丢失且未验证到新固件；提交状态不明，请现场确认，不要重复刷写")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="指定设备的广域网 MQTT OTA")
    parser.add_argument("--config", type=Path, required=True, help="设备专属 MQTT 配置")
    parser.add_argument("--device-id", required=True, help="必须与配置中的设备 ID 完全一致")
    parser.add_argument("--package", type=Path, required=True, help="已校验的 .fbgfw 固件包")
    args = parser.parse_args()
    config = load_remote_config(args.config)
    if args.device_id != config.device_id:
        parser.error("设备 ID 与配置不一致")
    package = load_package(args.package)
    secret = getpass.getpass("该设备 OTA 授权密钥：")
    with MqttOtaClient(config, secret) as updater:
        updater.install(package, lambda sent, total: print(f"\r{sent}/{total} 字节", end="", flush=True))
    print("\n广域网 OTA 已验证新版本和新启动标识")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
