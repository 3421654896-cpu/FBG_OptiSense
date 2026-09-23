"""Threaded MQTT transport for the remote FBG monitor.

The module intentionally has no Qt dependency. MQTT callbacks validate incoming
telemetry and replace one protected ``latest`` slot. The GUI polls that slot on
its own timer, so MQTT bursts can never create an unbounded Qt event queue.
"""

from __future__ import annotations

import os
import re
import socket
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import yaml

from fbg_remote_protocol import (
    decode_ack,
    decode_metadata,
    decode_telemetry,
    encode_mode_command,
)


AckCallback = Callable[[object], None]
StatusCallback = Callable[[str], None]
ConnectionCallback = Callable[[bool, str], None]
MetadataCallback = Callable[[object], None]
_SUBSCRIPTION_COUNT = 5


def validate_device_id(value: object) -> str:
    device_id = str(value)
    if not re.fullmatch(r"fbg-[A-Za-z0-9]{1,28}", device_id):
        raise ValueError(
            "device_id 必须为 fbg- 加1..28位 ASCII 字母或数字；"
            "后缀不能包含点、下划线或连字符"
        )
    return device_id


@dataclass(frozen=True)
class RemoteMqttConfig:
    """Connection settings with secrets resolved at load time."""

    device_id: str
    host: str
    topic_root: str = "fbg/v1"
    port: int = 8883
    keepalive_s: int = 30
    client_id_prefix: str = "fbg-viewer"
    username: Optional[str] = None
    password: Optional[str] = None
    ca_file: Optional[Path] = None
    client_cert: Optional[Path] = None
    client_key: Optional[Path] = None

    def __post_init__(self) -> None:
        # The board must flatten the command topic for BW20 compatibility.
        # A canonical ID keeps that mapping one-to-one across multiple boards.
        validate_device_id(self.device_id)

    @property
    def topic_prefix(self) -> str:
        return f"{self.topic_root}/{self.device_id}"

    @property
    def telemetry_topic(self) -> str:
        return f"{self.topic_prefix}/telemetry"

    @property
    def command_topic(self) -> str:
        # BW20 Combo-AT P1.0.22 cannot subscribe to a topic containing '/'.
        # Keep the command topic flat and deterministic; telemetry remains in
        # the structured fbg/v1/<device>/... namespace.
        source = self.device_id[4:] if self.device_id.startswith("fbg-") else self.device_id
        suffix = "".join(
            character
            for character in source
            if character.isascii() and character.isalnum()
        )
        return f"fbgcmd{suffix}"

    @property
    def ack_topic(self) -> str:
        return f"{self.topic_prefix}/ack"

    @property
    def status_topic(self) -> str:
        return f"{self.topic_prefix}/status"

    @property
    def stress_metadata_topic(self) -> str:
        return f"{self.topic_prefix}/metadata/stress"

    @property
    def temperature_metadata_topic(self) -> str:
        return f"{self.topic_prefix}/metadata/temperature"


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是 YAML 映射")
    return value


def _optional_path(value: Any, base_dir: Path) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _credential(
    section: Mapping[str, Any], literal_name: str, env_name_field: str, default_env: str
) -> Optional[str]:
    env_name = str(section.get(env_name_field, default_env) or default_env).strip()
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return env_value
    literal = section.get(literal_name)
    if literal is None or str(literal) == "":
        return None
    return str(literal)


def _config_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须是整数")
    return value


def load_remote_config(path: Union[os.PathLike, str]) -> RemoteMqttConfig:
    """Load a local YAML config without logging credential values."""

    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _mapping(data, "配置")
    broker = _mapping(root.get("broker"), "broker")
    tls = _mapping(root.get("tls"), "tls")
    credentials = _mapping(root.get("credentials"), "credentials")

    device_id = validate_device_id(root.get("device_id", ""))
    host = str(broker.get("host", "")).strip()
    if not host:
        raise ValueError("配置缺少 broker.host")
    topic_root = str(root.get("topic_prefix", "fbg/v1")).strip().strip("/")
    if not topic_root:
        raise ValueError("topic_prefix 清除首尾斜杠后不能为空")
    if any(marker in topic_root for marker in ("+", "#", ",", ">")):
        raise ValueError("topic_prefix 不能包含 MQTT 通配符 +/#、逗号或字符 >")
    if len(topic_root.encode("utf-8")) > 96:
        raise ValueError("topic_prefix 不能超过96个 UTF-8 字节")

    port = _config_integer(broker.get("port", 8883), "broker.port")
    if not 1 <= port <= 65535:
        raise ValueError("broker.port 必须在 1..65535")
    keepalive = _config_integer(
        broker.get("keepalive_s", 30), "broker.keepalive_s"
    )
    if not 10 <= keepalive <= 1200:
        raise ValueError("broker.keepalive_s 必须在 10..1200 秒")

    tls_enabled = tls.get("enabled", True)
    if not isinstance(tls_enabled, bool):
        raise ValueError("tls.enabled 必须是 true 或 false")
    if not tls_enabled:
        raise ValueError("公网遥测必须启用 TLS；请将 tls.enabled 设为 true")

    base_dir = config_path.parent
    ca_file = _optional_path(tls.get("ca_file"), base_dir)
    client_cert = _optional_path(tls.get("client_cert"), base_dir)
    client_key = _optional_path(tls.get("client_key"), base_dir)
    for label, item in (
        ("CA 证书", ca_file),
        ("客户端证书", client_cert),
        ("客户端私钥", client_key),
    ):
        if item is not None and not item.is_file():
            raise ValueError(f"{label}不存在: {item}")
    if (client_cert is None) != (client_key is None):
        raise ValueError("client_cert 和 client_key 必须同时配置")

    username = _credential(
        credentials, "username", "username_env", "FBG_MQTT_USERNAME"
    )
    password = _credential(
        credentials, "password", "password_env", "FBG_MQTT_PASSWORD"
    )
    if password is not None and username is None:
        raise ValueError("配置了 MQTT 密码时必须同时配置用户名")

    return RemoteMqttConfig(
        device_id=device_id,
        host=host,
        topic_root=topic_root,
        port=port,
        keepalive_s=keepalive,
        client_id_prefix=str(broker.get("client_id_prefix", "fbg-viewer")),
        username=username,
        password=password,
        ca_file=ca_file,
        client_cert=client_cert,
        client_key=client_key,
    )


class FbgMqttTransport:
    """MQTT subscriber/publisher with TLS verification and latest-only delivery."""

    def __init__(
        self,
        config: RemoteMqttConfig,
        *,
        on_ack: Optional[AckCallback] = None,
        on_status: Optional[StatusCallback] = None,
        on_connection: Optional[ConnectionCallback] = None,
        on_metadata: Optional[MetadataCallback] = None,
    ) -> None:
        self.config = config
        self.on_ack = on_ack
        self.on_status = on_status
        self.on_connection = on_connection
        self.on_metadata = on_metadata

        self._client = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[object] = None
        self._mqtt_module = None
        self._subscribe_mid: Optional[int] = None

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @staticmethod
    def _clean_client_component(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
        return cleaned.strip("-") or "viewer"

    def _notify_connection(self, connected: bool, detail: str) -> None:
        if self.on_connection is not None:
            try:
                self.on_connection(bool(connected), str(detail))
            except Exception:
                pass

    def _callback_is_current(self, client: object) -> bool:
        """Reject callbacks queued by a client that has already been stopped."""
        return client is not None and client is self._client

    def _build_client(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "缺少 paho-mqtt，请先执行: py -3 -m pip install -r requirements_remote.txt"
            ) from exc

        self._mqtt_module = mqtt
        host_name = self._clean_client_component(socket.gethostname())
        prefix = self._clean_client_component(self.config.client_id_prefix)
        device = self._clean_client_component(self.config.device_id)
        client_id = f"{prefix}-{device}-{host_name}-{os.getpid()}"
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )
        client.on_connect = self._on_connect
        client.on_connect_fail = self._on_connect_fail
        client.on_disconnect = self._on_disconnect
        client.on_subscribe = self._on_subscribe
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=1, max_delay=60)

        if self.config.username is not None:
            client.username_pw_set(self.config.username, self.config.password)

        context = ssl.create_default_context(
            cafile=str(self.config.ca_file) if self.config.ca_file else None
        )
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if self.config.client_cert is not None:
            context.load_cert_chain(
                certfile=str(self.config.client_cert),
                keyfile=str(self.config.client_key),
            )
        client.tls_set_context(context)
        return client

    def start(self) -> None:
        if self._client is not None:
            return
        self._stop.clear()
        try:
            self._client = self._build_client()
            self._client.connect_async(
                self.config.host, self.config.port, self.config.keepalive_s
            )
            self._client.loop_start()
            self._notify_connection(False, "正在连接 MQTT")
        except Exception:
            self._stop.set()
            self._client = None
            raise

    def stop(self) -> None:
        self._stop.set()
        client, self._client = self._client, None
        self._subscribe_mid = None
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
            try:
                client.loop_stop()
            except Exception:
                pass
        self._connected.clear()
        with self._latest_lock:
            self._latest_frame = None

    def _put_latest(self, frame: object) -> None:
        with self._latest_lock:
            self._latest_frame = frame

    def take_latest_telemetry(self) -> Optional[object]:
        """Atomically return and clear the newest decoded telemetry frame."""

        with self._latest_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    @staticmethod
    def _reason_ok(reason_code: object) -> bool:
        try:
            return int(reason_code) == 0
        except (TypeError, ValueError):
            return str(reason_code).lower() in {"success", "0"}

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if not self._callback_is_current(client):
            return
        self._subscribe_mid = None
        if not self._reason_ok(reason_code):
            self._connected.clear()
            self._notify_connection(False, f"MQTT 连接被拒绝: {reason_code}")
            return
        result, message_id = client.subscribe(
            [
                (self.config.telemetry_topic, 1),
                (self.config.ack_topic, 1),
                (self.config.status_topic, 1),
                (self.config.stress_metadata_topic, 1),
                (self.config.temperature_metadata_topic, 1),
            ]
        )
        mqtt = self._mqtt_module
        if mqtt is not None and result != mqtt.MQTT_ERR_SUCCESS:
            self._subscribe_mid = None
            self._connected.clear()
            self._notify_connection(False, f"MQTT 订阅失败，错误码 {result}")
            return
        self._subscribe_mid = int(message_id)
        self._connected.clear()
        self._notify_connection(False, "MQTT 已连接，正在确认主题权限")

    def _on_subscribe(
        self, client, userdata, message_id, reason_code_list, properties
    ) -> None:
        if not self._callback_is_current(client):
            return
        if self._subscribe_mid is None or int(message_id) != self._subscribe_mid:
            return
        reason_codes = list(reason_code_list or ())
        if len(reason_codes) != _SUBSCRIPTION_COUNT:
            self._subscribe_mid = None
            self._connected.clear()
            self._notify_connection(
                False,
                "MQTT 订阅回执不完整，未确认全部主题权限",
            )
            return
        denied = []
        for reason_code in reason_codes:
            try:
                failed = int(reason_code) >= 0x80
            except (TypeError, ValueError):
                text = str(reason_code).lower()
                failed = "error" in text or "denied" in text or "not authorized" in text
            if failed:
                denied.append(str(reason_code))
        if denied:
            self._subscribe_mid = None
            self._connected.clear()
            self._notify_connection(
                False,
                "MQTT 主题权限被拒绝，请检查远端账号 ACL：" + ", ".join(denied),
            )
            return
        self._subscribe_mid = None
        self._connected.set()
        self._notify_connection(True, "MQTT 已连接，主题权限正常")

    def _on_connect_fail(self, client, userdata) -> None:
        if not self._callback_is_current(client):
            return
        self._subscribe_mid = None
        self._connected.clear()
        self._notify_connection(False, "MQTT 连接失败，正在自动重试")

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ) -> None:
        if not self._callback_is_current(client):
            return
        self._subscribe_mid = None
        self._connected.clear()
        detail = "MQTT 已断开"
        if not self._stop.is_set():
            detail += "，正在自动重连"
        self._notify_connection(False, detail)

    def _on_message(self, client, userdata, message) -> None:
        if not self._callback_is_current(client):
            return
        try:
            if message.topic == self.config.telemetry_topic:
                self._put_latest(decode_telemetry(bytes(message.payload)))
            elif message.topic == self.config.ack_topic:
                ack = decode_ack(bytes(message.payload))
                if self.on_ack is not None:
                    self.on_ack(ack)
            elif message.topic == self.config.status_topic:
                status = bytes(message.payload).decode("utf-8", errors="replace")
                if self.on_status is not None:
                    self.on_status(status)
            elif message.topic in {
                self.config.stress_metadata_topic,
                self.config.temperature_metadata_topic,
            }:
                metadata = decode_metadata(bytes(message.payload))
                if self.on_metadata is not None:
                    self.on_metadata(metadata)
        except Exception as exc:
            self._notify_connection(self.connected, f"收到无效消息: {exc}")

    def publish_mode(self, mode: int, command_id: int) -> None:
        """Publish a non-retained QoS-1 mode command."""

        payload = encode_mode_command(command_id, mode)
        if not self.connected or self._client is None:
            raise RuntimeError("MQTT 尚未连接")
        info = self._client.publish(
            self.config.command_topic, payload=payload, qos=1, retain=False
        )
        mqtt = self._mqtt_module
        if mqtt is not None and info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT 发布失败，错误码 {info.rc}")


__all__ = [
    "FbgMqttTransport",
    "RemoteMqttConfig",
    "load_remote_config",
    "validate_device_id",
]
