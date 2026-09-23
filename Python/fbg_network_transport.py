"""Runtime selector for mutually-exclusive LAN and WAN transports."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from fbg_lan_transport import FbgLanTransport
from fbg_mqtt_transport import FbgMqttTransport, RemoteMqttConfig
from fbg_remote_protocol import encode_mode_command


NETWORK_LAN = "lan"
NETWORK_WAN = "mqtt"
NETWORK_MODES = (NETWORK_LAN, NETWORK_WAN)


class FbgSelectableTransport:
    """Own exactly one live receiver and expose the common viewer API."""

    def __init__(
        self,
        mqtt_config: Optional[RemoteMqttConfig],
        *,
        initial_mode: str = NETWORK_LAN,
        on_ack: Optional[Callable[[object], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_connection: Optional[Callable[[bool, str], None]] = None,
        on_metadata: Optional[Callable[[object], None]] = None,
    ) -> None:
        if initial_mode not in NETWORK_MODES:
            raise ValueError(f"不支持的网络方式：{initial_mode}")
        self.mqtt_config = mqtt_config
        self.initial_mode = initial_mode
        self.on_ack = on_ack
        self.on_status = on_status
        self.on_connection = on_connection
        self.on_metadata = on_metadata
        self._lock = threading.RLock()
        self._transport = None
        self._mode: Optional[str] = None
        self._generation = 0

    @property
    def network_mode(self) -> Optional[str]:
        with self._lock:
            return self._mode

    @property
    def connected(self) -> bool:
        with self._lock:
            transport = self._transport
        return bool(transport is not None and transport.connected)

    @property
    def board_ip(self) -> Optional[str]:
        """Last learned LAN board address, if the active path is LAN."""
        with self._lock:
            transport = self._transport
        return None if transport is None else getattr(transport, "board_ip", None)

    def _guard(self, generation: int, callback):
        def guarded(*args):
            with self._lock:
                current = generation == self._generation
            if current and callback is not None:
                callback(*args)

        return guarded

    def _build(self, mode: str, generation: int):
        callbacks = dict(
            on_ack=self._guard(generation, self.on_ack),
            on_status=self._guard(generation, self.on_status),
            on_connection=self._guard(generation, self.on_connection),
            on_metadata=self._guard(generation, self.on_metadata),
        )
        if mode == NETWORK_LAN:
            return FbgLanTransport(**callbacks)
        if self.mqtt_config is None:
            raise RuntimeError(
                "广域网模式需要 fbg_remote_config.yaml 中的 MQTT 配置"
            )
        return FbgMqttTransport(self.mqtt_config, **callbacks)

    def start(self) -> None:
        self.select_network(self.initial_mode)

    def select_network(self, mode: str) -> None:
        if mode not in NETWORK_MODES:
            raise ValueError(f"不支持的网络方式：{mode}")
        if mode == NETWORK_WAN and self.mqtt_config is None:
            raise RuntimeError(
                "未找到 MQTT 配置，不能切换到广域网模式"
            )

        with self._lock:
            if self._mode == mode and self._transport is not None:
                return
            old = self._transport
            self._transport = None
            self._mode = None
            self._generation += 1
            generation = self._generation
        if old is not None:
            old.stop()

        transport = self._build(mode, generation)
        try:
            transport.start()
        except Exception:
            transport.stop()
            raise
        installed = False
        with self._lock:
            # start() may block while another GUI/action thread selects a
            # newer path or stops reception.  Never let the older completion
            # resurrect a second live receiver after that newer decision.
            if generation == self._generation and self._transport is None:
                self._transport = transport
                self._mode = mode
                installed = True
        if not installed:
            transport.stop()

    def stop(self) -> None:
        with self._lock:
            transport, self._transport = self._transport, None
            self._mode = None
            self._generation += 1
        if transport is not None:
            transport.stop()

    def take_latest_telemetry(self):
        with self._lock:
            transport = self._transport
        return None if transport is None else transport.take_latest_telemetry()

    def publish_mode(self, mode: int, command_id: int) -> None:
        # Validate before delegating so a replacement/test transport cannot
        # silently coerce booleans or fractional command identifiers.
        encode_mode_command(command_id, mode)
        with self._lock:
            transport = self._transport
        if transport is None:
            raise RuntimeError("网络连接尚未启动")
        transport.publish_mode(mode, command_id)


__all__ = [
    "FbgSelectableTransport",
    "NETWORK_LAN",
    "NETWORK_MODES",
    "NETWORK_WAN",
]
