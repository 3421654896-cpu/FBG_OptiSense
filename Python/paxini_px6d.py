"""Paxini PX6D six-axis force sensor reader (RS485/USB-CDC serial).

Hardware-verified on COM9 (streaming device id 0x01, baud 921600 8N1):
  - One-shot "get frame" (cmd 0x01) answers only a 13-byte ACK, no forces.
  - cmd 0x02 data 0x01 starts auto-push: first an ACK (cmd 0x02), then a
    continuous stream of cmd-0x03 frames whose 24 data bytes are six float32
    LE values: Fx Fy Fz (N) and Mx My Mz (N·m).
  - cmd 0x02 data 0x00 stops the stream.
Frame = AA 55 | device_id | cmd | data... | RC8 (XOR over id..data).
A missing/unplugged sensor must never block stress capture: ``read_six_axis``
raises and the caller records ``paxini_force`` as null with the reason.
"""

from __future__ import annotations

import struct
import threading
import time


HEADER = b"\xAA\x55"
DEFAULT_DEVICE_ID = 0x01
DEFAULT_BAUD = 921600
PUSH_DATA_LENGTH = {0x02: 8, 0x03: 24, 0x10: 8}


def rc8(body: bytes) -> int:
    checksum = 0
    for byte in body:
        checksum ^= byte
    return checksum & 0xFF


def build_frame(cmd: int, data: bytes, device_id: int = DEFAULT_DEVICE_ID) -> bytes:
    body = bytes((device_id, cmd)) + data
    return HEADER + body + bytes((rc8(body),))


def parse_six_axis(payload: bytes):
    if len(payload) < 24:
        return None
    values = struct.unpack("<6f", payload[:24])
    if not all(value == value and abs(value) < 1e9 for value in values):
        return None
    return values


class PaxiniForceSensor:
    """Auto-push six-axis reader; keeps the newest frame ready for capture."""

    def __init__(
        self,
        port: str,
        *,
        baud: int = DEFAULT_BAUD,
        device_id: int = DEFAULT_DEVICE_ID,
    ):
        self.port = port
        self.baud = int(baud)
        self.device_id = int(device_id)
        self.last_error: str | None = None
        self.frames_received = 0
        self._ser = None
        self._lock = threading.Lock()
        self._latest = None
        self._latest_ns = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._tx_lock = threading.Lock()
        self._calibration_event = threading.Event()
        self._calibration_payload = None
        self.zero_count = 0
        self.last_zero_ns = None

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        """Open the port and start the auto-push pump thread."""

        with self._lock:
            if self._running:
                return
            import serial

            try:
                self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self._running = True
            self._thread = threading.Thread(
                target=self._pump, name="paxini-px6d", daemon=True
            )
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.write(build_frame(0x02, b"\x00", self.device_id))
                self._ser.flush()
                time.sleep(0.05)
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def close(self):
        self.stop()

    # ------------------------------------------------------------------- pump

    def _send(self, packet: bytes):
        try:
            with self._tx_lock:
                self._ser.write(packet)
                self._ser.flush()
        except Exception as exc:
            self.last_error = str(exc)

    def _pump(self):
        buffer = bytearray()
        # 0 = stop any foreign stream, 1 = start ours; then just listen.
        phase = 0
        phase_deadline = time.monotonic() + 0.15
        while self._running:
            try:
                waiting = self._ser.in_waiting
                if waiting:
                    buffer.extend(self._ser.read(waiting))
                if time.monotonic() >= phase_deadline:
                    if phase == 0:
                        self._send(build_frame(0x02, b"\x00", self.device_id))
                        phase = 1
                    elif phase == 1:
                        self._send(build_frame(0x02, b"\x01", self.device_id))
                        phase = 2
                    buffer.clear()
                    phase_deadline = time.monotonic() + 0.15
                consumed = self._extract(buffer)
                if not consumed:
                    time.sleep(0.004)
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.05)

    def _extract(self, buffer: bytearray) -> bool:
        start = buffer.find(HEADER)
        if start < 0:
            if len(buffer) > 512:
                del buffer[:-2]
            return False
        if start > 0:
            del buffer[:start]
            return True
        if len(buffer) < 5:
            return False
        cmd = buffer[3]
        expected = PUSH_DATA_LENGTH.get(cmd)
        if expected is None:
            del buffer[:1]
            return True
        if len(buffer) < 4 + expected + 1:
            return False
        payload = bytes(buffer[4 : 4 + expected])
        del buffer[: 4 + expected + 1]
        if cmd == 0x03:
            values = parse_six_axis(payload)
            if values is not None:
                self._latest = values
                self._latest_ns = time.monotonic_ns()
                self.frames_received += 1
                self.last_error = None
        elif cmd == 0x10:
            self._calibration_payload = payload
            self._calibration_event.set()
        return True

    # ------------------------------------------------------------------- read

    def calibrate_zero(self, timeout_s: float = 3.0):
        """Run the official PX6D recalibration/zero command (cmd 0x10).

        The official host sends this as a broadcast (device id 0x7F). Keep the
        sensor unloaded while it calibrates. Returns the 8-byte ACK payload.
        """

        if not self._running or self._ser is None:
            raise RuntimeError("PX6D未连接，无法归零")
        self._calibration_payload = None
        self._calibration_event.clear()
        self._send(build_frame(0x10, b"\x01", device_id=0x7F))
        if not self._calibration_event.wait(timeout_s):
            raise TimeoutError("PX6D归零命令未确认")
        self.zero_count += 1
        self.last_zero_ns = time.monotonic_ns()
        return self._calibration_payload

    def read_six_axis(self, timeout_s: float = 1.0):
        """Return the newest hardware-zeroed six-axis tuple; raise when stale."""

        deadline = time.monotonic() + timeout_s
        frames0 = self.frames_received
        while time.monotonic() < deadline:
            if self._latest is not None and self.frames_received > frames0:
                return self._latest
            if (
                self._latest is not None
                and self.frames_received == frames0
                and self._latest_ns is not None
                and (time.monotonic_ns() - self._latest_ns) / 1e6 < 500.0
            ):
                return self._latest
            time.sleep(0.01)
        raise TimeoutError(self.last_error or "PX6D 自动回传无数据")
