"""Passive TH1 extension of the existing 20-byte temperature monitor.

No port reads or commands here. The original temperature bytes and frame
length stay compatible with old clients; RPM always comes from FG feedback.
"""
import binascii
import logging
import struct
import threading
import time


def decode_thermal_frame(frame):
    if (len(frame) != 20 or frame[:2] != b'\xff\xff'
            or frame[2] not in (1, 2) or frame[3] != 1
            or frame[14:17] != b'TH\x01'
            or binascii.crc_hqx(frame[:18], 0xffff)
            != int.from_bytes(frame[18:20], 'big')):
        return None
    integer, fraction, rpm, duty, flags = struct.unpack('>hhHHH', frame[4:14])
    temperature = integer + fraction / 10000.0
    if not -55 <= temperature <= 125 or abs(fraction) >= 10000 or duty > 1000:
        return None
    return dict(temperature_c=temperature, fan_rpm=rpm,
                fan_duty_permille=duty, thermal_flags=flags,
                sensor_age_ms=frame[17] * 10, work_state=frame[2])


class ThermalStreamDecoder:
    """Observe a copy of mixed ACK/ADC/dense/finger bytes, including splits."""
    def __init__(self):
        self.buffer = bytearray()

    def reset(self):
        self.buffer.clear()

    def feed(self, data):
        self.buffer.extend(data)
        result = []
        while len(self.buffer) >= 20:
            start = self.buffer.find(b'\xff\xff')
            if start < 0:
                del self.buffer[:-1]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < 20:
                break
            values = decode_thermal_frame(self.buffer[:20])
            if values is None:
                del self.buffer[0]
            else:
                result.append(values)
                del self.buffer[:20]
        return result


class ThermalReadObserver:
    """Read-owner side tap, never a second reader; UI errors cannot stop scans."""
    def __init__(self, callback=None):
        self.callback = callback
        self.decoder = ThermalStreamDecoder()
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.decoder.reset()

    def feed(self, data):
        received = time.monotonic()
        with self._lock:
            values = self.decoder.feed(data)
            callback = self.callback
        if callback is not None:
            for value in values:
                value['received_monotonic'] = received
                try:
                    callback(value)
                except Exception:
                    logging.getLogger(__name__).exception('Thermal display callback failed')
