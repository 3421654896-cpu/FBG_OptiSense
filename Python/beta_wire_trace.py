"""Diagnostic native-USB trace; never changes reads, writes, or timeout policy."""

import hashlib
import time

from thermal_serial import ThermalSerial


class TraceSerial(ThermalSerial):
    def __init__(self, *args, **kwargs):
        self.trace = []
        self.raw = bytearray()
        super().__init__(*args, **kwargs)

    def read(self, size=1):
        data = super().read(size)
        if data:
            self.trace.append(dict(action='read', at=time.monotonic(),
                                   offset=len(self.raw), count=len(data)))
            self.raw.extend(data)
        return data

    def write(self, data):
        returned = super().write(data)
        self.trace.append(dict(action='write', at=time.monotonic(), count=len(data),
                               returned=returned, header_hex=data[:24].hex(),
                               sha256=hashlib.sha256(data).hexdigest()))
        return returned

    def reset_input_buffer(self):
        self.trace.append(dict(action='reset_input_buffer', at=time.monotonic()))
        return super().reset_input_buffer()
