"""Native CDC port with a non-consuming thermal monitor observer."""
import serial
from board_thermal_protocol import ThermalReadObserver


class ThermalSerial(serial.Serial):
    def __init__(self, *args, **kwargs):
        self.thermal_observer = ThermalReadObserver()
        super().__init__(*args, **kwargs)

    def read(self, size=1):
        data = super().read(size)
        self.thermal_observer.feed(data)
        return data

    def open(self):
        self.thermal_observer.reset()
        return super().open()

    def close(self):
        self.thermal_observer.reset()
        return super().close()

    def reset_input_buffer(self):
        self.thermal_observer.reset()
        return super().reset_input_buffer()
