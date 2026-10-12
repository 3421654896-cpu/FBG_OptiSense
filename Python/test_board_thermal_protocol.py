import binascii
import struct
import unittest
from unittest.mock import patch

from board_thermal_protocol import decode_thermal_frame, ThermalStreamDecoder, ThermalReadObserver
from thermal_serial import ThermalSerial


def thermal_frame(state=2, temperature=33.25, rpm=2340, duty=210, flags=17):
    integer = int(temperature)
    frame = (b'\xff\xff' + bytes((state, 1))
             + struct.pack('>hhHHH', integer, round((temperature-integer)*10000), rpm, duty, flags)
             + b'TH\x01\x05')
    return frame + binascii.crc_hqx(frame, 0xffff).to_bytes(2, 'big')


class ThermalProtocolTests(unittest.TestCase):
    def test_measurement_and_validity_fields(self):
        values = decode_thermal_frame(thermal_frame())
        self.assertEqual(values['temperature_c'], 33.25)
        self.assertEqual(values['fan_rpm'], 2340)
        self.assertEqual(values['fan_duty_permille'], 210)
        self.assertEqual(values['thermal_flags'], 17)
        self.assertEqual(values['sensor_age_ms'], 50)

    def test_signed_temperature_manual_mode(self):
        self.assertEqual(decode_thermal_frame(thermal_frame(state=1, temperature=-5.25))['temperature_c'], -5.25)

    def test_zero_rpm_not_replaced_by_pwm(self):
        self.assertEqual(decode_thermal_frame(thermal_frame(rpm=0, duty=1000))['fan_rpm'], 0)

    def test_reject_corrupt_crc_old_monitor_and_impossible_pwm(self):
        frame = thermal_frame()
        self.assertIsNone(decode_thermal_frame(frame[:-1]+bytes((frame[-1]^1,))))
        self.assertIsNone(decode_thermal_frame(b'\xff\xff\x02\x01'+bytes(16)))
        self.assertIsNone(decode_thermal_frame(thermal_frame(duty=1001)))

    def test_all_fragment_boundaries_and_mixed_payloads(self):
        first, second = thermal_frame(), thermal_frame(state=1)
        mixed = b'noise\xff\xff\x02\xd6' + bytes(120) + first + b'T45\xff' + second
        for split in range(len(mixed)+1):
            decoder = ThermalStreamDecoder()
            result = decoder.feed(mixed[:split]) + decoder.feed(mixed[split:])
            self.assertEqual([v['work_state'] for v in result], [2,1])
            self.assertLess(len(decoder.buffer), 20)

    def test_observer_reset_and_exception_do_not_affect_reader(self):
        result=[]
        observer=ThermalReadObserver(result.append)
        observer.feed(thermal_frame()[:10]); observer.reset()
        observer.feed(thermal_frame()[10:])
        self.assertEqual(result, [])
        observer.feed(thermal_frame())
        self.assertIn('received_monotonic', result[0])
        observer.callback=lambda _: (_ for _ in ()).throw(ValueError('test'))
        with self.assertLogs('board_thermal_protocol', level='ERROR'):
            observer.feed(thermal_frame())

    def test_usb_observer_preserves_bytes_and_issues_no_commands(self):
        port=ThermalSerial()
        result=[]; port.thermal_observer.callback=result.append
        payload=b'raw_adc' + thermal_frame() + b'raw_tail'
        with patch('serial.Serial.read', return_value=payload) as read, patch('serial.Serial.write') as write:
            self.assertEqual(port.read(200),payload)
            read.assert_called_once_with(200); write.assert_not_called()
        self.assertEqual(len(result),1)
        port.close()
