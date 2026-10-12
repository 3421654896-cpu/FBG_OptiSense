import unittest
from unittest.mock import patch

from beta_wire_trace import TraceSerial
from thermal_serial import ThermalSerial


class TraceTests(unittest.TestCase):
    def device(self):
        device = object.__new__(TraceSerial)
        device.trace = []
        device.raw = bytearray()
        device.is_open = False
        return device

    def test_reads_are_returned_unchanged_and_offsets_are_contiguous(self):
        device = self.device()
        with patch.object(ThermalSerial, 'read', side_effect=[b'abc', b'', b'de']) as read:
            self.assertEqual(device.read(20), b'abc')
            self.assertEqual(device.read(20), b'')
            self.assertEqual(device.read(20), b'de')
        self.assertEqual(read.call_count, 3)
        self.assertEqual(device.raw, b'abcde')
        self.assertEqual([(x['offset'], x['count']) for x in device.trace], [(0, 3), (3, 2)])

    def test_partial_write_return_is_not_changed(self):
        device = self.device()
        with patch.object(ThermalSerial, 'write', return_value=2) as write:
            self.assertEqual(device.write(b'abcd'), 2)
        write.assert_called_once_with(b'abcd')
        self.assertEqual(device.trace[0]['returned'], 2)
        self.assertEqual(device.trace[0]['count'], 4)
        self.assertEqual(device.trace[0]['header_hex'], '61626364')
        self.assertEqual(device.raw, b'')

    def test_reset_does_not_erase_diagnostic_history(self):
        device = self.device()
        device.raw.extend(b'previous response')
        with patch.object(ThermalSerial, 'reset_input_buffer') as reset:
            device.reset_input_buffer()
        reset.assert_called_once_with()
        self.assertEqual(device.raw, b'previous response')
        self.assertEqual(device.trace[0]['action'], 'reset_input_buffer')


if __name__ == '__main__':
    unittest.main()
