import binascii
import unittest
import temporary_test_mode as mode
from test_temporary_test_mode import reference, frame, seal


class PointWaitTests(unittest.TestCase):
    def setUp(self):
        table, x, y = reference()
        self.rows = mode.select_points(table, x, y)['rows']

    def test_legacy_packet_unchanged_and_extension_bound(self):
        old = mode.command(self.rows, 64, 123)
        waits = [650]*45
        waits[22] = 4000
        new = mode.command(self.rows, 64, 123, point_delays_us=waits)
        self.assertEqual(old[:568], new[:568])
        self.assertEqual(new[568:572], b'PWT1')
        self.assertEqual(int.from_bytes(new[616:618], 'big'), 4000)
        self.assertEqual(int.from_bytes(new[662:666], 'big'), binascii.crc32(new[:662]))
        self.assertEqual(new[666:], bytes(142))

    def test_invalid_waits_rejected(self):
        for waits in ([650]*44, [49]+[650]*44,
                      [15025]+[650]*44, [15001]+[650]*44, [651]+[650]*44, [True]+[650]*44):
            with self.assertRaises(ValueError):
                mode.command(self.rows, 64, 123, point_delays_us=waits)
        with self.assertRaises(ValueError):
            mode.command(self.rows, 64, 123, boundary_extra_us=25, point_delays_us=[650]*45)

    def test_decode_checks_each_actual_wait(self):
        mode.decode(frame(), 123, 456, point_delays_us=[600]*45)
        waits = [600]*45
        waits[22] = 4000
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            mode.decode(frame(), 123, 456, point_delays_us=waits)

    def test_long_wait_has_no_total_budget_gate(self):
        mode.command(self.rows, 512, 123, point_delays_us=[5000]*45)
        packet = mode.command(self.rows, 512, 123, point_delays_us=[15000]*45)
        self.assertEqual([int.from_bytes(packet[i:i+2], 'big') for i in range(572,662,2)], [15000]*45)
        mode.command(self.rows, 32, 123, first_delay_us=850, boundary_extra_us=3000)

    def test_wide_timestamps_beyond_65535_and_fragmented_transport(self):
        wire = bytearray(631)
        wire[:85] = frame()[:85]
        wire[3] |= 2
        wire[20:24] = (270000).to_bytes(4, 'big')
        for i in range(45):
            at = 85+i*12
            wire[at:at+2] = (100).to_bytes(2, 'big')
            wire[at+2:at+4] = (102).to_bytes(2, 'big')
            wire[at+4:at+8] = (i*6000+100).to_bytes(4, 'big')
            wire[at+8:at+12] = (i*6000+5300).to_bytes(4, 'big')
        wire = seal(wire)
        buffer = bytearray(wire[:450])
        self.assertEqual(list(mode.extract_frames(buffer)), [])
        buffer.extend(wire[450:])
        self.assertEqual(list(mode.extract_frames(buffer)), [wire])
        result = mode.decode(wire, 123, 456, point_delays_us=[5000]*45)
        self.assertEqual(result['records'][-1]['second_end_us'], 269300)
        broken = bytearray(wire)
        broken[3] &= ~2
        with self.assertRaises(ValueError):
            mode.decode(seal(broken), 123, 456, point_delays_us=[5000]*45)


if __name__ == '__main__':
    unittest.main()

