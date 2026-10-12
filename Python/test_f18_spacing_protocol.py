import unittest
from capture_fast_flank_stream import command as legacy_command
from f18_spacing_protocol import command, decode_data, decode_end, FIRMWARE
from test_capture_fast_flank_stream import data_frame, end_frame, _finish_crc


class SpacingProtocolTests(unittest.TestCase):
    def test_explicit_response_identity_required(self):
        raw=bytearray(data_frame()); raw[8:12]=FIRMWARE.to_bytes(4,'big'); raw[31]=2
        wire=_finish_crc(raw)
        decode_data(wire,tag=123,table_crc32=456,spacing_us=50)
        with self.assertRaises(ValueError): decode_data(wire,tag=123,table_crc32=456,spacing_us=600)
        old_data=bytearray(data_frame()); old_data[8:12]=(0x10036).to_bytes(4,'big'); old_data[31]=2
        with self.assertRaises(ValueError): decode_data(_finish_crc(old_data),tag=123,table_crc32=456,spacing_us=50)
        terminal=bytearray(end_frame()); terminal[8:12]=FIRMWARE.to_bytes(4,'big')
        decode_end(_finish_crc(terminal),tag=123,table_crc32=456)
        old_terminal=bytearray(end_frame()); old_terminal[8:12]=(0x10036).to_bytes(4,'big')
        with self.assertRaises(ValueError): decode_end(_finish_crc(old_terminal),tag=123,table_crc32=456)

    def test_only_explicit_spacing_bytes_change(self):
        original=legacy_command(256,123)
        for spacing in range(50,601,25):
            packet=command(256,123,spacing)
            self.assertEqual(packet[:19],original[:19])
            self.assertEqual(packet[22:],original[22:])
            self.assertEqual(packet[19],0x53)
            self.assertEqual(int.from_bytes(packet[20:22],'big'),spacing)
        self.assertEqual(original[19:],bytes(808-19))

    def test_invalid_requests_rejected(self):
        for value in (0,25,49,51,601,625,50.0,True):
            with self.assertRaises(ValueError): command(256,123,value)
        with self.assertRaises(ValueError): command(0,123,50)
        with self.assertRaises(ValueError): command(256,0,50)
        self.assertEqual(int.from_bytes(command(2048,123,50)[16:18],'big'),2048)
        with self.assertRaises(ValueError): command(2049,123,50)


if __name__=='__main__': unittest.main()
