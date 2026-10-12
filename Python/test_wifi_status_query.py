from __future__ import annotations

import unittest

import wifi_status_query as status


class WifiStatusQueryTests(unittest.TestCase):
    def test_command_is_read_only_fixed_frame(self):
        command = status.build_status_query()
        self.assertEqual(len(command), 808)
        self.assertEqual(command[:4], b"\xff\xff\x01\x21")
        self.assertEqual(command[4:], bytes(804))

    def test_decode_reply(self):
        frame = bytearray(20)
        frame[:5] = b"\xff\xff\x02\x21\x01"
        frame[5:13] = bytes((5, 1, 1, 0, 11, 1, 5, 0x35))
        frame[13] = 7
        frame[14:16] = (31).to_bytes(2, "big")
        frame[16:18] = (92).to_bytes(2, "big")
        frame[18:] = b"\xff\xef"
        decoded = status.find_status_reply(b"junk" + frame)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["last_error"], 11)
        self.assertEqual(decoded["raw_state"], 1)
        self.assertEqual(decoded["retry_exponent"], 5)
        self.assertTrue(decoded["wifi_connected"])
        self.assertFalse(decoded["mqtt_connected"])
        self.assertTrue(decoded["uart_tx_busy"])
        self.assertTrue(decoded["uart_error_pending"])
        self.assertTrue(decoded["lan_client_active"])
        self.assertEqual(decoded["uart_tx_complete_count"], 31)
        self.assertEqual(decoded["uart_rx_byte_count"], 92)

    def test_rejects_bad_trailer(self):
        frame = bytearray(20)
        frame[:5] = b"\xff\xff\x02\x21\x01"
        self.assertIsNone(status.find_status_reply(frame))


if __name__ == "__main__":
    unittest.main()
