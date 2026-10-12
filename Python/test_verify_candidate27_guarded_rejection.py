import unittest

from verify_candidate27_guarded_rejection import malformed


class GuardedRejectionTests(unittest.TestCase):
    def test_every_malformed_request_has_only_zero_current_codes(self):
        for name in ('bad_crc', 'guard_not_between', 'nonzero_tail',
                     'short_guard_hold', 'nonzero_v1_crc_slot'):
            packet = malformed(name, 123)
            self.assertEqual(len(packet), 808)
            self.assertTrue(all(packet[26 + local * 12:36 + local * 12] == bytes(10)
                                for local in range(27)))
            self.assertEqual(packet[354:364], bytes(10))


if __name__ == '__main__':
    unittest.main()
