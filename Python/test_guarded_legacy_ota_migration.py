import struct
import unittest

from guarded_legacy_ota_migration import vector_is_valid


class LegacyVectorTests(unittest.TestCase):
    def test_valid_legacy_vector(self):
        image = bytearray(b"\xff" * 32)
        struct.pack_into("<II", image, 0, 0x2000FE78, 0x0800024D)
        self.assertTrue(vector_is_valid(bytes(image), 0x08000000, 0x08040000))

    def test_rejects_invalid_stack_or_reset(self):
        cases = (
            (0x10000000, 0x0800024D),
            (0x2000FE78, 0x0800024C),
            (0x2000FE78, 0x08050001),
        )
        for stack, reset in cases:
            with self.subTest(stack=stack, reset=reset):
                image = bytearray(b"\xff" * 32)
                struct.pack_into("<II", image, 0, stack, reset)
                self.assertFalse(
                    vector_is_valid(bytes(image), 0x08000000, 0x08040000)
                )


if __name__ == "__main__":
    unittest.main()
