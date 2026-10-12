import struct
import unittest

from guarded_initial_ota_install import (
    BOOT_START,
    validate_bootloader_image,
)


def _valid_bootloader(length=1400):
    image = bytearray(b"\xff" * length)
    struct.pack_into("<II", image, 0, 0x20001000, BOOT_START + 0x101)
    return bytes(image)


class GuardedInitialOtaInstallTests(unittest.TestCase):
    def test_valid_bootloader_vector_is_accepted(self):
        stack, reset = validate_bootloader_image(_valid_bootloader())
        self.assertEqual(stack, 0x20001000)
        self.assertEqual(reset, BOOT_START + 0x101)

    def test_invalid_vector_or_size_is_rejected(self):
        cases = [bytearray(b"\xff" * 7), bytearray(_valid_bootloader())]
        struct.pack_into("<I", cases[1], 0, 0x10000000)
        missing_thumb = bytearray(_valid_bootloader())
        struct.pack_into("<I", missing_thumb, 4, BOOT_START + 0x100)
        outside_image = bytearray(_valid_bootloader())
        struct.pack_into("<I", outside_image, 4, BOOT_START + len(outside_image) + 1)
        cases.extend((missing_thumb, outside_image))
        for image in cases:
            with self.subTest(length=len(image)):
                with self.assertRaises(ValueError):
                    validate_bootloader_image(bytes(image))


if __name__ == "__main__":
    unittest.main()
