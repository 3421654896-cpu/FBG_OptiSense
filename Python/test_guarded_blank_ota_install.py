import unittest

from guarded_blank_ota_install import (
    APPLICATION_START,
    BOOT_START,
    FLASH_END,
    build_expected_flash,
)


class GuardedBlankOtaInstallTests(unittest.TestCase):
    def test_expected_image_keeps_config_gap_erased(self):
        blank = b"\xff" * (FLASH_END - BOOT_START)
        bootloader = b"boot"
        application = b"application"
        result = build_expected_flash(blank, bootloader, application)
        self.assertEqual(result[: len(bootloader)], bootloader)
        app_offset = APPLICATION_START - BOOT_START
        self.assertEqual(result[app_offset : app_offset + len(application)], application)
        self.assertEqual(result[len(bootloader) : app_offset], b"\xff" * (app_offset - len(bootloader)))

    def test_nonblank_flash_is_rejected(self):
        blank = bytearray(b"\xff" * (FLASH_END - BOOT_START))
        blank[0x4000] = 0
        with self.assertRaisesRegex(ValueError, "not completely erased"):
            build_expected_flash(bytes(blank), b"boot", b"application")


if __name__ == "__main__":
    unittest.main()
