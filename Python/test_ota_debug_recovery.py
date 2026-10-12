from __future__ import annotations

import binascii
import struct
import unittest

from ota_debug_recovery import verify_halted_ota_images


class GuardedDebugRecoveryTests(unittest.TestCase):
    def make_package(self):
        image = b"verified-image"
        return type(
            "Package",
            (),
            {
                "firmware_version": 0x00010056,
                "image_size": len(image),
                "image_crc32": binascii.crc32(image) & 0xFFFFFFFF,
                "link_address": 0x08008000,
                "image": image,
            },
        )()

    def metadata(self, package, *, installed: bool) -> bytes:
        words = [0xFFFFFFFF] * 16
        words[0] = 0 if installed else 0x4F544131
        words[1] = 1
        words[2] = 0xF2050052
        words[3] = package.firmware_version
        words[4] = package.image_size
        words[5] = package.image_crc32
        words[6] = package.link_address
        words[7] = 0x08040000
        words[8] = 0x12345678
        words[14] = 0 if installed else 0xFFFFFFFF
        raw = bytearray(struct.pack("<16I", *words))
        normalized = bytearray(raw)
        normalized[56:60] = b"\xff\xff\xff\xff"
        words[15] = binascii.crc32(normalized[4:60]) & 0xFFFFFFFF
        return struct.pack("<16I", *words)

    def test_pending_package_does_not_require_old_application_to_match(self):
        package = self.make_package()
        state = verify_halted_ota_images(
            package,
            self.metadata(package, installed=False),
            package.image,
            b"old-application",
            flash_status=0,
            flash_control=0x80000000,
        )
        self.assertEqual(state.phase, "pending")
        self.assertEqual(state.boot_token, 0x12345678)

    def test_installed_package_requires_application_match(self):
        package = self.make_package()
        with self.assertRaisesRegex(RuntimeError, "应用区"):
            verify_halted_ota_images(
                package,
                self.metadata(package, installed=True),
                package.image,
                b"old-application",
                flash_status=0,
                flash_control=0x80000000,
            )

    def test_unlocked_flash_is_rejected(self):
        package = self.make_package()
        with self.assertRaisesRegex(RuntimeError, "Flash"):
            verify_halted_ota_images(
                package,
                self.metadata(package, installed=False),
                package.image,
                b"old-application",
                flash_status=0,
                flash_control=0,
            )


if __name__ == "__main__":
    unittest.main()
