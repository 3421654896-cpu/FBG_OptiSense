import tempfile
import unittest
from pathlib import Path

from fbg_ota_protocol import OTA_LINK_ADDRESS
from make_initial_ota_installer import (
    BOOTLOADER_START,
    merge,
    read_hex,
    write_hex,
)


def _put_u32(memory, address, value):
    for offset, byte in enumerate(int(value).to_bytes(4, "little")):
        memory[address + offset] = byte


def _image(start, reset_offset=0x100):
    memory = {}
    _put_u32(memory, start, 0x20001000)
    _put_u32(memory, start + 4, start + reset_offset + 1)
    memory[start + reset_offset] = 0x00
    memory[start + reset_offset + 1] = 0xBF
    return memory


class InitialOtaInstallerTests(unittest.TestCase):
    def test_safe_bootloader_and_application_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boot = root / "boot.hex"
            app = root / "app.hex"
            merged = root / "merged.hex"
            write_hex(_image(BOOTLOADER_START), boot)
            write_hex(_image(OTA_LINK_ADDRESS), app)
            merge(boot, app, merged)
            memory = read_hex(merged)
            self.assertIn(BOOTLOADER_START, memory)
            self.assertIn(OTA_LINK_ADDRESS, memory)

    def test_application_cannot_write_bootloader_region(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boot = root / "boot.hex"
            app = root / "app.hex"
            merged = root / "merged.hex"
            write_hex(_image(BOOTLOADER_START), boot)
            invalid = _image(OTA_LINK_ADDRESS)
            invalid[BOOTLOADER_START + 0x200] = 0xA5
            write_hex(invalid, app)
            with self.assertRaisesRegex(ValueError, "应用区"):
                merge(boot, app, merged)

    def test_missing_eof_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.hex"
            path.write_text(":020000040800F2\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "EOF"):
                read_hex(path)


if __name__ == "__main__":
    unittest.main()
