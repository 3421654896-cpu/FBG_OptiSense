"""Tests for per-machine OTA settings and Windows credential protection."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ota_machine_profiles import (
    OtaMachineProfileStore,
    protect_secret,
    unprotect_secret,
)


def _protect_for_test(value: str) -> str:
    return base64.b64encode(("protected:" + value).encode("utf-8")).decode("ascii")


def _unprotect_for_test(value: str) -> str:
    raw = base64.b64decode(value).decode("utf-8")
    if not raw.startswith("protected:"):
        raise ValueError("bad test credential")
    return raw.removeprefix("protected:")


class OtaMachineProfileStoreTests(unittest.TestCase):
    def test_missing_saved_package_uses_newest_bundled_firmware(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            firmware = root / "firmware"
            firmware.mkdir()
            older = firmware / "JDSU_F205RE_v1.0.86.fbgfw"
            newest = firmware / "JDSU_F205RE_v1.0.100.fbgfw"
            older.write_bytes(b"older")
            newest.write_bytes(b"newest")
            with mock.patch(
                "ota_machine_profiles.resource_path",
                side_effect=lambda *parts: root.joinpath(*parts),
            ):
                store = OtaMachineProfileStore(
                    root / "ota_profiles.json",
                    protector=_protect_for_test,
                    unprotector=_unprotect_for_test,
                )
                profile = store.load_profile("machine_2")
            self.assertEqual(Path(profile["package_path"]), newest)

    def test_machine_settings_are_independent_and_selection_persists(self):
        # Package fallback has its own test. Keep this isolation check
        # independent of compatibility firmware present in a source clone.
        with tempfile.TemporaryDirectory() as folder, mock.patch(
            "ota_machine_profiles._bundled_firmware_package", return_value=None
        ):
            path = Path(folder) / "ota_profiles.json"
            store = OtaMachineProfileStore(
                path,
                protector=_protect_for_test,
                unprotector=_unprotect_for_test,
            )
            store.save_profile(
                "machine_1",
                board_ip="192.168.3.46",
                password="first-secret",
                package_path="one.fbgfw",
                mqtt_device_id="fbg-ONE",
            )
            store.save_profile(
                "machine_2",
                board_ip="192.168.3.47",
                password="second-secret",
                package_path="two.fbgfw",
            )
            restored = OtaMachineProfileStore(
                path,
                protector=_protect_for_test,
                unprotector=_unprotect_for_test,
            )
            self.assertEqual(restored.selected_machine_id, "machine_2")
            first = restored.load_profile("machine_1")
            second = restored.load_profile("machine_2")
            self.assertEqual(first["board_ip"], "192.168.3.46")
            self.assertEqual(first["password"], "first-secret")
            self.assertEqual(first["package_path"], "one.fbgfw")
            self.assertEqual(first["mqtt_device_id"], "fbg-ONE")
            self.assertEqual(second["board_ip"], "192.168.3.47")
            self.assertEqual(second["password"], "second-secret")
            self.assertEqual(second["package_path"], "two.fbgfw")
            self.assertEqual(restored.lan_board_ip("machine_1"), "192.168.3.46")
            self.assertEqual(restored.lan_board_ip("machine_2"), "192.168.3.47")

    def test_lan_target_fails_closed_and_never_needs_ota_password(self):
        with tempfile.TemporaryDirectory() as folder:
            store = OtaMachineProfileStore(
                Path(folder) / "ota_profiles.json",
                protector=_protect_for_test,
                unprotector=lambda _value: (_ for _ in ()).throw(
                    AssertionError("LAN target lookup must not decrypt credentials")
                ),
            )
            with self.assertRaisesRegex(ValueError, "板卡 IP"):
                store.lan_board_ip("machine_3")
            store.save_profile(
                "machine_3", board_ip="192.168.4.62",
                password="secret", package_path="firmware.fbgfw",
            )
            self.assertEqual(store.lan_board_ip("machine_3"), "192.168.4.62")
            store.save_profile(
                "machine_3", board_ip="not-an-ip",
                password="", package_path="firmware.fbgfw",
            )
            with self.assertRaisesRegex(ValueError, "板卡 IP"):
                store.lan_board_ip("machine_3")

    def test_password_is_never_written_as_plaintext(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ota_profiles.json"
            store = OtaMachineProfileStore(
                path,
                protector=_protect_for_test,
                unprotector=_unprotect_for_test,
            )
            store.save_profile(
                "machine_1",
                board_ip="192.168.3.46",
                password="plain-text-must-not-appear",
                package_path="firmware.fbgfw",
            )
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("plain-text-must-not-appear", raw)
            payload = json.loads(raw)
            self.assertIn(
                "password_dpapi", payload["machines"]["machine_1"]
            )

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI only")
    def test_windows_dpapi_round_trip(self):
        encrypted = protect_secret("FBG-OTA-一号机")
        self.assertNotIn("FBG-OTA", encrypted)
        self.assertEqual(unprotect_secret(encrypted), "FBG-OTA-一号机")


if __name__ == "__main__":
    unittest.main()
