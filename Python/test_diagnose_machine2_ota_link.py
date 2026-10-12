"""Offline tests for the optional LAN-only OTA diagnostic resume journal."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import diagnose_machine2_ota_link as diagnostic


class ResumeJournalTests(unittest.TestCase):
    def test_journal_reuses_only_matching_active_session_and_clears_nonce(self):
        package = SimpleNamespace(
            firmware_version=0x0001005A,
            image_size=182600,
            image_crc32=0xADD078AF,
        )
        seal = lambda value: "sealed:" + value[::-1]
        unseal = lambda value: value.removeprefix("sealed:")[::-1]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            diagnostic, "protect_secret", side_effect=seal
        ), mock.patch.object(diagnostic, "unprotect_secret", side_effect=unseal):
            path = Path(directory) / "resume.json"
            first, reused = diagnostic.resume_journal_nonce(
                path, machine_id="machine_2", host="192.168.4.24", package=package
            )
            self.assertFalse(reused)
            self.assertRegex(first, r"^[0-9A-F]{16}$")
            self.assertNotIn(first, path.read_text(encoding="utf-8"))

            second, reused = diagnostic.resume_journal_nonce(
                path, machine_id="machine_2", host="192.168.4.24", package=package
            )
            self.assertTrue(reused)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                diagnostic.resume_journal_nonce(
                    path, machine_id="machine_2", host="192.168.4.23", package=package
                )

            diagnostic.complete_resume_journal(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "complete")
            self.assertNotIn("nonce_dpapi", record)
            with self.assertRaisesRegex(RuntimeError, "not active"):
                diagnostic.resume_journal_nonce(
                    path, machine_id="machine_2", host="192.168.4.24", package=package
                )


if __name__ == "__main__":
    unittest.main()
