"""Safety regression tests for direct PI11210 host commands."""

from __future__ import annotations

import unittest

from laser_dac_safety import (
    FULLBAND_2001_CODE_LIMITS,
    PI11210_SAFE_CODE_LIMITS,
    validate_pi11210_codes,
)
from laser_power_calibration import LaserSerial
from validate_rc_against_single import COMMAND_SIZE, dac_command


class LaserDacSafetyTests(unittest.TestCase):
    def test_exact_authorized_limits_are_accepted(self) -> None:
        self.assertEqual(
            validate_pi11210_codes(PI11210_SAFE_CODE_LIMITS),
            PI11210_SAFE_CODE_LIMITS,
        )

    def test_malformed_or_unsafe_codes_are_rejected(self) -> None:
        invalid_rows = (
            [0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [PI11210_SAFE_CODE_LIMITS[0] + 1, 0, 0, 0, 0],
            [0, PI11210_SAFE_CODE_LIMITS[1] + 1, 0, 0, 0],
            [0, 0, PI11210_SAFE_CODE_LIMITS[2] + 1, 0, 0],
            [0, 0, 0, PI11210_SAFE_CODE_LIMITS[3] + 1, 0],
            [0, 0, 0, 0, PI11210_SAFE_CODE_LIMITS[4] + 1],
            [0.5, 0, 0, 0, 0],
            [float("nan"), 0, 0, 0, 0],
            [True, 0, 0, 0, 0],
        )
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_pi11210_codes(row)

    def test_direct_command_preserves_exact_big_endian_codes(self) -> None:
        codes = [1, 0x1234, 0x5678, 0x2345, 0x3456]
        packet = dac_command(codes)
        self.assertEqual(len(packet), COMMAND_SIZE)
        self.assertEqual(packet[:4], b"\xFF\xFF\x00\x01")
        self.assertEqual(packet[4:14], b"\x00\x01\x12\x34\x56\x78\x23\x45\x34\x56")

    def test_direct_command_reuses_same_safety_gate(self) -> None:
        with self.assertRaises(ValueError):
            dac_command([PI11210_SAFE_CODE_LIMITS[0] + 1, 0, 0, 0, 0])

    def test_calibration_sender_reuses_same_safety_gate(self) -> None:
        self.assertEqual(
            LaserSerial.validate_codes(list(PI11210_SAFE_CODE_LIMITS)),
            PI11210_SAFE_CODE_LIMITS,
        )
        for unsafe in (
            [-1, 0, 0, 0, 0],
            [0.5, 0, 0, 0, 0],
            [PI11210_SAFE_CODE_LIMITS[0] + 1, 0, 0, 0, 0],
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                LaserSerial.validate_codes(unsafe)

    def test_140_ma_profile_is_explicit_and_does_not_relax_default(self) -> None:
        codes = list(FULLBAND_2001_CODE_LIMITS)
        self.assertEqual(
            validate_pi11210_codes(codes, FULLBAND_2001_CODE_LIMITS),
            FULLBAND_2001_CODE_LIMITS,
        )
        with self.assertRaises(ValueError):
            validate_pi11210_codes(codes)
        with self.assertRaises(ValueError):
            validate_pi11210_codes(
                [FULLBAND_2001_CODE_LIMITS[0] + 1, 0, 0, 0, 0],
                FULLBAND_2001_CODE_LIMITS,
            )


if __name__ == "__main__":
    unittest.main()
