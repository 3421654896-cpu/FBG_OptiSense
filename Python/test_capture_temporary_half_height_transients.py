import binascii
from pathlib import Path
import unittest

from capture_temporary_half_height_transients import (
    CUSTOM_RESPONSE_LENGTH,
    custom_command,
    decode_custom,
    main_edge_metric,
)


def row(index, codes=None):
    return {
        "index": index,
        "codes": codes or [63351, 63351, 32767, 24575, 24575],
    }


def response(source, target, tag):
    packet = bytearray(b"\xd5\x5d\x03\x05")
    packet.extend(source["index"].to_bytes(2, "big"))
    packet.extend(target["index"].to_bytes(2, "big"))
    packet.extend((387).to_bytes(4, "big"))
    packet.extend(bytes((33, 6)))
    packet.extend(tag.to_bytes(4, "big"))
    packet.extend(b"\x00\x64" * 6)
    for sample in range(33):
        packet.extend((388 + sample * 25).to_bytes(4, "big"))
        packet.extend(b"\x00\x65" * 6)
    packet.extend(b"\x5d\xd5")
    assert len(packet) == CUSTOM_RESPONSE_LENGTH
    return bytes(packet)


class TemporaryHalfHeightTransientTests(unittest.TestCase):
    def test_request_binds_two_safe_rows_with_crc(self):
        source, target = row(119), row(126, [60000, 61000, 30000, 20000, 21000])
        packet = custom_command(source, target, 9)
        self.assertEqual(packet[:4], b"\xff\xff\x03\x01")
        self.assertEqual(packet[8], 5)
        self.assertEqual(packet[13:16], b"C2P")
        self.assertEqual(
            int.from_bytes(packet[36:40], "big"),
            binascii.crc32(packet[:36]),
        )
        self.assertFalse(any(packet[40:]))

    def test_request_rejects_current_above_temporary_ceiling(self):
        with self.assertRaisesRegex(ValueError, "145 mA"):
            custom_command(row(119, [63352, 1, 1, 1, 1]), row(126), 9)

    def test_tagged_board_timestamp_response(self):
        source, target = row(119), row(126)
        decoded = decode_custom(response(source, target, 9), source, target, 9)
        self.assertEqual(decoded["path_update_us"], 387)
        self.assertEqual(decoded["sample_start_after_target_write_us"][0], 1)
        self.assertEqual(len(decoded["direct_adc"]), 33)

    def test_firmware_custom_branch_is_crc_bound_and_tableless(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "ms5614t.c"
        ).read_text(encoding="utf-8")
        method = source.split("void runLaserSwitchTest(void)", 1)[1].split(
            "static void sendStressRawCapture", 1
        )[0]
        self.assertIn("aRxBuffer[8] != 5U", method)
        self.assertIn("CandidateRoute_Crc(aRxBuffer, 36U)", method)
        self.assertIn("customLimits[5]", method)
        self.assertIn("PI11210_ApplyCalibrationCodes(targetCodes)", method)
        self.assertNotIn("memcpy(Stress_Wave_DAC", method)

    def test_main_edge_gate_includes_dac_update_but_not_long_tail_settling(self):
        times = [1, 56, 111, 167, 222, 277, 332, 387, 443, 500,
                 650, 800, 1000, 1250, 1500, 2000, 2500, 3000,
                 4000, 5000, 6000, 8000, 10000, 12000, 16000,
                 20000, 25000, 30000, 40000, 50000, 60000, 80000,
                 100000]
        # CH1 crosses 90 % before 2 ms but keeps drifting afterwards.  The new
        # route rule intentionally accepts the completed main edge.
        ch1 = [0, 5, 15, 35, 60, 80, 92, 96, 98, 99] + [100] * 20 + [104, 108, 110]
        record = {
            "path_update_us": 484,
            "baseline_adc": [0, 0, 0, 0, 0, 0],
            "sample_start_after_target_write_us": times,
            "direct_adc": [[0, value, 0, 0, 0, 0] for value in ch1],
        }
        metric = main_edge_metric([record])
        self.assertTrue(metric["passed"])
        self.assertEqual(metric["classification"], "main_edge_complete")
        self.assertLessEqual(metric["main_edge_complete_from_switch_begin_us"], 3000)

    def test_unresolved_small_step_reports_detour_without_a_fixed_code_gate(self):
        times = list(range(0, 3300, 100))
        values = [0] * 4 + [35] + [10] * 28
        record = {
            "path_update_us": 484,
            "baseline_adc": [0, 0, 0, 0, 0, 0],
            "sample_start_after_target_write_us": times,
            "direct_adc": [[0, value, 0, 0, 0, 0] for value in values],
        }
        metric = main_edge_metric([record])
        self.assertTrue(metric["passed"])
        self.assertEqual(metric["classification"], "no_resolvable_edge")
        self.assertEqual(metric["early_excursion_codes"], 35)


if __name__ == "__main__":
    unittest.main()
