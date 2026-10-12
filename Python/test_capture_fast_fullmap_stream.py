import binascii
from pathlib import Path
import unittest

import capture_fast_fullmap_stream as stream


def _finish_crc(frame: bytearray):
    frame[-6:-2] = binascii.crc32(frame[:-6]).to_bytes(4, "big")
    frame[-2:] = b"\x9d\xd9"
    return bytes(frame)


def data_frame(sequence=0, tag=123, table_crc=456, start_us=0, interval_us=32_500, spacing_us=75, lan_compact=False):
    frame = bytearray(stream.DATA_LENGTH)
    flags = int(sequence >= stream.WARMUP_CYCLES) | (2 if lan_compact else 0)
    frame[:4] = bytes((0xD9, 0x9D, 3, flags))
    frame[4:8] = tag.to_bytes(4, "big")
    frame[8:12] = stream.FIRMWARE.to_bytes(4, "big")
    frame[12:16] = sequence.to_bytes(4, "big")
    frame[16:20] = start_us.to_bytes(4, "big")
    frame[20:24] = interval_us.to_bytes(4, "big")
    frame[24:28] = table_crc.to_bytes(4, "big")
    frame[28:34] = bytes((1, 45, 16, spacing_us // 25, 0, 2))
    rows_end = 40 + len(stream.ROWS)
    frame[40:rows_end] = bytes(stream.ROWS)
    for local in range(len(stream.ROWS)):
        at = rows_end + local * 8
        frame[at : at + 2] = (100 + local).to_bytes(2, "big")
        frame[at + 2 : at + 4] = (110 + local).to_bytes(2, "big")
        frame[at + 4 : at + 6] = (local * 700 + 100).to_bytes(2, "big")
        frame[at + 6 : at + 8] = (local * 700 + 500).to_bytes(2, "big")
    return _finish_crc(frame)


class FastFullMapStreamTests(unittest.TestCase):
    def test_explicit_gain_requires_matching_frame_provenance(self):
        for selector in (0,1,3):
            wire = stream.command(32,123,75,ch1_selector=selector)
            self.assertEqual(wire[21:23], bytes((0x47,selector)))
            frame = bytearray(data_frame(sequence=16))
            frame[33] = selector
            frame = _finish_crc(frame)
            with self.assertRaisesRegex(ValueError, 'feedback'):
                stream.decode_data(frame,tag=123,table_crc32=456,spacing_us=75)
            decoded = stream.decode_data(frame,tag=123,table_crc32=456,
                                         spacing_us=75,ch1_selector=selector)
            self.assertEqual(len(decoded['records']),45)
        for invalid in (-1,4,True):
            with self.assertRaises(ValueError):
                stream.command(32,123,75,ch1_selector=invalid)

    def test_invalid_duration_cap_rejected_before_device_read(self):
        for duration in (0, 4.9, 241, float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, 'duration cap'):
                stream.receive(None, cycles=4096, tag=1, table_crc32=1,
                               spacing_us=50, max_duration_s=duration)

    def test_exact_firmware_identity_tracks_source_and_rejects_other_versions(self):
        import re
        source = (Path(__file__).resolve().parents[1] / 'JDSU/Core/Inc/firmware_version.h').read_text('utf-8')
        value = re.search(r'APP_FIRMWARE_VERSION\s+(0x[0-9A-Fa-f]+)', source)
        self.assertIsNotNone(value)
        self.assertEqual(stream.FIRMWARE, int(value.group(1), 16))
        for version in (stream.FIRMWARE-1, stream.FIRMWARE+1):
            frame = bytearray(data_frame())
            frame[8:12] = version.to_bytes(4, 'big')
            with self.assertRaisesRegex(ValueError, 'firmware'):
                stream.decode_data(_finish_crc(frame), tag=123, table_crc32=456, spacing_us=75)

    def test_command_is_separate_bounded_and_self_describing(self):
        wire = stream.command(640, 0x12345678, 75)
        self.assertEqual(len(wire), 808)
        self.assertEqual(wire[:4], b"\xff\xff\x03\x09")
        self.assertEqual(wire[6:8], (5000).to_bytes(2, "big"))
        self.assertEqual(wire[12:16], b"F45!")
        self.assertEqual(wire[19:21], (75).to_bytes(2, "big"))
        self.assertFalse(any(wire[21:]))

    def test_spacing_validation(self):
        for value in (49, 74, 601):
            with self.assertRaises(ValueError):
                stream.command(32, 1, value)
        for value in (50, 75, 100, 600):
            self.assertEqual(len(stream.command(32, 1, value)), 808)

    def test_45_fresh_records_decode_in_installed_order(self):
        decoded = stream.decode_data(
            data_frame(sequence=16), tag=123, table_crc32=456, spacing_us=75
        )
        self.assertEqual(len(decoded["records"]), 45)
        self.assertEqual(tuple(item["row"] for item in decoded["records"]), stream.ROWS)
        self.assertTrue(decoded["minimum_warmup_complete"])
        self.assertFalse(decoded["lan_compact_first_only"])
        self.assertTrue(decoded["second_code_is_physical_sample"])

    def test_lan_compact_provenance_is_explicit(self):
        decoded = stream.decode_data(
            data_frame(sequence=16, lan_compact=True),
            tag=123,
            table_crc32=456,
            spacing_us=75,
        )
        self.assertTrue(decoded["lan_compact_first_only"])
        self.assertFalse(decoded["second_code_is_physical_sample"])
        self.assertFalse(decoded["point_timestamps_are_physical"])

    def test_spacing_provenance_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "spacing"):
            stream.decode_data(
                data_frame(sequence=16, spacing_us=100),
                tag=123,
                table_crc32=456,
                spacing_us=75,
            )

    def test_sampling_gate_uses_slowest_board_interval(self):
        frames = [
            stream.decode_data(
                data_frame(sequence=i, start_us=i * 32_500),
                tag=123,
                table_crc32=456,
                spacing_us=75,
            )
            for i in range(32)
        ]
        result = stream.analyze(frames)
        self.assertTrue(result["meets_sampling_rate_for_15hz"])
        self.assertEqual(result["fresh_point_count_per_cycle"], 45)
        frames[-1]["cycle_start_us"] += 1_000
        self.assertFalse(stream.analyze(frames)["meets_sampling_rate_for_15hz"])

    def test_firmware_preserves_full_map_path_contract(self):
        source = (Path(__file__).parent.parent / "JDSU" / "Core" / "Src" / "ms5614t.c").read_text(encoding="utf-8")
        self.assertIn('memcmp(&aRxBuffer[12], "F45!", 4U)', source)
        self.assertIn("FAST_FULLMAP_STREAM_POINTS STRESS_TABLE_POINT_COUNT", source)
        self.assertIn("STRESS_PATH_PRECONDITION_POINT_INDEX", source)
        self.assertIn("isStressSectionBoundary(row)", source)
        self.assertIn("aRxBuffer[3] == 0x09U", source)

    def test_failed_batch_experiment_is_not_reachable_in_production(self):
        source = (Path(__file__).parent.parent / "JDSU" / "Core" / "Src" / "ms5614t.c").read_text(encoding="utf-8")
        self.assertNotIn('"B45!"', source)
        self.assertNotIn("PI11210_WriteSequentialCodesRaw", source)
        self.assertNotIn("PI11210_ApplyFastBatchCalibrationCodes", source)
        self.assertNotIn("aRxBuffer[3] == 0x0EU", source)


if __name__ == "__main__":
    unittest.main()
