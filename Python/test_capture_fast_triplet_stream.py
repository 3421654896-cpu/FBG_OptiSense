import binascii
from pathlib import Path
import unittest

import capture_fast_triplet_stream as stream


def _finish_crc(frame: bytearray):
    frame[-6:-2] = binascii.crc32(frame[:-6]).to_bytes(4, "big")
    frame[-2:] = b"\x9d\xd9"
    return bytes(frame)


def data_frame(sequence=0, tag=123, table_crc=456, start_us=0, interval_us=31_500):
    frame = bytearray(stream.DATA_LENGTH)
    frame[:4] = bytes((0xD9, 0x9D, 3, int(sequence >= stream.WARMUP_CYCLES)))
    frame[4:8] = tag.to_bytes(4, "big")
    frame[8:12] = stream.FIRMWARE.to_bytes(4, "big")
    frame[12:16] = sequence.to_bytes(4, "big")
    frame[16:20] = start_us.to_bytes(4, "big")
    frame[20:24] = interval_us.to_bytes(4, "big")
    frame[24:28] = table_crc.to_bytes(4, "big")
    frame[28:34] = bytes((1, len(stream.ROWS), stream.WARMUP_CYCLES, 0, 0, 2))
    rows_end = 40 + len(stream.ROWS)
    frame[40:rows_end] = bytes(stream.ROWS)
    for local in range(len(stream.ROWS)):
        at = rows_end + local * 8
        frame[at : at + 2] = (100 + local).to_bytes(2, "big")
        frame[at + 2 : at + 4] = (110 + local).to_bytes(2, "big")
        frame[at + 4 : at + 6] = (local * 1000 + 100).to_bytes(2, "big")
        frame[at + 6 : at + 8] = (local * 1000 + 900).to_bytes(2, "big")
    return _finish_crc(frame)


class FastTripletStreamTests(unittest.TestCase):
    def test_command_is_separate_and_bounded(self):
        wire = stream.command(640, 0x12345678)
        self.assertEqual(len(wire), 808)
        self.assertEqual(wire[:4], b"\xff\xff\x03\x08")
        self.assertEqual(wire[12:16], b"F27!")
        self.assertEqual(wire[18], 16)
        self.assertFalse(any(wire[19:]))

    def test_27_records_decode_in_declared_order(self):
        decoded = stream.decode_data(
            data_frame(sequence=16, start_us=504_000),
            tag=123,
            table_crc32=456,
        )
        self.assertEqual(len(decoded["records"]), 27)
        self.assertEqual(tuple(item["row"] for item in decoded["records"]), stream.ROWS)
        self.assertEqual(tuple(item["row"] for item in decoded["records"][21:24]), (38, 37, 36))

    def test_sampling_gate_uses_slowest_board_interval(self):
        frames = [
            stream.decode_data(
                data_frame(sequence=index, start_us=index * 31_500),
                tag=123,
                table_crc32=456,
            )
            for index in range(32)
        ]
        result = stream.analyze(frames)
        self.assertTrue(result["meets_sampling_rate_for_15hz"])
        self.assertEqual(result["point_count_per_cycle"], 27)
        self.assertEqual(result["samples_per_grating"], 3)
        frames[-1]["cycle_start_us"] += 2_000
        self.assertFalse(stream.analyze(frames)["meets_sampling_rate_for_15hz"])

    def test_firmware_route_is_independent_from_legacy_diagnostics(self):
        source = (Path(__file__).parent.parent / "JDSU" / "Core" / "Src" / "ms5614t.c").read_text(encoding="utf-8")
        self.assertIn('memcmp(&aRxBuffer[12], "F27!", 4U)', source)
        self.assertIn("FAST_TRIPLET_STREAM_POINTS 27U", source)
        self.assertIn("FAST_TRIPLET_STREAM_FRAME_LENGTH 291U", source)
        self.assertIn("fastTripletRows", source)
        self.assertIn("aRxBuffer[3] == 0x07U || aRxBuffer[3] == 0x08U", source)
        self.assertIn("ROUTE_TEST_POINTS 18U", source)


if __name__ == "__main__":
    unittest.main()
