import binascii
from pathlib import Path
import unittest

import capture_fast_flank_stream as stream


def _finish_crc(frame: bytearray):
    frame[-6:-2] = binascii.crc32(frame[:-6]).to_bytes(4, "big")
    frame[-2:] = b"\x9d\xd9"
    return bytes(frame)


def data_frame(sequence=0, tag=123, table_crc=456, start_us=0):
    frame = bytearray(stream.DATA_LENGTH)
    frame[:4] = bytes((0xD9, 0x9D, 3, int(sequence >= stream.WARMUP_CYCLES)))
    frame[4:8] = tag.to_bytes(4, "big")
    frame[8:12] = stream.FIRMWARE.to_bytes(4, "big")
    frame[12:16] = sequence.to_bytes(4, "big")
    frame[16:20] = start_us.to_bytes(4, "big")
    frame[20:24] = (190).to_bytes(4, "big")
    frame[24:28] = table_crc.to_bytes(4, "big")
    frame[28:34] = bytes((1, len(stream.ROWS), stream.WARMUP_CYCLES, 0, 0, 2))
    frame[40:58] = bytes(stream.ROWS)
    for local in range(len(stream.ROWS)):
        at = 58 + local * 8
        frame[at : at + 2] = (100 + local).to_bytes(2, "big")
        frame[at + 2 : at + 4] = (110 + local).to_bytes(2, "big")
        frame[at + 4 : at + 6] = (local * 10 + 1).to_bytes(2, "big")
        frame[at + 6 : at + 8] = (local * 10 + 5).to_bytes(2, "big")
    return _finish_crc(frame)


def end_frame(cycles=32, tag=123, table_crc=456):
    frame = bytearray(stream.END_LENGTH)
    frame[:4] = b"\xd9\x9d\x04\x00"
    frame[4:8] = tag.to_bytes(4, "big")
    frame[8:12] = stream.FIRMWARE.to_bytes(4, "big")
    frame[12:14] = cycles.to_bytes(2, "big")
    frame[14:16] = cycles.to_bytes(2, "big")
    frame[18:22] = b"\x01\x01\x00\x02"
    frame[28:32] = (cycles * 21_000).to_bytes(4, "big")
    frame[32:36] = table_crc.to_bytes(4, "big")
    return _finish_crc(frame)


class FastFlankStreamTests(unittest.TestCase):
    def test_command_is_fixed_and_bounded(self):
        wire = stream.command(256, 0x12345678)
        self.assertEqual(len(wire), 808)
        self.assertEqual(wire[:4], b"\xff\xff\x03\x07")
        self.assertEqual(wire[4], 1)
        self.assertEqual(int.from_bytes(wire[6:8], "big"), 1500)
        self.assertEqual(wire[12:16], b"F15!")
        self.assertEqual(wire[18], 16)
        self.assertFalse(any(wire[19:]))
        for bad in (31, 1025):
            with self.assertRaises(ValueError):
                stream.command(bad, 1)

    def test_fragmented_frames_decode_with_board_time(self):
        data = data_frame(sequence=16, start_us=336_000)
        end = end_frame()
        buffer = bytearray(b"noise" + data[:41])
        self.assertEqual(list(stream.extract_frames(buffer)), [])
        buffer.extend(data[41:] + end)
        frames = list(stream.extract_frames(buffer))
        self.assertEqual(len(frames), 2)
        decoded = stream.decode_data(frames[0], tag=123, table_crc32=456)
        self.assertTrue(decoded["minimum_warmup_complete"])
        self.assertEqual(decoded["records"][14]["row"], 38)
        self.assertEqual(decoded["records"][15]["row"], 36)
        terminal = stream.decode_end(frames[1], tag=123, table_crc32=456)
        self.assertTrue(terminal["firmware_shutter_asserted"])
        self.assertEqual(terminal["stop_reason"], 0)

    def test_warmup_and_sampling_gate_are_explicit(self):
        frames = [
            stream.decode_data(
                data_frame(sequence=i, start_us=i * 21_000),
                tag=123,
                table_crc32=456,
            )
            for i in range(32)
        ]
        result = stream.analyze(frames)
        self.assertEqual(result["minimum_warmup_discarded_cycles"], 16)
        self.assertEqual(result["post_minimum_warmup_cycles"], 16)
        self.assertAlmostEqual(result["board_rate_hz"], 1_000_000 / 21_000)
        self.assertTrue(result["meets_sampling_rate_for_15hz"])
        self.assertFalse(result["absolute_optical_stability_verified"])
        self.assertFalse(result["physical_15hz_verified"])
        self.assertFalse(result["contact_area_verified"])

    def test_corruption_or_false_warmup_is_rejected(self):
        false_ready = bytearray(data_frame(sequence=0))
        false_ready[3] = 1
        false_ready = _finish_crc(false_ready)
        with self.assertRaisesRegex(ValueError, "warm-up"):
            stream.decode_data(false_ready, tag=123, table_crc32=456)
        corrupt = bytearray(data_frame(sequence=0))
        corrupt[90] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            stream.decode_data(bytes(corrupt), tag=123, table_crc32=456)

    def test_firmware_source_is_bounded_and_fail_closed(self):
        source = (Path(__file__).parent.parent / "JDSU" / "Core" / "Src" / "ms5614t.c").read_text(encoding="utf-8")
        self.assertIn('memcmp(&aRxBuffer[12], "F15!", 4U)', source)
        self.assertIn("FAST_FLANK_STREAM_MAX_CYCLES 1024U", source)
        self.assertIn("FAST_FLANK_STREAM_WARMUP_CYCLES 16U", source)
        self.assertIn("PI11210_SetSOAShutter(1U)", source)
        self.assertIn("WifiTransport_IsOtaActive()", source)
        self.assertIn("ADC_ReadMask(0x02U, samples)", source)
        self.assertIn("routeTestRows[1]", source)

    def test_receive_publishes_every_validated_frame_in_order(self):
        payload = b"".join(
            data_frame(sequence=index, start_us=index * 21_000)
            for index in range(32)
        ) + end_frame()

        class Device:
            def __init__(self, wire):
                self.wire = bytearray(wire)

            def read(self, count):
                result = bytes(self.wire[:count])
                del self.wire[:count]
                return result

        observed = []
        frames, terminal = stream.receive(
            Device(payload),
            cycles=32,
            tag=123,
            table_crc32=456,
            on_frame=lambda item: observed.append(item["sequence"]),
        )
        self.assertEqual(observed, list(range(32)))
        self.assertEqual(len(frames), 32)
        self.assertEqual(terminal["completed_cycles"], 32)

    def test_receive_stop_is_causal_and_does_not_accept_terminal(self):
        class Device:
            def read(self, _count):
                self.read_called = True
                return b""

        device = Device()
        with self.assertRaisesRegex(InterruptedError, "stop requested"):
            stream.receive(
                device,
                cycles=32,
                tag=123,
                table_crc32=456,
                should_stop=lambda: True,
            )
        self.assertFalse(hasattr(device, "read_called"))


if __name__ == "__main__":
    unittest.main()
