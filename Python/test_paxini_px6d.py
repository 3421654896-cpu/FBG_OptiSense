"""Unit tests for the Paxini PX6D auto-push reader (no hardware needed)."""

import struct
import unittest

import paxini_px6d as px


def push_frame(values):
    payload = struct.pack("<6f", *values)
    return (
        px.HEADER
        + bytes((0x00, 0x03))
        + payload
        + bytes((px.rc8(bytes((0x00, 0x03)) + payload),))
    )


class FrameHelperTests(unittest.TestCase):
    def test_rc8_is_xor_of_body(self):
        body = bytes((0x00, 0x02, 0x01))
        self.assertEqual(px.rc8(body), 0x00 ^ 0x02 ^ 0x01)

    def test_build_frame_layout(self):
        frame = px.build_frame(0x02, b"\x01", device_id=0x00)
        self.assertEqual(
            frame,
            b"\xAA\x55\x00\x02\x01" + bytes((0x00 ^ 0x02 ^ 0x01,)),
        )

    def test_parse_six_axis_decodes_six_little_endian_floats(self):
        payload = struct.pack("<6f", 1.5, -2.25, 30.0, 0.01, -0.02, 0.03)
        values = px.parse_six_axis(payload)
        self.assertAlmostEqual(values[0], 1.5, places=6)
        self.assertAlmostEqual(values[1], -2.25, places=6)
        self.assertEqual(values[2], 30.0)
        self.assertAlmostEqual(values[3], 0.01, places=6)
        self.assertAlmostEqual(values[4], -0.02, places=6)
        self.assertAlmostEqual(values[5], 0.03, places=6)

    def test_parse_six_axis_rejects_short_or_nan(self):
        self.assertIsNone(px.parse_six_axis(b"\x00" * 8))
        nan = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, float("nan"))
        self.assertIsNone(px.parse_six_axis(nan))


class PumpTests(unittest.TestCase):
    def test_extract_parses_push_frame(self):
        sensor = px.PaxiniForceSensor("COMX")
        buffer = bytearray(push_frame((0.5, -0.25, 12.0, 0.001, 0.0, -0.002)))
        self.assertTrue(sensor._extract(buffer))
        self.assertEqual(buffer, bytearray())
        self.assertEqual(sensor.frames_received, 1)
        self.assertAlmostEqual(sensor._latest[2], 12.0, places=6)

    def test_extract_skips_unknown_cmd(self):
        sensor = px.PaxiniForceSensor("COMX")
        buffer = bytearray(b"\xAA\x55\x00\x7E\x01\x02\x03")
        self.assertTrue(sensor._extract(buffer))
        self.assertEqual(sensor.frames_received, 0)
        self.assertEqual(buffer, bytearray(b"\x55\x00\x7E\x01\x02\x03"))

    def test_calibration_ack_sets_event(self):
        sensor = px.PaxiniForceSensor("COMX")
        payload = b"\x00" * 8
        frame = (
            px.HEADER + bytes((0x01, 0x10)) + payload
            + bytes((px.rc8(bytes((0x01, 0x10)) + payload),))
        )
        self.assertTrue(sensor._extract(bytearray(frame)))
        self.assertTrue(sensor._calibration_event.is_set())
        self.assertEqual(sensor._calibration_payload, payload)

    def test_calibrate_zero_uses_official_broadcast_command(self):
        sensor = px.PaxiniForceSensor("COMX")
        sensor._running = True
        sensor._ser = object()
        sent = []

        def send(packet):
            sent.append(packet)
            sensor._calibration_payload = b"\x00" * 8
            sensor._calibration_event.set()

        sensor._send = send
        payload = sensor.calibrate_zero(timeout_s=0.05)
        self.assertEqual(
            sent,
            [px.build_frame(0x10, b"\x01", device_id=0x7F)],
        )
        self.assertEqual(payload, b"\x00" * 8)
        self.assertEqual(sensor.zero_count, 1)
        self.assertIsNotNone(sensor.last_zero_ns)

    def test_read_six_axis_returns_latest_frame(self):
        sensor = px.PaxiniForceSensor("COMX")
        sensor._extract(bytearray(push_frame((1.0, 2.0, 3.0, 0.0, 0.0, 0.0))))
        values = sensor.read_six_axis(timeout_s=0.05)
        self.assertAlmostEqual(values[0], 1.0, places=6)
        self.assertAlmostEqual(values[2], 3.0, places=6)

    def test_read_six_axis_times_out_without_data(self):
        sensor = px.PaxiniForceSensor("COMX")
        with self.assertRaises(TimeoutError):
            sensor.read_six_axis(timeout_s=0.05)


if __name__ == "__main__":
    unittest.main()
