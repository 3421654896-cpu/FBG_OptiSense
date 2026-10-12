import struct
import unittest
import zlib

from fbg_remote_protocol import (
    ACK_ACCEPTED,
    ACK_APPLIED,
    ACK_REJECTED,
    HEADER_SIZE,
    MODE_STRESS,
    MODE_TEMPERATURE,
    MetadataFrame,
    ModeAck,
    ModeCommand,
    ProtocolError,
    TelemetryFrame,
    decode_mode_ack,
    decode_mode_command,
    decode_metadata,
    decode_telemetry,
    encode_mode_ack,
    encode_mode_command,
    encode_metadata,
    encode_telemetry,
)


def make_frame(mode, point_count, channel_mask):
    channel_count = bin(channel_mask).count("1")
    samples = tuple(
        tuple((point * 37 + channel * 101) & 0x0FFF for channel in range(channel_count))
        for point in range(point_count)
    )
    return TelemetryFrame(
        mode=mode,
        flags=0x03,
        boot_id=0x12345678,
        seq=0x10203040,
        uptime_ms=987654,
        temperature_mC=25123,
        table_crc32=0x89ABCDEF,
        gainmask0=0x115,
        gainmask1=0x12A,
        channel_mask=channel_mask,
        samples=samples,
        pd_voltage_mV=9000,
        pd_current_mA=3000,
        pd_power_limit_mW=27000,
        fan_rpm=4321,
        fan_duty_permille=1000,
        thermal_flags=0x0013,
        pd_flags=0x03,
        pd_protocol=0x08,
    )


class RemoteTelemetryProtocolTests(unittest.TestCase):
    def test_temperature_100_point_ch01_round_trip(self):
        source = make_frame(MODE_TEMPERATURE, 100, 0x03)
        encoded = encode_telemetry(source)

        self.assertEqual(len(encoded), HEADER_SIZE + 100 * 2 * 2 + 4)
        self.assertEqual(encoded[:4], b"FBG1")
        self.assertEqual(decode_telemetry(encoded), source)

    def test_stress_100_point_four_channel_round_trip(self):
        source = make_frame(MODE_STRESS, 100, 0x0F)
        encoded = encode_telemetry(source)

        self.assertEqual(len(encoded), HEADER_SIZE + 100 * 4 * 2 + 4)
        self.assertEqual(decode_telemetry(encoded), source)

    def test_legacy_v1_telemetry_remains_decodable(self):
        source = make_frame(MODE_TEMPERATURE, 2, 0x03)
        payload = b"".join(
            struct.pack(">H", code) for point in source.samples for code in point
        )
        legacy_header = struct.pack(
            ">4sBBBBHHHBBIIIiIHH",
            b"FBG1", 1, 1, source.mode, source.flags,
            40, len(payload), source.point_count, source.channel_mask, 1,
            source.boot_id, source.seq, source.uptime_ms, source.temperature_mC,
            source.table_crc32, source.gainmask0, source.gainmask1,
        )
        packet = legacy_header + payload
        packet += struct.pack(">I", zlib.crc32(packet) & 0xFFFFFFFF)
        decoded = decode_telemetry(packet)
        self.assertEqual(decoded.samples, source.samples)
        self.assertEqual(decoded.pd_power_limit_mW, 0)
        self.assertEqual(decoded.fan_rpm, 0)

    def test_crc_error_is_rejected(self):
        encoded = bytearray(encode_telemetry(make_frame(MODE_TEMPERATURE, 66, 0x03)))
        encoded[HEADER_SIZE + 17] ^= 0x01

        with self.assertRaisesRegex(ProtocolError, "CRC32 mismatch"):
            decode_telemetry(encoded)

    def test_truncated_and_trailing_frames_are_rejected(self):
        encoded = encode_telemetry(make_frame(MODE_TEMPERATURE, 66, 0x03))
        for malformed in (encoded[:-1], encoded + b"\x00"):
            with self.assertRaisesRegex(ProtocolError, "length"):
                decode_telemetry(malformed)

    def test_declared_point_count_must_match_payload(self):
        encoded = bytearray(encode_telemetry(make_frame(MODE_TEMPERATURE, 66, 0x03)))
        struct.pack_into(">H", encoded, 12, 67)
        struct.pack_into(">I", encoded, len(encoded) - 4, zlib.crc32(encoded[:-4]) & 0xFFFFFFFF)

        with self.assertRaisesRegex(ProtocolError, "payload_len"):
            decode_telemetry(encoded)

    def test_channel_mask_and_point_width_are_strict(self):
        bad_mask = make_frame(MODE_TEMPERATURE, 66, 0x03)
        bad_mask = TelemetryFrame(**{**bad_mask.__dict__, "channel_mask": 0x10})
        with self.assertRaisesRegex(ProtocolError, "channel_mask"):
            encode_telemetry(bad_mask)

        source = make_frame(MODE_TEMPERATURE, 66, 0x03)
        malformed_samples = list(source.samples)
        malformed_samples[10] = (123,)
        bad_width = TelemetryFrame(
            **{**source.__dict__, "samples": tuple(malformed_samples)}
        )
        with self.assertRaisesRegex(ProtocolError, "point 10"):
            encode_telemetry(bad_width)

    def test_header_channel_mask_mismatch_is_rejected(self):
        encoded = bytearray(encode_telemetry(make_frame(MODE_TEMPERATURE, 66, 0x03)))
        encoded[14] = 0x01
        struct.pack_into(">I", encoded, len(encoded) - 4, zlib.crc32(encoded[:-4]) & 0xFFFFFFFF)

        with self.assertRaisesRegex(ProtocolError, "payload_len"):
            decode_telemetry(encoded)


class RemoteModeProtocolTests(unittest.TestCase):
    def test_mode_must_be_an_exact_integer(self):
        with self.assertRaises(ProtocolError):
            encode_mode_command(1, False)
        with self.assertRaises(ProtocolError):
            encode_mode_command(1, 0.0)

    def test_stress_and_temperature_commands(self):
        cases = (
            (ModeCommand(0x00000001, MODE_STRESS), b"FCMD1|00000001|MODE|STRESS"),
            (
                ModeCommand(0x89ABCDEF, MODE_TEMPERATURE),
                b"FCMD1|89ABCDEF|MODE|TEMPERATURE",
            ),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(encode_mode_command(command), expected)
                self.assertEqual(decode_mode_command(expected), command)

    def test_all_ack_statuses_round_trip(self):
        for index, status in enumerate((ACK_ACCEPTED, ACK_APPLIED, ACK_REJECTED)):
            ack = ModeAck(
                command_id=0xA0 + index,
                status=status,
                mode=MODE_TEMPERATURE,
                applied_seq=0x1020 + index,
            )
            with self.subTest(status=status):
                encoded = encode_mode_ack(ack)
                self.assertEqual(decode_mode_ack(encoded), ack)
                self.assertTrue(encoded.startswith(b"FACK1|"))

    def test_malformed_command_and_ack_are_rejected(self):
        malformed = (
            b"FCMD1|00000001|MODE|UNKNOWN",
            b"FCMD1|1|MODE|STRESS",
            b"FCMD1|abcdef12|MODE|STRESS",
            b" FCMD1|00000001|MODE|STRESS",
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                decode_mode_command(value)

        with self.assertRaises(ProtocolError):
            decode_mode_ack(b"FACK1|00000001|DONE|STRESS|00000000")


class RemoteMetadataProtocolTests(unittest.TestCase):
    def test_temperature_100_point_metadata_round_trip(self):
        source = MetadataFrame(
            mode=MODE_TEMPERATURE,
            table_crc32=0x89ABCDEF,
            wavelength_pm=tuple(1_540_000 + index * 97 for index in range(100)),
        )
        encoded = encode_metadata(source)

        self.assertEqual(encoded[:4], b"FBGM")
        self.assertEqual(len(encoded), 12 + 100 * 4 + 4)
        self.assertEqual(decode_metadata(encoded), source)

    def test_stress_100_point_metadata_round_trip(self):
        source = MetadataFrame(
            mode=MODE_STRESS,
            table_crc32=0x10203040,
            wavelength_pm=tuple(1_520_000 + index * 101 for index in range(100)),
        )
        self.assertEqual(decode_metadata(encode_metadata(source)), source)

    def test_metadata_crc_error_is_rejected(self):
        source = MetadataFrame(MODE_TEMPERATURE, 0x12345678, (1_540_000, 1_540_100))
        encoded = bytearray(encode_metadata(source))
        encoded[13] ^= 0x20

        with self.assertRaisesRegex(ProtocolError, "metadata CRC32 mismatch"):
            decode_metadata(encoded)

    def test_metadata_truncation_and_declared_count_are_rejected(self):
        source = MetadataFrame(MODE_TEMPERATURE, 0x12345678, (1_540_000, 1_540_100))
        encoded = encode_metadata(source)
        with self.assertRaisesRegex(ProtocolError, "length"):
            decode_metadata(encoded[:-1])

        wrong_count = bytearray(encoded)
        struct.pack_into(">H", wrong_count, 6, 3)
        struct.pack_into(
            ">I",
            wrong_count,
            len(wrong_count) - 4,
            zlib.crc32(wrong_count[:-4]) & 0xFFFFFFFF,
        )
        with self.assertRaisesRegex(ProtocolError, "length"):
            decode_metadata(wrong_count)

    def test_metadata_mode_point_count_and_uint32_are_strict(self):
        with self.assertRaisesRegex(ProtocolError, "unsupported mode"):
            encode_metadata(MetadataFrame(2, 0, (1_540_000,)))
        with self.assertRaisesRegex(ProtocolError, "point_count"):
            encode_metadata(MetadataFrame(MODE_TEMPERATURE, 0, ()))
        with self.assertRaisesRegex(ProtocolError, "uint32"):
            encode_metadata(
                MetadataFrame(MODE_TEMPERATURE, 0, (1 << 32,))
            )
        with self.assertRaisesRegex(ProtocolError, "uint32"):
            encode_metadata(
                MetadataFrame(MODE_TEMPERATURE, -1, (1_540_000,))
            )


if __name__ == "__main__":
    unittest.main()
