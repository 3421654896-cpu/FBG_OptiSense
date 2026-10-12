import struct
import unittest
import zlib
from pathlib import Path

from fbg_lan_serial import (
    FbgLanSerial,
    RAW_COMPACT_ENABLE,
    RAW_TEMPORARY_BATCH_ENABLE,
    RAW_TEMPORARY_COMPACT_ENABLE,
    RAW_COMPACT_SCAN_HEADER,
    RAW_COMMAND_SIZE,
    RAW_KIND_COMMAND_ACK,
    RAW_KIND_COMPACT_F45,
    RAW_KIND_COMPACT_F45_BATCH,
    RAW_KIND_COMPACT_SCAN,
    RAW_KIND_DATA,
    RAW_KIND_TEMPORARY_BATCH,
    RAW_KIND_COMPACT_TEMPORARY_BATCH,
    RAW_KIND_TEMPORARY_CAP_ACK,
    RAW_KIND_TEMPORARY_COMPACT_CAP_ACK,
    RAW_KIND_READY,
    RAW_RESPONSE_HEADER,
    expand_compact_f45,
    expand_compact_f45_batch,
    expand_compact_scan,
    expand_compact_temporary_batch,
    expand_temporary_batch,
)
from frame_telemetry import BoardSequenceTracker, extract_raw_scan_frames
from verify_runtime_firmware import board_extension
from temporary_test_mode import decode as decode_temporary
from temporary_test_mode import extract_frames as extract_temporary_frames


def raw_packet(kind, sequence, payload=b""):
    header = RAW_RESPONSE_HEADER.pack(
        b"FBR1", 1, kind, sequence, len(payload)
    )
    body = header + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def native_ch1_scan(point_count=45, board_sequence=7, extension_length=142):
    frame = bytearray(b"\xEE\xEE" + int(point_count).to_bytes(2, "big"))
    for point in range(point_count):
        frame.extend(b"\x00\x00")
        frame.extend(((0x180 + point) & 0x0FFF).to_bytes(2, "big"))
        frame.extend(b"\x00\x00\x00\x00")
    # Preserve non-empty peak metadata as well as temperature/gain/status.
    frame.extend(b"\xAB")
    frame.extend(b"\x00")
    frame.extend(b"\x01" + struct.pack(">f", 1538.62))
    frame.extend(b"\x00\x00")
    frame.extend(b"\x00\x19\x00\x2A")  # temperature
    frame.extend(b"\x12\x34\x56\x78")  # actual gain masks
    extension = bytearray(index & 0xFF for index in range(extension_length))
    extension[16] = 0x02  # active mask
    extension[22:24] = b"\x01\x03"  # actual CH0/CH1 selectors
    extension[24:28] = b"\x00\x01\x00\x16"  # firmware 1.0.22
    extension[28:32] = int(board_sequence).to_bytes(4, "big")
    extension[32:36] = (56_177).to_bytes(4, "big")
    extension[36] = 1  # multirate extension version
    extension[44:48] = b"\xCA\xFE\xBA\xBE"  # opaque/table metadata marker
    extension[48:52] = (123_456).to_bytes(4, "big")
    if extension_length >= 154:
        # Future trailer fields (for example a wavelength-table CRC) must be
        # opaque to CS1 and survive byte-for-byte.
        extension[150:154] = b"\xA1\xB2\xC3\xD4"
    frame.extend(b"\xB5\x4D\x01" + bytes((len(extension),)))
    frame.extend(extension)
    frame.extend(b"\xFF\xEF")
    return bytes(frame)


def compact_ch1_payload(native):
    point_count = int.from_bytes(native[2:4], "big")
    sample_end = 4 + point_count * 8
    samples = b"".join(
        native[4 + point * 8 + 2 : 4 + point * 8 + 4]
        for point in range(point_count)
    )
    trailer = native[sample_end:]
    return RAW_COMPACT_SCAN_HEADER.pack(1, 0x02, point_count, len(trailer)) + samples + trailer


def compact_f45_payload(sequence=20, elapsed_us=31_000):
    payload = bytearray(128)
    payload[0] = 1
    payload[1] = 1 if sequence >= 16 else 0
    payload[2:6] = (0x10203040).to_bytes(4, "big")
    payload[6:10] = (0x0001002D).to_bytes(4, "big")
    payload[10:14] = int(sequence).to_bytes(4, "big")
    payload[14:18] = (123_456).to_bytes(4, "big")
    payload[18:22] = int(elapsed_us).to_bytes(4, "big")
    payload[22:26] = (0xA1B2C3D4).to_bytes(4, "big")
    payload[26:32] = bytes((1, 45, 16, 2, 0, 2))
    payload[32:38] = b"\x00" * 6
    for point in range(45):
        payload[38 + point * 2 : 40 + point * 2] = (300 + point).to_bytes(2, "big")
    return bytes(payload)


def native_temporary_frame(point_count=15, sequence=20, tag=123, route_crc=456,
                           wide=False, time_step=1000):
    frame = bytearray(46 + point_count * (13 if wide else 9))
    frame[:4] = bytes((0xD9, 0x9D, 14, int(sequence >= 16) | (2 if wide else 0)))
    for offset, value in (
        (4, tag),
        (8, 0x0001004D),
        (12, sequence),
        (16, 1_000_000),
        (20, point_count * time_step),
        (24, route_crc),
    ):
        frame[offset : offset + 4] = int(value).to_bytes(4, "big")
    frame[28:34] = bytes((24, point_count, 16, 2, 0, 2))
    frame[40 : 40 + point_count] = bytes(range(point_count))
    start = 40 + point_count
    for point in range(point_count):
        values = (300 + point, 302 + point,
                  point * time_step + 100, point * time_step + 800)
        at = start + point * (12 if wide else 8)
        frame[at : at + 2] = values[0].to_bytes(2, "big")
        frame[at + 2 : at + 4] = values[1].to_bytes(2, "big")
        width = 4 if wide else 2
        frame[at + 4 : at + 4 + width] = values[2].to_bytes(width, "big")
        frame[at + 4 + width : at + 4 + 2 * width] = values[3].to_bytes(width, "big")
    frame[-6:-2] = (zlib.crc32(frame[:-6]) & 0xFFFFFFFF).to_bytes(4, "big")
    frame[-2:] = b"\x9D\xD9"
    return bytes(frame)


def compact_temporary_payload(frames):
    """Mirror the MCU's TC4 layout, retaining the original native CRC."""

    parts = []
    for frame in frames:
        point_count = frame[29]
        wide = bool(frame[3] & 2)
        record_length = 12 if wide else 8
        compact = bytearray(frame[:40])
        for point in range(point_count):
            source = 40 + point_count + point * record_length
            first = int.from_bytes(frame[source : source + 2], "big")
            second = int.from_bytes(frame[source + 2 : source + 4], "big")
            compact.extend((first >> 4, ((first & 15) << 4) | (second >> 8), second & 255))
            if wide:
                compact.extend(frame[source + 5 : source + 8])
                compact.extend(frame[source + 9 : source + 12])
            else:
                compact.extend(frame[source + 4 : source + 8])
        compact.extend(frame[-6:-2])
        parts.append(bytes(compact))
    assert all(len(part) == len(parts[0]) for part in parts)
    return (bytes((1, len(frames))) + len(frames[0]).to_bytes(2, "big")
            + len(parts[0]).to_bytes(2, "big") + b"".join(parts))


class FbgLanSerialTests(unittest.TestCase):
    def test_compact_temporary_batch_preserves_every_byte_narrow_and_wide(self):
        for wide in (False, True):
            with self.subTest(wide=wide):
                originals = tuple(native_temporary_frame(
                    point_count=45, sequence=20 + index, wide=wide,
                    time_step=3500 if wide else 1000,
                ) for index in range(4))
                payload = compact_temporary_payload(originals)
                self.assertLessEqual(len(payload), 2048)
                self.assertEqual(expand_compact_temporary_batch(payload), originals)
                self.assertEqual(
                    decode_temporary(expand_compact_temporary_batch(payload)[0],
                                     123, 456, point_count=45),
                    decode_temporary(originals[0], 123, 456, point_count=45),
                )
                adapter = FbgLanSerial()
                adapter._network_buffer.extend(raw_packet(
                    RAW_KIND_COMPACT_TEMPORARY_BATCH, 110, payload
                ))
                adapter._drain_packets(received_at=12.5)
                stream = bytearray(adapter._receive_buffer)
                self.assertEqual(list(extract_temporary_frames(stream)), list(originals))
                self.assertFalse(stream)
                self.assertEqual(adapter.transport_stats["compact_temporary_frames"], 4)
                self.assertLess(adapter.transport_stats["wire_to_native_ratio"], 0.82)
                self.assertEqual(
                    {adapter.take_frame_arrival_info(frame)[1] for frame in originals}, {1}
                )

    def test_compact_temporary_batch_rejects_corrupt_frame_atomically(self):
        originals = tuple(native_temporary_frame(sequence=20 + index) for index in range(2))
        payload = bytearray(compact_temporary_payload(originals))
        payload[-6] ^= 1  # Outer packet CRC remains valid; native CRC must catch this.
        self.assertIsNone(expand_compact_temporary_batch(payload))
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_COMPACT_TEMPORARY_BATCH, 111, payload))
        adapter._drain_packets()
        self.assertEqual(adapter.in_waiting, 0)
        self.assertEqual(adapter.transport_stats["invalid_compact_temporary_packets"], 1)

    def test_compact_temporary_batch_rejects_bad_layout(self):
        frame = native_temporary_frame(point_count=15)
        payload = bytearray(compact_temporary_payload((frame,)))
        payload[4:6] = (2048).to_bytes(2, "big")
        self.assertIsNone(expand_compact_temporary_batch(payload))

    def test_temporary_batch_preserves_every_native_byte_and_arrival(self):
        originals = tuple(
            native_temporary_frame(point_count=45, sequence=20 + index)
            for index in range(4)
        )
        payload = bytes((1, 4)) + len(originals[0]).to_bytes(2, "big") + b"".join(originals)
        self.assertEqual(expand_temporary_batch(payload), originals)
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_TEMPORARY_BATCH, 100, payload))

        adapter._drain_packets(received_at=12.5)

        stream = bytearray(adapter._receive_buffer)
        self.assertEqual(list(extract_temporary_frames(stream)), list(originals))
        self.assertFalse(stream)
        self.assertEqual(adapter.transport_stats["data_packets"], 1)
        self.assertEqual(adapter.transport_stats["temporary_batch_frames"], 4)
        self.assertEqual(
            {adapter.take_frame_arrival_info(frame)[1] for frame in originals}, {1}
        )

    def test_temporary_batch_rejects_corrupt_native_frame_atomically(self):
        original = native_temporary_frame(point_count=15)
        damaged = bytearray(original)
        damaged[55] ^= 1
        payload = bytes((1, 2)) + len(original).to_bytes(2, "big") + original + damaged
        self.assertIsNone(expand_temporary_batch(payload))
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_TEMPORARY_BATCH, 101, payload))
        adapter._drain_packets()
        self.assertEqual(adapter.in_waiting, 0)
        self.assertEqual(adapter.transport_stats["invalid_temporary_batch_packets"], 1)

    def test_temporary_batch_accepts_three_wide_45_point_frames(self):
        originals = tuple(
            native_temporary_frame(point_count=45, sequence=30 + index, wide=True)
            for index in range(3)
        )
        self.assertEqual(len(originals[0]), 631)
        payload = bytes((1, 3)) + (631).to_bytes(2, "big") + b"".join(originals)
        self.assertEqual(expand_temporary_batch(payload), originals)
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_TEMPORARY_BATCH, 102, payload))
        adapter._drain_packets()
        self.assertEqual(list(extract_temporary_frames(bytearray(adapter._receive_buffer))),
                         list(originals))

    def test_firmware_temporary_batch_is_negotiated_and_backpressured(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")
        producer = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "ms5614t.c"
        ).read_text(encoding="utf-8")
        queue = source[
            source.index("uint8_t WifiTransport_QueueTemporaryFrame") :
            source.index("static uint8_t CommitLanRawF45Batch")
        ]
        self.assertIn("if(!lanRawTemporaryBatchEnabled)", queue)
        self.assertIn('WIFI_LAN_RAW_TEMPORARY_CAP     "FBGL1|CAPS|TB4"', source)
        self.assertIn("if(!CommitLanRawTemporaryBatch()) return 0U", queue)
        self.assertIn("WIFI_LAN_RAW_KIND_TEMPORARY_BATCH", source)
        self.assertIn('WIFI_LAN_RAW_TEMPORARY_COMPACT_CAP "FBGL1|CAPS|TC4"', source)
        self.assertIn("WIFI_LAN_RAW_KIND_COMPACT_TEMPORARY_BATCH", source)
        self.assertIn("LanCompactTemporaryLength(data, length)", queue)
        self.assertIn("EncodeLanCompactTemporary(", queue)
        self.assertIn("if(compact ||", queue)
        self.assertIn("return WifiTransport_QueueRawFrame(data, length)", queue)
        self.assertIn("WifiTransport_QueueTemporaryFrame(txBuffer, frameLength)", producer)
        self.assertIn("WifiTransport_FlushTemporaryFrames()", producer)

    def test_lan_raw_preserves_variable_temporary_frame_for_shared_parser(self):
        native = native_temporary_frame(point_count=15)
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_DATA, 44, native))

        adapter._drain_packets(received_at=14.0)

        stream = bytearray(adapter._receive_buffer)
        frames = list(extract_temporary_frames(stream))
        self.assertEqual(frames, [native])
        self.assertFalse(stream)
        parsed = decode_temporary(native, 123, 456, point_count=15)
        self.assertEqual(len(parsed["records"]), 15)
        self.assertEqual(parsed["records"][0]["first_code"], 300)
        self.assertEqual(parsed["records"][-1]["second_code"], 316)

    def test_firmware_allows_only_validated_temporary_custom_route_over_lan(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "ms5614t.c"
        ).read_text(encoding="utf-8")
        self.assertIn("(streamOverLan && !temporary45Stream)", source)
        self.assertIn(
            "!CandidateRoute_Parse(aRxBuffer, USART_RX_SIZE, &candidateRoute)",
            source,
        )
        self.assertNotIn(
            "if(streamOverLan || !CandidateRoute_Parse",
            source,
        )

    def test_firmware_applies_backpressure_to_unsent_large_raw_frames(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")
        raw_queue = source[
            source.index("uint8_t WifiTransport_QueueRawFrame") :
            source.index("uint8_t WifiTransport_IsLanRawClientActive")
        ]
        large_frame_branch = raw_queue[raw_queue.index("else\n    {") :]
        self.assertIn("if(lanRawScanReady) return 0U;", large_frame_branch)
        self.assertLess(
            large_frame_branch.index("if(lanRawScanReady) return 0U;"),
            large_frame_branch.index("BuildLanRawPacket"),
        )
        self.assertIn(
            "lanClientDeadlineMs = nextPublishMs + "
            "WIFI_LAN_CLIENT_TIMEOUT_MS;",
            raw_queue,
        )

    def test_compact_f45_restores_first_codes_and_marks_compatibility_fields(self):
        compact = compact_f45_payload()

        restored = expand_compact_f45(compact)

        self.assertIsNotNone(restored)
        self.assertEqual(len(restored), 453)
        self.assertEqual(restored[:4], b"\xD9\x9D\x03\x03")
        self.assertEqual(restored[40:85], bytes(range(45)))
        for point in range(45):
            record = 85 + point * 8
            self.assertEqual(restored[record : record + 2], (300 + point).to_bytes(2, "big"))
            self.assertEqual(restored[record + 2 : record + 4], restored[record : record + 2])
        self.assertEqual(restored[-2:], b"\x9D\xD9")
        self.assertEqual(
            int.from_bytes(restored[-6:-2], "big"),
            zlib.crc32(restored[:-6]) & 0xFFFFFFFF,
        )

    def test_compact_f45_packet_is_exposed_as_shared_native_stream(self):
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(
            raw_packet(RAW_KIND_COMPACT_F45, 91, compact_f45_payload())
        )

        adapter._drain_packets(received_at=12.5)

        restored = bytes(adapter._receive_buffer)
        self.assertEqual(len(restored), 453)
        self.assertEqual(adapter.transport_stats["compact_f45_packets"], 1)
        self.assertEqual(adapter.transport_stats["invalid_compact_f45_packets"], 0)
        self.assertLess(adapter.transport_stats["wire_to_native_ratio"], 0.4)

    def test_compact_f45_batch_restores_all_frames_in_sequence(self):
        compact_frames = tuple(
            compact_f45_payload(sequence=30 + index, elapsed_us=30_500 + index)
            for index in range(4)
        )
        batch = bytes((1, len(compact_frames))) + b"".join(compact_frames)

        restored = expand_compact_f45_batch(batch)

        self.assertIsNotNone(restored)
        self.assertEqual(len(restored), 4)
        self.assertEqual(
            [int.from_bytes(frame[12:16], "big") for frame in restored],
            [30, 31, 32, 33],
        )

        adapter = FbgLanSerial()
        adapter._network_buffer.extend(
            raw_packet(RAW_KIND_COMPACT_F45_BATCH, 92, batch)
        )
        adapter._drain_packets(received_at=13.5)

        self.assertEqual(bytes(adapter._receive_buffer), b"".join(restored))
        frames = list(restored)
        arrivals = [adapter.take_frame_arrival_info(frame) for frame in frames]
        self.assertTrue(all(arrival is not None for arrival in arrivals))
        self.assertEqual({arrival[1] for arrival in arrivals}, {1})
        self.assertEqual(adapter.transport_stats["data_packets"], 1)
        self.assertEqual(adapter.transport_stats["compact_f45_packets"], 4)
        self.assertEqual(adapter.transport_stats["invalid_compact_f45_packets"], 0)

    def test_malformed_compact_f45_is_never_exposed(self):
        malformed = bytearray(compact_f45_payload())
        malformed[27] = 44
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_COMPACT_F45, 1, malformed))

        adapter._drain_packets()

        self.assertEqual(adapter.in_waiting, 0)
        self.assertEqual(adapter.transport_stats["invalid_compact_f45_packets"], 1)

    def test_compact_ch1_scan_round_trips_every_native_field(self):
        native = native_ch1_scan()
        compact = compact_ch1_payload(native)

        restored = expand_compact_scan(compact)

        self.assertEqual(restored, native)
        self.assertLess(len(compact) + 16, len(native) + 16)
        self.assertEqual(len(native), 529)
        self.assertEqual(len(compact) + 16, 277)
        extension = board_extension(restored)
        self.assertIsNotNone(extension)
        self.assertEqual(extension[16], 0x02)
        self.assertEqual(extension[22:24], b"\x01\x03")
        self.assertEqual(extension[24:28], b"\x00\x01\x00\x16")
        self.assertEqual(extension[28:32], (7).to_bytes(4, "big"))
        self.assertEqual(extension[32:36], (56_177).to_bytes(4, "big"))
        self.assertEqual(extension[44:48], b"\xCA\xFE\xBA\xBE")
        self.assertEqual(extension, board_extension(native))

    def test_compact_scan_preserves_arbitrary_extension_and_trailer_length(self):
        native = native_ch1_scan(extension_length=173)
        compact = compact_ch1_payload(native)

        restored = expand_compact_scan(compact)

        self.assertEqual(restored, native)
        self.assertEqual(len(board_extension(restored)), 173)
        self.assertEqual(board_extension(restored)[150:154], b"\xA1\xB2\xC3\xD4")
        self.assertEqual(
            compact[RAW_COMPACT_SCAN_HEADER.size + 45 * 2 :],
            native[4 + 45 * 8 :],
        )

    def test_fragmented_and_sticky_compact_packets_restore_native_order(self):
        first = native_ch1_scan(point_count=10, board_sequence=10)
        second = native_ch1_scan(point_count=10, board_sequence=11)
        packet1 = raw_packet(RAW_KIND_COMPACT_SCAN, 81, compact_ch1_payload(first))
        packet2 = raw_packet(RAW_KIND_COMPACT_SCAN, 82, compact_ch1_payload(second))
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(packet1[:17])

        adapter._drain_packets()
        self.assertEqual(adapter.in_waiting, 0)

        adapter._network_buffer.extend(packet1[17:] + packet2)
        adapter._drain_packets()
        restored = bytearray(adapter._receive_buffer)
        frames = extract_raw_scan_frames(restored)
        self.assertEqual(frames, [first, second])
        self.assertFalse(restored)
        first_arrival = adapter.take_frame_arrival_info(first)
        second_arrival = adapter.take_frame_arrival_info(second)
        self.assertIsNotNone(first_arrival)
        self.assertIsNotNone(second_arrival)
        self.assertEqual(first_arrival[1], second_arrival[1])
        stats = adapter.transport_stats
        self.assertEqual(stats["data_packets"], 2)
        self.assertEqual(stats["compact_scan_packets"], 2)
        self.assertEqual(stats["invalid_compact_packets"], 0)
        self.assertLess(stats["wire_to_native_ratio"], 1.0)

    def test_out_of_order_wire_sequences_are_not_hidden_or_reused(self):
        later = native_ch1_scan(point_count=2, board_sequence=101)
        earlier = native_ch1_scan(point_count=2, board_sequence=100)
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(
            raw_packet(RAW_KIND_COMPACT_SCAN, 51, compact_ch1_payload(later))
            + raw_packet(RAW_KIND_COMPACT_SCAN, 50, compact_ch1_payload(earlier))
        )

        adapter._drain_packets()

        stream = bytearray(adapter._receive_buffer)
        frames = extract_raw_scan_frames(stream)
        self.assertEqual(frames, [later, earlier])
        tracker = BoardSequenceTracker()
        for frame in frames:
            tracker.observe(int.from_bytes(board_extension(frame)[28:32], "big"))
        self.assertEqual(tracker.stats().resets, 1)

    def test_arrival_time_is_captured_before_a_slow_reader_drains_buffer(self):
        first = native_ch1_scan(point_count=2, board_sequence=201)
        second = native_ch1_scan(point_count=2, board_sequence=202)
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(raw_packet(RAW_KIND_DATA, 1, first))
        adapter._drain_packets(received_at=10.000)
        adapter._network_buffer.extend(raw_packet(RAW_KIND_DATA, 2, second))
        adapter._drain_packets(received_at=10.064)

        # Both frames may already be waiting when the benchmark reads them;
        # their socket-thread arrival timestamps must remain distinct.
        first_arrival = adapter.take_frame_arrival_info(first)
        second_arrival = adapter.take_frame_arrival_info(second)
        self.assertEqual(first_arrival[0], 10.000)
        self.assertEqual(second_arrival[0], 10.064)
        self.assertNotEqual(first_arrival[1], second_arrival[1])

    def test_ready_negotiates_once_and_legacy_data_remains_compatible(self):
        adapter = FbgLanSerial()
        sent = []
        adapter._send = sent.append
        native = native_ch1_scan(point_count=2)
        adapter._network_buffer.extend(
            raw_packet(RAW_KIND_READY, 123)
            + raw_packet(RAW_KIND_READY, 123)
            + raw_packet(RAW_KIND_DATA, 1, native)
        )

        adapter._drain_packets()

        self.assertEqual(sent, [RAW_COMPACT_ENABLE])
        self.assertEqual(bytes(adapter._receive_buffer), native)
        self.assertEqual(adapter.transport_stats["compact_scan_packets"], 0)

    def test_malformed_or_crc_bad_compact_scan_is_never_exposed(self):
        native = native_ch1_scan(point_count=2)
        bad_mask = bytearray(compact_ch1_payload(native))
        bad_mask[1] = 0
        crc_bad = bytearray(
            raw_packet(RAW_KIND_COMPACT_SCAN, 2, compact_ch1_payload(native))
        )
        crc_bad[-1] ^= 1
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(
            raw_packet(RAW_KIND_COMPACT_SCAN, 1, bad_mask) + crc_bad
        )

        adapter._drain_packets()

        self.assertEqual(adapter.in_waiting, 0)
        self.assertEqual(adapter.transport_stats["invalid_compact_packets"], 1)

    def test_crc_framing_returns_native_usb_bytes_and_ready_state(self):
        adapter = FbgLanSerial()
        adapter._network_buffer.extend(
            raw_packet(RAW_KIND_READY, 0)
            + raw_packet(RAW_KIND_DATA, 7, b"\xff\xff\x01\x02")
        )

        adapter._drain_packets()

        self.assertTrue(adapter._raw_ready.is_set())
        self.assertEqual(adapter.in_waiting, 4)
        adapter._thread = object()
        self.assertEqual(adapter.read(4), b"\xff\xff\x01\x02")

    def test_full_808_byte_command_is_chunked_and_acknowledged(self):
        adapter = FbgLanSerial(write_timeout=0.3)
        adapter._thread = object()
        adapter._connected.set()
        commands = []

        def acknowledge(command):
            commands.append(command)
            fields = command.split(b"|", 4)
            request_id = int(fields[1], 16)
            offset = int(fields[2], 16)
            chunk_length = len(bytes.fromhex(fields[4].decode("ascii")))
            next_offset = offset + chunk_length
            adapter._network_buffer.extend(
                raw_packet(
                    RAW_KIND_COMMAND_ACK,
                    request_id,
                    next_offset.to_bytes(2, "big"),
                )
            )
            adapter._drain_packets()

        adapter._send = acknowledge
        data = bytes((index % 251) + 1 for index in range(RAW_COMMAND_SIZE))

        self.assertEqual(adapter.write(data), RAW_COMMAND_SIZE)
        self.assertEqual(len(commands), 5)
        self.assertTrue(commands[0].startswith(b"FUSB1|"))
        self.assertIn(b"|0000|0328|", commands[0])
        self.assertIn(b"|0320|0328|", commands[-1])

    def test_temporary_batch_capability_waits_for_control_ack(self):
        adapter = FbgLanSerial(write_timeout=0.3)
        adapter._thread = object()
        adapter._connected.set()
        adapter._compact_capability_sent = True
        sent = []

        def acknowledge(command):
            sent.append(command)
            if command == RAW_TEMPORARY_BATCH_ENABLE:
                adapter._network_buffer.extend(
                    raw_packet(RAW_KIND_TEMPORARY_CAP_ACK, 123)
                )
            elif command == RAW_TEMPORARY_COMPACT_ENABLE:
                adapter._network_buffer.extend(
                    raw_packet(RAW_KIND_TEMPORARY_COMPACT_CAP_ACK, 124)
                )
            else:
                fields = command.split(b"|", 4)
                request_id = int(fields[1], 16)
                offset = int(fields[2], 16)
                size = len(bytes.fromhex(fields[4].decode("ascii")))
                adapter._network_buffer.extend(
                    raw_packet(
                        RAW_KIND_COMMAND_ACK,
                        request_id,
                        (offset + size).to_bytes(2, "big"),
                    )
                )
            adapter._drain_packets()

        adapter._send = acknowledge
        self.assertEqual(adapter.write(bytes(808)), 808)
        self.assertEqual(sent[-2:], [RAW_TEMPORARY_BATCH_ENABLE, RAW_TEMPORARY_COMPACT_ENABLE])
        self.assertTrue(adapter._temporary_batch_capability_ready.is_set())
        self.assertTrue(adapter._temporary_compact_capability_ready.is_set())
        self.assertEqual(adapter.write(bytes(808)), 808)
        self.assertEqual(sent.count(RAW_TEMPORARY_BATCH_ENABLE), 1)
        self.assertEqual(sent.count(RAW_TEMPORARY_COMPACT_ENABLE), 1)

    def test_older_firmware_without_compact_ack_keeps_native_batch(self):
        adapter = FbgLanSerial(write_timeout=0.3)
        adapter._thread = object()
        adapter._connected.set()
        adapter._compact_capability_sent = True
        adapter._temporary_compact_capability_ready.wait = lambda timeout: False
        sent = []

        def acknowledge(command):
            sent.append(command)
            if command == RAW_TEMPORARY_BATCH_ENABLE:
                adapter._network_buffer.extend(raw_packet(RAW_KIND_TEMPORARY_CAP_ACK, 123))
            elif command != RAW_TEMPORARY_COMPACT_ENABLE:
                fields = command.split(b"|", 4)
                request_id = int(fields[1], 16)
                offset = int(fields[2], 16)
                size = len(bytes.fromhex(fields[4].decode("ascii")))
                adapter._network_buffer.extend(raw_packet(
                    RAW_KIND_COMMAND_ACK, request_id,
                    (offset + size).to_bytes(2, "big"),
                ))
            adapter._drain_packets()

        adapter._send = acknowledge
        self.assertEqual(adapter.write(bytes(808)), 808)
        self.assertTrue(adapter._temporary_batch_capability_ready.is_set())
        self.assertFalse(adapter._temporary_compact_capability_ready.is_set())
        self.assertEqual(adapter.write(bytes(808)), 808)
        self.assertEqual(sent.count(RAW_TEMPORARY_COMPACT_ENABLE), 1)

    def test_corrupt_packet_is_not_exposed_as_adc_data(self):
        adapter = FbgLanSerial()
        packet = bytearray(raw_packet(RAW_KIND_DATA, 1, b"unsafe"))
        packet[-1] ^= 0x01
        adapter._network_buffer.extend(packet)

        adapter._drain_packets()

        self.assertEqual(adapter.in_waiting, 0)

    def test_firmware_relearns_only_an_explicit_hello_and_locks_owner(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")

        # A missing Seed URC may be recovered, but arbitrary SocketDown data
        # must not become a control/OTA owner.
        self.assertIn(
            "if(!lanClientActive && lanSocketId < 0 && !lanSocketResetPending)",
            source,
        )
        self.assertIn(
            "!IsLanSessionHello(payload, (uint32_t)declaredLength)",
            source,
        )
        # A second seed is independently deleted; its delete result cannot
        # clear the selected owner's ConID or RAW state.
        self.assertIn(
            "lanClientActive && connectionId != lanSocketId",
            source,
        )
        self.assertIn("WIFI_LAN_DELETE_REJECTED", source)
        self.assertIn("CompleteLanChildDelete", source)

        # If a reconnect arrives in the same parser drain that expires the
        # former owner, the old ConID must be deleted first.  The still-open
        # newcomer is then promoted only as an unprivileged HELLO candidate.
        overlay = source[
            source.index("static uint8_t StartLanOverlayIfNeeded") :
            source.index("static void CompleteLanChildDelete")
        ]
        self.assertLess(
            overlay.index("if(lanSocketResetPending"),
            overlay.index("if(lanCleanupSocketId"),
        )
        completion = source[
            source.index("static void CompleteLanChildDelete") :
            source.index("static uint8_t IsScanMode", source.index("static void CompleteLanChildDelete"))
        ]
        self.assertIn("PromoteWaitingLanSocket();", completion)
        self.assertIn("lanClientDeadlineMs = WifiNowMs() +", source)
        self.assertIn("WIFI_LAN_STALE_DELETE_GRACE_MS", source)
        self.assertIn(
            "(void)ReuseSelectedLanSocketCandidate(connectionId);",
            source,
        )

    def test_firmware_compact_scan_is_opt_in_and_has_native_fallback(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")
        producer = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "ms5614t.c"
        ).read_text(encoding="utf-8")

        self.assertIn('WIFI_LAN_RAW_COMPACT_CAP       "FBGL1|CAPS|CS1"', source)
        self.assertIn("static uint8_t lanRawCompactScan = 0U;", source)
        self.assertIn("lanRawCompactScan = 0U;", source)
        self.assertIn("lanRawCompactScan = 1U;", source)
        compact = source[
            source.index("uint8_t WifiTransport_QueueRawScanFrame") :
            source.index("void WifiTransport_QueueScanFrame")
        ]
        self.assertGreaterEqual(
            compact.count("return WifiTransport_QueueRawFrame(data, length);"),
            3,
        )
        self.assertIn("WIFI_LAN_RAW_KIND_COMPACT_SCAN", compact)
        self.assertNotIn("142U", compact)
        self.assertNotIn("52U", compact)
        self.assertIn(
            "trailerLength = (uint16_t)(length - nativeSampleEnd);", compact
        )
        self.assertIn("memcpy(&lanRawScan[position], &data[nativeSampleEnd]", compact)
        self.assertIn("WifiTransport_QueueRawScanFrame(", producer)
        self.assertIn("stressActiveChannelMask", producer)
        self.assertIn("WIFI_LAN_RAW_KIND_COMPACT_F45", source)
        self.assertIn("WIFI_LAN_RAW_KIND_COMPACT_F45_BATCH", source)
        self.assertIn("WifiTransport_QueueFastFullMapFrame", source)
        self.assertIn("WifiTransport_FlushFastFullMapFrames", source)
        self.assertIn("WifiTransport_IsLanRawClientActive", producer)
        self.assertIn("WifiTransport_FlushFastFullMapFrames", producer)

        raw_hello = source[
            source.index(
                "if(declaredLength == (sizeof(WIFI_LAN_RAW_HELLO) - 1U)"
            ) : source.index(
                "if(declaredLength == (sizeof(WIFI_LAN_RAW_COMPACT_CAP) - 1U)"
            )
        ]
        self.assertIn("if(!compactAlreadyNegotiated)", raw_hello)

    def test_firmware_warm_wifi_probe_skips_only_matching_ip_association(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")

        self.assertIn("WIFI_STATE_WAIT_WARM_WIFI_QUERY", source)
        self.assertIn('"AT+WJAP?\\r\\n"', source)
        self.assertIn("status == 3", source)
        self.assertIn("wifiConfig.ssid", source)
        self.assertIn("WIFI_FORCED_24G_BSSID", source)

        startup_state = source[
            source.index("case WIFI_STATE_WAIT_ATE0:") :
            source.index("case WIFI_STATE_WAIT_MQTT_DISCONNECT:")
        ]
        self.assertIn("StartWarmWifiProbe();", startup_state)

        warm_state = source[
            source.index("case WIFI_STATE_WAIT_WARM_WIFI_QUERY:") :
            source.index("case WIFI_STATE_WAIT_WMODE:")
        ]
        self.assertIn('"AT+MQTTDISCONN\\r\\n"', warm_state)
        self.assertIn("!warmWifiStatusSeen || !warmWifiMatchesConfig", warm_state)
        self.assertIn("else if(timedOut", warm_state)

        continuation = source[
            source.index("static uint8_t ContinueAfterMqttDisconnect") :
            source.index("static uint8_t DeviceIdIsCanonical")
        ]
        self.assertIn(
            "warmWifiStatusSeen && warmWifiMatchesConfig && wifiConnected",
            continuation,
        )
        self.assertIn('"AT+SOCKETRECVCFG=1\\r\\n"', continuation)
        self.assertIn('"AT+WMODE=1,0\\r\\n"', continuation)

        mqtt_disconnect_state = source[
            source.index("case WIFI_STATE_WAIT_MQTT_DISCONNECT:") :
            source.index("case WIFI_STATE_WAIT_WARM_WIFI_QUERY:")
        ]
        self.assertGreaterEqual(
            mqtt_disconnect_state.count("ContinueAfterMqttDisconnect();"), 2
        )

        # An idempotent ERROR from MQTTDISCONN must consume the already saved
        # WJAP? result.  Unsupported/timed-out WJAP? still performs MQTT
        # cleanup, after which the continuation falls back to a normal join.
        error_path = source[
            source.index("if(responseError)") :
            source.index("if(StartLanOverlayIfNeeded(now))")
        ]
        self.assertIn("WIFI_STATE_WAIT_WARM_WIFI_QUERY", error_path)
        self.assertIn("WIFI_STATE_WAIT_MQTT_DISCONNECT", error_path)
        self.assertIn("ContinueAfterMqttDisconnect();", error_path)
        self.assertIn("warmWifiMatchesConfig = 0U;", source[
            source.index("if(strstr(line, \"+EVENT:WIFI_DISCONNECT\")") :
            source.index("ParseLanSocketLine(line);")
        ])

    def test_network_ota_reset_removed_but_usb_ack_then_reset_retained(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")

        self.assertNotIn('"AT+RST\\r\\n"', source)
        self.assertNotIn("RestartBw20ForOtaMcuReset", source)
        self.assertNotIn("PrepareBw20ForOtaMcuReset", source)

        self.assertIn(
            "QueueLanRawPriority(WIFI_LAN_RAW_KIND_READY,",
            source,
        )
        self.assertIn("OtaUpdate_Init(bootId);", source)
        for token in ("WIFI_PUBLISH_OTA", "DrainBw20RxFor", "ResetMcuAfterUsbDetach",
                      "OtaUpdate_HandleCommand", "otaResetAfterLanDelete", "wanOtaLastCommandMs"):
            self.assertNotIn(token, source)
        ota_source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "ota_update.c"
        ).read_text(encoding="utf-8")
        self.assertIn("metadata.reserved[0] = otaSession.session;", ota_source)
        self.assertIn("(unsigned long)otaRuntimeBootId", ota_source)
        self.assertNotIn("OtaUpdate_ReleaseCommittedSession", ota_source)
        from test_firmware_scan_scheduler import _function
        usb_update = _function(ota_source, "OtaUpdate_ProcessUsb")
        self.assertIn("if(cdc == NULL || cdc->TxState != 0U) return;", usb_update)
        self.assertIn("if(usbUpdateReboot) NVIC_SystemReset();", usb_update)
        self.assertIn("OtaUpdate_HandleCommand(command, length, key, OTA_TRANSPORT_USB", usb_update)
        usb_source = (Path(__file__).resolve().parents[1] / "JDSU/USB_DEVICE/App/usbd_cdc_if.c").read_text(encoding="utf-8")
        self.assertIn("OtaUpdate_QueueUsbFrame(usbAssemblyBuffer", usb_source)
        self.assertNotIn("OtaUpdate_QueueUsbFrame", source)

    def test_firmware_reaps_tcp_seeds_before_listener_recreate_after_ota_removal(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "JDSU" / "Core" / "Src" / "wifi_transport.c"
        ).read_text(encoding="utf-8")

        self.assertIn('"AT+SOCKET?\\r\\n"', source)
        self.assertIn("static int32_t lanSocketId = -1;", source)
        self.assertIn("static int32_t lanStartupStaleChildId = -1;", source)
        self.assertNotIn("connectionId <= 127", source)
        self.assertIn("WIFI_STATE_WAIT_LAN_QUERY_STALE", source)
        self.assertIn("WIFI_STATE_WAIT_LAN_DELETE_CHILD", source)
        self.assertIn("type == 5 && serverConnectionId == WIFI_LAN_SOCKET_ID", source)
        self.assertIn("WIFI_LAN_STARTUP_CLEANUP_MAX", source)
        self.assertIn("WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE", source)
        self.assertIn("WIFI_STATE_WAIT_LAN_VERIFY_CREATED", source)
        self.assertIn("WIFI_STATE_WAIT_LAN_CREATE_GRACE", source)
        self.assertIn("WIFI_STATE_WAIT_LAN_CHILD_DELETE_GRACE", source)
        self.assertIn("WIFI_LAN_SERVER_DELETE_VERIFY_MAX", source)
        self.assertIn('"+EVENT:SocketDissconnect,%d"', source)

        from test_firmware_scan_scheduler import _function
        startup_cleanup = _function(source, "ContinueLanStartupSocketCleanup")
        # A clean parent can be reused only if no child was found or reaped.
        # A stale child still takes the verified parent-recycle path below.
        self.assertEqual(startup_cleanup.count("ContinueAfterLanListenerReady();"), 1)
        self.assertRegex(startup_cleanup, r"if\(lanStartupCleanupAttempts == 0U && lanQuerySeeds == 0U\)"
                         r"\s*\{[\s\S]*?ContinueAfterLanListenerReady\(\);\s*return;\s*\}")
        self.assertLess(startup_cleanup.index("RestartBackoff(WIFI_ERROR_LAN_SOCKET);"),
                        startup_cleanup.index("ContinueAfterLanListenerReady();"))
        self.assertLess(startup_cleanup.index("ContinueAfterLanListenerReady();"),
                        startup_cleanup.index("WIFI_STATE_WAIT_LAN_DELETE_STALE"))
        self.assertIn("ScheduleLanListenerCreate(1U);", startup_cleanup)
        self.assertIn('"AT+SOCKETDEL=%d\\r\\n"', startup_cleanup)
        self.assertIn("lanCleanupOverflow = 1U;", source)

        rejected_owner = source[
            source.index("static void RejectSelectedLanSocket") :
            source.index("static uint8_t ReuseSelectedLanSocketCandidate")
        ]
        self.assertIn("lanCleanupOverflow = 1U;", rejected_owner)
        self.assertIn("lanCleanupCanPromote = 0U;", rejected_owner)
        self.assertNotIn("if(otaWasActive)", rejected_owner)
        self.assertIn(
            "!lanSocketResetPending || lanCleanupOverflow",
            source,
        )
        self.assertIn(
            "if(OtaUpdate_IsActive() || lanCleanupOverflow) canPromote = 0U;",
            source,
        )

        auto_delete = source[
            source.index('if(sscanf(line, "+EVENT:SocketAutoDel,%d"') :
            source.index('if(sscanf(line, "+EVENT:SocketDisconnect,%d"')
        ]
        self.assertIn("lanCleanupOverflow = 1U;", auto_delete)
        self.assertIn("lanCleanupCanPromote = 0U;", auto_delete)

        process_start = source.index("void WifiTransport_Process(void)")
        server_delete = source[
            source.index("case WIFI_STATE_WAIT_LAN_DELETE_STALE:", process_start) :
            source.index("case WIFI_STATE_WAIT_LAN_CREATE:", process_start)
        ]
        self.assertIn("StartLanServerGoneVerification(1U);", server_delete)
        self.assertIn("WIFI_STATE_WAIT_LAN_SERVER_DELETE_GRACE", server_delete)

        listener_create = source[
            source.index("static void VerifyCreatedLanListener") :
            source.index("static void StartLanServerGoneVerification")
        ]
        self.assertIn('"AT+SOCKET?\\r\\n"', listener_create)
        self.assertIn("ContinueCreatedLanListenerVerification", source)
        self.assertIn("lanStartupServerStatus >= 0", source)
        self.assertIn("WifiNowMs() + 3000U", source)
        self.assertIn("ScheduleLanListenerCreate", source)
        listener_ready = source[
            source.index("static void ContinueAfterLanListenerReady") :
            source.index("static void StartLanListenerCreate")
        ]
        self.assertNotIn("lanSocketId = -1;", listener_ready)
        self.assertNotIn("lanCleanupSocketId = -1;", listener_ready)

        receive_state = source[
            source.index("case WIFI_STATE_WAIT_LAN_RECV_CFG:") :
            source.index("case WIFI_STATE_WAIT_LAN_DELETE_STALE:")
        ]
        self.assertIn("StartLanStartupSocketQuery();", receive_state)
        child_delete = source[
            source.index("case WIFI_STATE_WAIT_LAN_DELETE_CHILD:", process_start) :
            source.index("case WIFI_STATE_WAIT_LAN_DELETE_STALE:", process_start)
        ]
        self.assertIn("ScheduleLanStartupChildRequery();", child_delete)
        self.assertIn("WIFI_STATE_WAIT_LAN_CHILD_DELETE_GRACE", child_delete)

        self.assertNotIn("otaResetAfterLanDelete", source)
        self.assertNotIn("WIFI_PUBLISH_OTA", source)
        self.assertIn("WIFI_STATE_LAN_DELETE", source)
        self.assertIn('"AT+SOCKETDEL=%d\\r\\n"', source)


if __name__ == "__main__":
    unittest.main()
