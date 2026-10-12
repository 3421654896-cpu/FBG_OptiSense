import unittest
from unittest import mock
from threading import Thread

from realtime_frame_queue import LatestFrameBuffer, TimestampedFrame


class LatestFrameBufferTests(unittest.TestCase):
    def test_receipt_and_age_share_monotonic_not_qpc_clock(self):
        # Reproduce Windows where QPC is ahead of GetTickCount64. Receipt
        # must not land in the future relative to the event/capture clock.
        with (mock.patch("realtime_frame_queue.time.monotonic_ns", side_effect=[1_000_000, 3_000_000]),
              mock.patch("realtime_frame_queue.time.perf_counter_ns", return_value=12_000_000) as qpc):
            buffer = LatestFrameBuffer()
            buffer.put(b"frame")
            frame = buffer.take(latest_only=True)
            self.assertEqual(frame.received_ns, 1_000_000)
            self.assertEqual(frame.age_ms(), 2.0)
            qpc.assert_not_called()

    def test_fifo_mode_keeps_calibration_frame_order(self):
        buffer = LatestFrameBuffer(maxlen=4)
        buffer.put(b"first", received_ns=10)
        buffer.put(b"second", received_ns=20)

        self.assertEqual(buffer.take(latest_only=False).payload, b"first")
        self.assertEqual(buffer.take(latest_only=False).payload, b"second")
        self.assertEqual(buffer.stats().dropped, 0)

    def test_latest_mode_skips_stale_complete_frames_and_counts_them(self):
        buffer = LatestFrameBuffer(maxlen=4)
        for index in range(3):
            buffer.put(bytes([index]), received_ns=100 + index)

        newest = buffer.take(latest_only=True)

        self.assertEqual(newest.payload, b"\x02")
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.stats().consumer_skipped, 2)
        self.assertEqual(buffer.stats().dropped, 2)

    def test_overflow_and_consumer_skips_are_reported_separately(self):
        buffer = LatestFrameBuffer(maxlen=2)
        buffer.put(b"a", received_ns=1)
        buffer.put(b"b", received_ns=2)
        buffer.put(b"c", received_ns=3)

        self.assertEqual(buffer.take(latest_only=True).payload, b"c")
        stats = buffer.stats()
        self.assertEqual(stats.overflow_dropped, 1)
        self.assertEqual(stats.consumer_skipped, 1)
        self.assertEqual(stats.dropped, 2)

    def test_frame_age_uses_monotonic_nanoseconds(self):
        frame = TimestampedFrame(payload=b"x", received_ns=1_000_000)
        self.assertEqual(frame.age_ms(now_ns=3_500_000), 2.5)
        self.assertEqual(frame.age_ms(now_ns=500_000), 0.0)

    def test_clear_can_reset_session_statistics(self):
        buffer = LatestFrameBuffer(maxlen=1)
        buffer.put(b"a", received_ns=1)
        buffer.put(b"b", received_ns=2)
        buffer.clear(reset_counters=True)

        self.assertEqual(buffer.stats().received, 0)
        self.assertEqual(buffer.stats().dropped, 0)
        self.assertEqual(len(buffer), 0)

    def test_payload_is_snapshotted_before_parser_buffer_changes(self):
        buffer = LatestFrameBuffer(maxlen=1)
        parser_buffer = bytearray(b"abc")
        buffer.put(parser_buffer, received_ns=1)
        parser_buffer[0] = ord("z")

        self.assertEqual(buffer.take(latest_only=False).payload, b"abc")

    def test_multiple_producers_do_not_lose_statistics(self):
        buffer = LatestFrameBuffer(maxlen=1000)

        def produce(prefix):
            for index in range(250):
                buffer.put(bytes((prefix, index & 0xFF)), received_ns=index)

        producers = [Thread(target=produce, args=(index,)) for index in range(4)]
        for producer in producers:
            producer.start()
        for producer in producers:
            producer.join()

        stats = buffer.stats()
        self.assertEqual(stats.received, 1000)
        self.assertEqual(stats.pending, 1000)
        self.assertEqual(stats.dropped, 0)


if __name__ == "__main__":
    unittest.main()
