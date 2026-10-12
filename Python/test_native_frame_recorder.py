import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from native_frame_recorder import NativeFrameRecorder
from realtime_frame_queue import LatestFrameBuffer


class NativeRecorderTests(unittest.TestCase):
    def test_gui_latest_only_drops_never_drop_native_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            buffer = LatestFrameBuffer(maxlen=2)
            output = Path(directory) / "native.jsonl"
            recorder = NativeFrameRecorder(buffer, output)
            for n in range(400):
                buffer.put(n.to_bytes(2, "big"), received_ns=n + 1)
            self.assertEqual(buffer.take(latest_only=True).payload, (399).to_bytes(2, "big"))
            self.assertEqual(buffer.stats().dropped, 399)
            stats = recorder.close()
            self.assertEqual(stats["written"], 400)
            rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
            self.assertEqual([r["host_received_monotonic_ns"] for r in rows], list(range(1, 401)))
            self.assertEqual([bytes.fromhex(r["payload_hex"]) for r in rows], [n.to_bytes(2, "big") for n in range(400)])
            buffer.put(b"later")
            self.assertEqual(recorder.close()["offered"], 400)
            with self.assertRaises(FileExistsError):
                NativeFrameRecorder(buffer, output)

    def test_tap_overflow_is_visible_and_does_not_block_ui(self):
        buffer = LatestFrameBuffer()
        tap = buffer.subscribe(capacity=1)
        buffer.put(b"first", received_ns=1)
        buffer.put(b"second", received_ns=2)
        self.assertEqual(tap.stats()["dropped"], 1)
        self.assertEqual(buffer.take(latest_only=True).payload, b"second")
        self.assertEqual(buffer.stats().last_received_ns, 2)
        buffer.unsubscribe(tap)
        buffer.put(b"third", received_ns=3)
        self.assertEqual(tap.stats()["offered"], 2)
        buffer.clear()
        self.assertIsNone(buffer.stats().last_received_ns)

    def test_capture_overflow_fails_close_not_silently_training(self):
        gate = threading.Event()
        original_run = NativeFrameRecorder._run
        def delayed_run(recorder):
            gate.wait(timeout=2)
            original_run(recorder)
        with tempfile.TemporaryDirectory() as directory:
            buffer = LatestFrameBuffer()
            with mock.patch.object(NativeFrameRecorder, "_run", delayed_run):
                recorder = NativeFrameRecorder(buffer, Path(directory) / "native.jsonl", capacity=1)
                buffer.put(b"first")
                buffer.put(b"overflow")
                gate.set()
                with self.assertRaisesRegex(RuntimeError, "recording incomplete"):
                    recorder.close()
                self.assertEqual(recorder.stats()["dropped"], 1)
                buffer.put(b"unsubscribed")
                self.assertEqual(recorder.stats()["offered"], 2)

    def test_disk_error_is_reported(self):
        fake_file = mock.Mock()
        fake_file.write.side_effect = OSError("disk full")
        with mock.patch.object(Path, "open", return_value=fake_file):
            buffer = LatestFrameBuffer()
            recorder = NativeFrameRecorder(buffer, "unused.jsonl")
            buffer.put(b"first")
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                recorder.close()
            self.assertEqual(recorder.stats()["written"], 0)
            fake_file.close.assert_called_once()

    def test_invalid_capacity_does_not_create_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.jsonl"
            with self.assertRaises(ValueError):
                NativeFrameRecorder(LatestFrameBuffer(), path, capacity=0)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
