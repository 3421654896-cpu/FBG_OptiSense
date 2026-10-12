import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fast_fullmap_live


class _Device:
    def __init__(self):
        self.dtr = False
        self.writes = []

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)

    def flush(self):
        return None

    def reset_input_buffer(self):
        return None

    def read(self, _size):
        return b"wire"


class _FeedbackWorker:
    def __init__(self, *_args, **_kwargs):
        pass

    def _feedback_command_and_ack(self):
        return True

    def _interruptible_wait(self, _seconds):
        return True


class FastFullMapLiveTests(unittest.TestCase):
    def _common_patches(self):
        return (
            mock.patch.object(
                fast_fullmap_live.app, "EqualIntervalWorker", _FeedbackWorker
            ),
            mock.patch.object(fast_fullmap_live, "_safe_disarm", return_value=True),
            mock.patch.object(fast_fullmap_live, "_close_shutter", return_value=True),
        )

    def test_complete_stream_keeps_only_summary_and_publishes_every_frame(self):
        frames = [b"\xd9\x9d\x03" + bytes((index,)) for index in range(32)]
        terminal_wire = b"\xd9\x9d\x04\x00"
        decoded = [
            {
                "sequence": index,
                "cycle_start_us": index * 32_000,
                "minimum_warmup_complete": index >= 16,
                "i2c_error_delta": 0,
                "spi_error_delta": 0,
            }
            for index in range(32)
        ]
        terminal = {
            "status": 0,
            "stop_reason": 0,
            "firmware_shutter_asserted": True,
            "requested_cycles": 32,
            "completed_cycles": 32,
            "i2c_error_delta": 0,
            "spi_error_delta": 0,
        }
        published = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            feedback, disarm, shutter = self._common_patches()
            with (
                feedback,
                disarm as disarm_mock,
                shutter as shutter_mock,
                mock.patch.object(
                    fast_fullmap_live.fullmap,
                    "extract_frames",
                    return_value=iter(frames + [terminal_wire]),
                ),
                mock.patch.object(
                    fast_fullmap_live.fullmap,
                    "decode_data",
                    side_effect=decoded,
                ),
                mock.patch.object(
                    fast_fullmap_live.fullmap.flank,
                    "decode_end",
                    return_value=terminal,
                ),
            ):
                report = fast_fullmap_live.execute_realtime_device(
                    _Device(), output, cycles=32, spacing_us=50,
                    on_frame=published.append,
                )

            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertEqual(report["frame_count"], 32)
            self.assertEqual(report["frames_retained"], 0)
            self.assertNotIn("frames", report)
            self.assertEqual(len(published), 32)
            self.assertTrue(report["analysis"]["meets_sampling_rate_for_15hz"])
            self.assertFalse(report["analysis"]["physical_15hz_verified"])
            self.assertFalse(report["analysis"]["contact_area_verified"])
            self.assertEqual(disarm_mock.call_count, 2)
            self.assertEqual(shutter_mock.call_count, 2)
            self.assertEqual(saved["frame_count"], 32)

    def test_operator_stop_still_performs_exact_shutdown_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cancelled.json"
            feedback, disarm, shutter = self._common_patches()
            with (
                feedback,
                disarm as disarm_mock,
                shutter as shutter_mock,
            ):
                with self.assertRaises(InterruptedError):
                    fast_fullmap_live.execute_realtime_device(
                        _Device(), output, cycles=32, should_stop=lambda: True
                    )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("InterruptedError", saved["error"])
            self.assertEqual(saved["frame_count"], 0)
            self.assertTrue(saved["safety_cleanup"]["exact_disarm_ack"])
            self.assertTrue(saved["safety_cleanup"]["exact_soa_shutter_ack"])
            self.assertEqual(disarm_mock.call_count, 2)
            self.assertEqual(shutter_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
