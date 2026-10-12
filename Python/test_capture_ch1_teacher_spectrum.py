import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import capture_ch1_teacher_spectrum as capture


class TeacherSpectrumCheckpointTests(unittest.TestCase):
    def test_first_use_creates_nested_record_directory_and_writes_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'beta_data' / 'outputs' / 'temporary_test' / 'switch_90pct.json'
            self.assertFalse(output.parent.exists())
            capture.save_checkpoint(output, {'complete': False, 'captures': []})
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),
                             {'complete': False, 'captures': []})
            self.assertFalse(output.with_suffix('.json.tmp').exists())
            capture.save_checkpoint(output, {'complete': True, 'captures': [1]})
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['captures'], [1])

    def test_invalid_parent_is_not_hidden_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)/'not_a_directory'
            parent.touch()
            with self.assertRaises(OSError):
                capture.save_checkpoint(parent/'switch.json', {'captures': []})
            self.assertTrue(parent.is_file())

    def test_extended_reference_window_can_settle_after_old_35_sample_limit(self):
        # The first 40 frames wander.  A clean tail becomes stable only after
        # the old hard 35-frame cutoff, but before the caller's extended cap.
        wandering = [
            (100, 100, 100, 100 + 2 * i, 100, 100)
            for i in range(40)
        ]
        frames = iter(wandering + [(100,) * 6 for _ in range(40)])
        worker = mock.Mock(running=True)
        worker._rx_buffer = bytearray()
        worker._interruptible_wait.return_value = True
        worker._read_available_frames.side_effect = lambda: (next(frames),)
        point = SimpleNamespace(index=9, target_nm=1525.18, measured_nm=1525.18,
                                codes=(1, 2, 3, 4, 5))
        with mock.patch.object(capture, 'command_with_link_retry', return_value=0.0), \
             mock.patch.object(capture.app, 'decode_single_value_monitor_frame',
                               side_effect=lambda frame: frame), \
             mock.patch.object(capture.time, 'sleep'):
            row = capture.acquire_settled_ch1(
                worker, point, .2, max_samples=80, point_timeout_s=3.0
            )
        self.assertTrue(row['stable'])
        self.assertGreater(row['sample_count'], 35)
        self.assertEqual(row['sample_limit'], 80)
        self.assertEqual(row['point_timeout_s'], 3.0)

    def test_optional_raw_samples_preserve_direct_channel_order(self):
        worker = mock.Mock(running=True)
        worker._read_available_frames.return_value = [b'frame'] * 25
        point = SimpleNamespace(index=1, target_nm=1525.02, measured_nm=1525.021,
                                codes=(0, 0, 0, 0, 0))
        with mock.patch.object(capture.app, 'decode_single_value_monitor_frame',
                               return_value=(100, 110, 0, 40, 0, 0)):
            result = capture.acquire_settled_ch1(worker, point, .3,
                                                retain_monitor_samples=True)
            self.assertTrue(result['stable'])
            self.assertEqual(result['monitor_samples'], [[100, 110, 0, 40, 0, 0]] * result['sample_count'])
            self.assertEqual(result['monitor_channels'][3], 'CH1')
            self.assertEqual(result['ch1_adc_code'], 40)
            default = capture.acquire_settled_ch1(worker, point, .3)
            self.assertNotIn('monitor_samples', default)

    def test_short_windows_reader_lock_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher.json"
            with mock.patch.object(
                capture,
                "save_json",
                side_effect=(PermissionError("busy"), None),
            ) as save:
                capture.save_checkpoint(output, {"rows": {}})
            self.assertEqual(save.call_count, 2)

    def test_non_locking_write_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher.json"
            with mock.patch.object(
                capture,
                "save_json",
                side_effect=OSError("disk failed"),
            ):
                with self.assertRaisesRegex(OSError, "disk failed"):
                    capture.save_checkpoint(output, {"rows": {}})


if __name__ == "__main__":
    unittest.main()
