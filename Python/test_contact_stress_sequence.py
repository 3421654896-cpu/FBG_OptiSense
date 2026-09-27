"""Tests for the multi-depth contact-stress sequence dataset acquisition."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import app_JDSU  # Match application import order before constructing QApplication.
from PyQt5 import QtWidgets

from contact_stress_sequence import (
    DEPTH_STEPS_MM,
    PHASES,
    build_depth_targets,
    stress_sequence_request_payload,
)


class DepthTargetTests(unittest.TestCase):
    def test_depth_steps_are_ascending_within_press_limit(self):
        self.assertEqual(len(DEPTH_STEPS_MM), 5)
        self.assertEqual(list(DEPTH_STEPS_MM), sorted(DEPTH_STEPS_MM))
        for depth in DEPTH_STEPS_MM:
            self.assertGreater(depth, 0.0)
            self.assertLessEqual(depth, 0.80)

    def test_build_depth_targets_reuses_one_xy_for_all_depths(self):
        point = {
            "point_index": 3,
            "model_x": 0.05,
            "model_y": 0.27,
            "work_x": 1.0,
            "work_y": 2.0,
            "contact_z": 10.0,
            "contact_machine_z": 15.0,
            "target_machine_x": 6.0,
            "target_machine_y": 7.0,
            "calibration_machine_minus_work": (5.0, 5.0, 5.0),
            "xy_feed_mm_min": 60.0,
            "press_feed_mm_min": 15.0,
            "z_travel_feed_mm_min": 60.0,
            "dwell_seconds": 0.4,
        }

        targets = build_depth_targets(point)

        self.assertEqual(len(targets), len(DEPTH_STEPS_MM))
        for depth, target in zip(DEPTH_STEPS_MM, targets):
            self.assertEqual(target["point_index"], 3)
            self.assertEqual(target["work_x"], 1.0)
            self.assertEqual(target["work_y"], 2.0)
            self.assertEqual(target["contact_z"], 10.0)
            self.assertEqual(target["stress_depth_mm"], depth)
            self.assertEqual(target["depth_mm"], depth)
        # Every depth re-presses the same physical point: XY must not move.
        self.assertEqual(
            len({(t["work_x"], t["work_y"]) for t in targets}),
            1,
        )

    def test_payload_requests_one_session_per_depth_sequence(self):
        point = {
            "point_index": 1,
            "model_x": 0.0,
            "model_y": 0.0,
            "work_x": 0.0,
            "work_y": 0.0,
            "contact_z": 5.0,
        }
        payload = stress_sequence_request_payload(
            [point],
            depth_steps_mm=DEPTH_STEPS_MM,
            dwell_seconds=0.4,
            repeats=1,
        )

        self.assertEqual(payload["kind"], "contact_stress_sequence")
        self.assertEqual(payload["depth_steps_mm"], list(DEPTH_STEPS_MM))
        self.assertEqual(payload["repeats"], 1)
        # One location is visited once; the depth ladder is expanded in the
        # execution queue, not in the unique location list.
        self.assertEqual(len(payload["points"]), 1)
        self.assertEqual(len(payload["queue"]), len(DEPTH_STEPS_MM))
        self.assertEqual(
            [item["stress_depth_mm"] for item in payload["queue"]],
            list(DEPTH_STEPS_MM),
        )


class SequenceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_session_layout_uses_event_and_frame_files(self):
        from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame

        wavelengths = [1525.0 + 0.02 * i for i in range(45)]
        with tempfile.TemporaryDirectory() as folder:
            writer = FingerSessionWriter(
                Path(folder) / "session",
                manifest_extra={"kind": "contact_stress_sequence"},
            )
            writer.write_event(ContactEvent(kind="session_start"))
            writer.write_frame(
                SpectrumFrame(
                    sequence=1,
                    wavelengths_nm=wavelengths,
                    raw_adc_codes={1: list(range(45))},
                    adc_value_kind="firmware_rc_estimate",
                    quality={"stress_depth_mm": 0.10, "phase": "hold"},
                )
            )
            writer.close()

            manifest = json.loads(
                (Path(folder) / "session" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            events = (
                Path(folder) / "session" / "events.jsonl"
            ).read_text(encoding="utf-8").strip().splitlines()
            frames = (
                Path(folder) / "session" / "frames.jsonl"
            ).read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(manifest["schema"], "fbg-finger-session/v1")
        self.assertEqual(len(events), 1)
        self.assertEqual(len(frames), 1)
        frame_record = json.loads(frames[0])
        self.assertEqual(frame_record["quality"]["stress_depth_mm"], 0.10)


if __name__ == "__main__":
    unittest.main()
