import unittest
import os
import json
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5 import QtWidgets
from commission_finger_center_once import CenterRun, complete_baseline_snapshot, validate_start
import commission_finger_center_once as commission
from mechanical_finger_3d import load_finger_layout, load_machine_touch_samples


class CenterCommissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.target = {"work_x": -6.66, "work_y": 13.39, "contact_z": 0,
                       "depth_mm": 0.1, "calibration_machine_minus_work": [-13, -50, 14]}
        self.status = {"connected": True, "stopped": True, "moving": False,
                       "work_x": -7, "work_y": 7, "work_z": 4,
                       "machine_x": -20, "machine_y": -43, "machine_z": 18,
                       "x_homed": True, "y_homed": True, "z_homed": True}

    def test_confirmed_stationary_geometry_passes(self):
        validate_start(self.status, self.target)

    def test_sparse_cached_points_are_not_complete_baseline(self):
        full = {"acquisition_profile": "MAP", "fresh_point_mask": [True] * 45,
                "point_age_frames": [0] * 45, "bandwidth_discontinuity": False}
        self.assertTrue(complete_baseline_snapshot(full))
        reference = {**full, "bandwidth_discontinuity": True}
        self.assertTrue(complete_baseline_snapshot(reference, full_map_reference=True))
        self.assertFalse(complete_baseline_snapshot(full, full_map_reference=True))
        self.assertFalse(complete_baseline_snapshot(reference))
        for change in ({"acquisition_profile": "TRACK13_WIDE"},
                       {"fresh_point_mask": [True] * 44 + [False]},
                       {"point_age_frames": [0] * 44 + [1]},
                       {"fresh_point_mask": None}, {"point_age_frames": None},
                       {"bandwidth_discontinuity": True}):
            with self.subTest(change=change):
                self.assertFalse(complete_baseline_snapshot({**full, **change}))

    def test_unsafe_states_are_rejected_without_motion(self):
        for change in ({"estop": True}, {"estop_led": True}, {"stopped": False},
                       {"moving": True}, {"connected": False}, {"z_homed": False},
                       {"x_pos_limit": True}, {"machine_x": -31}, {"machine_z": 17.8},
                       {"machine_z": float("nan")}, {"work_x": -25, "machine_x": -38}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_start({**self.status, **change}, self.target)

    def test_work_zero_change_does_not_invalidate_absolute_fixture(self):
        shifted_work = {
            **self.status,
            "work_x": 120.0,
            "work_y": -75.0,
            "work_z": 40.0,
        }
        validate_start(shifted_work, self.target)

    def test_other_depths_are_not_authorised_by_this_run(self):
        for depth in (0, 0.2, 0.8, 0.801, -0.1, float("nan")):
            with self.subTest(depth=depth), self.assertRaises(RuntimeError):
                validate_start(self.status, {**self.target, "depth_mm": depth})

    def test_target_comes_from_current_calibrated_full_face_plan(self):
        fixture = SimpleNamespace(machine_touch_samples=load_machine_touch_samples(),
                                  layout=load_finger_layout())
        dummy = SimpleNamespace(window=SimpleNamespace(finger_controller=fixture))
        target = CenterRun.make_target(dummy)
        self.assertAlmostEqual(target["model_x"], 0)
        self.assertAlmostEqual(target["model_y"], .32)
        self.assertEqual(target["depth_mm"], .1)
        self.assertEqual(target["press_feed_mm_min"], 5)
        self.assertLessEqual(target["dwell_seconds"], 2)
        with mock.patch("commission_finger_center_once.generate_silicone_coverage_plan", return_value=[]):
            with self.assertRaises(RuntimeError):
                CenterRun.make_target(dummy)

    def test_offline_full_recording_lifecycle_dispatches_exactly_once(self):
        # Real window/writer integration, fake device and no hardware methods.
        from fbg_unified_app import UnifiedFbgStudio, SOURCE_LOCAL
        from fbg_network_transport import NETWORK_LAN
        window = UnifiedFbgStudio(initial_source=NETWORK_LAN, demo=True)
        self.app.processEvents()
        window.current_source = SOURCE_LOCAL
        with tempfile.TemporaryDirectory() as directory:
            run = CenterRun(window, Path(directory))
            run.timer.stop()
            run.mark = mock.Mock()
            target = run.make_target()
            offsets = target["calibration_machine_minus_work"]
            status = dict(self.status)
            for index, axis in enumerate("xyz"):
                status[f"machine_{axis}"] = status[f"work_{axis}"] + offsets[index]
            window._last_mach3_status = status
            page = window.local_controller.page_peak
            page.data = [[100] * 45 for _ in range(4)]
            page.adc = [[0.06] * 45 for _ in range(4)]
            page.last_frame_received_monotonic_ns = time.monotonic_ns()
            page.frame_generation = 1
            with (mock.patch.object(window.mach3_emergency, "press_target") as dispatch,
                  mock.patch.object(commission.local, "ser_open", False),
                  mock.patch.object(commission.local, "ser", SimpleNamespace(is_open=False))):
                run.begin_recording()
                self.assertIsNone(run.failure)
                run.phase_started -= 4
                for _ in range(62):
                    page.frame_generation += 1
                    page.last_frame_received_monotonic_ns = time.monotonic_ns()
                    run.tick()
                    window._capture_coverage_phase_frame()
                dispatch.assert_called_once()
                self.assertEqual(run.summary["motion_commands_started"], 1)
                for phase in ("contact", "hold", "release", "release_complete"):
                    page.frame_generation += 1
                    page.last_frame_received_monotonic_ns = time.monotonic_ns()
                    window._begin_coverage_phase(phase, target, status)
                run.finished_motion(True, "offline simulated retraction", status)
                window._complete_coverage_point(window._coverage_token(), "offline")
                run.tick()
                self.assertEqual(run.state, "finished")
                self.assertIsNone(run.failure)
                self.assertGreater(run.summary["saved_frames"], 60)
                for phase in ("pre_contact", "contact", "hold", "release", "release_complete"):
                    self.assertGreater(run.summary["phase_frame_counts"][phase], 0)
                manifest = json.loads((Path(directory) / "session/manifest.json").read_text("utf-8"))
                self.assertTrue(manifest["metadata"]["only_one_press"])
                self.assertFalse(run.summary["optical_shutter_confirmed"])
                dispatch.assert_called_once()
        window.close()
        self.app.processEvents()

    def test_acquisition_only_never_dispatches_motion(self):
        from fbg_unified_app import UnifiedFbgStudio, SOURCE_LOCAL
        from fbg_network_transport import NETWORK_LAN
        window = UnifiedFbgStudio(initial_source=NETWORK_LAN, demo=True)
        self.app.processEvents()
        window.current_source = SOURCE_LOCAL
        with tempfile.TemporaryDirectory() as directory:
            run = CenterRun(window, Path(directory), acquisition_only=True)
            run.timer.stop()
            run.mark = mock.Mock()
            offsets = run.make_target()["calibration_machine_minus_work"]
            status = dict(self.status)
            for index, axis in enumerate("xyz"):
                status[f"machine_{axis}"] = status[f"work_{axis}"] + offsets[index]
            window._last_mach3_status = status
            page = window.local_controller.page_peak
            page.data = [[100] * 45 for _ in range(4)]
            page.adc = [[0.06] * 45 for _ in range(4)]
            page.frame_generation = 1
            page.last_frame_received_monotonic_ns = time.monotonic_ns()
            with (mock.patch.object(window.mach3_emergency, "press_target") as dispatch,
                  mock.patch.object(commission.local, "ser_open", False),
                  mock.patch.object(commission.local, "ser", SimpleNamespace(is_open=False))):
                run.begin_recording()
                run.phase_started -= 4
                for _ in range(62):
                    page.frame_generation += 1
                    page.last_frame_received_monotonic_ns = time.monotonic_ns()
                    run.tick()
                self.assertEqual(run.state, "observing")
                self.assertFalse(run.motion_sent)
                # Observation may report reception gaps, but can never recover by moving.
                page.last_frame_received_monotonic_ns = None
                run.phase_started -= 21
                run.tick()
                self.assertEqual(run.state, "finished")
                self.assertTrue(run.summary["stationary_observation_complete"])
                self.assertEqual(run.summary["motion_commands_started"], 0)
                self.assertNotIn("motion_success", run.summary)
                dispatch.assert_not_called()
                manifest = json.loads((Path(directory) / "session/manifest.json").read_text("utf-8"))
                self.assertFalse(manifest["metadata"]["training_eligible"])
                self.assertFalse(manifest["metadata"]["only_one_press"])
        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
