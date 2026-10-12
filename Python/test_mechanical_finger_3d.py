import os
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from mechanical_finger_3d import (
    DEFAULT_LAYOUT_PATH,
    DEFAULT_MESH_PATH,
    DEFAULT_PRESS_TOOL_PATH,
    DEFAULT_TOUCH_CALIBRATION_PATH,
    MODEL_X_HALF_SPAN_MM,
    MODEL_Y_HALF_SPAN_MM,
    PRESS_FACE_RADIUS_MM,
    SILICONE_LOWER_LANDMARK_X,
    SILICONE_LOWER_LANDMARK_Y,
    SILICONE_TIP_FILL,
    SILICONE_X_RADIUS,
    SILICONE_Y_CENTER,
    SILICONE_Y_RADIUS,
    MechanicalFinger3DWindow,
    _silicone_pad_mesh,
    generate_silicone_coverage_plan,
    load_finger_layout,
    load_machine_touch_samples,
    load_stl_mesh,
    machine_coordinates,
    machine_pose_from_touch_samples,
    machine_target_from_model_point,
    ordered_sensor_peaks,
    prepare_press_tool_mesh,
    prepare_thumb_mesh,
    rotation_from_positive_z,
    relative_effort_label,
    silicone_contains_point,
    silicone_surface_height,
    stress_strength,
    wavelength_shifts_pm,
)
from PyQt5 import QtWidgets


class MechanicalFingerModelTests(unittest.TestCase):
    def test_relative_effort_is_displacement_only(self):
        for clearance, percent in [(4.0, '0.0%'), (0.0, '0.0%'), (-0.2, '25.0%'), (-0.8, '100.0%')]:
            label = relative_effort_label(clearance)
            self.assertIn(percent, label)
            self.assertIn('非实测力、非光谱预测', label)
        self.assertIn('超出标定范围', relative_effort_label(-0.81))
        for value in [float('nan'), float('inf'), -float('inf')]:
            self.assertIn('未知', relative_effort_label(value))

    @classmethod
    def setUpClass(cls):
        cls.qt_app = QtWidgets.QApplication.instance()
        if cls.qt_app is None:
            cls.qt_app = QtWidgets.QApplication([])

    def test_default_layout_maps_all_nine_peaks_once(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        self.assertEqual(layout.channel, 1)
        self.assertEqual(len(layout.sensors), 9)
        self.assertEqual(
            sorted(sensor.peak_index for sensor in layout.sensors), list(range(9))
        )
        np.testing.assert_allclose(
            ordered_sensor_peaks(np.arange(1.0, 10.0), layout),
            np.arange(1.0, 10.0),
        )

    def test_mapping_is_configurable_without_changing_peak_values(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        sensors = list(layout.sensors)
        sensors[0] = replace(sensors[0], peak_index=8)
        sensors[8] = replace(sensors[8], peak_index=0)
        swapped = replace(layout, sensors=tuple(sensors))
        mapped = ordered_sensor_peaks(np.arange(100.0, 109.0), swapped)
        self.assertEqual(mapped[0], 108.0)
        self.assertEqual(mapped[8], 100.0)

    def test_shift_uses_current_spectrum_and_optional_common_mode(self):
        baseline = np.arange(9, dtype=float) + 1500.0
        current = baseline + 0.010
        current[4] += 0.090
        raw = wavelength_shifts_pm(current, baseline)
        corrected = wavelength_shifts_pm(current, baseline, remove_common_mode=True)
        np.testing.assert_allclose(raw[[0, 4]], (10.0, 100.0), atol=1e-8)
        np.testing.assert_allclose(corrected[[0, 4]], (0.0, 90.0), atol=1e-8)

    def test_strength_respects_activation_and_full_scale(self):
        result = stress_strength((0.0, 5.0, 52.5, 100.0, -100.0), 5.0, 100.0)
        np.testing.assert_allclose(result, (0.0, 0.0, 0.5, 1.0, 1.0))

    def test_contact_estimate_renders_nine_weights_and_area_disclaimer(self):
        window = MechanicalFinger3DWindow()
        estimate = SimpleNamespace(
            response_weights=(0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0),
            contact_detected=True,
            x_mm=-5.0,
            y_mm=12.0,
            confidence=0.82,
            effective_response_area_mm2=17.5,
            acquisition_profile="MAP",
        )
        window.update_from_contact_estimate(estimate, frame_number=7)
        self.assertEqual(len(window.contact_response_strength), 9)
        self.assertEqual(int(np.argmax(window.contact_response_strength)), 4)
        self.assertIn("有效响应面积（估计影响范围）", window.contact_metrics_label.text())
        self.assertIn("压头名义接触面积：50.27 mm²", window.contact_metrics_label.text())
        self.assertIn("非实际接触面积实测", window.contact_metrics_label.text())
        self.assertIn("82%", window.contact_metrics_label.text())
        window.close()
        window.deleteLater()

    def test_fast_flank_result_is_rendered_as_change_without_area(self):
        window = MechanicalFinger3DWindow()
        result = SimpleNamespace(
            sequence=42,
            one_frame_delta_pm=(0.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 0.0),
            activation_threshold_pm=(5.0,) * 9,
            contact_detected=True,
            provisional_x_mm=-6.2,
            provisional_y_mm=12.1,
            confidence=0.76,
            sampling_gate_for_15hz=True,
            physical_contact_area_mm2=None,
        )
        window.update_from_fast_flank_result(result)
        self.assertEqual(int(np.argmax(window.contact_response_strength)), 4)
        self.assertIn("检测到压力变化", window.contact_metrics_label.text())
        self.assertIn("15 Hz采样率门槛通过", window.contact_metrics_label.text())
        self.assertIn("压头名义接触面积：50.27 mm²", window.contact_metrics_label.text())
        self.assertIn("非实际接触面积实测", window.contact_metrics_label.text())
        window.close()
        window.deleteLater()

    def test_fast_flank_controls_explain_baseline_and_safe_stop(self):
        window = MechanicalFinger3DWindow()
        window.set_fast_flank_running(True)
        self.assertFalse(window.fast_flank_start_button.isEnabled())
        self.assertTrue(window.fast_flank_stop_button.isEnabled())
        self.assertFalse(window.coverage_start_button.isEnabled())
        window.set_fast_flank_progress(16, 1024)
        self.assertIn("最短预热", window.fast_flank_status_label.text())
        window.set_fast_flank_progress(48, 1024)
        self.assertIn("32/64", window.fast_flank_status_label.text())
        window.set_fast_flank_progress(81, 1024)
        self.assertIn("可以施压", window.fast_flank_status_label.text())
        window.finish_fast_flank("已确认关光", success=True)
        self.assertTrue(window.fast_flank_start_button.isEnabled())
        self.assertFalse(window.fast_flank_stop_button.isEnabled())
        self.assertIn("已确认关光", window.fast_flank_status_label.text())
        window.close()
        window.deleteLater()

    def test_physical_reference_import_is_qualified_but_not_claimed_as_pass(self):
        window = MechanicalFinger3DWindow()
        payload = {
            "schema": "fbg-physical-reference/v1",
            "quantity": "force",
            "unit": "N",
            "source_device": "load-cell-01",
            "calibration_id": "LOAD-CAL-1",
            "synchronization_method": "shared-host-monotonic-callback",
            "maximum_clock_error_ms": 0.4,
            "timebase": "host_monotonic_ns",
            "synchronized": True,
            "samples": [
                {"monotonic_ns": index * 10_000_000, "value": float(index)}
                for index in range(12)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(window.load_physical_reference_file(path))
        text = window.truth_readiness_label.text()
        self.assertIn("load-cell-01", text)
        self.assertIn("LOAD-CAL-1", text)
        self.assertIn("仍需", text)
        window.close()
        window.deleteLater()

    def test_unsynchronised_reference_fails_closed_in_ui(self):
        window = MechanicalFinger3DWindow()
        payload = {
            "schema": "fbg-physical-reference/v1",
            "quantity": "force",
            "unit": "N",
            "source_device": "load-cell-01",
            "calibration_id": "LOAD-CAL-1",
            "synchronization_method": "none",
            "maximum_clock_error_ms": 0.4,
            "timebase": "host_monotonic_ns",
            "synchronized": False,
            "samples": [
                {"monotonic_ns": index * 10_000_000, "value": float(index)}
                for index in range(12)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(window.load_physical_reference_file(path))
        self.assertIn("未声明已同步", window.truth_readiness_label.text())
        self.assertIsNone(window.physical_reference_trace)
        window.close()
        window.deleteLater()

    def test_contact_area_truth_import_reports_count_and_levels(self):
        window = MechanicalFinger3DWindow()
        payload = {
            "schema": "fbg-contact-area-truth/v1",
            "records": [
                {
                    "event_id": f"press-{index:03d}",
                    "area_mm2": (7.0, 14.0, 21.0)[index % 3],
                    "source": "pressure_film",
                    "measurement_id": f"film-{index:03d}",
                    "calibration_id": "FILM-CAL-1",
                    "independent": True,
                }
                for index in range(12)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "area.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(window.load_contact_area_truth_file(path))
        text = window.truth_readiness_label.text()
        self.assertIn("12次", text)
        self.assertIn("3个面积等级", text)
        self.assertIn("FILM-CAL-1", text)
        window.close()
        window.deleteLater()

    def test_same_window_runs_complete_ch1_physical_acceptance(self):
        window = MechanicalFinger3DWindow()
        times_s = np.arange(0.0, 1.001, 1.0 / 60.0)
        timestamps = np.rint(times_s * 1_000_000_000).astype(np.int64)
        angles = 2.0 * math.pi * 15.0 * times_s
        reference = {
            "schema": "fbg-physical-reference/v1",
            "quantity": "force",
            "unit": "N",
            "source_device": "load-cell-01",
            "calibration_id": "LOAD-CAL-1",
            "synchronization_method": "shared-host-monotonic-callback",
            "maximum_clock_error_ms": 0.2,
            "timebase": "host_monotonic_ns",
            "synchronized": True,
            "samples": [
                {"monotonic_ns": int(t), "value": float(math.sin(a))}
                for t, a in zip(timestamps, angles)
            ],
        }
        areas = {
            "schema": "fbg-contact-area-truth/v1",
            "records": [
                {
                    "event_id": f"press-{index:03d}",
                    "area_mm2": (8.0, 16.0, 24.0)[index % 3],
                    "source": "pressure_film",
                    "measurement_id": f"film-{index:03d}",
                    "calibration_id": "FILM-CAL-1",
                    "independent": True,
                }
                for index in range(12)
            ],
        }
        session = {
            "schema": "fbg-physical-acceptance-session/v1",
            "frequency_hz": 15.0,
            "low_frequency_gain_output_per_input": 2.0,
            "independent_held_out_test": True,
            "position_truth_source": "mach3_calibrated",
            "ch1_frames": [
                {
                    "monotonic_ns": int(t),
                    "response_value": float(
                        2.0 * math.sin(a + math.radians(-5.0))
                    ),
                }
                for t, a in zip(timestamps, angles)
            ],
            "spatial_records": [
                {
                    "event_id": f"press-{index:03d}",
                    "true_x_mm": float(index % 4) * 3.0,
                    "true_y_mm": float(index // 4) * 3.0,
                    "predicted_x_mm": float(index % 4) * 3.0 + 0.1,
                    "predicted_y_mm": float(index // 4) * 3.0 - 0.1,
                    "predicted_area_mm2": (8.0, 16.0, 24.0)[index % 3]
                    + 0.2,
                }
                for index in range(12)
            ],
            "limits": {
                "maximum_amplitude_error_db": 1.0,
                "maximum_phase_error_deg": 10.0,
                "minimum_tone_r_squared": 0.95,
                "maximum_position_p95_error_mm": 1.0,
                "maximum_area_p95_error_mm2": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {
                name: Path(directory) / f"{name}.json"
                for name in ("reference", "areas", "session")
            }
            for name, payload in (
                ("reference", reference),
                ("areas", areas),
                ("session", session),
            ):
                paths[name].write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(window.load_physical_reference_file(paths["reference"]))
            self.assertFalse(window.run_physical_acceptance_button.isEnabled())
            self.assertTrue(window.load_contact_area_truth_file(paths["areas"]))
            self.assertTrue(window.run_physical_acceptance_button.isEnabled())
            self.assertTrue(
                window.load_physical_acceptance_session_file(paths["session"])
            )
            self.assertTrue(window.physical_acceptance_report_path.is_file())
            saved_report = json.loads(
                window.physical_acceptance_report_path.read_text(encoding="utf-8")
            )
            self.assertTrue(saved_report["end_to_end_goal_verified"])
            self.assertEqual(
                set(saved_report["input_files"]),
                {
                    "pressure_reference",
                    "contact_area_truth",
                    "ch1_acceptance_session",
                },
            )
            self.assertTrue(
                all(
                    len(item["sha256"]) == 64
                    for item in saved_report["input_files"].values()
                )
            )
        text = window.truth_readiness_label.text()
        self.assertIn("物理验收：全部通过", text)
        self.assertIn("15 Hz 通过", text)
        self.assertIn("位置P95", text)
        self.assertIn("面积P95", text)
        window.close()
        window.deleteLater()

    def test_acceptance_rejects_reference_changed_after_import(self):
        window = MechanicalFinger3DWindow()
        reference = {
            "schema": "fbg-physical-reference/v1",
            "quantity": "force",
            "unit": "N",
            "source_device": "load-cell-01",
            "calibration_id": "LOAD-CAL-1",
            "synchronization_method": "shared-host-monotonic-callback",
            "maximum_clock_error_ms": 0.2,
            "timebase": "host_monotonic_ns",
            "synchronized": True,
            "samples": [
                {"monotonic_ns": index * 10_000_000, "value": float(index)}
                for index in range(12)
            ],
        }
        areas = {
            "schema": "fbg-contact-area-truth/v1",
            "records": [
                {
                    "event_id": f"press-{index:03d}",
                    "area_mm2": (8.0, 16.0, 24.0)[index % 3],
                    "source": "pressure_film",
                    "measurement_id": f"film-{index:03d}",
                    "calibration_id": "FILM-CAL-1",
                    "independent": True,
                }
                for index in range(12)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.json"
            area_path = Path(directory) / "areas.json"
            session_path = Path(directory) / "session.json"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            area_path.write_text(json.dumps(areas), encoding="utf-8")
            session_path.write_text("{}", encoding="utf-8")
            self.assertTrue(window.load_physical_reference_file(reference_path))
            self.assertTrue(window.load_contact_area_truth_file(area_path))
            reference_path.write_text(
                json.dumps(reference, indent=2), encoding="utf-8"
            )
            self.assertFalse(
                window.load_physical_acceptance_session_file(session_path)
            )
        self.assertIn("发生变化", window.truth_readiness_label.text())
        window.close()
        window.deleteLater()

    def test_real_solidworks_thumb_mesh_is_deployable(self):
        vertices, faces = load_stl_mesh(DEFAULT_MESH_PATH)
        self.assertGreater(len(vertices), 4000)
        self.assertGreater(len(faces), 9000)
        self.assertLess(int(np.max(faces)), len(vertices))

        prepared, normalized = prepare_thumb_mesh(vertices)
        lower = np.min(prepared, axis=0)
        upper = np.max(prepared, axis=0)
        span = upper - lower
        np.testing.assert_allclose(
            (lower[:2] + upper[:2]) * 0.5, (0.0, 0.0), atol=1e-12
        )
        self.assertAlmostEqual(float(lower[2]), 0.0)
        np.testing.assert_allclose(
            np.max(np.abs(normalized), axis=0), (1.0, 1.0), atol=1e-7
        )
        self.assertAlmostEqual(float(span[0] * 0.5), MODEL_X_HALF_SPAN_MM)
        self.assertAlmostEqual(float(span[1] * 0.5), MODEL_Y_HALF_SPAN_MM)

    def test_silicone_skin_follows_the_photographed_incline(self):
        model_height = 30.695908
        tip = silicone_surface_height(0.0, 0.55, model_height)
        middle = silicone_surface_height(0.0, 0.20, model_height)
        root = silicone_surface_height(0.0, -0.15, model_height)
        self.assertLess(tip, middle)
        self.assertLess(middle, root)
        self.assertGreater(
            silicone_surface_height(0.0, 0.20, model_height),
            silicone_surface_height(0.65, 0.20, model_height),
        )

    def test_all_fbg_marks_stay_inside_the_recessed_silicone_boundary(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        for sensor in layout.sensors:
            pocket_radius = (abs(sensor.x) / SILICONE_X_RADIUS) ** 4 + (
                abs(sensor.y - SILICONE_Y_CENTER) / SILICONE_Y_RADIUS
            ) ** 4
            self.assertLess(pocket_radius, 1.0, sensor.name)

    def test_fbg_marks_form_a_regular_three_by_three_grid(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        grid = np.asarray([(sensor.x, sensor.y) for sensor in layout.sensors])
        np.testing.assert_allclose(grid[0:3, 1], grid[0, 1])
        np.testing.assert_allclose(grid[3:6, 1], grid[3, 1])
        np.testing.assert_allclose(grid[6:9, 1], grid[6, 1])
        np.testing.assert_allclose(grid[[0, 3, 6], 0], grid[0, 0])
        np.testing.assert_allclose(grid[[1, 4, 7], 0], grid[1, 0])
        np.testing.assert_allclose(grid[[2, 5, 8], 0], grid[2, 0])
        self.assertAlmostEqual(
            grid[1, 0], 0.5 * (grid[0, 0] + grid[2, 0]), delta=1e-6
        )
        self.assertAlmostEqual(
            grid[3, 1], 0.5 * (grid[0, 1] + grid[6, 1]), delta=1e-6
        )

    def test_click_target_accepts_silicone_and_rejects_outside(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        samples = load_machine_touch_samples(DEFAULT_TOUCH_CALIBRATION_PATH)
        self.assertTrue(silicone_contains_point(0.0, SILICONE_Y_CENTER))
        self.assertFalse(silicone_contains_point(0.9, SILICONE_Y_CENTER))

        g5 = layout.sensors[4]
        target = machine_target_from_model_point(
            g5.x, g5.y, samples, layout.sensors
        )
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target[0], -5.7162403744, places=6)
        self.assertAlmostEqual(target[1], 12.0375628409, places=6)
        self.assertAlmostEqual(target[2], 0.0)
        self.assertIsNone(
            machine_target_from_model_point(0.9, 0.9, samples, layout.sensors)
        )

    def test_silicone_tip_fills_only_the_central_crescent(self):
        _, _, normalized = _silicone_pad_mesh(
            (np.asarray((-1.0, -1.0)), np.asarray((1.0, 1.0))),
            30.0,
            radial_rings=8,
            angular_segments=32,
        )
        tip_index = int(np.argmax(normalized[:, 1]))
        self.assertAlmostEqual(float(normalized[tip_index, 0]), 0.0, places=7)
        self.assertAlmostEqual(
            float(normalized[tip_index, 1]),
            SILICONE_Y_CENTER + SILICONE_Y_RADIUS + SILICONE_TIP_FILL,
            places=7,
        )

    def test_v2_press_tool_is_deployable_and_tip_is_at_origin(self):
        vertices, faces = load_stl_mesh(DEFAULT_PRESS_TOOL_PATH)
        self.assertEqual(len(faces), 312)
        prepared = prepare_press_tool_mesh(vertices)
        lower = np.min(prepared, axis=0)
        upper = np.max(prepared, axis=0)
        np.testing.assert_allclose(
            (lower[:2] + upper[:2]) * 0.5, (0.0, 0.0), atol=1e-12
        )
        self.assertAlmostEqual(float(lower[2]), 0.0)
        np.testing.assert_allclose(upper - lower, (8.0, 7.993512, 25.0))

    def test_press_face_rotation_tracks_the_finger_surface_normal(self):
        normal = np.asarray((0.0, 1.0, 1.0), dtype=float)
        normal /= np.linalg.norm(normal)
        angle, axis = rotation_from_positive_z(normal)

        self.assertAlmostEqual(angle, 45.0)
        np.testing.assert_allclose(axis, (-1.0, 0.0, 0.0), atol=1e-12)

    def test_two_silicone_boundary_points_map_the_full_model(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        samples = load_machine_touch_samples(DEFAULT_TOUCH_CALIBRATION_PATH)
        self.assertEqual(len(samples), 2)
        self.assertIsNone(samples[0].sensor_id)
        self.assertIsNone(samples[1].sensor_id)
        self.assertEqual(samples[0].surface_kind, "silicone")
        self.assertEqual(samples[1].surface_kind, "silicone")
        self.assertAlmostEqual(samples[0].x_norm, SILICONE_LOWER_LANDMARK_X)
        self.assertAlmostEqual(samples[1].x_norm, -SILICONE_LOWER_LANDMARK_X)
        self.assertAlmostEqual(samples[0].y_norm, SILICONE_LOWER_LANDMARK_Y)
        self.assertAlmostEqual(samples[1].y_norm, SILICONE_LOWER_LANDMARK_Y)
        self.assertTrue(
            silicone_contains_point(
                SILICONE_LOWER_LANDMARK_X, SILICONE_LOWER_LANDMARK_Y
            )
        )

        right_status = {
            "work_x": 0.0,
            "work_y": 0.0,
            "work_z": 0.0,
        }
        np.testing.assert_allclose(machine_coordinates(right_status), (0.0, 0.0, 0.0))
        pose = machine_pose_from_touch_samples(
            right_status, samples, layout.sensors
        )
        self.assertIsNotNone(pose)
        self.assertEqual(pose.nearest_sensor_id, 9)
        self.assertAlmostEqual(pose.x_norm, SILICONE_LOWER_LANDMARK_X)
        self.assertAlmostEqual(pose.y_norm, SILICONE_LOWER_LANDMARK_Y)
        self.assertAlmostEqual(pose.clearance_mm, 0.0)
        self.assertEqual(pose.mapping_kind, "anchor")
        self.assertEqual(pose.surface_kind, "silicone")

        left_retracted = {
            "work_x": -13.85,
            "work_y": 0.2765625,
            "work_z": 1.771875,
            "machine_x": -26.9484375,
            "machine_y": -49.828125,
            "machine_z": 15.7765625,
        }
        left_pose = machine_pose_from_touch_samples(
            left_retracted, samples, layout.sensors
        )
        self.assertIsNotNone(left_pose)
        self.assertAlmostEqual(left_pose.x_norm, -SILICONE_LOWER_LANDMARK_X)
        self.assertAlmostEqual(left_pose.y_norm, SILICONE_LOWER_LANDMARK_Y)
        self.assertAlmostEqual(left_pose.clearance_mm, 1.771875)
        self.assertEqual(left_pose.mapping_kind, "anchor")

        offsets = np.asarray(
            [
                (
                    sample.machine_x - sample.work_x,
                    sample.machine_y - sample.work_y,
                    sample.machine_z - sample.work_z,
                )
                for sample in samples
            ],
            dtype=float,
        )
        np.testing.assert_allclose(offsets[0], offsets[1], atol=1e-12)

        outside = dict(right_status, work_x=20.0)
        self.assertIsNone(
            machine_pose_from_touch_samples(outside, samples, layout.sensors)
        )

        midpoint_status = {
            "work_x": -6.925,
            "work_y": 0.13828125,
            "work_z": 2.0,
        }
        midpoint_pose = machine_pose_from_touch_samples(
            midpoint_status, samples, layout.sensors
        )
        self.assertIsNotNone(midpoint_pose)
        self.assertAlmostEqual(midpoint_pose.x_norm, 0.0, places=9)
        self.assertAlmostEqual(
            midpoint_pose.y_norm, SILICONE_LOWER_LANDMARK_Y, places=9
        )
        self.assertAlmostEqual(midpoint_pose.clearance_mm, 2.0)
        self.assertEqual(midpoint_pose.mapping_kind, "similarity")

        g5 = layout.sensors[4]
        g5_target = machine_target_from_model_point(
            g5.x, g5.y, samples, layout.sensors
        )
        g5_pose = machine_pose_from_touch_samples(
            {
                "work_x": g5_target[0],
                "work_y": g5_target[1],
                "work_z": 2.0,
            },
            samples,
            layout.sensors,
        )
        self.assertIsNotNone(g5_pose)
        self.assertEqual(g5_pose.nearest_sensor_id, 5)
        self.assertAlmostEqual(g5_pose.x_norm, g5.x, places=9)
        self.assertAlmostEqual(g5_pose.y_norm, g5.y, places=9)
        self.assertAlmostEqual(g5_pose.clearance_mm, 2.0)
        self.assertEqual(g5_pose.mapping_kind, "similarity")

    def test_route_from_boundary_anchor_to_sensor_grid_stays_continuous(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        samples = load_machine_touch_samples(DEFAULT_TOUCH_CALIBRATION_PATH)
        g5 = layout.sensors[4]
        g5_target = machine_target_from_model_point(
            g5.x, g5.y, samples, layout.sensors
        )
        poses = []
        for fraction in np.linspace(0.0, 1.0, 21):
            status = {
                "work_x": g5_target[0] * fraction,
                "work_y": g5_target[1] * fraction,
                "work_z": 4.0,
            }
            pose = machine_pose_from_touch_samples(status, samples, layout.sensors)
            self.assertIsNotNone(pose, fraction)
            poses.append(np.asarray((pose.x_norm, pose.y_norm)))

        step_sizes = np.linalg.norm(np.diff(np.asarray(poses), axis=0), axis=1)
        self.assertLess(float(np.max(step_sizes)), 0.04)

    def test_machine_absolute_coordinates_ignore_work_offset_changes(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        samples = load_machine_touch_samples(DEFAULT_TOUCH_CALIBRATION_PATH)
        g5 = layout.sensors[4]
        g5_target = machine_target_from_model_point(
            g5.x, g5.y, samples, layout.sensors
        )
        offset = np.asarray((-13.0984375, -50.1046875, 14.0046875))
        base = {
            "work_x": g5_target[0],
            "work_y": g5_target[1],
            "work_z": 2.0,
            "machine_x": g5_target[0] + offset[0],
            "machine_y": g5_target[1] + offset[1],
            "machine_z": 2.0 + offset[2],
        }
        shifted_work = dict(base, work_x=93.0, work_y=-71.0, work_z=12.0)
        first = machine_pose_from_touch_samples(base, samples, layout.sensors)
        shifted = machine_pose_from_touch_samples(
            shifted_work, samples, layout.sensors
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(shifted)
        np.testing.assert_allclose(
            (first.x_norm, first.y_norm, first.clearance_mm),
            (shifted.x_norm, shifted.y_norm, shifted.clearance_mm),
            atol=1e-12,
        )

    def test_four_mm_plan_keeps_entire_press_face_inside_silicone(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        samples = load_machine_touch_samples(DEFAULT_TOUCH_CALIBRATION_PATH)
        plan = generate_silicone_coverage_plan(
            samples, layout.sensors, spacing_mm=4.0
        )
        self.assertGreater(len(plan), 0)
        for point in plan:
            self.assertTrue(
                silicone_contains_point(point["model_x"], point["model_y"])
            )
            centre_machine = np.asarray(
                (point["work_x"], point["work_y"]), dtype=float
            )
            for angle in np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False):
                perimeter_machine = centre_machine + PRESS_FACE_RADIUS_MM * np.asarray(
                    (np.cos(angle), np.sin(angle)), dtype=float
                )
                # The current calibration consists of the two lower silicone
                # anchors, so recover the normalized position with the same
                # similarity map used by production.
                pose = machine_pose_from_touch_samples(
                    {
                        "work_x": float(perimeter_machine[0]),
                        "work_y": float(perimeter_machine[1]),
                        "work_z": float(point["contact_z"]),
                    },
                    samples,
                    layout.sensors,
                )
                self.assertIsNotNone(pose)
                self.assertTrue(silicone_contains_point(pose.x_norm, pose.y_norm))

    def test_coverage_spacing_is_guarded(self):
        layout = load_finger_layout(DEFAULT_LAYOUT_PATH)
        samples = load_machine_touch_samples(DEFAULT_TOUCH_CALIBRATION_PATH)
        with self.assertRaises(ValueError):
            generate_silicone_coverage_plan(samples, layout.sensors, spacing_mm=1.9)
        with self.assertRaises(ValueError):
            generate_silicone_coverage_plan(samples, layout.sensors, spacing_mm=8.1)


if __name__ == "__main__":
    unittest.main()
