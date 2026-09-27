"""Tests for the fixed 5x5 grid fed by temporary-test-mode live frames."""

import os
import sys
from pathlib import Path

# Same Qt plugin bootstrap as app_JDSU.py; without an explicit library path
# the venv interpreter can crash on offscreen startup under some sessions.
qt_platform_plugins = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "PyQt5"
    / "Qt5"
    / "plugins"
    / "platforms"
)
if qt_platform_plugins.is_dir():
    sys.path.append(str(qt_platform_plugins.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from manual_stress_capture import (  # noqa: E402
    FINGER_POINT_LAYOUT,
    FingerSurfaceMap,
    ForceVectorWidget,
    ManualStressCaptureWindow,
    build_training_views,
    extract_frame_vector,
)


class FakeTemporaryPage:
    """Minimal stand-in for TemporaryTestWindow's listener surface."""

    def __init__(self):
        self.frame_listeners = []
        self.reference_wavelengths = [1525.0 + 0.02 * i for i in range(45)]


def temporary_frame(sequence=16, base=200):
    """One decoded temporary-mode frame shaped like temporary_test_mode.decode."""

    return {
        "kind": "data",
        "sequence": sequence,
        "cycle_start_us": sequence * 50_000,
        "elapsed_us": 45_000,
        "records": [
            {
                "first_code": base + point,
                "second_code": base + point + 500,
                "write_end_us": point * 100,
                "second_end_us": point * 100 + 60,
            }
            for point in range(45)
        ],
        "i2c_errors": 0,
        "spi_errors": 0,
        "firmware": 0x010056,
        "feedback_selector": 2,
        "adc_channel": 1,
        "signal_channel_name": "CH1",
        "quality": {
            "warmed_up": sequence >= 16,
            "clipped": False,
            "unsettled": False,
            "stability_verified": False,
            "peak_at_edge": False,
        },
    }


class FixedGridCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths(
            [str(qt_platform_plugins.parent)]
        ) if qt_platform_plugins.is_dir() else None
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_force_vector_widget_keeps_xyz_and_connection_state(self):
        widget = ForceVectorWidget()
        try:
            self.assertFalse(widget.connected)
            widget.set_force(1.25, -2.5, 3.75, connected=True)
            self.assertTrue(widget.connected)
            self.assertEqual(widget.force_xyz, (1.25, -2.5, 3.75))
            widget.set_force(None, None, None, connected=False)
            self.assertFalse(widget.connected)
            self.assertEqual(widget.force_xyz, (0.0, 0.0, 0.0))
        finally:
            widget.close()

    def test_capture_page_embeds_force_vector_beside_model(self):
        with tempfile.TemporaryDirectory() as folder:
            page = FakeTemporaryPage()
            window = ManualStressCaptureWindow(
                page,
                wavelength_provider=lambda: list(page.reference_wavelengths),
                output_root=Path(folder),
            )
            try:
                self.assertIsInstance(window.force_vector, ForceVectorWidget)
                # The panel lives beside the model (not overlaid on it): its
                # parent is the card frame, NOT the map widget, and the row
                # layout keeps them as siblings that cannot cover each other.
                self.assertIsNot(window.force_vector.parent(), window.surface_map)
                self.assertIs(
                    window.force_vector.parent(), window.surface_map.parentWidget()
                )
                self.assertIs(
                    window.force_model_layout.itemAt(0).widget(),
                    window.force_vector,
                )
                self.assertIs(
                    window.force_model_layout.itemAt(1).widget(),
                    window.surface_map,
                )
                self.assertIs(window.zero_button, window.force_vector.zero_button)
                self.assertIs(window.zero_button.parentWidget(), window.force_vector)
                self.assertEqual(window.zero_button.text(), "标定归零")
                self.assertFalse(hasattr(window.surface_map, "force_arrows"))
            finally:
                window.close()

    def test_surface_layout_has_rectangular_5_by_5_numbered_points(self):
        self.assertEqual([point[0] for point in FINGER_POINT_LAYOUT], list(range(1, 26)))
        coordinates = [(point[1], point[2]) for point in FINGER_POINT_LAYOUT]
        self.assertEqual(len(set(coordinates)), 25)
        x_levels = sorted({point[1] for point in FINGER_POINT_LAYOUT})
        y_levels = sorted({point[2] for point in FINGER_POINT_LAYOUT})
        self.assertEqual(len(x_levels), 5)
        self.assertEqual(len(y_levels), 5)
        self.assertEqual(
            set(coordinates),
            {(x, y) for y in y_levels for x in x_levels},
        )

    def test_finger_surface_widget_uses_original_3d_model_view(self):
        surface = FingerSurfaceMap(on_selected=lambda _label: None)
        try:
            self.assertTrue(surface.uses_real_3d_model)
            self.assertIsNotNone(surface.view)
            self.assertEqual(surface.marker_positions.shape, (25, 3))
            self.assertGreater(float(surface.marker_positions[:, 2].min()), 0.0)
        finally:
            surface.close()

    def test_surface_marker_buttons_do_not_overlap(self):
        surface = FingerSurfaceMap(on_selected=lambda _label: None)
        try:
            surface.resize(520, 560)
            surface.show()
            self.app.processEvents()
            surface._position_buttons()
            rectangles = [button.geometry() for button in surface.buttons]
            self.assertTrue(all(button.isVisible() for button in surface.buttons))
            overlaps = [
                (left, right)
                for left in range(len(rectangles))
                for right in range(left + 1, len(rectangles))
                if rectangles[left].intersects(rectangles[right])
            ]
            self.assertEqual(overlaps, [])
        finally:
            surface.close()

    def test_finger_surface_widget_owns_all_numbered_markers(self):
        selected = []
        surface = FingerSurfaceMap(on_selected=selected.append)
        try:
            self.assertEqual(len(surface.buttons), 25)
            self.assertEqual(len(surface.text_items), 25)
            self.assertEqual([item.text for item in surface.text_items],
                             [str(index) for index in range(1, 26)])
            surface.buttons[12].click()
            self.assertEqual(selected, [13])
        finally:
            surface.close()

    def test_build_training_views_keeps_only_stable_adc_and_derived_fits(self):
        wavelengths = [1525.0 + 0.02 * i for i in range(45)]
        frame = temporary_frame(sequence=16, base=300)
        views = build_training_views(frame, wavelengths)

        self.assertEqual(views["stable_adc"], list(range(800, 845)))
        self.assertEqual(len(views["stable_voltage_v"]), 45)
        self.assertEqual(len(views["stable_optical_power_dbm"]), 45)
        self.assertNotIn("first", views)
        self.assertNotIn("raw_adc", views)
        self.assertEqual(views["conversion"]["feedback_selector"], 2)
        self.assertEqual(views["conversion"]["transimpedance_ohm"], 5000.0)
        self.assertEqual(len(views["makima_peak_fits"]), 9)
        first_fit = views["makima_peak_fits"][0]
        self.assertEqual(first_fit["fit_method"], "makima_piecewise_cubic_5point")
        self.assertEqual(len(first_fit["input_adc"]), 5)
        self.assertNotIn("input_first_adc", first_fit)
        self.assertGreater(len(first_fit["curve_x_nm"]), 5)
        self.assertEqual(len(first_fit["curve_x_nm"]), len(first_fit["curve_adc"]))
        self.assertEqual(len(first_fit["curve_adc"]), len(first_fit["curve_voltage_v"]))
        self.assertEqual(len(first_fit["curve_adc"]), len(first_fit["curve_dbm"]))

    def test_extract_frame_vector_returns_45_second_codes(self):
        vector = extract_frame_vector(temporary_frame(base=300))
        self.assertEqual(len(vector), 45)
        self.assertEqual(vector[0], 800)
        self.assertEqual(vector[-1], 844)
        with self.assertRaises(ValueError):
            extract_frame_vector({"records": []})

    def test_page_has_fixed_numbered_5_by_5_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            page = FakeTemporaryPage()
            window = ManualStressCaptureWindow(
                page,
                wavelength_provider=lambda: list(page.reference_wavelengths),
                output_root=Path(folder),
            )
            window.attach()
            try:
                self.assertEqual(len(window.grid_buttons), 25)
                self.assertEqual(
                    [button.toolTip() for button in window.grid_buttons],
                    [f"手指表面位置编号 {index}" for index in range(1, 26)],
                )
                self.assertEqual(window.grid_shape, (5, 5))
                self.assertIn(window._on_temporary_frame, page.frame_listeners)
            finally:
                window.close()

    def test_one_capture_writes_one_45_point_vector_with_number_label(self):
        with tempfile.TemporaryDirectory() as folder:
            page = FakeTemporaryPage()
            window = ManualStressCaptureWindow(
                page,
                wavelength_provider=lambda: list(page.reference_wavelengths),
                output_root=Path(folder),
            )
            window.attach()
            # Deterministic fake force sensor; the real one talks to COM8.
            window.force_sensor_enabled = False
            window.force_last_error = None
            window._read_force_sample = lambda: (0.12, -0.05, 2.5, 0.0, 0.0, 0.0)
            try:
                window.select_grid_point(13)
                for listener in page.frame_listeners:
                    listener(temporary_frame(sequence=16, base=200))
                self.assertEqual(window.frames_seen, 1)
                self.assertEqual(len(window.latest_values), 45)
                self.assertTrue(window.capture_current_press())
                session_dir = window.session_dir
                window.close_session()
            finally:
                window.close()

            frames = [
                json.loads(line)
                for line in (session_dir / "frames.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame["quality"]["press_label"], 13)
        self.assertEqual(frame["quality"]["grid_row"], 3)
        self.assertEqual(frame["quality"]["grid_column"], 3)
        self.assertEqual(frame["quality"]["sample_index"], 1)
        self.assertEqual(frame["quality"]["source_frame_sequence"], 16)
        self.assertEqual(len(frame["raw_adc_codes"]["1"]), 45)
        self.assertEqual(frame["raw_adc_codes"]["1"][0], 700)
        self.assertEqual(frame["raw_adc_codes"]["1"][-1], 744)
        self.assertEqual(frame["adc_value_kind"], "direct_adc")
        # Only the stable second read and its derived representations are kept.
        self.assertEqual(frame["quality"]["stable_adc"][0], 700)
        self.assertNotIn("adc_first_code", frame["quality"])
        self.assertEqual(len(frame["quality"]["stable_voltage_v"]), 45)
        self.assertAlmostEqual(
            frame["quality"]["stable_voltage_v"][0],
            700 * 2.5 / 4096,
            places=9,
        )
        self.assertEqual(len(frame["quality"]["stable_dbm"]), 45)
        self.assertTrue(
            all(isinstance(v, float) for v in frame["quality"]["stable_dbm"])
        )
        self.assertEqual(frame["quality"]["feedback_selector"], 2)
        self.assertEqual(frame["quality"]["transimpedance_ohm"], 5000.0)
        views = frame["quality"]["training_views"]
        self.assertEqual(views["stable_adc"][0], 700)
        self.assertEqual(len(views["stable_voltage_v"]), 45)
        self.assertEqual(len(views["stable_optical_power_dbm"]), 45)
        self.assertNotIn("raw_adc", views)
        self.assertEqual(len(views["makima_peak_fits"]), 9)
        self.assertIn("curve_adc", views["makima_peak_fits"][0])
        self.assertIn("curve_voltage_v", views["makima_peak_fits"][0])
        self.assertIn("curve_dbm", views["makima_peak_fits"][0])
        # The Paxini PX6D six-axis force read at the capture instant.
        force = frame["quality"]["paxini_force"]
        self.assertIsNotNone(force)
        self.assertAlmostEqual(force["fx_n"], 0.12, places=6)
        self.assertAlmostEqual(force["fz_n"], 2.5, places=6)
        self.assertIsNone(frame["quality"]["force_error"])

    def test_capture_records_force_error_when_sensor_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            page = FakeTemporaryPage()
            window = ManualStressCaptureWindow(
                page,
                wavelength_provider=lambda: list(page.reference_wavelengths),
                output_root=Path(folder),
            )
            window.attach()

            def broken_read():
                raise TimeoutError("PX6D cmd 0x01 无应答")

            window._read_force_sample = broken_read
            try:
                window.select_grid_point(1)
                for listener in page.frame_listeners:
                    listener(temporary_frame(sequence=16, base=200))
                # Missing force data must NOT block the capture.
                self.assertTrue(window.capture_current_press())
                session_dir = window.session_dir
                window.close_session()
            finally:
                window.close()

            frames = [
                json.loads(line)
                for line in (session_dir / "frames.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
        self.assertEqual(len(frames), 1)
        self.assertIsNone(frames[0]["quality"]["paxini_force"])
        self.assertIn("无应答", frames[0]["quality"]["force_error"])

    def test_capture_requires_selected_number_and_fresh_45_point_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            page = FakeTemporaryPage()
            window = ManualStressCaptureWindow(
                page,
                wavelength_provider=lambda: list(page.reference_wavelengths),
                output_root=Path(folder),
            )
            window.attach()
            try:
                self.assertFalse(window.capture_current_press())
                self.assertIn("编号", window.capture_status_label.text())
                window.select_grid_point(1)
                # No frame delivered yet -> must refuse with the link hint.
                self.assertFalse(window.capture_current_press())
                self.assertIn("临时模式", window.capture_status_label.text())
                # A wrong-sized frame is counted, never captured.
                for listener in page.frame_listeners:
                    listener({"records": [{"second_code": 1}] * 9})
                self.assertEqual(window.rejected_non45_frames, 1)
                self.assertIsNone(window.latest_values)
            finally:
                window.close()

    def test_stale_frame_refuses_capture(self):
        import time

        with tempfile.TemporaryDirectory() as folder:
            page = FakeTemporaryPage()
            window = ManualStressCaptureWindow(
                page,
                wavelength_provider=lambda: list(page.reference_wavelengths),
                output_root=Path(folder),
            )
            window.attach()
            try:
                window.select_grid_point(2)
                for listener in page.frame_listeners:
                    listener(temporary_frame())
                # Simulate an old frame by rewinding the receipt timestamp.
                window.last_frame_monotonic = time.monotonic() - 5.0
                self.assertFalse(window.capture_current_press())
                self.assertIn("过期", window.capture_status_label.text())
            finally:
                window.close()

    def test_bad_frame_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            window = ManualStressCaptureWindow(
                object(),
                wavelength_provider=lambda: [1525.0] * 45,
                output_root=Path(folder),
            )
            with self.assertRaises(TypeError):
                window.attach()
            window.close()


if __name__ == "__main__":
    unittest.main()
