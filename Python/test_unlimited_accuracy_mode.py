import os
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_JDSU
import fbg_unified_app


class UnlimitedAccuracyModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (
            app_JDSU.QtWidgets.QApplication.instance()
            or app_JDSU.QtWidgets.QApplication([])
        )

    def test_operational_table_and_guards_are_complete_and_safe(self):
        points = app_JDSU.load_fullband_accuracy_table()
        guards = app_JDSU.load_fullband_transition_guards(points)
        self.assertEqual(len(points), 2001)
        self.assertAlmostEqual(points[0].target_nm, 1525.00)
        self.assertAlmostEqual(points[-1].target_nm, 1565.00)
        np.testing.assert_allclose(
            np.diff([point.target_nm for point in points]), 0.02, atol=1e-10
        )
        self.assertTrue(
            np.all(np.diff([point.measured_nm for point in points]) > 0.0)
        )
        self.assertEqual(set(guards), {241, 1854})
        self.assertEqual(
            points[0].codes[0], app_JDSU.FULLBAND_ACCURACY_GAIN_MAX_CODE
        )
        self.assertEqual(
            [len(guards[index].precondition_codes) for index in sorted(guards)],
            [2, 2],
        )
        limits = app_JDSU._fullband_accuracy_code_limits()
        self.assertTrue(
            all(
                0 <= code <= maximum
                for point in points
                for code, maximum in zip(point.codes, limits)
            )
        )

    def test_robust_stability_gate_accepts_quiet_data_and_rejects_drift(self):
        quiet = np.tile([500, 800, 1200, 1600, 300, 700], (20, 1)).astype(float)
        quiet[::4, 2] += 1
        median, sigma, drift, stable = app_JDSU.fullband_accuracy_statistics(quiet)
        self.assertTrue(stable)
        self.assertEqual(median.shape, (6,))
        self.assertTrue(np.all(sigma <= 1.0))
        self.assertTrue(np.all(drift <= 1.0))

        drifting = quiet.copy()
        drifting[:, 3] += np.arange(len(drifting)) * 3
        self.assertFalse(app_JDSU.fullband_accuracy_statistics(drifting)[3])

    def test_transition_guards_reject_fractional_dac_codes(self):
        points = app_JDSU.load_fullband_accuracy_table()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = json.loads(
                (Path(app_JDSU.__file__).with_name(
                    "fullband_equal_power_operational_2001_transition_guards.json"
                )).read_text(encoding="utf-8")
            )
            payload["guards"][0]["precondition_codes"][0][2] = 1.5
            (root / "fullband_equal_power_operational_2001_transition_guards.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                app_JDSU.load_fullband_transition_guards(points, root)

    def test_worker_never_truncates_fractional_dac_codes(self):
        class Port:
            def __init__(self):
                self.writes = []

            def write(self, payload):
                self.writes.append(payload)

        port = Port()
        worker = app_JDSU.UnlimitedAccuracyWorker(port, (), {})
        with self.assertRaises(ValueError):
            worker._command_and_ack((1, 2, 3.5, 4, 5))
        self.assertEqual(port.writes, [])

    def test_145_ma_codes_are_confined_to_fullband_commands(self):
        codes = (
            app_JDSU.FULLBAND_ACCURACY_GAIN_MAX_CODE,
            app_JDSU.FULLBAND_ACCURACY_SOA_MAX_CODE,
            0,
            0,
            0,
        )
        with self.assertRaises(ValueError):
            app_JDSU.build_single_value_dac_command(codes)
        command = app_JDSU.build_single_value_dac_command(
            codes, fullband_2001=True
        )
        returned = tuple(
            (command[4 + channel * 2] << 8) | command[5 + channel * 2]
            for channel in range(5)
        )
        self.assertEqual(returned, codes)

        with self.assertRaises(ValueError):
            app_JDSU.build_single_value_dac_command(
                (codes[0] + 1, *codes[1:]), fullband_2001=True
            )

    def test_ack_mismatch_reports_requested_and_returned_codes(self):
        requested = (100, 200, 300, 400, 500)
        returned = (99, 200, 300, 400, 500)
        frame = bytearray(20)
        frame[:4] = b"\xFF\xFF\x02\x00"
        for channel, code in enumerate(returned):
            frame[4 + channel * 2] = code >> 8
            frame[5 + channel * 2] = code & 0xFF

        class Port:
            def write(self, _payload):
                return None

        worker = app_JDSU.UnlimitedAccuracyWorker(Port(), (), {})
        with mock.patch.object(
            worker, "_read_available_frames", return_value=(bytes(frame),)
        ), mock.patch.object(
            app_JDSU.time,
            "monotonic",
            side_effect=(0.0, 0.0, 3.0, 4.0, 4.0, 7.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "请求.*最近回读") as caught:
                worker._command_and_ack(requested)
        self.assertIn("GAIN=100", str(caught.exception))
        self.assertIn("GAIN=99", str(caught.exception))

    def test_equal_interval_worker_discards_old_frames_and_takes_one_adc_frame(self):
        values = (111, 222, 1000, 2000, 3000, 4000)
        frame = bytearray(20)
        frame[:4] = b"\xFF\xFF\x02\x02"
        for channel, code in enumerate(values):
            frame[4 + channel * 2] = code >> 8
            frame[5 + channel * 2] = code & 0xFF

        class Port:
            def __init__(self):
                self.reset_count = 0

            def reset_input_buffer(self):
                self.reset_count += 1

        points = app_JDSU.load_fullband_accuracy_table()
        port = Port()
        worker = app_JDSU.EqualIntervalWorker(port, points, {}, settle_s=0.375)
        with mock.patch.object(worker, "_command_and_ack", return_value=1.0), \
             mock.patch.object(worker, "_interruptible_wait", return_value=True), \
             mock.patch.object(
                 worker, "_read_available_frames", return_value=(bytes(frame),)
             ):
            result = worker._acquire_point(1, points[0])
        self.assertEqual(port.reset_count, 1)
        self.assertEqual(result["sample_count"], 1)
        self.assertTrue(result["single_sample"])
        self.assertEqual(result["adc_codes"], tuple(float(v) for v in values[2:]))
        self.assertAlmostEqual(result["settle_s"], 0.375)

    def test_operational_table_rejects_nonfinite_measured_wavelength(self):
        source = Path(app_JDSU.__file__).with_name(
            "fullband_equal_power_operational_2001.csv"
        )
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = reader.fieldnames
        rows[-1]["measured_wavelength_nm"] = "Infinity"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / source.name
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "有限数"):
                app_JDSU.load_fullband_accuracy_table(target.parent)

    def test_profiled_2001_table_accepts_145_ma_but_not_one_code_more(self):
        fieldnames = [
            "limit_profile", "index", "target_wavelength_nm",
            "measured_wavelength_nm", "single_mode_pass", "gain_code",
            "soa_code", "phase_code", "wavelength_a_code", "wavelength_b_code",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "fullband_equal_power_operational_2001.csv"
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for index in range(2001):
                    wavelength = 1525.0 + 0.02 * index
                    writer.writerow({
                        "limit_profile": app_JDSU.FULLBAND_ACCURACY_LIMIT_PROFILE,
                        "index": index,
                        "target_wavelength_nm": f"{wavelength:.2f}",
                        "measured_wavelength_nm": f"{wavelength:.8f}",
                        "single_mode_pass": 1,
                        "gain_code": 63351,
                        "soa_code": 63351,
                        "phase_code": 0,
                        "wavelength_a_code": 0,
                        "wavelength_b_code": 0,
                    })
            points = app_JDSU.load_fullband_accuracy_table(target.parent)
            self.assertEqual(points[0].codes[:2], (63351, 63351))

            text = target.read_text(encoding="utf-8")
            target.write_text(text.replace(",63351,63351,", ",63352,63351,", 1),
                              encoding="utf-8")
            with self.assertRaises(ValueError):
                app_JDSU.load_fullband_accuracy_table(target.parent)

    def test_monitor_decoder_keeps_pdt_pdr_before_ch0_to_ch3(self):
        values = (101, 202, 303, 404, 505, 606)
        frame = bytearray(20)
        frame[:4] = b"\xFF\xFF\x02\x02"
        for channel, code in enumerate(values):
            frame[4 + channel * 2] = code >> 8
            frame[5 + channel * 2] = code & 0xFF
        self.assertEqual(
            app_JDSU.decode_single_value_monitor_frame(bytes(frame)), values
        )
        self.assertEqual(
            app_JDSU.decode_single_value_rt_frame(bytes(frame)), values[2:]
        )

    def test_page_draws_exactly_one_more_point_per_result(self):
        page = app_JDSU.UnlimitedAccuracyWindow()
        page._on_scan_started(1)
        base = {
            "scan_id": 1,
            "total": 2001,
            "target_nm": 1525.0,
            "wavelength_nm": page.points[0].measured_nm,
            "voltages": (0.1, 0.2, 0.3, 0.4),
            "sigma_volts": (0.001, 0.001, 0.001, 0.001),
            "stable": True,
            "saturated_channels": (),
            "sample_count": 15,
            "elapsed_s": 0.5,
        }
        first = dict(base, index=0)
        page._on_point_ready(first)
        self.assertEqual(page.progress_bar.value(), 1)
        self.assertEqual(len(page.curves[0].xData), 1)
        self.assertTrue(np.isnan(page.values[0, 1]))

        second = dict(
            base,
            index=1,
            target_nm=1525.02,
            wavelength_nm=page.points[1].measured_nm,
        )
        page._on_point_ready(second)
        self.assertEqual(page.progress_bar.value(), 2)
        self.assertEqual(len(page.curves[0].xData), 2)
        page.deleteLater()

    def test_equal_interval_fit_button_requires_a_complete_scan_and_marks_peaks(self):
        page = app_JDSU.EqualIntervalWindow()
        self.assertFalse(page.precise_fit_btn.isEnabled())
        x = page.wavelengths
        page.values[:] = 0.08
        expected = (1532.47228, 1544.45520)
        for centre, amplitude in zip(expected, (0.52, 0.66), strict=True):
            page.values[1] += amplitude * np.exp(
                -0.5 * ((x - centre) / 0.28) ** 2
            )
        page.completed_points = len(page.points)
        page._on_scan_completed(1, 10.0, 0)
        self.assertTrue(page.precise_fit_btn.isEnabled())

        page.fit_reflection_spectrum()
        centres = [fit.center_nm for fit in page.precise_fit_results[1]]
        self.assertEqual(len(centres), 2)
        np.testing.assert_allclose(centres, expected, atol=0.004)
        self.assertGreater(len(page.precise_fit_items), 0)

        page.clear_spectrum()
        self.assertFalse(page.precise_fit_btn.isEnabled())
        page.deleteLater()

    def test_auto_calibration_replaces_wavelength_table_page(self):
        window = app_JDSU.MainWindow(auto_acquire_local=False)
        self.assertEqual(window.stack.count(), 7)
        self.assertIs(window.stack.widget(2), window.page_ap6150)
        self.assertIs(window.stack.widget(4), window.page_unlimited_accuracy)
        self.assertIs(window.stack.widget(5), window.page_equal_interval)
        self.assertIs(window.stack.widget(6), window.page_temporary_test)
        self.assertEqual(window.menu_page.actions()[2].text(), "自动校准模式")
        self.assertEqual(window.menu_page.actions()[4].text(), "无限准确模式")
        self.assertEqual(window.menu_page.actions()[5].text(), "等间隔模式")
        self.assertFalse(window.menu_page.actions()[5].isVisible())
        self.assertEqual(window.menu_page.actions()[6].text(), "手指应力测试模式")
        self.assertEqual(fbg_unified_app.LOCAL_MODE_NAMES[-1], "手指应力测试模式")
        self.assertEqual(fbg_unified_app.LOCAL_MODE_NAMES[2], "自动校准")
        self.assertIn("145 mA", window.page_ap6150.status_label.text())
        self.assertAlmostEqual(window.page_equal_interval.settle_spin.value(), 0.2)
        gain_labels = (
            window.page_peak.auto_scale_label,
            window.page_temperature_peak.auto_scale_label,
            window.page_extra.gain_label,
            window.page_unlimited_accuracy.gain_label,
            window.page_equal_interval.gain_label,
        )
        for label in gain_labels:
            self.assertIn("模拟", label.text())
            self.assertIn("数字", label.text())
        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
