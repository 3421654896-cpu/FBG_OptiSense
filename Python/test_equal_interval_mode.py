import os
import json
import shutil
import tempfile
import unittest
from unittest import mock
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_JDSU
import mode_table_manager


class _BufferedPort:
    def __init__(self, payload=b""):
        self.payload = bytearray(payload)
        self.writes = []
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self.payload)

    def read(self, size):
        data = bytes(self.payload[:size])
        del self.payload[:size]
        return data

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def reset_input_buffer(self):
        self.payload.clear()


def _feedback_ack(ch0_selector, ch1_selector, status=app_JDSU.ACK_VALUE):
    frame = bytearray(app_JDSU.SINGLE_VALUE_FRAME_SIZE)
    frame[:4] = b"\xFF\xFF\x02\x03"
    frame[4] = int(ch0_selector)
    frame[5] = int(ch1_selector)
    frame[6] = int(status)
    return bytes(frame)


def _stress_feedback_ack(ch0_selector, ch1_selector, status=app_JDSU.ACK_VALUE):
    frame = bytearray(app_JDSU.SINGLE_VALUE_FRAME_SIZE)
    frame[:4] = b"\xFF\xFF\x02\x04"
    frame[4] = int(ch0_selector)
    frame[5] = int(ch1_selector)
    frame[6] = int(status)
    return bytes(frame)


class EqualIntervalModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = app_JDSU.QtWidgets.QApplication.instance()
        if cls.qt_app is None:
            cls.qt_app = app_JDSU.QtWidgets.QApplication([])
        cls.points = app_JDSU.load_fullband_accuracy_table()
        cls.x = np.asarray([point.measured_nm for point in cls.points], dtype=float)

    def test_feedback_command_and_ack_round_trip(self):
        command = app_JDSU.build_extra_feedback_command(3, 2)
        self.assertEqual(len(command), app_JDSU.tx_size)
        self.assertEqual(command[:6], b"\xFF\xFF\x01\x06\x03\x02")
        self.assertEqual(
            app_JDSU.decode_extra_feedback_ack(_feedback_ack(3, 2)),
            (3, 2),
        )
        self.assertIsNone(
            app_JDSU.decode_extra_feedback_ack(
                _feedback_ack(3, 2, app_JDSU.ACK_ERROR_VALUE)
            )
        )
        self.assertEqual(
            app_JDSU.decode_extra_feedback_status(
                _feedback_ack(3, 2, app_JDSU.ACK_ERROR_VALUE)
            ),
            ((3, 2), app_JDSU.ACK_ERROR_VALUE),
        )
        with self.assertRaises(ValueError):
            app_JDSU.build_extra_feedback_command(4, 1)

    def test_extra_stream_parser_accepts_feedback_ack(self):
        expected = _feedback_ack(0, 3)
        buffer = bytearray(b"noise" + expected)
        self.assertEqual(app_JDSU.extract_single_value_frames(buffer), [expected])
        self.assertEqual(buffer, bytearray())

    def test_stress_feedback_command_and_io_ack_round_trip(self):
        command = app_JDSU.build_stress_feedback_command(2, 0)
        self.assertEqual(len(command), app_JDSU.tx_size)
        self.assertEqual(command[:6], b"\xFF\xFF\x01\x07\x02\x00")
        expected = _stress_feedback_ack(2, 0)
        buffer = bytearray(b"noise" + expected)
        self.assertEqual(app_JDSU.extract_single_value_frames(buffer), [expected])
        self.assertEqual(
            app_JDSU.decode_stress_feedback_status(expected),
            ((2, 0), app_JDSU.ACK_VALUE),
        )
        with self.assertRaises(ValueError):
            app_JDSU.build_stress_feedback_command(-1, 1)

    def test_stress_ui_has_four_manual_digital_gains_and_real_analogue_controls(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        self.assertEqual(len(page.digital_gain_spins), 4)
        page.digital_gain_spins[2].setValue(3.25)
        self.assertAlmostEqual(page.voltage_scalars[2], 3.25)
        self.assertEqual(page._selected_stress_feedback_selectors(), (1, 1))
        page.stress_feedback_buttons[0].click()
        self.assertEqual(page._selected_stress_feedback_selectors(), (3, 1))
        self.assertIn("20 kΩ", page.stress_feedback_buttons[0].text())
        self.assertEqual(len(page.stress_fixed_feedback_buttons), 2)
        self.assertTrue(all(not item.isEnabled() for item in page.stress_fixed_feedback_buttons))
        self.assertIn("固定2 kΩ", page.stress_fixed_feedback_buttons[0].text())
        page.deleteLater()

    def test_stress_peak_table_compacts_every_current_ch0_peak(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        page.visible_plot_channels = {0}
        page.fbg_channels = {0}
        page.peak_fits[0] = [
            SimpleNamespace(valid=True),
            SimpleNamespace(valid=False),
            SimpleNamespace(valid=True),
            SimpleNamespace(valid=True),
        ]
        # P2's current raw fit is malformed, but its live tracker still carries
        # a recent trustworthy centre.  The display must not flicker to 3 peaks.
        page.waves[0] = [1527.1, 1530.2, 1534.2, 1538.3] + [0.0] * 11
        page._update_peak_value_table()

        self.assertEqual(
            [page.num_labels[0][index].text() for index in range(4)],
            ["1527.1000", "1530.2000", "1534.2000", "1538.3000"],
        )
        self.assertTrue(all(not label.isHidden() for label in page.num_labels[0][:4]))
        self.assertTrue(all(label.isHidden() for label in page.num_labels[0][4:]))
        self.assertTrue(all(not label.isHidden() for label in page.peak_header_labels[:4]))
        self.assertTrue(all(label.isHidden() for label in page.peak_header_labels[4:]))
        page.deleteLater()

    def test_stress_firmware_has_no_amplitude_autoranging_path(self):
        firmware = (
            Path(app_JDSU.__file__).resolve().parents[1]
            / "JDSU"
            / "Core"
            / "Src"
            / "ms5614t.c"
        ).read_text(encoding="utf-8")
        self.assertNotIn("stressUpdateGainForNextFrame", firmware)
        self.assertNotIn("STRESS_GAIN_TO_20K_CODE", firmware)
        self.assertNotIn("STRESS_GAIN_TO_40K_CODE", firmware)
        self.assertIn("else if(aRxBuffer[3] == 0x07)", firmware)
        self.assertIn("stressManualFeedbackArmed = 1U", firmware)

    def test_certified_path_sensitive_point_uses_hidden_fullband_precondition(self):
        root = Path(app_JDSU.__file__).resolve().parents[1]
        firmware = (root / "JDSU/Core/Src/ms5614t.c").read_text(
            encoding="utf-8"
        )
        header = (root / "JDSU/Core/Inc/stress_table.h").read_text(
            encoding="utf-8"
        )
        table = (root / "JDSU/Core/Src/stress_table.c").read_text(
            encoding="utf-8"
        )
        certificate = json.loads(
            (
                root
                / "Python/artifacts/ch1_measured_template_selection_v4_certified_20260905.json"
            ).read_text(encoding="utf-8")
        )
        installed = json.loads(
            (root / "Python/mode_tables_from_fullband_2001.json").read_text(
                encoding="utf-8"
            )
        )
        predecessors = certificate["sparse_sequence_validation"][
            "certified_hidden_predecessors"
        ]
        self.assertEqual(len(predecessors), 1)
        predecessor = predecessors[0]
        self.assertTrue(predecessor["certified"])
        self.assertEqual(predecessor["runtime_path"],
                         "previous_sparse->hidden_fullband_predecessor->target")
        point_index = int(predecessor["mode_index"])
        predecessor_codes = [int(value) for value in predecessor["predecessor_codes"]]
        self.assertEqual(point_index, 2)
        self.assertEqual(
            int(installed["modes"]["stress"]["rows"][point_index]["fullband_index"]),
            int(predecessor["fullband_index"]),
        )
        self.assertIn(
            f"#define STRESS_PATH_PRECONDITION_POINT_INDEX {point_index}U",
            header,
        )
        self.assertIn("#define STRESS_PATH_PRECONDITION_HOLD_US 5000U", header)
        self.assertIn(
            "{" + ",".join(str(value) for value in predecessor_codes) + "};",
            table,
        )
        self.assertIn(
            "activeScanPointIndex == STRESS_PATH_PRECONDITION_POINT_INDEX",
            firmware,
        )
        self.assertIn("Stress_Path_Precondition_DAC", firmware)
        self.assertIn(
            "Stress_Wave_DAC[activeScanPointIndex]", firmware
        )
        self.assertIn("delay_us(STRESS_PATH_PRECONDITION_HOLD_US);", firmware)

    def test_worker_requires_matching_mcu_feedback_confirmation(self):
        port = _BufferedPort(_feedback_ack(3, 2))
        worker = app_JDSU.EqualIntervalWorker(
            port,
            self.points,
            {},
            settle_s=0.2,
            feedback_selectors=(3, 2),
        )
        self.assertTrue(worker._feedback_command_and_ack())
        self.assertEqual(
            port.writes,
            [app_JDSU.build_extra_feedback_command(3, 2)],
        )
        worker.deleteLater()

    def test_equal_interval_buttons_create_worker_with_selected_feedback(self):
        page = app_JDSU.EqualIntervalWindow()
        self.assertEqual(page._selected_feedback_selectors(), (1, 1))
        page._start_feedback_apply = Mock()
        page.ch0_feedback_btn.click()
        page._start_feedback_apply.assert_called_once_with()
        worker = page._create_worker()
        self.assertEqual(worker.feedback_selectors, (3, 1))
        self.assertIn("CH0：20 kΩ", page.ch0_feedback_btn.text())
        page._set_scan_parameters_enabled(False)
        self.assertFalse(page.ch0_feedback_btn.isEnabled())
        self.assertFalse(page.ch1_feedback_btn.isEnabled())
        worker.deleteLater()
        page.deleteLater()

    def test_periodic_feedback_verification_runs_before_point_33(self):
        class VerificationWorker(app_JDSU.EqualIntervalWorker):
            def __init__(self, points):
                super().__init__(
                    _BufferedPort(),
                    points,
                    {},
                    settle_s=0.0,
                    feedback_selectors=(1, 1),
                )
                self.verification_count = 0

            def _interruptible_wait(self, _duration_s):
                return True

            def _feedback_command_and_ack(self):
                self.verification_count += 1
                return True

            def _acquire_point(self, scan_id, point):
                return {"scan_id": scan_id, "index": point.index}

            def _safe_shutter(self):
                pass

        worker = VerificationWorker(self.points[:33])
        worker.run()
        self.assertEqual(worker.verification_count, 2)
        worker.deleteLater()

    def test_firmware_feedback_ack_reads_chose_pins(self):
        firmware = (
            Path(app_JDSU.__file__).resolve().parents[1]
            / "JDSU"
            / "Core"
            / "Src"
            / "ms5614t.c"
        ).read_text(encoding="utf-8")
        self.assertIn("static uint8_t PD_ReadFeedbackSelector", firmware)
        self.assertIn("HAL_GPIO_ReadPin(CHOISE_0_A_PORT", firmware)
        self.assertIn(
            "extraFeedbackSelector[0] = PD_ReadFeedbackSelector(0U);",
            firmware,
        )

    def test_temperature_samples_ch0_ch1_with_pipelined_adc_transaction(self):
        firmware = (
            Path(app_JDSU.__file__).resolve().parents[1]
            / "JDSU"
            / "Core"
            / "Src"
            / "ms5614t.c"
        ).read_text(encoding="utf-8")
        precision_body = firmware.split(
            "void sampleVoltagePrecision(uint16_t i){", 1
        )[1].split("void sampleTemperature(void){", 1)[0]
        self.assertIn("ADC_ReadTwo(samples[sampleIndex])", precision_body)
        self.assertNotIn(
            "samples[sampleIndex][0] = ADC_Write_Read(0U)", precision_body
        )
        self.assertIn("PRECISION_ADC_SAMPLE_COUNT", precision_body)
        self.assertIn("PRECISION_ADC_TRIM_EACH_SIDE", precision_body)

    def test_firmware_scan_counts_follow_generated_table_macros(self):
        root = Path(app_JDSU.__file__).resolve().parents[1]
        acquisition = (root / "JDSU/Core/Src/ms5614t.c").read_text(
            encoding="utf-8"
        )
        wireless = (root / "JDSU/Core/Src/wifi_transport.c").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "? STRESS_TABLE_POINT_COUNT : TEMPERATURE_TABLE_POINT_COUNT;",
            acquisition,
        )
        self.assertIn(
            "#define WIFI_TEMPERATURE_POINT_COUNT     TEMPERATURE_TABLE_POINT_COUNT",
            wireless,
        )

    def test_equal_interval_range_selects_only_closed_target_interval(self):
        selected = app_JDSU.select_fullband_target_range(
            self.points, 1530.00, 1530.10
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[0].target_nm, 1530.00)
        self.assertEqual(selected[-1].target_nm, 1530.10)
        self.assertEqual(selected[0].index, 250)
        self.assertEqual(selected[-1].index, 255)

    def test_equal_interval_range_controls_default_to_full_table_and_lock(self):
        page = app_JDSU.EqualIntervalWindow()
        self.assertAlmostEqual(page.start_wavelength_spin.value(), 1525.0)
        self.assertAlmostEqual(page.end_wavelength_spin.value(), 1565.0)
        page.start_wavelength_spin.setValue(1540.0)
        page.end_wavelength_spin.setValue(1540.2)
        selected = page._selected_range_points()
        self.assertEqual(len(selected), 11)
        page._set_scan_parameters_enabled(False)
        self.assertFalse(page.start_wavelength_spin.isEnabled())
        self.assertFalse(page.end_wavelength_spin.isEnabled())
        page.deleteLater()

    def test_equal_interval_range_rejects_reverse_or_empty_range(self):
        with self.assertRaisesRegex(ValueError, "开始波长"):
            app_JDSU.select_fullband_target_range(self.points, 1540.0, 1539.0)
        with self.assertRaisesRegex(ValueError, "没有0.02 nm标定点"):
            app_JDSU.select_fullband_target_range(
                self.points, 1525.001, 1525.019
            )

    @staticmethod
    def _synthetic_spectrum(x, peak_count=9):
        centers = np.linspace(1527.0, 1563.0, peak_count)
        channels = np.full((4, x.size), 0.045, dtype=float)
        for peak_index, center in enumerate(centers):
            sigma = 0.18 + 0.01 * (peak_index % 3)
            shape = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            channels[1] += (0.42 + 0.010 * peak_index) * shape
        return channels, centers

    @staticmethod
    def _mark_sparse_validation_passed(report):
        rows = report["stress"]["rows"]
        validation = report["sparse_sequence_validation"]
        validation.update({
            "status": "passed",
            "validation_method": "hardware_sparse_order_replay",
            "validated_point_count": len(rows),
            "validated_repeats": 2,
            "validated_fullband_indices": [
                row["fullband_index"] for row in rows
            ],
            "per_point_results": [
                {
                    "mode_index": mode_index,
                    "fullband_index": row["fullband_index"],
                    "passed": True,
                    "requires_hidden_predecessor": False,
                }
                for mode_index, row in enumerate(rows)
            ],
        })
        return validation

    def test_thumb_defaults_to_ch1_nine_peaks_stress45_temperature99(self):
        values, centers = self._synthetic_spectrum(self.x)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x,
            values,
            self.points,
            feedback_selectors=(1, 3),
        )
        self.assertEqual(report["schema"], "equal_interval_auto_mode_selection_v4")
        self.assertEqual(report["fbg_channels"], [1])
        self.assertEqual(
            report["selection"]["method"], "measured_template_fisher_v1"
        )
        self.assertEqual(report["stress"]["point_count"], 45)
        self.assertEqual(report["temperature"]["point_count"], 99)
        self.assertEqual(report["detected_peak_count"], 9)
        self.assertEqual(report["stress"]["points_per_peak"], [5] * 9)
        self.assertEqual(report["temperature"]["points_per_peak"], [11] * 9)
        self.assertEqual(report["analog_feedback"]["ch0_kohm"], 40)
        self.assertEqual(report["analog_feedback"]["ch1_kohm"], 20)
        self.assertEqual(
            report["sparse_sequence_validation"]["status"], "pending"
        )
        self.assertEqual(
            len(report["sparse_sequence_validation"]["stress_table_fingerprint_sha256"]),
            64,
        )
        fitted = np.asarray(
            [peak["center_nm"] for peak in report["temperature"]["peaks"]]
        )
        np.testing.assert_allclose(fitted, centers, atol=0.015)
        for peak in report["stress"]["peaks"]:
            self.assertEqual(
                len(peak["dense_wavelength_nm"]),
                len(peak["dense_template_signal_v"]),
            )
            self.assertGreater(len(peak["dense_wavelength_nm"]), 5)
            self.assertEqual(len(peak["selected_fullband_indices"]), 5)
        for mode, count in (("stress", 45), ("temperature", 99)):
            rows = report[mode]["rows"]
            indices = [row["fullband_index"] for row in rows]
            self.assertEqual(len(indices), count)
            self.assertEqual(len(set(indices)), count)
            self.assertEqual(indices, sorted(indices))
            for peak_number in range(1, report["detected_peak_count"] + 1):
                peak_rows = [row for row in rows if row["peak_number"] == peak_number]
                center = peak_rows[0]["fitted_center_nm"]
                wavelengths = [row["measured_wavelength_nm"] for row in peak_rows]
                self.assertLess(min(wavelengths), center)
                self.assertGreater(max(wavelengths), center)
                self.assertEqual(
                    [row["selection_role"] for row in peak_rows].count("peak_body"),
                    len(peak_rows) - 2,
                )
                self.assertEqual(peak_rows[0]["selection_role"], "left_guard")
                self.assertEqual(peak_rows[-1]["selection_role"], "right_guard")
                body_offsets = [
                    abs(row["offset_sigma"])
                    for row in peak_rows
                    if row["selection_role"] == "peak_body"
                ]
                self.assertLessEqual(max(body_offsets), 2.05)
                for row in peak_rows:
                    self.assertEqual(len(row["source_adc_v_by_channel"]), 4)
                    self.assertAlmostEqual(
                        row["source_adc_v"],
                        row["source_adc_v_by_channel"][row["source_channel"]],
                    )
                    self.assertEqual(row["source_channel"], 1)
                    source = self.points[row["fullband_index"]]
                    self.assertEqual(row["codes"], list(source.codes))
                    self.assertEqual(
                        row["selection_method"], "measured_template_fisher_v1"
                    )

    def test_default_five_and_eleven_point_templates_are_two_sided_symmetric(self):
        fit = SimpleNamespace(center_nm=1540.0, sigma_nm=0.18)
        for count in (5, 11):
            targets, _guard, _core = app_JDSU._peak_sampling_targets(fit, count)
            offsets = np.asarray([target - fit.center_nm for target, _role in targets])
            np.testing.assert_allclose(offsets, -offsets[::-1], atol=1e-12)
            self.assertEqual(targets[0][1], "left_guard")
            self.assertEqual(targets[-1][1], "right_guard")

    def test_default_contract_rejects_non_nine_peak_sensor(self):
        values, centers = self._synthetic_spectrum(self.x, peak_count=8)
        with self.assertRaisesRegex(ValueError, "9个峰不一致"):
            app_JDSU.select_mode_points_from_dense_spectrum(
                self.x,
                values,
                self.points,
            )

    def test_v4_cannot_replace_old_table_until_exact_sparse_order_is_validated(self):
        values, _centers = self._synthetic_spectrum(self.x)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points
        )
        with self.assertRaisesRegex(ValueError, "尚未通过实际稀疏顺序"):
            mode_table_manager.validate_candidate_report(report)

        validation = self._mark_sparse_validation_passed(report)
        normalized = mode_table_manager.validate_candidate_report(report)
        self.assertEqual(
            normalized["sparse_sequence_validation"]["status"], "passed"
        )
        index, codes = mode_table_manager._stress_path_precondition(
            normalized["stress"]["rows"],
            normalized["sparse_sequence_validation"],
        )
        self.assertEqual(index, 45)
        self.assertEqual(codes, [0] * 5)

    def test_v4_rejects_uncertified_or_stale_hidden_predecessor(self):
        values, _centers = self._synthetic_spectrum(self.x)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points
        )
        validation = self._mark_sparse_validation_passed(report)
        row = report["stress"]["rows"][0]
        predecessor = self.points[max(0, row["fullband_index"] - 1)]
        validation["certified_hidden_predecessors"] = [{
            "mode_index": 0,
            "fullband_index": row["fullband_index"],
            "predecessor_fullband_index": predecessor.index,
            "predecessor_codes": list(predecessor.codes),
            "hold_us": 5000,
            "runtime_path": "previous_sparse->hidden_fullband_predecessor->target",
            "validated_repeats": 2,
            "certified": False,
        }]
        validation["per_point_results"][0][
            "requires_hidden_predecessor"
        ] = True
        validation["per_point_results"][0][
            "runtime_path"
        ] = "previous_sparse->hidden_fullband_predecessor->target"
        validation["per_point_results"][0]["hidden_predecessor"] = {
            "fullband_index": predecessor.index,
            "codes": list(predecessor.codes),
            "hold_us": 5000,
        }
        with self.assertRaisesRegex(ValueError, "明确的实测认证"):
            mode_table_manager.validate_candidate_report(report)

        validation["certified_hidden_predecessors"][0]["certified"] = True
        normalized = mode_table_manager.validate_candidate_report(report)
        index, codes = mode_table_manager._stress_path_precondition(
            normalized["stress"]["rows"],
            normalized["sparse_sequence_validation"],
        )
        self.assertEqual(index, 0)
        self.assertEqual(codes, list(predecessor.codes))

        report["stress"]["rows"][0]["codes"][0] -= 1
        with self.assertRaisesRegex(ValueError, "DAC码与2001点标定表不一致"):
            mode_table_manager.validate_candidate_report(report)

    def test_v4_rejects_sparse_validation_from_another_45_point_table(self):
        values, _centers = self._synthetic_spectrum(self.x)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points
        )
        validation = self._mark_sparse_validation_passed(report)
        validation["stress_table_fingerprint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "45点表指纹不匹配"):
            mode_table_manager.validate_candidate_report(report)

    def test_explicit_eight_peak_contract_can_use_dynamic_budgets(self):
        values, centers = self._synthetic_spectrum(self.x, peak_count=8)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x,
            values,
            self.points,
            expected_peaks=8,
            stress_point_count=48,
            temperature_point_count=104,
        )
        self.assertEqual(report["detected_peak_count"], 8)
        self.assertEqual(report["stress"]["points_per_peak"], [6] * 8)
        self.assertEqual(report["temperature"]["points_per_peak"], [13] * 8)
        fitted = np.asarray(
            [peak["center_nm"] for peak in report["temperature"]["peaks"]]
        )
        np.testing.assert_allclose(fitted, centers, atol=0.015)

    def test_tiny_extra_bump_is_not_promoted_to_a_grating_peak(self):
        values, centers = self._synthetic_spectrum(self.x, peak_count=8)
        tiny_center = 1545.0
        tiny_shape = np.exp(-0.5 * ((self.x - tiny_center) / 0.16) ** 2)
        values[0] += 0.030 * tiny_shape
        values[1] += 0.025 * tiny_shape
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x,
            values,
            self.points,
            expected_peaks=8,
        )
        self.assertEqual(report["detected_peak_count"], 8)
        fitted = np.asarray(
            [peak["center_nm"] for peak in report["temperature"]["peaks"]]
        )
        np.testing.assert_allclose(fitted, centers, atol=0.015)
        rejected = sum(
            item["rejected_as_tiny_count"]
            for item in report["detection"]["temperature_channels"]
        )
        self.assertGreaterEqual(rejected, 1)

    def test_low_gain_ch1_peaks_do_not_require_a_40mv_absolute_floor(self):
        centers = np.linspace(1527.0, 1563.0, 9)
        values = np.full((4, self.x.size), 0.004, dtype=float)
        for peak_index, center in enumerate(centers):
            shape = np.exp(-0.5 * ((self.x - center) / 0.18) ** 2)
            values[1] += (0.013 + 0.0015 * peak_index) * shape
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points
        )
        self.assertEqual(report["detected_peak_count"], 9)
        self.assertEqual(report["detection"]["minimum_amplitude_v"], 0.0)

    def test_integrated_optimizer_accepts_asymmetric_rippled_ch1_templates(self):
        centers = np.linspace(1527.5, 1562.5, 9)
        values = np.full((4, self.x.size), 0.006, dtype=float)
        for peak_index, center in enumerate(centers):
            local = self.x - center
            envelope = (0.24 + 0.01 * peak_index) * np.exp(
                -0.5 * (local / 0.20) ** 2
            )
            values[1] += envelope * (
                1.0
                + 0.22 * np.tanh(local / 0.10)
                + 0.09 * np.sin(2.0 * np.pi * local / 0.10)
            )
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points
        )
        self.assertEqual(report["detected_peak_count"], 9)
        self.assertEqual(report["stress"]["points_per_peak"], [5] * 9)
        self.assertTrue(
            all(
                peak["fit_model"] == "measured_template_not_gaussian"
                for peak in report["stress"]["peaks"]
            )
        )
        for peak_number in range(1, 10):
            rows = [
                row for row in report["stress"]["rows"]
                if row["peak_number"] == peak_number
            ]
            roles = {row["template_role"] for row in rows}
            self.assertIn("left_shift_sensitive", roles)
            self.assertIn("right_shift_sensitive", roles)
            self.assertIn("peak_amplitude_anchor", roles)

    def test_default_thumb_selection_ignores_strong_ch0_peaks(self):
        values, _centers = self._synthetic_spectrum(self.x)
        for center in np.linspace(1528.0, 1560.0, 5):
            values[0] += 1.2 * np.exp(-0.5 * ((self.x - center) / 0.15) ** 2)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points
        )
        self.assertEqual(report["fbg_channels"], [1])
        self.assertTrue(
            all(
                row["source_channel"] == 1
                for mode in ("stress", "temperature")
                for row in report[mode]["rows"]
            )
        )

    def test_dynamic_point_budget_enforces_fit_and_lan_safety_limits(self):
        values, _centers = self._synthetic_spectrum(self.x)
        with self.assertRaisesRegex(ValueError, "不能少于45"):
            app_JDSU.select_mode_points_from_dense_spectrum(
                self.x, values, self.points, stress_point_count=44
            )
        with self.assertRaisesRegex(ValueError, "5～240"):
            app_JDSU.select_mode_points_from_dense_spectrum(
                self.x, values, self.points, temperature_point_count=241
            )

    def test_installer_accepts_legacy_v2_fixed_size_report(self):
        values, _centers = self._synthetic_spectrum(self.x)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x,
            values,
            self.points,
            stress_point_count=63,
            temperature_point_count=100,
        )
        report["schema"] = "equal_interval_auto_mode_selection_v2"
        report.pop("fbg_channels")
        normalized = mode_table_manager.validate_candidate_report(report)
        self.assertEqual(normalized["stress"]["point_count"], 63)
        self.assertEqual(normalized["temperature"]["point_count"], 100)

    def test_path_precondition_only_applies_to_exact_audited_neighbourhood(self):
        index, codes = mode_table_manager._stress_path_precondition(
            [{"measured_wavelength_nm": 1538.610, "precondition_codes": [1] * 5}]
        )
        self.assertEqual(index, 1)
        self.assertEqual(codes, [0] * 5)
        index, codes = mode_table_manager._stress_path_precondition(
            [{"measured_wavelength_nm": 1538.619, "precondition_codes": [2] * 5}]
        )
        self.assertEqual(index, 0)
        self.assertEqual(codes, [2] * 5)

    def test_candidate_can_replace_both_firmware_tables_with_backup(self):
        values, _centers = self._synthetic_spectrum(self.x, peak_count=8)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points, expected_peaks=8
        )
        source_root = Path(app_JDSU.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            target_root = Path(temporary)
            for relative in mode_table_manager.MODE_TABLE_FILES:
                source = source_root / relative
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            calibration = Path("Python/fullband_equal_power_operational_2001.csv")
            (target_root / calibration).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / calibration, target_root / calibration)

            result = mode_table_manager.apply_candidate_mode_tables(
                report, repository=target_root
            )
            header = (target_root / "JDSU/Core/Inc/stress_table.h").read_text(
                encoding="utf-8"
            )
            self.assertIn("#define STRESS_SEGMENT_COUNT 8U", header)
            self.assertIn("#define TEMPERATURE_SEGMENT_COUNT 8U", header)
            self.assertIn("#define TEMPERATURE_FEEDBACK_SELECTOR_CH0 1U", header)
            self.assertIn("#define TEMPERATURE_FEEDBACK_SELECTOR_CH1 1U", header)
            self.assertIn("#define TEMPERATURE_ADC_SETTLE_US 12000U", header)
            self.assertIn("#define TEMPERATURE_PRECONDITION_HOLD_US 5000U", header)
            self.assertIn("#define STRESS_PATH_PRECONDITION_HOLD_US 5000U", header)
            self.assertIn("STRESS_PATH_PRECONDITION_POINT_INDEX", header)
            self.assertIn("Stress_Path_Precondition_DAC[5]", header)
            self.assertIn(
                "Temperature_Precondition_DAC[TEMPERATURE_TABLE_POINT_COUNT][5]",
                header,
            )
            source_text = (
                target_root / "JDSU/Core/Src/stress_table.c"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "const uint16_t Temperature_Precondition_DAC"
                "[TEMPERATURE_TABLE_POINT_COUNT][5]",
                source_text,
            )
            self.assertIn(
                "const uint16_t Stress_Path_Precondition_DAC[5]", source_text
            )
            self.assertTrue((result.backup_dir / "manifest.json").is_file())
            stress_axis = app_JDSU.yaml.safe_load(
                (target_root / "Python/stress_wave_const.yaml").read_text(
                    encoding="utf-8"
                )
            )["Wave_DATA"]
            temperature_axis = app_JDSU.yaml.safe_load(
                (target_root / "Python/wave_const.yaml").read_text(
                    encoding="utf-8"
                )
            )["Wave_DATA"]
            self.assertIn("#define STRESS_TABLE_POINT_COUNT 45U", header)
            self.assertIn("#define TEMPERATURE_TABLE_POINT_COUNT 99U", header)
            self.assertEqual(len(stress_axis), 45)
            self.assertEqual(len(temperature_axis), 99)
            formal = app_JDSU.json.loads(
                (target_root / "Python/mode_tables_from_fullband_2001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(formal["detected_peak_count"], 8)
            self.assertEqual(len(formal["modes"]["stress"]["rows"]), 45)
            self.assertEqual(len(formal["modes"]["temperature"]["rows"]), 99)
            first_temperature = formal["modes"]["temperature"]["rows"][0]
            predecessor = self.points[first_temperature["fullband_index"] - 1]
            self.assertEqual(
                first_temperature["precondition_codes"], list(predecessor.codes)
            )

    def test_candidate_rejects_dac_row_not_in_audited_table(self):
        values, _centers = self._synthetic_spectrum(self.x, peak_count=8)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points, expected_peaks=8
        )
        report["stress"]["rows"][0]["codes"][0] -= 1
        with self.assertRaisesRegex(ValueError, "DAC码与2001点标定表不一致"):
            mode_table_manager.validate_candidate_report(report)

    def test_desktop_axis_changes_only_after_matching_ota_completes(self):
        values, _centers = self._synthetic_spectrum(self.x, peak_count=8)
        report = app_JDSU.select_mode_points_from_dense_spectrum(
            self.x, values, self.points, expected_peaks=8
        )
        source_root = Path(app_JDSU.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temporary:
            target_root = Path(temporary)
            for relative in mode_table_manager.MODE_TABLE_FILES:
                source = source_root / relative
                target = target_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            calibration = Path("Python/fullband_equal_power_operational_2001.csv")
            shutil.copy2(source_root / calibration, target_root / calibration)
            axis_path = target_root / "Python/stress_wave_const.yaml"
            original_axis = axis_path.read_text(encoding="utf-8")
            package = target_root / "outputs/ota/JDSU_F205RE_v9.9.9.fbgfw"
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_bytes(b"validated-by-mocked-builder")

            with mock.patch.object(
                mode_table_manager,
                "build_mode_table_ota",
                return_value=(package, "9.9.9"),
            ):
                result = mode_table_manager.apply_candidate_and_build(
                    report, repository=target_root
                )

            self.assertEqual(axis_path.read_text(encoding="utf-8"), original_axis)
            self.assertIsNotNone(result.pending_runtime_dir)
            self.assertTrue(
                os.path.samefile(
                    mode_table_manager.pending_mode_table_package(target_root), package
                )
            )
            self.assertFalse(
                mode_table_manager.activate_pending_runtime_tables(
                    target_root / "outputs/ota/different.fbgfw",
                    repository=target_root,
                )
            )
            self.assertTrue(
                mode_table_manager.activate_pending_runtime_tables(
                    package, repository=target_root
                )
            )
            self.assertNotEqual(axis_path.read_text(encoding="utf-8"), original_axis)
            self.assertIsNone(mode_table_manager.pending_mode_table_package(target_root))


if __name__ == "__main__":
    unittest.main()
