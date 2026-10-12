import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from dataclasses import replace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_JDSU
from machine_profile import get_runtime_machine_id, set_runtime_machine_id


HERE = Path(__file__).resolve().parent


class SingleValueDirectDacTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = app_JDSU.QtWidgets.QApplication.instance()
        if cls.qt_app is None:
            cls.qt_app = app_JDSU.QtWidgets.QApplication([])

    def setUp(self):
        self._old_ap_open = app_JDSU.ap_open
        self._old_switch_mode_enable = app_JDSU.switch_mode_enable
        self._old_machine_id = get_runtime_machine_id()
        set_runtime_machine_id('machine_1')
        app_JDSU.ap_open = False
        app_JDSU.switch_mode_enable = True
        # Hardware IO and modal dialogs are forbidden in the offline suite.
        # Individual tests replace these guards with simulated acknowledgments.
        for name in ('ensure_serial_open', 'serial_write'):
            guard = mock.patch.object(app_JDSU, name, side_effect=AssertionError(
                'Offline tests must not access a physical device'))
            guard.start()
            self.addCleanup(guard.stop)
        for name in ('warning', 'critical'):
            guard = mock.patch.object(app_JDSU.QMessageBox, name)
            guard.start()
            self.addCleanup(guard.stop)

    def tearDown(self):
        app_JDSU.ap_open = self._old_ap_open
        app_JDSU.switch_mode_enable = self._old_switch_mode_enable
        set_runtime_machine_id(self._old_machine_id)

    def test_all_stress_table_codes_round_trip_exactly(self):
        source = json.loads(
            (HERE / "stress_shape_power_dynamic_v3.json").read_text(encoding="utf-8")
        )
        limits = (
            app_JDSU.PI11210_GAIN_MAX_CODE,
            app_JDSU.PI11210_SOA_SOURCE_MAX_CODE,
            app_JDSU.PI11210_PHASE_MAX_CODE,
            app_JDSU.PI11210_WAVELENGTH_MAX_CODE,
            app_JDSU.PI11210_WAVELENGTH_MAX_CODE,
        )
        for row in source["rows"]:
            for code, maximum in zip(row["codes"], limits):
                expected = min(int(code), maximum)
                self.assertEqual(
                    app_JDSU.normalize_pi11210_calibration_code(
                        str(code), maximum
                    ),
                    expected,
                )
                self.assertEqual(
                    app_JDSU.normalize_pi11210_calibration_code(
                        f"{code}.000", maximum
                    ),
                    expected,
                )

    def test_non_integral_and_out_of_range_codes_are_rejected(self):
        for value in ("", "1.5", "nan", "-1", "58983"):
            with self.assertRaises(ValueError):
                app_JDSU.parse_pi11210_dac_code(
                    value, app_JDSU.PI11210_GAIN_MAX_CODE
                )

    def test_authorized_channel_limits_are_accepted(self):
        cases = (
            (app_JDSU.PI11210_GAIN_MAX_CODE, app_JDSU.PI11210_GAIN_MAX_CODE),
            (
                app_JDSU.PI11210_SOA_SOURCE_MAX_CODE,
                app_JDSU.PI11210_SOA_SOURCE_MAX_CODE,
            ),
            (app_JDSU.PI11210_PHASE_MAX_CODE, app_JDSU.PI11210_PHASE_MAX_CODE),
            (
                app_JDSU.PI11210_WAVELENGTH_MAX_CODE,
                app_JDSU.PI11210_WAVELENGTH_MAX_CODE,
            ),
        )
        for code, maximum in cases:
            self.assertEqual(
                app_JDSU.parse_pi11210_dac_code(str(code), maximum), code
            )

    def test_single_value_excel_row_populates_exact_codes(self):
        page = app_JDSU.extraWindow()
        page.excel_adc_value("58982\t54273\t16406\t6144\t2458")
        self.assertEqual(
            (page.gain, page.soa, page.phase, page.wavelena, page.wavelenb),
            (58981, 54273, 16406, 6144, 2458),
        )
        self.assertEqual(page.gain_dac_text.text(), "58981")
        self.assertEqual(page.wavelenb_dac_text.text(), "2458")
        page.deleteLater()

    def test_nearest_wavelength_uses_meter_verified_axis(self):
        points = app_JDSU.load_fullband_accuracy_table(HERE)
        point = app_JDSU.nearest_fullband_accuracy_point(points, 1528.461)
        self.assertEqual(point.index, 173)
        self.assertEqual(point, min(points, key=lambda row: abs(row.measured_nm - 1528.461)))
        # A nominal-grid search would choose the first row here; measured
        # wavelengths require the second.  No historical CSV value is frozen.
        synthetic = (replace(points[0], measured_nm=1525.019),
                     replace(points[1], measured_nm=1525.001))
        self.assertEqual(app_JDSU.nearest_fullband_accuracy_point(
            synthetic, 1525.001).index, 1)
        with self.assertRaisesRegex(ValueError, "1525.00–1565.00"):
            app_JDSU.nearest_fullband_accuracy_point(points, 1524.9)

    def test_wavelength_entry_populates_all_five_codes(self):
        page = app_JDSU.extraWindow()
        page.auto_wavelength_spin.setValue(1528.461)
        point = page.preview_wavelength_match()
        self.assertIsNotNone(point)
        self.assertEqual(point.index, 173)
        self.assertEqual(
            (
                page.gain,
                page.soa,
                page.phase,
                page.wavelena,
                page.wavelenb,
            ),
            point.codes,
        )
        self.assertIn(f"{point.measured_nm:.8f} nm", page.auto_wavelength_match_label.text())
        self.assertIn(f"{(point.measured_nm - 1528.461) * 1000:+.3f} pm",
                      page.auto_wavelength_match_label.text())
        page.deleteLater()

    def test_single_value_feedback_buttons_cover_all_four_tia_ranges(self):
        page = app_JDSU.extraWindow()
        self.assertEqual(page._selected_feedback_selectors(), (1, 1))
        with mock.patch.object(page, "_apply_single_value_feedback"):
            expected = (
                (3, "CH0：20 kΩ"),
                (2, "CH0：5 kΩ"),
                (0, "CH0：2 kΩ"),
                (1, "CH0：40 kΩ"),
            )
            for selector, label in expected:
                page.ch0_feedback_btn.click()
                self.assertEqual(
                    page._selected_feedback_selectors(), (selector, 1)
                )
                self.assertEqual(page.ch0_feedback_btn.text(), label)
        page.deleteLater()

    def test_single_value_feedback_change_is_written_and_io_verified(self):
        page = app_JDSU.extraWindow()

        def acknowledge(command):
            self.assertEqual(
                command, app_JDSU.build_extra_feedback_command(3, 1)
            )
            page.feedback_res_queue.put(((3, 1), app_JDSU.ACK_VALUE))

        with mock.patch.object(
            page, "_start_single_value_monitor", return_value=True
        ), mock.patch.object(
            app_JDSU, "serial_write", side_effect=acknowledge
        ) as write:
            page.ch0_feedback_btn.click()
        write.assert_called_once()
        self.assertEqual(page.confirmed_feedback_selectors, (3, 1))
        self.assertFalse(page.feedback_selection_dirty)
        self.assertIn("IO已核验", page.feedback_status_label.text())
        self.assertIn("CH0 20 kΩ", page.gain_label.text())
        self.assertIn("CH1 40 kΩ", page.gain_label.text())
        page.deleteLater()

    def test_auto_wavelength_output_replays_fullband_predecessor_then_target(self):
        page = app_JDSU.extraWindow()
        page.auto_wavelength_spin.setValue(1528.461)
        point = page.preview_wavelength_match()
        predecessor = page.fullband_dac_rows[point.index - 1]
        response = bytearray(20)
        for channel, code in enumerate(point.codes):
            response[4 + channel * 2] = code >> 8
            response[5 + channel * 2] = code & 0xFF
        with mock.patch.object(
            page, "_start_single_value_monitor", return_value=True
        ), mock.patch.object(
            page, "_write_direct_dac_codes", return_value=bytes(response)
        ) as send_codes, mock.patch.object(app_JDSU.time, "sleep") as wait:
            page.output_by_wavelength()
        self.assertEqual(
            [call.args[0] for call in send_codes.call_args_list],
            [predecessor, point.codes],
        )
        wait.assert_called_once_with(app_JDSU.SINGLE_VALUE_TABLE_PRECONDITION_S)
        self.assertTrue(all(call.kwargs.get('fullband_2001')
                            for call in send_codes.call_args_list))
        self.assertIn("2001点表", page.dac_label.text())
        self.assertIn("已输出", page.auto_wavelength_match_label.text())
        page.deleteLater()

    @staticmethod
    def _command_codes(command):
        return tuple(int.from_bytes(command[4 + channel * 2:6 + channel * 2], 'big')
                     for channel in range(5))

    @classmethod
    def _ack_command(cls, page, command, *, flags=0x87, codes=None):
        echoed = cls._command_codes(command) if codes is None else codes
        response = bytearray(20)
        for channel, code in enumerate(echoed):
            response[4 + channel * 2:6 + channel * 2] = int(code).to_bytes(2, 'big')
        response[14] = command[14]
        response[15] = flags
        page.res_queue.put(bytes(response))

    def test_all_three_machine_tables_encode_without_clipping_under_145ma(self):
        for machine_id in ('machine_1', 'machine_2', 'machine_3'):
            points = app_JDSU.load_fullband_accuracy_table(machine_id=machine_id)
            self.assertEqual(len(points), 2001)
            for point in points:
                with self.subTest(machine=machine_id, point=point.index):
                    command = app_JDSU.build_single_value_dac_command(
                        point.codes, fullband_2001=True)
                    self.assertEqual(self._command_codes(command), point.codes)

    def test_machine_two_auto_output_sends_exact_145ma_codes_and_confirms_ack(self):
        set_runtime_machine_id('machine_2')
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        for requested in (1525.0, 1528.461, 1545.0, 1565.0):
            with self.subTest(wavelength=requested):
                page.auto_wavelength_spin.setValue(requested)
                point = page.preview_wavelength_match()
                predecessor = page.fullband_dac_rows[(point.index - 1) % 2001]
                with mock.patch.object(page, '_start_single_value_monitor', return_value=True), \
                        mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                                          self._ack_command(page, command)) as write, \
                        mock.patch.object(app_JDSU.time, 'sleep'):
                    self.assertTrue(page.output_by_wavelength())
                self.assertEqual([self._command_codes(call.args[0])
                                  for call in write.call_args_list], [predecessor, point.codes])
                self.assertIn('已输出', page.auto_wavelength_match_label.text())
                self.assertIn('非电流回读', page.soa_state.text())
                self.assertIsNone(page._auto_output_fullband_index)

    def test_auto_receiver_start_callbacks_preserve_145ma_preview_codes(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        point = page.preview_wavelength_match()
        def start_receiver():
            for edit in (page.gain_text, page.soa_text, page.phase_text,
                         page.wavelena_text, page.wavelenb_text):
                edit.textChanged.emit(edit.text())
            self.assertEqual(page.gain, point.codes[0])
            self.assertNotIn('无效', page.gain_state.text())
            return True
        with mock.patch.object(page, '_start_single_value_monitor', side_effect=start_receiver), \
                mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                                  self._ack_command(page, command)), \
                mock.patch.object(app_JDSU.time, 'sleep'):
            self.assertTrue(page.output_by_wavelength())

    def test_145ma_writer_rejects_every_over_limit_channel_before_serial_write(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        limits = app_JDSU._fullband_accuracy_code_limits()
        for channel, maximum in enumerate(limits):
            codes = [100, 200, 300, 400, 500]
            codes[channel] = maximum + 1
            with self.subTest(channel=channel), mock.patch.object(app_JDSU, 'serial_write') as write:
                with self.assertRaises(ValueError):
                    page._write_direct_dac_codes(codes, fullband_2001=True)
                write.assert_not_called()

    def test_manual_writer_does_not_inherit_auto_145ma_permission(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        with mock.patch.object(page, '_start_single_value_monitor', return_value=True), \
                mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                                  self._ack_command(page, command)), \
                mock.patch.object(app_JDSU.time, 'sleep'):
            self.assertTrue(page.output_by_wavelength())
        with mock.patch.object(page, '_start_single_value_monitor') as start, \
                mock.patch.object(app_JDSU, 'serial_write') as write:
            self.assertFalse(page.write_data_MCU())
        start.assert_not_called()
        write.assert_not_called()
        page.excel_adc_value('58981\t58981\t300\t400\t500')
        with mock.patch.object(page, '_start_single_value_monitor', return_value=True), \
                mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                                  self._ack_command(page, command)) as write:
            self.assertTrue(page.write_data_MCU())
        self.assertEqual(self._command_codes(write.call_args.args[0]),
                         (58981, 58981, 300, 400, 500))

    def test_auto_context_rejects_edited_row_without_falling_back_to_manual_write(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        point = page.preview_wavelength_match()
        page._auto_output_fullband_index = point.index
        page.gain_text.setText('100')
        with mock.patch.object(page, '_start_single_value_monitor') as start, \
                mock.patch.object(app_JDSU, 'serial_write') as write:
            self.assertFalse(page.write_data_MCU())
        start.assert_not_called()
        write.assert_not_called()

    def test_auto_context_rejects_previous_machine_until_table_reload(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        set_runtime_machine_id('machine_2')
        with mock.patch.object(page, '_start_single_value_monitor') as start, \
                mock.patch.object(app_JDSU, 'serial_write') as write:
            self.assertFalse(page.output_by_wavelength())
        start.assert_not_called()
        write.assert_not_called()
        self.assertIsNone(page._auto_output_fullband_index)
        page.reload_calibration_table()
        self.assertEqual(page._fullband_machine_id, 'machine_2')
        with mock.patch.object(page, '_start_single_value_monitor', return_value=True), \
                mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                                  self._ack_command(page, command)), \
                mock.patch.object(app_JDSU.time, 'sleep'):
            self.assertTrue(page.output_by_wavelength())

    def test_auto_output_does_not_claim_success_after_timeout_or_device_errors(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        for mode in ('mismatch', 'write_failure', 'offline', 'overtemperature', 'timeout', 'serial'):
            with self.subTest(failure=mode):
                with mock.patch.object(page, '_start_single_value_monitor', return_value=True), \
                        mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                                          self._ack_command(page, command)), \
                        mock.patch.object(app_JDSU.time, 'sleep'):
                    self.assertTrue(page.output_by_wavelength())
                def reject(command):
                    if mode == 'serial':
                        raise OSError('simulated disconnected port')
                    if mode == 'mismatch':
                        codes = list(self._command_codes(command))
                        codes[0] = 58981  # old firmware clamping 145 mA to 135 mA
                        self._ack_command(page, command, codes=codes)
                    else:
                        flags = {'write_failure': 0x83, 'offline': 0x80,
                                 'overtemperature': 0xa7}.get(mode, 0x87)
                        self._ack_command(page, command, flags=flags)
                queue_context = (mock.patch.object(page.res_queue, 'get', side_effect=app_JDSU.Empty)
                                 if mode == 'timeout' else mock.patch.object(
                                     app_JDSU.time, 'sleep'))
                with mock.patch.object(page, '_start_single_value_monitor', return_value=True), \
                        mock.patch.object(app_JDSU, 'serial_write', side_effect=reject), queue_context:
                    self.assertFalse(page.output_by_wavelength())
                self.assertNotIn('已输出', page.auto_wavelength_match_label.text())
                self.assertIsNone(page._auto_output_fullband_index)

    def test_fullband_145ma_writer_still_forces_soa_shutter_to_zero(self):
        page = app_JDSU.extraWindow()
        self.addCleanup(page.deleteLater)
        codes = (63351, 63351, 300, 400, 500)
        with mock.patch.object(app_JDSU, 'serial_write', side_effect=lambda command:
                              self._ack_command(page, command)) as write:
            page._write_direct_dac_codes(codes, fullband_2001=True,
                                        soa_mode=app_JDSU.PI11210_SOA_SHUTTER_MODE)
        self.assertEqual(self._command_codes(write.call_args.args[0]),
                         (63351, 0, 300, 400, 500))
        self.assertEqual(write.call_args.args[0][14], app_JDSU.PI11210_SOA_SHUTTER_MODE)

    def test_write_auto_starts_single_value_receiver_before_dac_command(self):
        page = app_JDSU.extraWindow()
        page.excel_adc_value('0\t0\t0\t0\t0')
        events = []
        response = bytearray(20)
        live_thread = mock.Mock()
        live_thread.is_alive.return_value = True

        def start_receiver():
            events.append("receiver")
            page.recv_thread = live_thread

        with mock.patch.object(
            app_JDSU, "ensure_serial_open", side_effect=lambda: events.append("open")
        ), mock.patch.object(
            app_JDSU.ser,
            "reset_input_buffer",
            side_effect=lambda: events.append("reset"),
        ), mock.patch.object(
            app_JDSU,
            "send_work_mode_command",
            side_effect=lambda mode: events.append(f"mode-{mode}"),
        ), mock.patch.object(
            page, "start_recv_thread", side_effect=start_receiver
        ), mock.patch.object(
            page,
            "_write_direct_dac_codes",
            side_effect=lambda *args, **kwargs: events.append("write") or bytes(response),
        ) as write:
            page.write_data_MCU()
        self.assertEqual(events, ["open", "reset", "mode-2", "receiver", "write"])
        self.assertTrue(app_JDSU.ap_open)
        self.assertFalse(app_JDSU.switch_mode_enable)
        self.assertEqual(page.monitor_btn.text(), "停止")
        write.assert_called_once()
        page.deleteLater()

    def test_write_does_not_send_when_single_value_startup_fails(self):
        page = app_JDSU.extraWindow()
        page.excel_adc_value('0\t0\t0\t0\t0')
        with mock.patch.object(
            page, "_start_single_value_monitor", return_value=False
        ) as start, mock.patch.object(page, "_write_direct_dac_codes") as write:
            page.write_data_MCU()
        start.assert_called_once_with()
        write.assert_not_called()
        page.deleteLater()

    def test_active_single_value_receiver_is_not_started_twice(self):
        page = app_JDSU.extraWindow()
        app_JDSU.ap_open = True
        page.recv_thread = mock.Mock()
        page.recv_thread.is_alive.return_value = True
        with mock.patch.object(app_JDSU, "ensure_serial_open") as open_port, \
                mock.patch.object(page, "start_recv_thread") as start_thread:
            self.assertTrue(page._start_single_value_monitor())
        open_port.assert_not_called()
        start_thread.assert_not_called()
        page.deleteLater()

    def test_single_value_stop_forces_all_five_channels_to_zero(self):
        page = app_JDSU.extraWindow()
        app_JDSU.ap_open = True
        app_JDSU.switch_mode_enable = False
        events = []

        with mock.patch.object(
            page, "_single_value_monitor_is_active", return_value=True
        ), mock.patch.object(
            page,
            "_write_direct_dac_codes",
            side_effect=lambda *args, **kwargs: events.append(
                ("write", args, kwargs)
            ),
        ) as write, mock.patch.object(
            page,
            "stop_recv_thread",
            side_effect=lambda: events.append(("stop",)) or True,
        ), mock.patch.object(
            app_JDSU,
            "release_serial_if_allowed",
            side_effect=lambda: events.append(("release",)),
        ):
            page.on_work()

        write.assert_called_once_with(
            (0, 0, 0, 0, 0),
            soa_mode=app_JDSU.PI11210_SOA_SHUTTER_MODE,
        )
        self.assertEqual([event[0] for event in events], ["write", "stop", "release"])
        self.assertFalse(app_JDSU.ap_open)
        self.assertTrue(app_JDSU.switch_mode_enable)
        self.assertIn("五路全零", page.dac_label.text())
        page.deleteLater()

    def test_single_value_monitor_startup_failure_returns_false(self):
        page = app_JDSU.extraWindow()
        with mock.patch.object(
            app_JDSU, "ensure_serial_open", side_effect=OSError("port unavailable")
        ), mock.patch.object(app_JDSU.QMessageBox, "critical") as critical, \
                mock.patch.object(app_JDSU, "serial_write") as write:
            self.assertFalse(page._start_single_value_monitor())
        self.assertFalse(app_JDSU.ap_open)
        self.assertTrue(app_JDSU.switch_mode_enable)
        write.assert_not_called()
        critical.assert_called_once()
        page.deleteLater()

    def test_single_value_direct_write_accepts_matching_readback(self):
        page = app_JDSU.extraWindow()
        codes = (100, 200, 300, 400, 500)
        response = bytearray(20)
        for channel, code in enumerate(codes):
            response[4 + channel * 2] = code >> 8
            response[5 + channel * 2] = code & 0xFF
        page.res_queue.put(bytes(response))
        with mock.patch.object(app_JDSU, "serial_write") as write:
            actual = page._write_direct_dac_codes(codes)
        self.assertEqual(actual, bytes(response))
        write.assert_called_once_with(app_JDSU.build_single_value_dac_command(codes))
        page.deleteLater()

    def test_single_value_direct_write_reports_readback_timeout(self):
        page = app_JDSU.extraWindow()
        codes = (100, 200, 300, 400, 500)
        with mock.patch.object(app_JDSU, "serial_write") as write, \
                mock.patch.object(page.res_queue, "get", side_effect=app_JDSU.Empty):
            with self.assertRaisesRegex(RuntimeError, "2秒内未收到"):
                page._write_direct_dac_codes(codes)
        write.assert_called_once_with(app_JDSU.build_single_value_dac_command(codes))
        page.deleteLater()

    def test_table_point_is_preconditioned_by_its_predecessor(self):
        page = app_JDSU.extraWindow()
        target = page.stress_dac_rows[11]
        page.excel_adc_value("\t".join(str(value) for value in target))
        target_index = page.stress_dac_row_index[target]
        predecessor = page.stress_dac_rows[target_index - 1]
        response = bytearray(20)
        for channel, code in enumerate(target):
            response[4 + channel * 2] = code >> 8
            response[5 + channel * 2] = code & 0xFF
        with mock.patch.object(
            page, "_start_single_value_monitor", return_value=True
        ), mock.patch.object(
            page, "_write_direct_dac_codes", return_value=bytes(response)
        ) as send_codes, mock.patch.object(app_JDSU.time, "sleep") as wait:
            page.write_data_MCU()
        self.assertEqual(
            [call.args[0] for call in send_codes.call_args_list],
            [predecessor, target],
        )
        wait.assert_called_once_with(app_JDSU.SINGLE_VALUE_TABLE_PRECONDITION_S)
        self.assertIn(f"第{target_index + 1}点", page.dac_label.text())
        page.deleteLater()

    @staticmethod
    def _frame(frame_type, fill=0):
        frame = bytearray([fill] * app_JDSU.SINGLE_VALUE_FRAME_SIZE)
        frame[:4] = bytes((0xFF, 0xFF, 0x02, frame_type))
        return bytes(frame)

    def test_single_value_stream_parser_handles_fragmented_and_glued_frames(self):
        expected = [self._frame(0, 1), self._frame(1, 2), self._frame(2, 3)]
        stream = b"noise" + b"".join(expected)
        chunks = (stream[:1], stream[1:7], stream[7:19], stream[19:41], stream[41:])
        buffer = bytearray()
        actual = []
        for chunk in chunks:
            buffer.extend(chunk)
            actual.extend(app_JDSU.extract_single_value_frames(buffer))
        self.assertEqual(actual, expected)
        self.assertEqual(buffer, bytearray())

    def test_single_value_stream_parser_recovers_from_truncated_frame(self):
        valid = self._frame(2, 7)
        buffer = bytearray(b"\xFF\xFF\x02\x00\x01\x02\x03" + valid)
        self.assertEqual(app_JDSU.extract_single_value_frames(buffer), [valid])
        self.assertEqual(buffer, bytearray())

    def test_single_value_stream_parser_accepts_saturated_error_counter(self):
        valid = bytearray(self._frame(0, 0))
        valid[18:20] = b"\xFF\xFF"
        buffer = bytearray(valid)
        self.assertEqual(
            app_JDSU.extract_single_value_frames(buffer), [bytes(valid)]
        )
        self.assertEqual(buffer, bytearray())

    def test_stable_reference_uses_exact_single_value_command(self):
        codes = (58982, 54273, 16406, 6144, 2458)
        command = app_JDSU.build_single_value_dac_command(codes)
        self.assertEqual(len(command), app_JDSU.tx_size)
        self.assertEqual(command[:4], b"\xFF\xFF\x00\x01")
        returned = tuple(
            (command[4 + channel * 2] << 8) | command[5 + channel * 2]
            for channel in range(5)
        )
        self.assertEqual(returned, (58981, 54273, 16406, 6144, 2458))

    def test_single_value_command_rejects_fractional_or_boolean_codes(self):
        for invalid in (1.5, True, "2.25"):
            codes = [100, 200, 300, 400, 500]
            codes[2] = invalid
            with self.assertRaises(ValueError):
                app_JDSU.build_single_value_dac_command(codes)

    def test_wavelength_meter_excel_rows_use_the_same_safe_builder(self):
        worker = app_JDSU.APWorker("")
        worker.configure_excel_columns(
            (
                "unused",
                "code_GAIN",
                "code_SOA",
                "code_PHASE",
                "code_wavelengthA",
                "code_wavelengthB",
            )
        )
        row = [None, 58982, 54273, 16406, 6144, 2458]
        command, _info = worker.excel_operate(iter([row]))
        self.assertEqual(
            command,
            app_JDSU.build_single_value_dac_command(tuple(row[1:6])),
        )
        bad_row = list(row)
        bad_row[3] = 1.5
        with self.assertRaisesRegex(ValueError, "Excel"):
            worker.excel_operate(iter([bad_row]))
        worker.deleteLater()

    def test_current_conversion_rejects_nonfinite_or_invalid_scale(self):
        for current, scale in ((float("nan"), 150), (1, 0), (1, -150)):
            with self.assertRaises(ValueError):
                app_JDSU.pi11210_current_to_code(current, scale)

    def test_soa_shutter_command_uses_safe_zero_current_gate(self):
        codes = (58982, 58982, 32768, 24576, 24576)
        command = app_JDSU.build_single_value_dac_command(
            codes, soa_mode=app_JDSU.PI11210_SOA_SHUTTER_MODE
        )
        self.assertEqual(command[14], app_JDSU.PI11210_SOA_SHUTTER_MODE)
        returned_soa = (command[6] << 8) | command[7]
        self.assertEqual(returned_soa, app_JDSU.PI11210_SOA_SHUTTER_CODE)

    def test_single_value_rejects_reported_pi11210_write_failure(self):
        page = app_JDSU.extraWindow()
        codes = (100, 200, 300, 400, 500)
        response = bytearray(20)
        for channel, code in enumerate(codes):
            response[4 + channel * 2] = code >> 8
            response[5 + channel * 2] = code & 0xFF
        response[14] = app_JDSU.PI11210_SOA_SOURCE_MODE
        response[15] = 0x80 | 0x01 | 0x02  # status valid, online, ID valid; write failed
        response[16:18] = b"\x0A\x10"
        response[18:20] = b"\x00\x03"
        page.res_queue.put(bytes(response))
        with mock.patch.object(app_JDSU, "serial_write"):
            with self.assertRaisesRegex(RuntimeError, "写入失败"):
                page._write_direct_dac_codes(codes)
        page.deleteLater()

    def test_single_value_writer_rejects_fraction_before_serial_write(self):
        page = app_JDSU.extraWindow()
        with mock.patch.object(app_JDSU, "serial_write") as write:
            with self.assertRaises(ValueError):
                page._write_direct_dac_codes((100, 200, 1.5, 400, 500))
        write.assert_not_called()
        page.deleteLater()

    def test_single_value_rt_decoder_returns_ch0_through_ch3(self):
        frame = bytearray(self._frame(2))
        expected = (123, 456, 789, 1023)
        for channel, code in enumerate(expected):
            frame[8 + channel * 2] = code >> 8
            frame[9 + channel * 2] = code & 0xFF
        self.assertEqual(
            app_JDSU.decode_single_value_rt_frame(bytes(frame)), expected
        )
        frame[3] = 1
        self.assertIsNone(app_JDSU.decode_single_value_rt_frame(bytes(frame)))

    def test_reference_channel_filter_keeps_only_visible_fbg_peaks(self):
        fits = [
            [SimpleNamespace(valid=False)],
            [SimpleNamespace(valid=True)],
            [SimpleNamespace(valid=True)],
            [SimpleNamespace(valid=True)],
        ]
        selected = app_JDSU.channels_with_valid_peaks(
            fits,
            visible_channels={0, 1, 3},
            fbg_channels={0, 1, 2},
        )
        self.assertEqual(selected, {1})

    def test_stress_reference_offset_is_fixed_per_point_and_clipped(self):
        reference = np.asarray([
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ])
        frames = np.repeat(reference[None, :, :], 8, axis=0)
        frames[:, 1, :] += np.asarray([0.02, -0.04, 0.30])
        correction = app_JDSU.compute_fixed_stress_reference_offset(
            frames, reference, channels={1}
        )
        np.testing.assert_allclose(correction[0], 0.0)
        np.testing.assert_allclose(correction[1], [0.02, -0.04, 0.15])
        np.testing.assert_allclose(correction[2:], 0.0)

    def test_stress_page_has_white_solid_reference_overlay(self):
        page = app_JDSU.GraphWindow(precision_mode=False)
        self.assertEqual(page.stable_reference_btn.text(), "显示稳定单值曲线")
        self.assertEqual(len(page.stable_reference_curves), 4)
        for curve in page.stable_reference_curves:
            self.assertEqual(curve.opts["pen"].color().name(), "#ffffff")
            self.assertEqual(
                curve.opts["pen"].style(), app_JDSU.QtCore.Qt.SolidLine
            )
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
