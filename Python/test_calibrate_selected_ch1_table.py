from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calibrate_selected_ch1_table import (
    ABSOLUTE_LIMITS_MA,
    FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
    SPARSE_DIRECT_PATH,
    SPARSE_HIDDEN_PATH,
    CalibrationSettings,
    CandidatePoint,
    CeilingSeed,
    SelectedCh1Calibrator,
    currents_ma,
    build_hidden_predecessor_request,
    configure_sparse_sequence_path,
    execute,
    normalize_candidate_rows,
    run_stage,
    safe_current_to_code,
    summarize_ceiling,
    summarize_probe,
    summarize_sparse_sequence,
)
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import Reading


def codes(gain=120.0, soa=100.0, phase=5.0, wave_a=20.0, wave_b=20.0):
    return tuple(
        safe_current_to_code(value, channel)
        for channel, value in enumerate((gain, soa, phase, wave_a, wave_b))
    )


def fullband_points(predecessor_codes=None):
    state = predecessor_codes or codes(110.0, 90.0, 4.5, 18.0, 18.0)
    return tuple(SimpleNamespace(codes=state) for _ in range(2001))


class FakeDevice:
    def __init__(self, **_kwargs):
        self.connected = False
        self.is_open = False
        self.board_ip = "fake-board"
        self.writes = []

    def open(self):
        self.connected = True
        self.is_open = True

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)

    def reset_input_buffer(self):
        return None

    def close(self):
        self.connected = False
        self.is_open = False


class NeverConnectedDevice(FakeDevice):
    def open(self):
        self.connected = False
        self.is_open = True


class FakeState:
    def __init__(self):
        self.codes = codes()
        self.commands = []


class FakeWorker:
    def __init__(self, port, _points, _guards, _repeat=False, state=None):
        self.port = port
        self.state = state or FakeState()
        self.running = True
        self._rx_buffer = bytearray()
        self._last_codes = self.state.codes

    def _interruptible_wait(self, _seconds):
        return True

    def _command_and_ack(self, command_codes, soa_mode=0):
        exact = validate_pi11210_codes(command_codes)
        if soa_mode == 1:
            exact = (exact[0], 0, exact[2], exact[3], exact[4])
        self.state.codes = exact
        self.state.commands.append((exact, soa_mode))
        self._last_codes = exact
        return 1.0


class LinearMeter:
    identity = "YOKOGAWA,AQ6150,FAKE,01.03"

    def __init__(self, state, *, smsr=35.0, wavelength_jitter_pm=None, fail=False):
        self.state = state
        self.smsr = smsr
        self.fail = fail
        self.closed = False
        self.jitter = list(wavelength_jitter_pm or [])
        self.measurements = 0

    def measure(self, target_nm, select_main_peak=False):
        self.measurements += 1
        if self.fail:
            raise RuntimeError("simulated meter failure")
        current = currents_ma(self.state.codes)
        jitter_pm = self.jitter.pop(0) if self.jitter else 0.0
        wavelength_error_pm = -75.0 * (current[2] - 5.0) + jitter_pm
        power_mw = current[1] / 100.0
        peak_count = 1 if self.smsr >= 90.0 else 2
        smsr = None if peak_count == 1 else self.smsr
        return Reading(
            wavelength_nm=target_nm + wavelength_error_pm / 1000.0,
            wavelength_error_pm=wavelength_error_pm,
            power_mw=power_mw,
            power_dbm=0.0,
            peak_count=peak_count,
            side_mode_suppression_db=smsr,
        )

    def close(self):
        self.closed = True


class SoaCeilingMeter(LinearMeter):
    def measure(self, target_nm, select_main_peak=False):
        soa_ma = currents_ma(self.state.codes)[1]
        self.smsr = 35.0 if soa_ma <= 110.01 else 12.0
        return super().measure(target_nm, select_main_peak)


class AffinePhaseMeter(LinearMeter):
    def __init__(self, state, *, phase_slope_pm_per_ma, zero_phase_ma, smsr=35.0):
        super().__init__(state, smsr=smsr)
        self.phase_slope_pm_per_ma = phase_slope_pm_per_ma
        self.zero_phase_ma = zero_phase_ma

    def measure(self, target_nm, select_main_peak=False):
        self.measurements += 1
        current = currents_ma(self.state.codes)
        wavelength_error_pm = self.phase_slope_pm_per_ma * (
            current[2] - self.zero_phase_ma
        )
        peak_count = 1 if self.smsr >= 90.0 else 2
        smsr = None if peak_count == 1 else self.smsr
        return Reading(
            wavelength_nm=target_nm + wavelength_error_pm / 1000.0,
            wavelength_error_pm=wavelength_error_pm,
            power_mw=current[1] / 100.0,
            power_dbm=0.0,
            peak_count=peak_count,
            side_mode_suppression_db=smsr,
        )


class CoupledCalibrationMeter(LinearMeter):
    """Power follows SOA while the PHASE root drifts as SOA is reduced."""

    def __init__(self, state, *, smsr=35.0):
        super().__init__(state, smsr=smsr)

    def measure(self, target_nm, select_main_peak=False):
        self.measurements += 1
        current = currents_ma(self.state.codes)
        power_mw = 0.35 + 0.0111 * current[1]
        root_phase_ma = 0.81 + 0.004 * (135.0 - current[1])
        wavelength_error_pm = -330.0 * (current[2] - root_phase_ma)
        peak_count = 1 if self.smsr >= 90.0 else 2
        smsr = None if peak_count == 1 else self.smsr
        return Reading(
            wavelength_nm=target_nm + wavelength_error_pm / 1000.0,
            wavelength_error_pm=wavelength_error_pm,
            power_mw=power_mw,
            power_dbm=0.0,
            peak_count=peak_count,
            side_mode_suppression_db=smsr,
        )


class InterruptingMeter(LinearMeter):
    def measure(self, target_nm, select_main_peak=False):
        raise KeyboardInterrupt("simulated operator interrupt")


class OneTargetBadMeter(LinearMeter):
    def __init__(self, state, bad_target_nm):
        super().__init__(state, smsr=35.0)
        self.bad_target_nm = bad_target_nm

    def measure(self, target_nm, select_main_peak=False):
        self.smsr = 12.0 if abs(target_nm - self.bad_target_nm) < 1e-9 else 35.0
        return super().measure(target_nm, select_main_peak)


class OneTargetWrongBranchMeter(LinearMeter):
    def __init__(self, state, bad_target_nm):
        super().__init__(state, smsr=35.0)
        self.bad_target_nm = bad_target_nm

    def measure(self, target_nm, select_main_peak=False):
        reading = super().measure(target_nm, select_main_peak)
        if abs(target_nm - self.bad_target_nm) >= 1e-9:
            return reading
        return Reading(
            wavelength_nm=target_nm + 0.080,
            wavelength_error_pm=80.0,
            power_mw=reading.power_mw,
            power_dbm=reading.power_dbm,
            peak_count=2,
            side_mode_suppression_db=35.0,
        )


class CountedSmsrMeter(LinearMeter):
    """Drop SMSR at/after one visit of one exact state."""

    def __init__(self, state, suspect_codes, *, low_from_visit, persistent=False):
        super().__init__(state, smsr=35.0)
        self.suspect_codes = tuple(suspect_codes)
        self.low_from_visit = int(low_from_visit)
        self.persistent = persistent
        self.visits = 0

    def measure(self, target_nm, select_main_peak=False):
        if tuple(self.state.codes) == self.suspect_codes:
            self.visits += 1
            low = self.visits >= self.low_from_visit and (
                self.persistent or self.visits == self.low_from_visit
            )
            self.smsr = 18.0 if low else 35.0
        else:
            self.smsr = 35.0
        return super().measure(target_nm, select_main_peak)


def make_args(input_path, output_path, mode="probe-only", **overrides):
    defaults = dict(
        input=Path(input_path),
        output=Path(output_path),
        mode=mode,
        retry_rejected=False,
        retry_failed_calibration=False,
        continue_on_calibration_failure=False,
        probe_report=None,
        operational_json=Path("does-not-exist.json"),
        target_power_mw=None,
        host="fake",
        gpib="fake-gpib",
        connect_timeout_s=0.1,
        settle_s=0.30,
        predecessor_settle_s=0.08,
        minimum_smsr_db=20.0,
        probe_branch_tolerance_pm=50.0,
        wavelength_tolerance_pm=2.0,
        power_guard_fraction=0.03,
        ceiling_guard_fraction=0.01,
        power_tolerance_fraction=0.02,
        power_tolerance_mw=0.010,
        max_phase_delta_ma=0.60,
        max_phase_step_ma=0.10,
        max_soa_delta_ma=12.0,
        max_soa_step_ma=1.50,
        ceiling_soa_step_ma=2.0,
        confirmation_wavelength_span_pm=2.0,
        confirmation_power_span_fraction=0.03,
        maximum_iterations=16,
        sequence_rounds=2,
        hidden_predecessor_order=None,
        hidden_predecessor_hold_us=FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class CandidateParsingTests(unittest.TestCase):
    def test_accepts_stress_rows_and_plain_rows(self):
        row = {
            "fullband_index": 123,
            "target_wavelength_nm": 1527.46,
            "codes": list(codes()),
        }
        nested = normalize_candidate_rows({"stress": {"rows": [row]}})
        plain = normalize_candidate_rows(
            {"rows": [{**row, "target_nm": row["target_wavelength_nm"]}]}
        )
        self.assertEqual(nested[0].fullband_index, 123)
        self.assertEqual(plain[0].codes, nested[0].codes)

    def test_accepts_new_nine_peak_points_source_index_format(self):
        payload = {
            "stress": {
                "peaks": [
                    {
                        "peak_number": peak,
                        "points": [
                            {
                                "source_index": peak - 1,
                                "wavelength_nm": 1525.0 + (peak - 1) * 0.02,
                                "dac_codes": list(codes()),
                            }
                        ],
                    }
                    for peak in range(1, 10)
                ]
            }
        }
        points = normalize_candidate_rows(payload)
        self.assertEqual(len(points), 9)
        self.assertEqual([point.fullband_index for point in points], list(range(9)))

    def test_new_peak_group_format_requires_exactly_nine_peaks(self):
        with self.assertRaisesRegex(ValueError, "恰好包含9个峰"):
            normalize_candidate_rows({"peaks": []})

    def test_rejects_any_code_above_authorised_limit(self):
        invalid = list(codes())
        invalid[0] = 58982
        with self.assertRaisesRegex(ValueError, "实际命令上限"):
            normalize_candidate_rows(
                {
                    "rows": [
                        {
                            "fullband_index": 0,
                            "target_nm": 1525.0,
                            "codes": invalid,
                        }
                    ]
                }
            )


class ProbeTests(unittest.TestCase):
    def test_probe_uses_predecessor_then_target_and_guarded_common_power(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = LinearMeter(state, smsr=35.0)
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        points = (
            CandidatePoint(0, 100, 1527.0, codes(soa=100.0)),
            CandidatePoint(1, 200, 1529.0, codes(soa=105.0)),
        )
        rows = {str(point.order): engine.probe(point) for point in points}
        summary = summarize_probe(rows, 0.03)
        self.assertAlmostEqual(summary["recommended_common_power_mw"], 0.97, 3)
        self.assertTrue(summary["all_points_qualified"])
        self.assertEqual(len(state.commands), 4)
        self.assertEqual(state.commands[0][0], fullband_points()[99].codes)
        self.assertEqual(state.commands[1][0], points[0].codes)

    def test_two_peak_low_smsr_is_rejected_and_no_common_power_is_certified(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        engine = SelectedCh1Calibrator(
            worker,
            LinearMeter(state, smsr=12.0),
            fullband_points(),
            CalibrationSettings(),
        )
        row = engine.probe(CandidatePoint(0, 12, 1525.24, codes()))
        summary = summarize_probe({"0": row}, 0.03)
        self.assertFalse(row["qualified"])
        self.assertIsNone(summary["recommended_common_power_mw"])


class CalibrationTests(unittest.TestCase):
    def test_measures_candidate_and_ceiling_then_selects_nearest_clean_power_seed(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = LinearMeter(state, smsr=35.0)
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        candidate = codes(soa=100.0, phase=5.0, wave_a=20.0, wave_b=20.0)
        ceiling = codes(soa=135.0, phase=5.0, wave_a=21.0, wave_b=21.0)
        point = CandidatePoint(0, 50, 1526.0, candidate)

        result = engine.calibrate(point, 1.05, ceiling_codes=ceiling)

        self.assertTrue(result["passed"])
        self.assertEqual(result["seed_source"], "candidate")
        self.assertEqual(
            [probe["source"] for probe in result["seed_probes"]],
            ["candidate", "ceiling_probe"],
        )
        self.assertEqual(result["candidate_codes"], list(candidate))
        self.assertEqual(result["selected_seed_codes"], list(candidate))
        self.assertEqual(result["codes"][3:], list(candidate[3:]))
        target_commands = [
            state.commands[index][0] for index in range(1, len(state.commands), 2)
        ]
        self.assertIn(candidate, target_commands)
        self.assertIn(ceiling, target_commands)

    def test_ceiling_seed_can_win_and_identical_seed_is_not_probed_twice(self):
        candidate = codes(soa=100.0, phase=5.0, wave_a=20.0, wave_b=20.0)
        ceiling = codes(soa=135.0, phase=5.0, wave_a=21.0, wave_b=21.0)

        state = FakeState()
        engine = SelectedCh1Calibrator(
            FakeWorker(FakeDevice(), (), {}, state=state),
            LinearMeter(state, smsr=35.0),
            fullband_points(),
            CalibrationSettings(),
        )
        result = engine.calibrate(
            CandidatePoint(0, 50, 1526.0, candidate),
            1.32,
            ceiling_codes=ceiling,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["seed_source"], "ceiling_probe")
        self.assertEqual(result["selected_seed_codes"], list(ceiling))
        self.assertEqual(result["codes"][3:], list(ceiling[3:]))

        duplicate_state = FakeState()
        duplicate_meter = LinearMeter(duplicate_state, smsr=35.0)
        duplicate_engine = SelectedCh1Calibrator(
            FakeWorker(FakeDevice(), (), {}, state=duplicate_state),
            duplicate_meter,
            fullband_points(),
            CalibrationSettings(),
        )
        duplicate_result = duplicate_engine.calibrate(
            CandidatePoint(0, 50, 1526.0, candidate),
            1.0,
            ceiling_codes=candidate,
        )
        self.assertTrue(duplicate_result["passed"])
        self.assertEqual(len(duplicate_result["seed_probes"]), 1)
        self.assertEqual(duplicate_result["seed_source"], "candidate")

    def test_transient_low_smsr_is_latched_and_never_accepted_as_final(self):
        state = FakeState()
        candidate = codes(soa=100.0, phase=5.0)
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = CountedSmsrMeter(
            state, candidate, low_from_visit=3, persistent=False
        )
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )

        result = engine.calibrate(
            CandidatePoint(0, 50, 1526.0, candidate), 1.0
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["trajectory_classification"], "transient_side_mode_event")
        self.assertIn(list(candidate), result["forbidden_codes"])
        self.assertFalse(result["smsr_diagnostics"][-1]["accept_state_for_final"])
        self.assertEqual(len(result["smsr_diagnostics"][-1]["repeat_readings"]), 2)

    def test_repeated_low_smsr_is_classified_as_unstable_and_rejected(self):
        state = FakeState()
        candidate = codes(soa=100.0, phase=5.0)
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = CountedSmsrMeter(
            state, candidate, low_from_visit=3, persistent=True
        )
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )

        result = engine.calibrate(
            CandidatePoint(0, 50, 1526.0, candidate), 1.0
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["trajectory_classification"], "soa_phase_trajectory_unstable"
        )
        self.assertTrue(result["candidate_replacement_required"])
        self.assertLess(
            min(
                reading["normalized_smsr_db"]
                for reading in result["smsr_diagnostics"][-1]["repeat_readings"]
            ),
            20.0,
        )

    def test_low_smsr_on_soa_trajectory_is_latched_and_clean_state_rolled_back(self):
        state = FakeState()
        candidate = codes(soa=100.0, phase=5.0)
        suspect_values = list(candidate)
        suspect_values[1] -= 655  # exact first 1.5 mA bounded SOA step
        suspect = tuple(suspect_values)
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = CountedSmsrMeter(
            state, suspect, low_from_visit=1, persistent=True
        )
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )

        result = engine.calibrate(
            CandidatePoint(0, 50, 1526.0, candidate), 0.90
        )

        self.assertFalse(result["passed"])
        event = result["smsr_diagnostics"][-1]
        self.assertEqual(event["context"], "calibration_soa_outer")
        self.assertEqual(event["suspect_codes"], list(suspect))
        self.assertEqual(event["rollback_codes"], list(candidate))
        self.assertTrue(event["rollback_confirmed_clean"])
        self.assertIn(list(suspect), result["forbidden_codes"])

    def test_resumed_run_blocks_inherited_low_smsr_exact_state(self):
        state = FakeState()
        candidate = codes(soa=100.0, phase=5.0)
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        engine = SelectedCh1Calibrator(
            worker, LinearMeter(state), fullband_points(), CalibrationSettings()
        )

        result = engine.calibrate(
            CandidatePoint(0, 50, 1526.0, candidate),
            1.0,
            inherited_forbidden_codes=[candidate],
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["candidate_replacement_required"])
        self.assertEqual(result["seed_probes"][0]["source"], "candidate")
        self.assertTrue(
            result["seed_probes"][0]["blocked_by_prior_low_smsr_latch"]
        )
        self.assertEqual(meter := engine.meter.measurements, 0)

    def test_retry_stage_inherits_forbidden_codes_from_row_and_retry_history(self):
        point = CandidatePoint(0, 50, 1526.0, codes(soa=100.0))
        first_bad = codes(soa=98.5)
        older_bad = codes(soa=97.0)

        def failed_row(bad_codes):
            return {
                "passed": False,
                "history": [
                    {
                        "codes": list(bad_codes),
                        "reading": {
                            "wavelength_nm": 1526.0,
                            "power_mw": 1.0,
                            "peak_count": 2,
                            "side_mode_suppression_db": 12.0,
                        },
                    }
                ],
            }

        class RecordingCalibrator:
            settings = CalibrationSettings()

            def __init__(self):
                self.kwargs = None

            def calibrate(self, candidate, target_power_mw, **kwargs):
                self.kwargs = kwargs
                return {
                    "passed": True,
                    "codes": list(candidate.codes),
                    "final_wavelength_error_pm": 0.0,
                    "final_smsr_min_db": 35.0,
                    "final_power_mw": target_power_mw,
                }

        recorder = RecordingCalibrator()
        report = {
            "calibration": {
                "rows": {"0": failed_row(first_bad)},
                "retry_history": {"0": [failed_row(older_bad)]},
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            ok = run_stage(
                mode="calibrate",
                candidates=[point],
                calibrator=recorder,
                report=report,
                output=Path(folder) / "checkpoint.json",
                target_power_mw=1.0,
                calibration_ceiling_codes={0: codes(soa=135.0)},
                retry_failed_calibration=True,
            )
        self.assertTrue(ok)
        inherited = {
            tuple(item) for item in recorder.kwargs["inherited_forbidden_codes"]
        }
        self.assertEqual(inherited, {first_bad, older_bad})
        self.assertEqual(
            tuple(recorder.kwargs["ceiling_codes"]), codes(soa=135.0)
        )

    def test_staged_controller_can_reduce_soa_beyond_old_symmetric_envelope(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = CoupledCalibrationMeter(state)
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        point = CandidatePoint(
            0,
            50,
            1526.0,
            codes(gain=135.0, soa=135.0, phase=0.81, wave_a=30.0, wave_b=30.0),
        )

        result = engine.calibrate(point, 1.66725)

        self.assertTrue(result["passed"])
        self.assertLess(result["currents_ma"][1], 123.0)
        self.assertLessEqual(abs(result["final_wavelength_error_pm"]), 2.0)
        self.assertAlmostEqual(
            result["control_power_tolerance_mw"], 1.66725 * 0.01, places=6
        )
        self.assertAlmostEqual(
            result["final_power_mw"],
            1.66725,
            delta=result["control_power_tolerance_mw"],
        )
        self.assertTrue(
            any(
                item.get("context") == "calibration_phase_inner"
                for item in result["history"]
            )
        )
        target_commands = [
            state.commands[index][0] for index in range(1, len(state.commands), 2)
        ]
        for sent in target_commands:
            self.assertEqual(sent[0], point.codes[0])
            self.assertEqual(sent[3:], point.codes[3:])
            for actual, maximum in zip(currents_ma(sent), ABSOLUTE_LIMITS_MA):
                self.assertLessEqual(actual, maximum + 1e-9)

    def test_joint_tuning_converges_and_requires_two_replays(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = LinearMeter(state, smsr=35.0)
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        point = CandidatePoint(0, 50, 1526.0, codes(soa=105.0, phase=4.8))
        result = engine.calibrate(point, 1.0)
        self.assertTrue(result["passed"])
        self.assertGreaterEqual(len(result["confirmations"]), 2)
        self.assertLessEqual(abs(result["final_wavelength_error_pm"]), 2.0)
        self.assertAlmostEqual(result["final_power_mw"], 1.0, delta=0.01)
        # Every optical observation is a pair of commands.
        self.assertEqual(len(state.commands), meter.measurements * 2)

    def test_all_tuning_commands_respect_hard_limits_and_lock_branch_channels(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = LinearMeter(state, smsr=35.0)
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        point = CandidatePoint(
            0,
            77,
            1526.54,
            codes(gain=135.0, soa=134.99, phase=9.9, wave_a=30.0, wave_b=30.0),
        )
        engine.calibrate(point, 1.5)
        for sent, _mode in state.commands:
            values = currents_ma(sent)
            for actual, maximum in zip(values, ABSOLUTE_LIMITS_MA):
                self.assertLessEqual(actual, maximum + 1e-9)
        target_commands = [
            state.commands[index][0] for index in range(1, len(state.commands), 2)
        ]
        for sent in target_commands:
            self.assertEqual(sent[0], point.codes[0])
            self.assertEqual(sent[3:], point.codes[3:])

    def test_repeatability_span_failure_rejects_final_codes(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        # First reading is already converged; the two confirmations span 4 pm.
        meter = LinearMeter(state, smsr=35.0, wavelength_jitter_pm=[0.0, -2.0, 2.0])
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        result = engine.calibrate(
            CandidatePoint(0, 88, 1526.76, codes(soa=100.0, phase=5.0)), 1.0
        )
        self.assertFalse(result["passed"])
        self.assertGreater(result["final_wavelength_span_pm"], 2.0)


class CeilingAndSequenceTests(unittest.TestCase):
    def test_ceiling_phase_controller_learns_steep_slope_without_oscillation(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = AffinePhaseMeter(
            state, phase_slope_pm_per_ma=-330.0, zero_phase_ma=5.0
        )
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        seed_codes = codes(soa=135.0, phase=4.98)
        point = CandidatePoint(0, 90, 1526.8, seed_codes)

        result = engine.ceiling_probe(
            point, CeilingSeed(codes=seed_codes, source="fake_audit")
        )

        self.assertTrue(result["qualified"])
        self.assertLessEqual(
            abs(result["ceiling_reading"]["wavelength_error_pm"]), 2.0
        )
        corrections = [
            item
            for item in result["history"]
            if item["action"] == "phase_correction"
        ]
        self.assertLessEqual(len(corrections), 3)
        self.assertTrue(
            any(item["controller"] == "measured_bracket_secant" for item in corrections)
        )

    def test_ceiling_phase_controller_rejects_real_upper_phase_boundary(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = AffinePhaseMeter(
            state, phase_slope_pm_per_ma=-150.0, zero_phase_ma=10.05
        )
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings()
        )
        seed_codes = codes(soa=135.0, phase=10.0)
        point = CandidatePoint(0, 90, 1526.8, seed_codes)

        result = engine.ceiling_probe(
            point, CeilingSeed(codes=seed_codes, source="fake_audit")
        )

        self.assertFalse(result["qualified"])
        for sent, _mode in state.commands:
            self.assertLessEqual(sent[2], safe_current_to_code(10.0, 2))

    def test_ceiling_probe_steps_only_to_measured_clean_state_then_rolls_back(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = SoaCeilingMeter(state)
        settings = CalibrationSettings(ceiling_soa_step_ma=2.0)
        engine = SelectedCh1Calibrator(worker, meter, fullband_points(), settings)
        point = CandidatePoint(0, 90, 1526.8, codes(soa=100.0, phase=5.0))
        seed = CeilingSeed(codes=codes(soa=108.0, phase=5.0), source="fake_audit")
        result = engine.ceiling_probe(point, seed)
        self.assertTrue(result["qualified"])
        self.assertAlmostEqual(result["clean_ceiling_power_mw"], 1.10, places=2)
        self.assertTrue(
            any(
                item["action"] == "rollback_after_rejected_step"
                for item in result["history"]
            )
        )
        # The rejected 112 mA state is never reported as the ceiling.
        self.assertLessEqual(result["ceiling_currents_ma"][1], 110.01)
        summary = summarize_ceiling({"0": result}, 0.01)
        self.assertAlmostEqual(summary["recommended_common_power_mw"], 1.089, places=3)
        self.assertFalse(summary["uses_unmeasured_power_extrapolation"])

    def test_sparse_sequence_uses_previous_sparse_point_and_marks_bad_path(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        meter = OneTargetBadMeter(state, bad_target_nm=1525.02)
        engine = SelectedCh1Calibrator(
            worker, meter, fullband_points(), CalibrationSettings(sequence_rounds=2)
        )
        points = (
            CandidatePoint(0, 0, 1525.0, codes(soa=100.0)),
            CandidatePoint(1, 1, 1525.02, codes(soa=101.0)),
            CandidatePoint(2, 2, 1525.04, codes(soa=102.0)),
        )
        rounds = {}
        for round_index in range(2):
            round_rows = rounds[str(round_index)] = {}
            for point in points:
                predecessor = (point.order - 1) % len(points)
                round_rows[str(point.order)] = engine.sparse_sequence_observation(
                    point,
                    point.codes,
                    points[predecessor].codes,
                    round_index=round_index,
                    predecessor_order=predecessor,
                )
        first = rounds["0"]["0"]
        self.assertTrue(first["wraparound"])
        self.assertEqual(first["predecessor_codes"], list(points[-1].codes))
        bad = rounds["0"]["1"]
        self.assertFalse(bad["passed"])
        self.assertTrue(bad["requires_replacement_or_hidden_predecessor"])
        summary = summarize_sparse_sequence(rounds, point_count=3, required_rounds=2)
        self.assertFalse(summary["all_points_runtime_path_certified"])
        self.assertEqual(summary["failed_orders"], [1])

    def test_sparse_sequence_can_replay_one_exact_hidden_fullband_predecessor(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        audited = list(fullband_points())
        hidden_codes = codes(soa=88.0, phase=4.25)
        audited[140] = SimpleNamespace(codes=hidden_codes)
        engine = SelectedCh1Calibrator(
            worker, LinearMeter(state), tuple(audited), CalibrationSettings()
        )
        sparse_previous = codes(soa=104.0, phase=1.4)
        target = CandidatePoint(2, 141, 1527.82, codes(soa=96.0, phase=4.8))

        row = engine.sparse_sequence_observation(
            target,
            target.codes,
            sparse_previous,
            round_index=0,
            predecessor_order=1,
            hidden_predecessor_fullband_index=140,
            hidden_predecessor_codes=hidden_codes,
        )

        self.assertEqual(
            [command for command, _mode in state.commands[-3:]],
            [sparse_previous, hidden_codes, target.codes],
        )
        self.assertTrue(row["used_hidden_predecessor"])
        self.assertEqual(row["runtime_path"], SPARSE_HIDDEN_PATH)
        self.assertEqual(
            row["hidden_predecessor"],
            {
                "fullband_index": 140,
                "codes": list(hidden_codes),
                "hold_us": 5000,
            },
        )

    def test_default_sparse_path_has_no_hidden_command(self):
        state = FakeState()
        worker = FakeWorker(FakeDevice(), (), {}, state=state)
        engine = SelectedCh1Calibrator(
            worker, LinearMeter(state), fullband_points(), CalibrationSettings()
        )
        point = CandidatePoint(1, 141, 1527.82, codes(soa=96.0))
        previous = codes(soa=104.0)
        row = engine.sparse_sequence_observation(
            point,
            point.codes,
            previous,
            round_index=0,
            predecessor_order=0,
        )
        self.assertEqual(
            [command for command, _mode in state.commands[-2:]],
            [previous, point.codes],
        )
        self.assertFalse(row["used_hidden_predecessor"])
        self.assertEqual(row["runtime_path"], SPARSE_DIRECT_PATH)
        self.assertNotIn("hidden_predecessor", row)

    def test_hidden_path_change_archives_complete_old_rounds_before_reset(self):
        old_rounds = {
            "0": {"0": {"passed": True}, "1": {"passed": False}},
            "1": {"0": {"passed": True}, "1": {"passed": False}},
        }
        report = {
            "sparse_sequence": {
                "code_source": "calibration",
                "rounds": old_rounds,
                "summary": {"all_points_runtime_path_certified": False},
            },
            "final_runtime_certified": False,
        }
        request = {
            "mode_order": 2,
            "fullband_index": 141,
            "predecessor_fullband_index": 140,
            "predecessor_codes": list(codes()),
            "hold_us": 5000,
            "runtime_path": SPARSE_HIDDEN_PATH,
        }
        configure_sparse_sequence_path(report, request)
        sequence = report["sparse_sequence"]
        self.assertEqual(sequence["rounds"], {})
        self.assertEqual(sequence["summary"], {})
        self.assertNotIn("code_source", sequence)
        self.assertEqual(sequence["hidden_predecessor_request"], request)
        archived = sequence["retry_history"][0]
        self.assertEqual(archived["rounds"], old_rounds)
        self.assertEqual(
            archived["reason"], "hidden_predecessor_configuration_changed"
        )
        self.assertEqual(len(archived["sha256"]), 64)

    def test_hidden_request_is_single_and_hold_is_firmware_locked(self):
        candidates = (
            CandidatePoint(0, 140, 1527.80, codes()),
            CandidatePoint(1, 141, 1527.82, codes()),
        )
        with self.assertRaisesRegex(ValueError, "最多只允许1个"):
            build_hidden_predecessor_request(
                candidates, fullband_points(), [0, 1], 5000
            )
        with self.assertRaisesRegex(ValueError, "5000 us"):
            build_hidden_predecessor_request(
                candidates, fullband_points(), [1], 4999
            )


class ExecuteSafetyTests(unittest.TestCase):
    def test_keyboard_interrupt_still_confirms_shutter_and_records_interrupt(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "fullband_index": 1,
                                "target_nm": 1525.02,
                                "codes": list(codes()),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()
            shutter_calls = []

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            with self.assertRaises(KeyboardInterrupt):
                execute(
                    make_args(source, output),
                    device_factory=FakeDevice,
                    meter_factory=lambda _resource: InterruptingMeter(state),
                    fullband_points=fullband_points(),
                    worker_factory=worker_factory,
                    shutter_callback=lambda device, worker: (
                        shutter_calls.append((device, worker)) or True
                    ),
                )

            self.assertEqual(len(shutter_calls), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["shutter_confirmed"])
            self.assertEqual(report["sessions"][-1]["status"], "interrupted")

    def test_clean_ceiling_rejections_return_two_after_confirmed_shutter(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "fullband_index": 1,
                                "target_nm": 1525.02,
                                "codes": list(codes()),
                            },
                            {
                                "fullband_index": 2,
                                "target_nm": 1525.04,
                                "codes": list(codes()),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            result = execute(
                make_args(source, output, mode="ceiling-probe"),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: LinearMeter(state, smsr=12.0),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )

            self.assertEqual(result, 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["shutter_confirmed"])
            self.assertEqual(
                report["sessions"][-1]["status"], "completed_with_rejections"
            )

    def test_retry_rejected_remeasures_only_failure_and_archives_prior_row(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "fullband_index": 1,
                                "target_nm": 1525.02,
                                "codes": list(codes()),
                            },
                            {
                                "fullband_index": 2,
                                "target_nm": 1525.04,
                                "codes": list(codes()),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            first_result = execute(
                make_args(source, output, mode="ceiling-probe"),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: OneTargetBadMeter(
                    state, bad_target_nm=1525.04
                ),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )
            self.assertEqual(first_result, 2)
            first_measurements = len(state.commands)
            first_report = json.loads(output.read_text(encoding="utf-8"))
            qualified_before_retry = first_report["ceiling_probe"]["rows"]["0"]
            self.assertTrue(qualified_before_retry["qualified"])
            self.assertFalse(
                first_report["ceiling_probe"]["rows"]["1"]["qualified"]
            )

            second_result = execute(
                make_args(
                    source,
                    output,
                    mode="ceiling-probe",
                    retry_rejected=True,
                ),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: LinearMeter(state, smsr=35.0),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )

            self.assertEqual(second_result, 0)
            self.assertGreater(len(state.commands), first_measurements)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["ceiling_probe"]["rows"]["0"]["qualified"])
            self.assertEqual(
                report["ceiling_probe"]["rows"]["0"], qualified_before_retry
            )
            self.assertNotIn("0", report["ceiling_probe"]["retry_history"])
            archived = report["ceiling_probe"]["retry_history"]["1"]
            self.assertEqual(len(archived), 1)
            self.assertFalse(archived[0]["qualified"])
            self.assertTrue(report["shutter_confirmed"])

    def test_calibration_fail_fast_then_retry_failure_and_continue_missing_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "fullband_index": 1,
                                "target_nm": 1525.02,
                                "codes": list(codes(soa=135.0)),
                            },
                            {
                                "fullband_index": 2,
                                "target_nm": 1525.04,
                                "codes": list(codes(soa=135.0)),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            ceiling_result = execute(
                make_args(source, output, mode="ceiling-probe"),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: LinearMeter(state, smsr=35.0),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )
            self.assertEqual(ceiling_result, 0)

            failed_result = execute(
                make_args(source, output, mode="calibrate"),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: OneTargetWrongBranchMeter(
                    state, bad_target_nm=1525.02
                ),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )
            self.assertEqual(failed_result, 2)
            failed_report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(failed_report["calibration"]["rows"]), {"0"})
            self.assertFalse(failed_report["calibration"]["rows"]["0"]["passed"])
            self.assertTrue(failed_report["shutter_confirmed"])

            retry_result = execute(
                make_args(
                    source,
                    output,
                    mode="calibrate",
                    retry_failed_calibration=True,
                ),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: LinearMeter(state, smsr=35.0),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )
            self.assertEqual(retry_result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(report["calibration"]["rows"]), {"0", "1"})
            self.assertTrue(
                all(row["passed"] for row in report["calibration"]["rows"].values())
            )
            archived = report["calibration"]["retry_history"]["0"]
            self.assertEqual(len(archived), 1)
            self.assertFalse(archived[0]["passed"])
            self.assertNotIn("1", report["calibration"]["retry_history"])
            self.assertTrue(report["shutter_confirmed"])

    def test_connection_timeout_still_attempts_safe_shutter_before_close(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "fullband_index": 1,
                                "target_nm": 1525.02,
                                "codes": list(codes()),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()
            shutter_calls = []

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            with self.assertRaisesRegex(RuntimeError, "未建立局域网"):
                execute(
                    make_args(source, output, connect_timeout_s=0.01),
                    device_factory=NeverConnectedDevice,
                    meter_factory=lambda _resource: LinearMeter(state),
                    fullband_points=fullband_points(),
                    worker_factory=worker_factory,
                    shutter_callback=lambda device, worker: (
                        shutter_calls.append((device, worker)) or False
                    ),
                )
            self.assertEqual(len(shutter_calls), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["shutter_confirmed"])

    def test_meter_failure_still_attempts_shutter_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "fullband_index": 1,
                                "target_nm": 1525.02,
                                "codes": list(codes()),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()
            shutter_calls = []

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            with self.assertRaisesRegex(RuntimeError, "simulated meter failure"):
                execute(
                    make_args(source, output),
                    device_factory=FakeDevice,
                    meter_factory=lambda _resource: LinearMeter(state, fail=True),
                    fullband_points=fullband_points(),
                    worker_factory=worker_factory,
                    shutter_callback=lambda device, worker: (
                        shutter_calls.append((device, worker)) or False
                    ),
                )
            self.assertEqual(len(shutter_calls), 1)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["shutter_confirmed"])
            self.assertEqual(report["sessions"][-1]["status"], "failed")

    def test_clean_probe_is_incremental_and_shutter_confirmed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "candidate.json"
            output = Path(folder) / "report.json"
            source.write_text(
                json.dumps(
                    {
                        "stress": {
                            "rows": [
                                {
                                    "fullband_index": 1,
                                    "target_wavelength_nm": 1525.02,
                                    "codes": list(codes()),
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = FakeState()

            def worker_factory(port, points, guards, repeat):
                return FakeWorker(port, points, guards, repeat, state=state)

            result = execute(
                make_args(source, output),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: LinearMeter(state, smsr=35.0),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )
            self.assertEqual(result, 0)
            first_command_count = len(state.commands)
            result = execute(
                make_args(source, output),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: LinearMeter(state, smsr=35.0),
                fullband_points=fullband_points(),
                worker_factory=worker_factory,
                shutter_callback=lambda _device, _worker: True,
            )
            self.assertEqual(result, 0)
            self.assertEqual(len(state.commands), first_command_count)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["shutter_confirmed"])
            self.assertFalse(report["writes_runtime_table"])
            self.assertFalse(report["writes_firmware"])


if __name__ == "__main__":
    unittest.main()
