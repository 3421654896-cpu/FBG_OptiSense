from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from calibrate_selected_ch1_table import validate_and_hash_fullband_points
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import Reading
from validate_ch1_g1_hidden_predecessor import (
    EXPECTED_CALIBRATED_CODES,
    EXPECTED_TARGET_POWER_MW,
    HIDDEN_CODES,
    POWER_TOLERANCE_MW,
    _validate_hold_schedule,
    execute,
)


class FakeDevice:
    def __init__(self, **_kwargs):
        self.connected = False
        self.is_open = False
        self.board_ip = "fake-board"
        self.writes = []
        self.closed = False

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
        self.closed = True


class FakeState:
    def __init__(self):
        self.codes = EXPECTED_CALIBRATED_CODES[0]
        self.commands = []
        self.hidden_hold_ms = None
        self.order2_count = 0


class FakeWorker:
    def __init__(self, port, _points, _guards, _repeat=False, *, state):
        self.port = port
        self.state = state
        self._rx_buffer = bytearray()
        self._last_codes = state.codes

    def _interruptible_wait(self, seconds):
        if tuple(self.state.codes) == HIDDEN_CODES:
            self.state.hidden_hold_ms = round(float(seconds) * 1000.0)
        return True

    def _command_and_ack(self, command_codes, soa_mode=0):
        exact = validate_pi11210_codes(command_codes)
        if soa_mode == 1:
            exact = (exact[0], 0, exact[2], exact[3], exact[4])
        self.state.codes = exact
        self.state.commands.append((exact, soa_mode))
        self._last_codes = exact
        return 0.001


class HoldAwareMeter:
    identity = "YOKOGAWA,AQ6150,FAKE,01.03"

    def __init__(
        self,
        state,
        *,
        minimum_hidden_hold_ms=5,
        fail_order2_numbers=(),
        fail_order3=False,
        interrupt=False,
    ):
        self.state = state
        self.minimum_hidden_hold_ms = minimum_hidden_hold_ms
        self.fail_order2_numbers = set(fail_order2_numbers)
        self.fail_order3 = fail_order3
        self.interrupt = interrupt
        self.closed = False

    def measure(self, target_nm, select_main_peak=False):
        if self.interrupt:
            raise KeyboardInterrupt()
        self.assert_main_peak = select_main_peak
        is_order2 = abs(float(target_nm) - 1527.82) < 1e-9
        if is_order2:
            self.state.order2_count += 1
            hidden_ok = (
                self.state.hidden_hold_ms is not None
                and self.state.hidden_hold_ms >= self.minimum_hidden_hold_ms
                and self.state.order2_count not in self.fail_order2_numbers
            )
            error_pm = 0.4 if hidden_ok else -348.1
        else:
            error_pm = 4.0 if self.fail_order3 else -0.5
        return Reading(
            wavelength_nm=float(target_nm) + error_pm / 1000.0,
            wavelength_error_pm=error_pm,
            power_mw=EXPECTED_TARGET_POWER_MW + POWER_TOLERANCE_MW / 4.0,
            power_dbm=2.2,
            peak_count=2,
            secondary_power_mw=0.0005,
            side_mode_suppression_db=32.0,
        )

    def close(self):
        self.closed = True


class ExactFakeShutter:
    def __init__(self):
        self.calls = 0

    def __call__(self, _device, worker):
        self.calls += 1
        worker._command_and_ack(worker._last_codes, soa_mode=1)
        return worker._last_codes[1] == 0


def make_fullband_points():
    safe = (50000, 40000, 10000, 10000, 10000)
    points = [
        SimpleNamespace(target_nm=1525.0 + index * 0.02, codes=safe)
        for index in range(2001)
    ]
    points[140] = SimpleNamespace(target_nm=1527.8, codes=HIDDEN_CODES)
    return tuple(points)


def write_evidence(folder: Path, fullband_points):
    candidate = folder / "candidate.json"
    calibration = folder / "calibration.json"
    output = folder / "audit.json"
    first_indices = [118, 122, 141, 170]
    remaining = [value for value in range(178, 2001)][:41]
    indices = first_indices + remaining
    rows = []
    for order, index in enumerate(indices):
        rows.append(
            {
                "fullband_index": index,
                "target_wavelength_nm": 1525.0 + index * 0.02,
                "codes": list(
                    EXPECTED_CALIBRATED_CODES.get(
                        order, (50000, 40000, 10000, 10000, 10000)
                    )
                ),
            }
        )
    candidate.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    calibration_rows = {
        str(order): {
            "order": order,
            "codes": list(
                EXPECTED_CALIBRATED_CODES.get(
                    order, (50000, 40000, 10000, 10000, 10000)
                )
            ),
            "passed": True,
        }
        for order in range(45)
    }
    calibration.write_text(
        json.dumps(
            {
                "candidate_source": {
                    "path": str(candidate),
                    "sha256": candidate_hash,
                    "point_count": 45,
                },
                "fullband_predecessor_sha256": validate_and_hash_fullband_points(
                    fullband_points
                ),
                "shutter_confirmed": True,
                "calibration": {
                    "target_power_mw": EXPECTED_TARGET_POWER_MW,
                    "summary": {"all_points_passed": True},
                    "rows": calibration_rows,
                },
            }
        ),
        encoding="utf-8",
    )
    return candidate, calibration, output


def make_args(candidate, calibration, output, *, hold_ms):
    return argparse.Namespace(
        candidate=candidate,
        calibration_report=calibration,
        output=output,
        host="fake",
        gpib="fake-gpib",
        connect_timeout_s=1.0,
        hold_ms=hold_ms,
    )


class HiddenPredecessorValidationTests(unittest.TestCase):
    def test_escalates_and_accepts_first_ten_of_ten_hold(self):
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            points = make_fullband_points()
            candidate, calibration, output = write_evidence(folder, points)
            state = FakeState()
            shutter = ExactFakeShutter()
            meter = HoldAwareMeter(state, minimum_hidden_hold_ms=20)

            result = execute(
                make_args(candidate, calibration, output, hold_ms=[5, 10, 20, 40]),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: meter,
                fullband_points=points,
                worker_factory=lambda port, p, g, repeat: FakeWorker(
                    port, p, g, repeat, state=state
                ),
                shutter_callback=shutter,
            )

            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["accepted_hidden_hold_ms"], 20)
            self.assertEqual(list(report["hold_results"]), ["5", "10", "20"])
            self.assertEqual(report["hold_results"]["5"]["completed_trials"], 10)
            self.assertEqual(report["hold_results"]["5"]["passed_trials"], 0)
            self.assertEqual(report["hold_results"]["20"]["passed_trials"], 10)
            self.assertTrue(report["shutter_confirmed"])
            self.assertEqual(report["status"], "complete")
            self.assertTrue(meter.closed)
            trial = report["hold_results"]["20"]["batches"][0]["trials"][0]
            self.assertEqual(
                [command["label"] for command in trial["commands"]],
                [
                    "order0_idx118",
                    "order1_idx122",
                    "hidden_idx140",
                    "order2_idx141",
                    "order3_idx170",
                ],
            )
            self.assertEqual(trial["commands"][2]["codes"], list(HIDDEN_CODES))
            self.assertEqual(trial["commands"][2]["hold_ms"], 20.0)
            self.assertTrue(all(c["exact_ack"] for c in trial["commands"]))
            # Two reset shutters per attempted hold plus the unconditional exit shutter.
            self.assertEqual(shutter.calls, 7)

    def test_one_failure_rejects_hold_even_when_nine_trials_pass(self):
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            points = make_fullband_points()
            candidate, calibration, output = write_evidence(folder, points)
            state = FakeState()
            shutter = ExactFakeShutter()
            meter = HoldAwareMeter(
                state,
                minimum_hidden_hold_ms=5,
                fail_order2_numbers={4},
            )

            result = execute(
                make_args(candidate, calibration, output, hold_ms=[5, 10]),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: meter,
                fullband_points=points,
                worker_factory=lambda port, p, g, repeat: FakeWorker(
                    port, p, g, repeat, state=state
                ),
                shutter_callback=shutter,
            )

            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["hold_results"]["5"]["passed_trials"], 9)
            self.assertFalse(report["hold_results"]["5"]["accepted"])
            self.assertEqual(report["hold_results"]["10"]["passed_trials"], 10)
            self.assertEqual(report["accepted_hidden_hold_ms"], 10)

    def test_order3_failure_rejects_otherwise_clean_hidden_path(self):
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            points = make_fullband_points()
            candidate, calibration, output = write_evidence(folder, points)
            state = FakeState()
            shutter = ExactFakeShutter()
            meter = HoldAwareMeter(state, fail_order3=True)

            result = execute(
                make_args(candidate, calibration, output, hold_ms=[5]),
                device_factory=FakeDevice,
                meter_factory=lambda _resource: meter,
                fullband_points=points,
                worker_factory=lambda port, p, g, repeat: FakeWorker(
                    port, p, g, repeat, state=state
                ),
                shutter_callback=shutter,
            )

            self.assertEqual(result, 2)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(report["accepted_hidden_hold_ms"])
            self.assertEqual(report["status"], "rejected")
            trial = report["hold_results"]["5"]["batches"][0]["trials"][0]
            self.assertTrue(all(trial["order2"]["gates"].values()))
            self.assertFalse(trial["order3"]["gates"]["wavelength"])
            self.assertFalse(trial["passed"])
            self.assertTrue(report["shutter_confirmed"])

    def test_keyboard_interrupt_still_shutters_and_marks_audit(self):
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            points = make_fullband_points()
            candidate, calibration, output = write_evidence(folder, points)
            state = FakeState()
            shutter = ExactFakeShutter()
            meter = HoldAwareMeter(state, interrupt=True)

            with self.assertRaises(KeyboardInterrupt):
                execute(
                    make_args(candidate, calibration, output, hold_ms=[5]),
                    device_factory=FakeDevice,
                    meter_factory=lambda _resource: meter,
                    fullband_points=points,
                    worker_factory=lambda port, p, g, repeat: FakeWorker(
                        port, p, g, repeat, state=state
                    ),
                    shutter_callback=shutter,
                )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "interrupted")
            self.assertTrue(report["shutter_confirmed"])
            self.assertEqual(shutter.calls, 1)
            self.assertTrue(meter.closed)

    def test_hold_schedule_is_limited_and_strictly_increasing(self):
        self.assertEqual(_validate_hold_schedule([5, 20, 80]), (5, 20, 80))
        for invalid in ([10, 5], [5, 5], [1], [], [5.5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_hold_schedule(invalid)


if __name__ == "__main__":
    unittest.main()
