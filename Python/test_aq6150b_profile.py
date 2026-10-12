"""Deterministic AQ6150B setup and blocking-FETCH tests."""

from __future__ import annotations

import unittest
from unittest import mock

import laser_power_calibration as calibration


class FakeInstrument:
    def __init__(self):
        self.timeout = None
        self.writes: list[str] = []
        self.queries: list[str] = []

    def write(self, command: str) -> None:
        self.writes.append(command)

    def query(self, command: str) -> str:
        self.queries.append(command)
        average_count = next(
            (item.rsplit(" ", 1)[-1] for item in reversed(self.writes)
             if item.startswith(":CALC2:COUN ")),
            "1",
        )
        replies = {
            "*IDN?": "YOKOGAWA,AQ6150B,TEST,01.03",
            ":SENS:CORR:DEV?": "NARROW",
            ":SENS:URAT?": "NORMAL",
            ":CALC2:ASE?": "1",
            ":CALC2:COUN?": f"+{average_count}",
            ":CALC2:PTHR:MOD?": "RELATIVE",
            ":CALC2:PTHR?": "+4.00000000E+001",
            ":CALC2:PEXC?": "+1.00000000E+001",
            ":FETC:ARR:POW:WAV?": "2,1.550000000E-6,1.551000000E-6",
            ":FETC:ARR:POW?": "2,3.010299957,-26.98970004",
        }
        return replies[command]

    def close(self) -> None:
        return


class FakeManager:
    def __init__(self, instrument: FakeInstrument):
        self.instrument = instrument

    def open_resource(self, _resource: str) -> FakeInstrument:
        return self.instrument

    def close(self) -> None:
        return


class AQ6150BProfileTests(unittest.TestCase):
    def setUp(self):
        self.instrument = FakeInstrument()
        manager = FakeManager(self.instrument)
        self.manager_patch = mock.patch.object(
            calibration.pyvisa, "ResourceManager", return_value=manager
        )
        self.manager_patch.start()

    def tearDown(self):
        self.manager_patch.stop()

    def test_constructor_applies_and_verifies_reproducible_profile(self):
        meter = calibration.AQ6150B("GPIB0::7::INSTR")
        self.assertEqual(meter.measurement_profile["update_rate"], "NORMAL")
        self.assertEqual(meter.measurement_profile["peak_threshold_relative_db"], 40.0)
        self.assertIn("*CLS", self.instrument.writes)
        self.assertNotIn(":TRIG:SOUR BUS", self.instrument.writes)
        for command in (
            ":SENS:CORR:DEV NARR",
            ":SENS:URAT NORM",
            ":CALC2:ASE ON",
            ":CALC2:COUN 1",
            ":CALC2:PTHR:MOD REL",
            ":CALC2:PTHR 40",
            ":CALC2:PEXC 10",
        ):
            self.assertIn(command, self.instrument.writes)

    def test_fetch_arrays_select_strongest_peak_without_fixed_sleep(self):
        meter = calibration.AQ6150B("GPIB0::7::INSTR")
        with mock.patch.object(calibration.time, "sleep") as sleep:
            reading = meter.measure(1550.0, select_main_peak=True)
        sleep.assert_not_called()
        self.assertEqual(self.instrument.writes.count(":INIT"), 1)
        self.assertNotIn("*TRG", self.instrument.writes)
        self.assertAlmostEqual(reading.wavelength_nm, 1550.0, places=6)
        self.assertAlmostEqual(reading.power_mw, 2.0, places=6)
        self.assertEqual(reading.peak_count, 2)
        self.assertAlmostEqual(reading.side_mode_suppression_db, 30.0, places=6)

    def test_constructor_accepts_and_verifies_instrument_averaging(self):
        meter = calibration.AQ6150B("GPIB0::7::INSTR", average_count=3)
        self.assertEqual(meter.measurement_profile["average_count"], 3)
        self.assertIn(":CALC2:COUN 3", self.instrument.writes)

    def test_constructor_rejects_out_of_range_average_count_before_hardware(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            calibration.AQ6150B("GPIB0::7::INSTR", average_count=0)


if __name__ == "__main__":
    unittest.main()
