import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import probe_ch1_g4_candidates as probe


class CandidateScreenTests(unittest.TestCase):
    def test_real_candidates_use_safe_existing_codes_and_no_install(self):
        rows, _ = probe.load_installed_rows()
        fullband = probe.app.load_fullband_accuracy_table()
        cases = probe.build_cases(rows, fullband)
        self.assertEqual(len(cases), 9)
        for case in cases:
            self.assertEqual(set(case['sources']),
                             {'held_installed_previous', 'held_fullband_previous'})
            self.assertEqual(case['sources']['held_installed_previous'], rows[case['slot']-1]['codes'])
            if case['name'].startswith('fullband_'):
                self.assertEqual(tuple(case['codes']), fullband[case['index']].codes)

    def test_output_and_bounds_rejected_before_opening_hardware(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(probe.serial, 'Serial') as serial:
            existing = Path(tmp) / 'existing.json'
            existing.touch()
            with self.assertRaises(FileExistsError):
                probe.execute(existing)
            for repeats in (0, 3, True):
                with self.assertRaises(ValueError):
                    probe.execute(Path(tmp)/'new.json', repeats=repeats)
            serial.assert_not_called()

    def test_missing_dac_ack_rejected(self):
        worker = SimpleNamespace(_command_and_ack=lambda _: None)
        with self.assertRaises(RuntimeError):
            probe.checked_command(worker, [0]*5)

    def test_meter_screen_is_not_no_secondary_peak_or_fast_certificate(self):
        reading = SimpleNamespace(wavelength_nm=1540., power_mw=1.7, peak_count=2, side_mode_suppression_db=30.)
        result = probe.optical_review(reading, 1540.)
        self.assertTrue(result['meets_existing_20db_screen'])
        self.assertTrue(result['detectable_secondary_peak'])
        self.assertFalse(result['fast_path_qualified'])
        reading.side_mode_suppression_db = 10.
        self.assertFalse(probe.optical_review(reading, 1540.)['meets_existing_20db_screen'])
        reading.wavelength_nm = float('nan')
        with self.assertRaises(ValueError):
            probe.optical_review(reading, 1540.)

    def test_meter_failure_still_disarms_and_shutters(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(probe.serial, 'Serial'), \
                patch.object(probe.app, 'EqualIntervalWorker', return_value=Mock()), \
                patch.object(probe, '_safe_disarm', return_value=True) as disarm, \
                patch.object(probe, '_close_shutter', return_value=True) as shutter, \
                patch.object(probe, 'AQ6150B', side_effect=RuntimeError('meter offline')):
            output = Path(tmp)/'failure.json'
            with self.assertRaisesRegex(RuntimeError, 'meter offline'):
                probe.execute(output)
            self.assertEqual(disarm.call_count, 2)
            shutter.assert_called_once()
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertFalse(saved['complete'])
            self.assertTrue(saved['safety_cleanup']['exact_soa_shutter_ack'])

    def test_complete_report_cannot_hide_missing_trials(self):
        rows, _ = probe.load_installed_rows()
        cases = probe.build_cases(rows, probe.app.load_fullband_accuracy_table())
        report = dict(schema='ch1_g4_static_candidate_screen_v1', cases=cases,
                      captures=[], repeats=2, complete=True)
        with self.assertRaisesRegex(ValueError, 'missing trials'):
            probe.summarize_candidates(report)


if __name__ == '__main__':
    unittest.main()
