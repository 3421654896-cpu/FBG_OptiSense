import unittest
from copy import deepcopy
from pathlib import Path
import app_JDSU as app
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_current_dense9_teacher import make_plan
from analyze_current_dense9_teacher import validated_groups


class Dense9Tests(unittest.TestCase):
    def test_plan_keeps_recalibrated_anchors_distinct(self):
        points = app.load_fullband_accuracy_table(Path(__file__).parent)
        rows, _ = load_installed_rows()
        before = deepcopy(rows)
        plan = make_plan(points, rows)
        self.assertEqual(len(plan), 9)
        self.assertEqual(rows, before)
        for segment in plan:
            self.assertTrue(20 <= len(segment['indices']) <= 90)
            self.assertEqual(len(segment['anchor_indices']), 5)
            for index, row in zip(segment['anchor_indices'], segment['installed_anchor_rows']):
                self.assertAlmostEqual(points[index].target_nm, row['target_wavelength_nm'])
                self.assertEqual(index in segment['anchors_differing_from_dense_codes'],
                                 tuple(points[index].codes) != tuple(row['codes']))

    def test_invalid_target_stops_before_hardware(self):
        points = app.load_fullband_accuracy_table(Path(__file__).parent)
        rows, _ = load_installed_rows()
        rows[0]['target_wavelength_nm'] += .001
        with self.assertRaises(ValueError):
            make_plan(points, rows)

    def test_extended_reference_is_bounded_and_invalid_margin_rejected(self):
        points = app.load_fullband_accuracy_table(Path(__file__).parent)
        rows, _ = load_installed_rows()
        original = make_plan(points, rows)
        expanded = make_plan(points, rows, .16)
        for before, after in zip(original, expanded):
            self.assertTrue(set(before['indices']) < set(after['indices']))
            self.assertLessEqual(len(after['indices']), 90)
            self.assertEqual(before['installed_anchor_rows'], after['installed_anchor_rows'])
        for margin in (float('nan'), 1.0, 0.0, -.1):
            with self.assertRaises(ValueError): make_plan(points, rows, margin)

    def test_incomplete_or_unsafe_capture_cannot_be_candidate(self):
        for data in ({}, {'complete': True}, {'complete': True, 'disarm_ack': True,
                'shutter_ack': True, 'cnc_commands': [], 'dense_ch1_kohm': 5}):
            with self.assertRaises(ValueError):
                validated_groups(data)


if __name__ == '__main__':
    unittest.main()
