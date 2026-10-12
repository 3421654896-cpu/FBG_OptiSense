import unittest

import numpy as np

from capture_fast_flank_stream import ROWS
from fast_flank_features import FastFlankBaseline, apparent_shift_pm, audit_static_stream


def rows_fixture():
    rows = []
    for index in range(45):
        local = index % 5
        rows.append(
            {
                "template_signal_v": 0.20 + local * 0.01,
                "template_slope_v_per_nm": 3.0 if local < 2 else -3.0,
            }
        )
    return rows


def frame(sequence, *, changed_grating=None, start_us=None):
    records = []
    for row in ROWS:
        code = 200
        if changed_grating is not None and row == changed_grating * 5 + 3:
            code = 230
        records.append({"row": row, "second_code": code})
    return {
        "sequence": sequence,
        "cycle_start_us": sequence * 21_000 if start_us is None else start_us,
        "minimum_warmup_complete": sequence >= 16,
        "records": records,
    }


class FastFlankFeatureTests(unittest.TestCase):
    def test_apparent_shift_requires_fixed_route(self):
        values = apparent_shift_pm(frame(16), rows_fixture())
        self.assertEqual(values.shape, (9,))
        self.assertTrue(np.all(np.isfinite(values)))
        bad = frame(16)
        bad["records"] = list(reversed(bad["records"]))
        with self.assertRaisesRegex(ValueError, "fixed"):
            apparent_shift_pm(bad, rows_fixture())

    def test_baseline_is_only_learned_from_declared_no_contact_and_freezes(self):
        baseline = FastFlankBaseline(rows_fixture(), baseline_frames=16)
        for sequence in range(32):
            result = baseline.update(frame(sequence), declared_no_contact=True)
        self.assertTrue(result.baseline_ready)
        baseline.freeze_for_contact()
        contact = baseline.update(frame(32, changed_grating=4), declared_no_contact=False)
        self.assertTrue(contact.baseline_frozen)
        self.assertIsNone(contact.physical_contact_area_mm2)
        self.assertNotAlmostEqual(contact.relative_shift_pm[4], 0.0)
        with self.assertRaisesRegex(ValueError, "frozen"):
            baseline.update(frame(33), declared_no_contact=True)

    def test_sequence_gap_invalidates_baseline(self):
        baseline = FastFlankBaseline(rows_fixture(), baseline_frames=16)
        baseline.update(frame(0), declared_no_contact=True)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            baseline.update(frame(2), declared_no_contact=True)
        self.assertFalse(baseline.ready)

    def test_static_audit_separates_drift_from_false_15hz_tone(self):
        rows = rows_fixture()
        frames = []
        for sequence in range(128):
            item = frame(sequence)
            # Synthetic slow drift plus a known 15 Hz false tone on G1 right.
            time_s = item["cycle_start_us"] / 1e6
            item["records"] = [dict(record) for record in item["records"]]
            item["records"][1]["second_code"] = round(
                200 + 0.05 * sequence + 4 * np.sin(2 * np.pi * 15 * time_s)
            )
            frames.append(item)
        result = audit_static_stream(frames, rows)
        self.assertEqual(result["post_minimum_warmup_frames"], 112)
        self.assertGreater(result["per_grating"]["G1"]["static_fitted_15hz_amplitude_pm"], 0)
        self.assertFalse(result["physical_15hz_verified"])
        self.assertFalse(result["contact_area_verified"])


if __name__ == "__main__":
    unittest.main()
