"""Offline regressions for press labels; no hardware or motion is opened."""
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import numpy as np

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from test_finger_features import synthetic_spectrum
from train_finger_localizer import (
    _area_label, _event_contact, _event_group, load_session_training_samples,
)


class SessionLabelTests(unittest.TestCase):
    def frame(self, sequence, timestamp, *, shift=0.0, **kwargs):
        wavelengths, codes = synthetic_spectrum(shift_nm=shift)
        settings = dict(transimpedance_ohm={1: 2000}, table_crc32=123,
                        packet_crc_ok=True, adc_value_kind="direct_adc")
        settings.update(kwargs)
        return SpectrumFrame(sequence, wavelengths, codes,
                             monotonic_ns=timestamp, **settings)

    def prefix(self):
        return [ContactEvent("baseline", event_id="base", monotonic_ns=1),
                *[self.frame(i, i + 1) for i in range(1, 4)],
                ContactEvent("contact", event_id="press", monotonic_ns=10,
                             metadata={"model_xy_mm": [1, 2]})]

    def load(self, records, metadata=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture"
            with FingerSessionWriter(path, session_id="trial", manifest_extra=metadata) as writer:
                for record in records:
                    if isinstance(record, ContactEvent):
                        writer.write_event(record)
                    else:
                        writer.write_frame(record)
            return load_session_training_samples(path)

    def test_quarantined_session_refused_before_any_frames_are_loaded(self):
        records = self.prefix() + [self.frame(i, i + 7) for i in range(4, 9)]
        with (mock.patch("train_finger_localizer._read_jsonl") as read_records,
              self.assertRaisesRegex(ValueError, "ineligible for training")):
            self.load(records, {"training_eligible": False})
        read_records.assert_not_called()

    def test_crc_override_cannot_bypass_session_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            with FingerSessionWriter(directory, manifest_extra={"training_eligible": False}):
                pass
            with self.assertRaisesRegex(ValueError, "ineligible for training"):
                load_session_training_samples(directory, table_crc_override=123)

    def test_training_eligibility_requires_a_real_boolean_when_declared(self):
        for value in ("false", "true", "", 0, 1, None, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                self.load([], {"training_eligible": value})

    def test_allowed_session_still_runs_normal_quality_checks(self):
        records = self.prefix() + [self.frame(i, i + 7) for i in range(4, 9)]
        legacy, _, _ = self.load(records)
        allowed, _, _ = self.load(records, {"training_eligible": True})
        np.testing.assert_array_equal([s.features for s in legacy], [s.features for s in allowed])
        with self.assertRaisesRegex(ValueError, "frames and labelled contact events"):
            self.load([], {"training_eligible": True})

    def test_release_start_is_unknown_not_a_negative_label(self):
        self.assertIsNone(_event_contact({"kind": "release"}))
        self.assertIsNone(_event_contact({"kind": "contact", "metadata": {"contact": None}}))
        self.assertIs(_event_contact({"kind": "release_complete"}), False)
        with self.assertRaisesRegex(ValueError, "boolean"):
            _event_contact({"kind": "hold", "metadata": {"contact": "false"}})

    def test_legacy_coverage_phases_share_one_press_group(self):
        a = dict(event_id="contact-id", metadata={"point_index": 5, "repetition": 2})
        b = dict(event_id="hold-id", metadata={"point_index": 5, "repetition": 2})
        self.assertEqual(_event_group(a), _event_group(b))
        b["metadata"]["repetition"] = 3
        self.assertNotEqual(_event_group(a), _event_group(b))

    def test_area_truth_requires_independent_event_link_and_audit_ids(self):
        event = {
            "metadata": {
                "measured_contact_area_mm2": 12.5,
                "area_truth_provenance": {
                    "source": "pressure_film",
                    "measurement_id": "PF-0007",
                    "calibration_id": "PF-CAL-2026",
                    "linked_event_id": "press-7",
                    "independent": True,
                },
            }
        }
        label = _area_label(event, True, "press-7")
        self.assertEqual(label.value_mm2, 12.5)
        self.assertEqual(label.source, "pressure_film")
        self.assertTrue(label.traceable)
        self.assertTrue(label.independent)
        self.assertEqual(label.measurement_id, "PF-0007")
        self.assertEqual(label.calibration_id, "PF-CAL-2026")

        wrong_event = _area_label(event, True, "press-8")
        self.assertFalse(wrong_event.traceable)
        event["metadata"]["area_truth_provenance"]["independent"] = "true"
        with self.assertRaisesRegex(ValueError, "independent flag"):
            _area_label(event, True, "press-7")

    def test_future_release_frames_cannot_change_earlier_features(self):
        records = self.prefix() + [self.frame(i, i + 7, shift=.003) for i in range(4, 9)]
        before, _, report = self.load(records)
        records += [ContactEvent("released", monotonic_ns=20)]
        records += [self.frame(i, i + 12, shift=-.004) for i in range(9, 15)]
        after, _, after_report = self.load(records)
        self.assertEqual(report.baseline_frames, 3)
        self.assertEqual(after_report.baseline_frames, 3)
        np.testing.assert_array_equal(
            [s.features for s in before if s.contact],
            [s.features for s in after if s.contact],
        )
        self.assertEqual(report.skipped_baseline_calibration, 3)
        self.assertTrue(all(s.contact for s in before))

    def test_release_and_unknown_events_break_previous_contact_label(self):
        for kind in ("release", "motion_abort"):
            with self.subTest(kind=kind):
                records = self.prefix() + [self.frame(i, i + 7) for i in range(4, 7)]
                records += [ContactEvent(kind, monotonic_ns=20)]
                records += [self.frame(i, i + 14) for i in range(7, 12)]
                samples, _, report = self.load(records)
                self.assertEqual(len(samples), 1)
                self.assertTrue(samples[0].contact)
                self.assertEqual(report.skipped_unlabelled, 5)

    def test_post_contact_negative_frames_cannot_supply_missing_baseline(self):
        records = [ContactEvent("contact", monotonic_ns=1)]
        records += [self.frame(i, i + 1) for i in range(1, 4)]
        records += [ContactEvent("released", monotonic_ns=10)]
        records += [self.frame(i, i + 7) for i in range(4, 10)]
        with self.assertRaisesRegex(ValueError, "before first contact"):
            self.load(records)

    def test_bad_crc_cannot_enter_reference(self):
        records = self.prefix()
        records[2] = replace(records[2], packet_crc_ok=False)
        records += [self.frame(i, i + 7) for i in range(4, 9)]
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.load(records)

    def test_mixed_baseline_gain_cannot_be_medianed_as_raw_codes(self):
        records = self.prefix()
        records[2] = replace(records[2], transimpedance_ohm={1: 40000})
        records += [self.frame(i, i + 7) for i in range(4, 9)]
        with self.assertRaisesRegex(ValueError, "baseline.*transimpedance"):
            self.load(records)

    def test_repeated_baseline_device_frame_cannot_count_as_three(self):
        records = self.prefix()
        records[1:4] = [replace(frame, device_sequence=20) for frame in records[1:4]]
        records += [self.frame(i, i + 7) for i in range(4, 9)]
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.load(records)

    def test_cached_frames_and_gaps_reset_causal_history(self):
        for change in ("cached", "gap", "boot", "discontinuity"):
            with self.subTest(change=change):
                frames = [self.frame(i, i + 7, device_sequence=i, device_boot_id=1)
                          for i in range(4, 10)]
                if change == "cached":
                    frames[2] = replace(frames[2], acquisition_profile="SURVEY",
                                        fresh_point_mask=[False] * 45)
                elif change == "gap":
                    frames[2:] = [replace(f, device_sequence=f.device_sequence + 2)
                                  for f in frames[2:]]
                elif change == "boot":
                    frames[2:] = [replace(f, device_boot_id=2) for f in frames[2:]]
                else:
                    frames[2] = replace(frames[2], bandwidth_discontinuity=True)
                samples, _, report = self.load(self.prefix() + frames)
                self.assertEqual(len(samples), 1 if change in {"cached", "discontinuity"} else 2)
                self.assertGreater(report.causal_resets, 0)

    def test_duplicate_device_frame_is_not_a_new_sample(self):
        frames = [self.frame(i, i + 7, device_sequence=7) for i in range(4, 9)]
        samples, _, report = self.load(self.prefix() + frames)
        self.assertEqual(samples, [])
        self.assertGreater(report.causal_resets, 0)

    def test_ui_snapshot_time_does_not_relabel_an_old_received_frame(self):
        frames = [self.frame(i, i + 7, frame_received_monotonic_ns=4,
                             quality={"phase_event_id": "press", "coverage_phase": "contact"})
                  for i in range(4, 7)]
        frames += [self.frame(i, i + 7, frame_received_monotonic_ns=i + 7,
                              quality={"phase_event_id": "press", "coverage_phase": "contact"})
                   for i in range(7, 10)]
        samples, _, report = self.load(self.prefix() + frames)
        self.assertEqual(len(samples), 1)
        self.assertEqual(report.skipped_timing_mismatch, 3)

    def test_coverage_without_receive_timestamp_is_unlabelled(self):
        frames = [self.frame(i, i + 7, quality={"coverage_phase": "contact"})
                  for i in range(4, 9)]
        samples, _, report = self.load(self.prefix() + frames)
        self.assertEqual(samples, [])
        self.assertEqual(report.skipped_timing_mismatch, 5)

    def test_batched_receive_times_do_not_form_a_timed_causal_window(self):
        frames = [self.frame(i, i + 7, frame_received_monotonic_ns=11)
                  for i in range(4, 9)]
        samples, _, report = self.load(self.prefix() + frames)
        self.assertEqual(samples, [])
        self.assertGreater(report.causal_resets, 0)


if __name__ == "__main__":
    unittest.main()
