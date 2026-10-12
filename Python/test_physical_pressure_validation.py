import math
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from physical_pressure_validation import (
    AcceptanceLimits,
    ContactAreaTruthSet,
    PhysicalAcceptanceSession,
    PhysicalReferenceTrace,
    align_physical_reference,
    assess_aligned_frequency_response,
    assess_frequency_response,
    assess_spatial_accuracy,
    evaluate_end_to_end_acceptance,
    evaluate_physical_acceptance_session,
    load_contact_area_truth,
)


class PhysicalPressureValidationTests(unittest.TestCase):
    @staticmethod
    def _area_payload():
        return {
            "schema": "fbg-contact-area-truth/v1",
            "records": [
                {
                    "event_id": f"press-{index:03d}",
                    "area_mm2": (8.0, 16.0, 24.0)[index % 3],
                    "source": "pressure_film",
                    "measurement_id": f"film-{index:03d}",
                    "calibration_id": "FILM-CAL-2026-09",
                    "independent": True,
                }
                for index in range(12)
            ],
        }

    @staticmethod
    def _acceptance_inputs():
        times_s = np.arange(0.0, 1.001, 1.0 / 60.0)
        monotonic_ns = np.rint(times_s * 1_000_000_000).astype(np.int64)
        angle = 2.0 * math.pi * 15.0 * times_s
        reference = PhysicalReferenceTrace(
            monotonic_ns=monotonic_ns,
            values=np.sin(angle),
            quantity="force",
            unit="N",
            source_device="load-cell-01",
            calibration_id="LOAD-CAL-1",
            synchronization_method="shared-host-monotonic-callback",
            maximum_clock_error_ms=0.2,
            synchronized=True,
        )
        area_truth = ContactAreaTruthSet.from_mapping(
            PhysicalPressureValidationTests._area_payload()
        )
        positions = [
            (-6.0, -6.0),
            (-2.0, -6.0),
            (2.0, -6.0),
            (6.0, -6.0),
            (-6.0, -2.0),
            (-2.0, -2.0),
            (2.0, -2.0),
            (6.0, -2.0),
            (-6.0, 2.0),
            (-2.0, 2.0),
            (2.0, 2.0),
            (6.0, 2.0),
        ]
        session = PhysicalAcceptanceSession.from_mapping(
            {
                "schema": "fbg-physical-acceptance-session/v1",
                "frequency_hz": 15.0,
                "low_frequency_gain_output_per_input": 2.0,
                "minimum_cycles": 8.0,
                "maximum_reference_gap_ms": 20.0,
                "maximum_allowed_clock_error_ms": 2.0,
                "independent_held_out_test": True,
                "position_truth_source": "mach3_calibrated",
                "ch1_frames": [
                    {
                        "monotonic_ns": int(timestamp),
                        "response_value": float(
                            2.0 * math.sin(value + math.radians(-5.0))
                        ),
                    }
                    for timestamp, value in zip(monotonic_ns, angle)
                ],
                "spatial_records": [
                    {
                        "event_id": f"press-{index:03d}",
                        "true_x_mm": x,
                        "true_y_mm": y,
                        "predicted_x_mm": x + 0.1,
                        "predicted_y_mm": y - 0.1,
                        "predicted_area_mm2": (8.0, 16.0, 24.0)[index % 3]
                        + 0.2,
                    }
                    for index, (x, y) in enumerate(positions)
                ],
                "limits": {
                    "maximum_amplitude_error_db": 1.0,
                    "maximum_phase_error_deg": 10.0,
                    "minimum_tone_r_squared": 0.95,
                    "maximum_position_p95_error_mm": 1.0,
                    "maximum_area_p95_error_mm2": 1.0,
                },
            }
        )
        return reference, area_truth, session

    def _frequency(self, *, gain=2.0, lag_deg=-12.0, calibrated=True):
        times = np.arange(0.0, 1.001, 1.0 / 60.0)
        angle = 2.0 * math.pi * 15.0 * times
        reference = np.sin(angle)
        response = gain * np.sin(angle + math.radians(lag_deg))
        monotonic_ns = np.rint(times * 1_000_000_000).astype(np.int64)
        trace = PhysicalReferenceTrace(
            monotonic_ns=monotonic_ns,
            values=reference,
            quantity="pressure",
            unit="kPa",
            source_device="calibrated-load-cell-01",
            calibration_id="CAL-2026-0906",
            synchronization_method="shared-host-monotonic-callback",
            maximum_clock_error_ms=0.2,
            synchronized=True,
        )
        aligned = align_physical_reference(
            monotonic_ns,
            response,
            trace,
        )
        return assess_aligned_frequency_response(
            aligned,
            frequency_hz=15.0,
            low_frequency_gain_output_per_input=(gain if calibrated else None),
        )

    def _spatial(self, *, area_source="pressure_film", held_out=True):
        truth_xy = np.asarray(
            [(x, y) for x in (-5.0, 0.0, 5.0) for y in (-5.0, 0.0, 5.0)]
            * 2
        )
        predicted_xy = truth_xy + np.asarray([0.2, -0.1])
        truth_area = np.tile([8.0, 16.0, 24.0], 6)
        predicted_area = truth_area + 0.5
        return assess_spatial_accuracy(
            truth_xy,
            predicted_xy,
            truth_area,
            predicted_area,
            independent_held_out_test=held_out,
            position_truth_source="mach3_calibrated",
            area_truth_source=area_source,
        )

    def test_known_tone_reports_gain_phase_and_real_nyquist_gate(self):
        evidence = self._frequency()
        self.assertTrue(evidence.nyquist_timing_pass)
        self.assertTrue(evidence.reference_provenance_valid)
        self.assertEqual(evidence.reference_calibration_id, "CAL-2026-0906")
        self.assertAlmostEqual(evidence.gain_relative_to_low_frequency, 1.0, places=8)
        self.assertAlmostEqual(evidence.amplitude_error_db, 0.0, places=8)
        self.assertAlmostEqual(evidence.phase_difference_deg, -12.0, places=8)
        self.assertFalse(evidence.physical_15hz_verified)

    def test_boolean_sync_claim_without_traceable_reference_fails_closed(self):
        times = np.arange(0.0, 1.001, 1.0 / 60.0)
        signal = np.sin(2.0 * math.pi * 15.0 * times)
        frequency = assess_frequency_response(
            times,
            signal,
            signal,
            frequency_hz=15.0,
            low_frequency_gain_output_per_input=1.0,
            synchronized_physical_reference=True,
        )
        report = evaluate_end_to_end_acceptance(
            frequency,
            self._spatial(),
            AcceptanceLimits(1.0, 15.0, 0.95, 1.0, 2.0),
        )
        self.assertFalse(frequency.reference_provenance_valid)
        self.assertFalse(report["physical_15hz_verified"])

    def test_alignment_rejects_unsynchronised_and_excess_clock_error(self):
        frame_times = np.arange(12, dtype=np.int64) * 10_000_000
        response = np.arange(12, dtype=float)
        common = dict(
            monotonic_ns=frame_times,
            values=response,
            quantity="force",
            unit="N",
            source_device="load-cell",
            calibration_id="CAL-1",
            synchronization_method="shared-host-clock",
            maximum_clock_error_ms=0.5,
        )
        with self.assertRaisesRegex(ValueError, "not declared synchronized"):
            align_physical_reference(
                frame_times, response, PhysicalReferenceTrace(**common)
            )
        with self.assertRaisesRegex(ValueError, "clock error exceeds"):
            align_physical_reference(
                frame_times,
                response,
                PhysicalReferenceTrace(
                    **{**common, "maximum_clock_error_ms": 3.0}, synchronized=True
                ),
            )

    def test_alignment_interpolates_and_reports_discarded_frames(self):
        reference_times = (
            np.arange(20, dtype=np.int64) * 10_000_000 + 100_000_000
        )
        trace = PhysicalReferenceTrace(
            monotonic_ns=reference_times,
            values=np.arange(20, dtype=float),
            quantity="force",
            unit="N",
            source_device="load-cell",
            calibration_id="CAL-2",
            synchronization_method="shared-host-clock",
            maximum_clock_error_ms=0.5,
            synchronized=True,
        )
        frame_times = (
            np.arange(21, dtype=np.int64) * 10_000_000 + 95_000_000
        )
        aligned = align_physical_reference(
            frame_times,
            np.arange(frame_times.size, dtype=float),
            trace,
        )
        self.assertEqual(aligned.discarded_outside_reference, 2)
        self.assertEqual(aligned.discarded_reference_gap, 0)
        self.assertEqual(len(aligned.reference_values), 19)
        self.assertAlmostEqual(aligned.reference_values[0], 0.5)
        self.assertAlmostEqual(aligned.reference_values[-1], 18.5)

    def test_missing_slow_sensitivity_cannot_pass_amplitude(self):
        frequency = self._frequency(calibrated=False)
        report = evaluate_end_to_end_acceptance(
            frequency,
            self._spatial(),
            AcceptanceLimits(1.0, 15.0, 0.95, 1.0, 2.0),
        )
        self.assertFalse(report["checks"]["amplitude_calibration_available"])
        self.assertFalse(report["physical_15hz_verified"])

    def test_nominal_tool_area_is_not_accepted_as_physical_truth(self):
        spatial = self._spatial(area_source="nominal_indenter")
        self.assertFalse(spatial.area_truth_valid)
        report = evaluate_end_to_end_acceptance(
            self._frequency(),
            spatial,
            AcceptanceLimits(1.0, 15.0, 0.95, 1.0, 2.0),
        )
        self.assertFalse(report["physical_contact_area_verified"])

    def test_all_three_goal_claims_pass_only_on_complete_evidence(self):
        report = evaluate_end_to_end_acceptance(
            self._frequency(),
            self._spatial(),
            AcceptanceLimits(1.0, 15.0, 0.95, 1.0, 2.0),
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["physical_15hz_verified"])
        self.assertTrue(report["position_accuracy_verified"])
        self.assertTrue(report["physical_contact_area_verified"])
        self.assertTrue(report["end_to_end_goal_verified"])

    def test_non_independent_spatial_samples_fail_closed(self):
        report = evaluate_end_to_end_acceptance(
            self._frequency(),
            self._spatial(held_out=False),
            AcceptanceLimits(1.0, 15.0, 0.95, 1.0, 2.0),
        )
        self.assertFalse(report["checks"]["independent_spatial_test"])
        self.assertFalse(report["position_accuracy_verified"])

    def test_contact_area_truth_loads_only_traceable_independent_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "area.json"
            path.write_text(json.dumps(self._area_payload()), encoding="utf-8")
            truth = load_contact_area_truth(path)
        self.assertIsInstance(truth, ContactAreaTruthSet)
        self.assertEqual(len(truth.records), 12)
        self.assertEqual(truth.distinct_area_count, 3)
        self.assertEqual(truth.calibration_ids, ("FILM-CAL-2026-09",))

    def test_contact_area_truth_rejects_nominal_or_non_boolean_claims(self):
        nominal = self._area_payload()
        nominal["records"][0]["source"] = "nominal_indenter"
        with self.assertRaisesRegex(ValueError, "untrusted"):
            ContactAreaTruthSet.from_mapping(nominal)
        string_boolean = self._area_payload()
        string_boolean["records"][0]["independent"] = "true"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            ContactAreaTruthSet.from_mapping(string_boolean)

    def test_contact_area_truth_rejects_duplicate_events_and_one_area_level(self):
        duplicate = self._area_payload()
        duplicate["records"][1]["event_id"] = duplicate["records"][0]["event_id"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            ContactAreaTruthSet.from_mapping(duplicate)
        one_level = self._area_payload()
        for record in one_level["records"]:
            record["area_mm2"] = 12.0
        with self.assertRaisesRegex(ValueError, "3 distinct"):
            ContactAreaTruthSet.from_mapping(one_level)

    def test_complete_acceptance_session_pairs_frequency_position_and_area(self):
        reference, area_truth, session = self._acceptance_inputs()
        report = evaluate_physical_acceptance_session(
            session, reference, area_truth
        )
        self.assertTrue(report["end_to_end_goal_verified"])
        self.assertTrue(report["physical_15hz_verified"])
        self.assertTrue(report["position_accuracy_verified"])
        self.assertTrue(report["physical_contact_area_verified"])
        self.assertEqual(report["paired_spatial_event_count"], 12)
        self.assertEqual(report["area_calibration_ids"], ["FILM-CAL-2026-09"])

    def test_acceptance_session_rejects_stale_area_event_link(self):
        reference, area_truth, session = self._acceptance_inputs()
        records = list(session.spatial_records)
        records[0] = type(records[0])(
            **{**records[0].__dict__, "event_id": "stale-event"}
        )
        broken = type(session)(**{**session.__dict__, "spatial_records": tuple(records)})
        with self.assertRaisesRegex(ValueError, "must match exactly"):
            evaluate_physical_acceptance_session(broken, reference, area_truth)


if __name__ == "__main__":
    unittest.main()
