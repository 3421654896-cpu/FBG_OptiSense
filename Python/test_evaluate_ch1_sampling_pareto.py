import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate_ch1_sampling_pareto as pareto


class SamplingParetoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teacher = pareto.load_teacher_capture()
        cls.timing = pareto.load_timing_reference()
        cls.plans = pareto.build_candidate_plans(cls.teacher)

    def test_uniform_plans_cover_all_nine_peaks_with_exact_budgets(self):
        self.assertTrue(
            self.teacher.source[
                "dac_rows_verified_against_normalized_2001_audit"
            ]
        )
        expected = {
            "uniform_4x9_36": (36, (4,) * 9),
            "uniform_5x9_45": (45, (5,) * 9),
            "uniform_6x9_54": (54, (6,) * 9),
        }
        for name, (total, counts) in expected.items():
            with self.subTest(name=name):
                plan = self.plans[name]
                self.assertEqual(plan.total_points, total)
                self.assertEqual(plan.points_per_peak, counts)
                self.assertTrue(np.all(np.diff(plan.selected_indices) > 0))
                for peak in plan.peaks:
                    self.assertIn(peak.peak_index, peak.selected_indices)
                    self.assertTrue(np.any(peak.selected_indices < peak.peak_index))
                    self.assertTrue(np.any(peak.selected_indices > peak.peak_index))

    def test_teacher_dac_row_must_match_normalized_2001_audit(self):
        payload = json.loads(pareto.DEFAULT_TEACHER.read_text(encoding="utf-8"))
        first_key = next(iter(payload["rows"]))
        payload["rows"][first_key]["dac_codes"][2] += 1
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "teacher.json"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact normalized audit row"):
                pareto.load_teacher_capture(tampered)

    def test_adaptive_plan_keeps_identifiable_floor_and_45_total(self):
        plan = self.plans["adaptive_45"]
        self.assertEqual(plan.total_points, 45)
        self.assertEqual(len(plan.points_per_peak), 9)
        self.assertGreaterEqual(min(plan.points_per_peak), 4)
        self.assertEqual(sum(plan.points_per_peak), 45)

    def test_timing_projection_distinguishes_map_rate_and_pressure_nyquist(self):
        fast = pareto.project_timing(36, self.timing)
        five = pareto.project_timing(45, self.timing)
        six = pareto.project_timing(54, self.timing)
        self.assertGreater(fast["predicted_mean_cycle_hz"], 15.0)
        self.assertLess(five["predicted_mean_cycle_hz"], 15.0)
        self.assertLess(six["predicted_mean_cycle_hz"], five["predicted_mean_cycle_hz"])
        self.assertLess(fast["predicted_mean_cycle_hz"], 30.0)

    def test_hybrid_timing_keeps_map_and_pressure_requirements_separate(self):
        dual = pareto.project_global_plus_roi_schedule(6, self.timing)
        self.assertTrue(dual["mean_complete_map_15_hz_gate"])
        self.assertTrue(dual["mean_pressure_nyquist_gate"])
        self.assertFalse(dual["joint_p99_projection_gate"])

        roi = pareto.project_roi_only_schedule(6, self.timing)
        self.assertTrue(roi["mean_pressure_nyquist_gate"])
        self.assertFalse(roi["complete_map_15_hz_gate"])

        periodic = pareto.project_periodic_global_roi_schedule(
            54, 5.0, 6, self.timing
        )
        self.assertFalse(periodic["complete_map_15_hz_gate"])
        self.assertTrue(periodic["pressure_nyquist_gate"])

    def test_path_audit_includes_repeated_frame_wrap(self):
        plan = self.plans["uniform_4x9_36"]
        metrics = pareto.plan_path_metrics(plan, self.teacher)
        span = int(
            self.teacher.source_indices[plan.selected_indices[-1]]
            - self.teacher.source_indices[plan.selected_indices[0]]
        )
        self.assertGreaterEqual(
            metrics["maximum_fullband_index_jump_including_frame_wrap"], span
        )
        self.assertTrue(
            metrics["all_sparse_transitions_require_hardware_replay_certification"]
        )

    def test_small_deterministic_evaluation_is_finite_and_does_not_apply_tables(self):
        report = pareto.evaluate_pareto(
            self.teacher, self.timing, trials=8, seed=12345
        )
        self.assertEqual(report["schema"], "ch1_sampling_pareto_evaluation_v1")
        self.assertTrue(report["offline_only"])
        self.assertFalse(report["modifies_mode_table"])
        self.assertEqual(len(report["candidates"]), 4)
        self.assertIn("uniform_4x9_36", [item["id"] for item in report["candidates"]])
        self.assertIsNone(report["recommendation"]["production_default_fixed_table"])
        self.assertIn(
            "global36_plus_roi6",
            report["hybrid_schedules"][
                "simultaneous_15hz_map_and_15hz_pressure_candidates"
            ],
        )
        serialized = __import__("json").dumps(report, allow_nan=False)
        self.assertGreater(len(serialized), 1000)


if __name__ == "__main__":
    unittest.main()
