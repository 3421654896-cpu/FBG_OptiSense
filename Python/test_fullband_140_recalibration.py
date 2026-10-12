"""Offline gates for the dedicated 145 mA, 2001-point recalibration path."""

from __future__ import annotations

import argparse
import unittest
import json
import tempfile
from pathlib import Path

import export_fullband_calibration as export
import fullband_equal_power_model as model
from laser_dac_safety import FULLBAND_2001_CODE_LIMITS


class Fullband145RecalibrationTests(unittest.TestCase):
    def test_new_repair_round_links_prior_audit_without_copying_bulky_history(self):
        source_max = {
            "codes": [63351, 63351, 1000, 2000, 3000],
            "reading": {"power_mw": 2.4},
        }
        row = {
            "success": False,
            "codes": [63351, 60000, 1000, 2000, 3000],
            "reading": {"wavelength_nm": 1525.0, "power_mw": 2.0},
            "history": [{"large": list(range(50))}],
            "own_seed_repair_attempts": [{"strategy": "old"}],
            "current_continuation_attempted": True,
            "original_failure": {
                "original_failure": {"source_max_power_row": source_max}
            },
            "certification_power_prediction_unverified": True,
        }
        payload = {"rows": {"1525.000000": row}}
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "prior.json"
            source.write_text("{}", encoding="utf-8")
            self.assertEqual(model.compact_new_repair_input(payload, source), 1)
            self.assertEqual(payload["repair_parent_artifact"], str(source.resolve()))
        self.assertEqual(row["source_max_power_row"], source_max)
        self.assertNotIn("history", row)
        self.assertNotIn("original_failure", row)
        self.assertNotIn("own_seed_repair_attempts", row)
        self.assertNotIn("current_continuation_attempted", row)
        self.assertTrue(row["certification_power_prediction_unverified"])
        self.assertEqual(row["codes"][1], 60000)
        self.assertEqual(row["reading"]["power_mw"], 2.0)

    def test_resume_gate_never_reuses_a_reading_from_before_predicted_code_change(self):
        row = {
            "success": False,
            "codes": [63351, 62000, 1000, 2000, 3000],
            "reading": {
                "wavelength_nm": 1525.0005,
                "power_mw": 2.056,
                "side_mode_suppression_db": 30.0,
            },
            "certification_power_prediction_unverified": True,
        }
        rows = {"1525.000000": row}
        self.assertEqual(
            model.reclassify_resumed_release_rows(rows, 2.056, 0.005), 0
        )
        self.assertFalse(row["success"])

        row.pop("certification_power_prediction_unverified")
        self.assertEqual(
            model.reclassify_resumed_release_rows(rows, 2.056, 0.005), 1
        )
        self.assertTrue(row["success"])
        self.assertEqual(
            row["repair_attempts"][-1]["strategy"],
            "resume_existing_live_reading_release_gate_v1",
        )

    def test_exact_145_ma_integer_ceiling_never_rounds_over(self):
        self.assertEqual(model.current_to_code(145.0, 0), 63351)
        self.assertEqual(model.current_to_code(145.0, 1), 63351)
        self.assertEqual(model.current_to_code(999.0, 0), 63351)
        self.assertLess(model.code_to_current(63351, 0), 145.0)
        self.assertEqual(
            model.open_fullband_laser.__name__, "open_fullband_laser"
        )
        self.assertEqual(FULLBAND_2001_CODE_LIMITS[:2], (63351, 63351))

    def test_cli_current_above_145_is_rejected_before_hardware(self):
        model.validate_requested_currents(
            argparse.Namespace(gain_ma=145.0, soa_ma=145.0)
        )
        with self.assertRaisesRegex(ValueError, "0..145 mA"):
            model.validate_requested_currents(
                argparse.Namespace(gain_ma=145.001, soa_ma=145.0)
            )
        with self.assertRaisesRegex(ValueError, "lower"):
            model.validate_requested_currents(
                argparse.Namespace(soa_min_ma=145.0, soa_max_ma=145.0)
            )

    @staticmethod
    def complete_rows(power_mw=2.0):
        rows = {}
        for index, target in enumerate(
            model.target_grid(
                model.TARGET_START_NM,
                model.TARGET_STOP_NM,
                model.TARGET_STEP_NM,
            )
        ):
            rows[f"{target:.6f}"] = {
                "success": True,
                "target_nm": target,
                "codes": [63351, 63351, 0, 0, 0],
                "reading": {"power_mw": power_mw + index * 1e-6},
            }
        return rows

    def test_common_power_uses_weakest_point_and_rounds_down(self):
        rows = self.complete_rows(2.0)
        result = model.recommend_common_power_target(
            rows, headroom_relative=0.005, resolution_mw=0.001
        )
        self.assertEqual(result["tested"], 2001)
        self.assertEqual(result["common_ceiling_mw"], 2.0)
        self.assertEqual(result["weakest_target_nm"], 1525.0)
        self.assertEqual(result["recommended_target_power_mw"], 1.99)

    def test_common_power_can_lift_target_inside_release_tolerance(self):
        rows = self.complete_rows(1.0)
        result = model.recommend_common_power_target(
            rows,
            headroom_relative=0.0,
            lift_relative=0.015,
            resolution_mw=0.001,
        )
        self.assertEqual(result["common_ceiling_mw"], 1.0)
        self.assertAlmostEqual(result["recommended_target_power_mw"], 1.015)
        weakest_relative_error = (
            result["common_ceiling_mw"]
            / result["recommended_target_power_mw"]
            - 1.0
        )
        self.assertGreater(weakest_relative_error, -0.015)
        self.assertLessEqual(weakest_relative_error, 0.0)

    def test_current_continuation_retries_with_soa_first_v2(self):
        source = Path(model.__file__).read_text(encoding="utf-8")
        self.assertIn(
            'continuation_strategy = "own_branch_current_continuation_v2_soa_first"',
            source,
        )
        split_start = source.index("attenuation_splits = (")
        soa_first = source.index("(0.0, 1.0)", split_start)
        joint_fallback = source.index("(0.75, 0.25)", split_start)
        self.assertLess(soa_first, joint_fallback)

    def test_repair_parser_accepts_phase_candidate_table(self):
        args = model.build_parser().parse_args([
            "repair-equal-power", "--phase-candidates", "branches.json",
        ])
        self.assertEqual(args.phase_candidates, Path("branches.json"))

    def test_common_power_refuses_incomplete_or_failed_evidence(self):
        rows = self.complete_rows()
        rows.pop(next(iter(rows)))
        with self.assertRaisesRegex(ValueError, "exactly 2001"):
            model.recommend_common_power_target(rows)

        rows = self.complete_rows()
        rows["1545.000000"]["success"] = False
        with self.assertRaisesRegex(ValueError, "not validated"):
            model.recommend_common_power_target(rows)

    def test_common_power_refuses_a_code_over_145_ma(self):
        rows = self.complete_rows()
        rows["1565.000000"]["codes"][0] = 63352
        with self.assertRaisesRegex(ValueError, "exceeds 145 mA"):
            model.recommend_common_power_target(rows)

    def test_firmware_absolute_limit_is_145_and_only_t45_opts_in(self):
        root = Path(__file__).resolve().parents[1]
        firmware = (root / "JDSU/Core/Src/ms5614t.c").read_text(encoding="utf-8")
        routes = (root / "JDSU/Core/Inc/candidate_route_protocol.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("case IDAC5: maximumCode = 63351U", firmware)
        self.assertIn("case IDAC6: maximumCode = 63351U", firmware)
        self.assertIn("legacyLimits[5] = {58981U, 58981U", routes)
        self.assertIn("temporaryLimits[5] = {63351U, 63351U", routes)
        self.assertIn(
            "> temporaryLimits[channel]",
            routes,
        )
        self.assertEqual(routes.count("legacyLimits[channel]"), 2)

    def test_operational_export_refuses_old_135_ma_or_failed_rows(self):
        rows = self.complete_rows()
        for row in rows.values():
            row["currents_ma"] = [139.99, 139.99, 0.0, 0.0, 0.0]
            row["wavelength_error_pm"] = 0.0
            row["power_relative_error"] = 0.0
            row["reading"].update({
                "wavelength_nm": row["target_nm"],
                "side_mode_suppression_db": 30.0,
                "peak_count": 1,
            })
        payload = {
            "limits_ma": {"gain_ma": 135.0, "soa_ma": 135.0},
            "target_power_mw": 2.0,
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measured 145 mA"):
                export.load_source(path)

            payload["limits_ma"] = {"gain_ma": 145.0, "soa_ma": 145.0}
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(export.load_source(path)["rows"]), 2001)

            payload["rows"]["1545.000000"]["reading"]["power_mw"] = 2.011
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "±0.5%"):
                export.load_source(path)

            payload["rows"]["1545.000000"]["reading"]["power_mw"] = 2.0
            payload["rows"]["1545.000000"]["success"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has not passed"):
                export.load_source(path)

    def test_release_requires_forward_and_reverse_exact_code_passes(self):
        source_rows = list(self.complete_rows().values())
        validation_rows = {
            f"{row['target_nm']:.6f}": {
                "success": True,
                "target_nm": row["target_nm"],
                "codes": list(row["codes"]),
                "reading": {"power_mw": 2.0},
            }
            for row in source_rows
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "forward.json"
            payload = {
                "direction": "forward",
                "limits_ma": {"gain_ma": 145.0, "soa_ma": 145.0},
                "target_power_mw": 2.0,
                "rows": validation_rows,
                "summary": {"tested": 2001, "passed": 2001},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = export.validate_repeatability_pass(
                path, source_rows, 2.0, "forward"
            )
            self.assertEqual(result["passed"], 2001)

            payload["direction"] = "reverse"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the forward"):
                export.validate_repeatability_pass(
                    path, source_rows, 2.0, "forward"
                )

            payload["direction"] = "forward"
            payload["rows"]["1525.000000"]["codes"][0] -= 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different DAC codes"):
                export.validate_repeatability_pass(
                    path, source_rows, 2.0, "forward"
                )


if __name__ == "__main__":
    unittest.main()
