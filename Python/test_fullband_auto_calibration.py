"""Offline safety and resume gates for the operator auto-calibration page."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import fullband_auto_calibration as auto
import fullband_equal_power_model as model
from fullband_auto_calibration_widget import FullbandAutoCalibrationWorker


class FullbandAutoCalibrationTests(unittest.TestCase):
    @staticmethod
    def _rows(success: bool = True) -> dict[str, dict]:
        return {
            f"{1525.0 + index * 0.02:.6f}": {
                "success": success,
                "target_nm": 1525.0 + index * 0.02,
                "codes": [63351, 63351, 0, 0, 0],
                "reading": {
                    "wavelength_nm": 1525.0 + index * 0.02,
                    "power_mw": 2.0,
                    "peak_count": 1,
                    "side_mode_suppression_db": None,
                },
            }
            for index in range(auto.TARGET_COUNT)
        }

    def test_state_rejects_an_old_current_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text(
                json.dumps({"profile": "fullband_2001_140mA_v1"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "其他校准配置"):
                auto.load_state(path)

    def test_state_save_retries_a_transient_windows_reader_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            real_replace = auto.os.replace
            calls = 0

            def flaky_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError("temporary progress-reader race")
                return real_replace(source, destination)

            with mock.patch.object(auto.os, "replace", side_effect=flaky_replace), \
                    mock.patch.object(auto.time, "sleep") as sleep:
                auto.save_state(path, {"profile": auto.PROFILE})
            self.assertEqual(calls, 2)
            sleep.assert_called_once_with(0.05)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["profile"], auto.PROFILE
            )

    def test_complete_rows_requires_145_ma_and_all_points(self):
        payload = {
            "limits_ma": {"gain_ma": 145.0, "soa_ma": 145.0},
            "rows": self._rows(),
            "certification_guard_band_expansion": {
                "validation_source": "validation_forward_round1.json",
                "targets": ["1545.000000"],
            },
            "repair_guard_band_targets": {
                "wavelength_tolerance_pm": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(auto.require_complete_rows(path, label="test")["rows"]), 2001)

            payload["limits_ma"]["gain_ma"] = 140.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "145 mA"):
                auto.require_complete_rows(path, label="test")

    def test_repair_misses_can_only_be_retested_with_a_complete_safe_grid(self):
        rows = self._rows()
        for row in rows.values():
            row["currents_ma"] = [145.0, 145.0, 0.0, 0.0, 0.0]
        rows["1537.880000"]["success"] = False
        rows["1553.420000"]["success"] = False
        payload = {
            "limits_ma": {"gain_ma": 145.0, "soa_ma": 145.0},
            "target_power_mw": 1.94,
            "certification_rework_round": 1,
            "neighbor_repair_summary": {"tested": 2001, "failed": 2},
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "repair.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                len(auto.require_retestable_repair_rows(
                    path, label="repair", certification_round=1,
                    target_power_mw=1.94,
                )["rows"]),
                2001,
            )
            with self.assertRaisesRegex(RuntimeError, "失败点"):
                auto.require_complete_rows(path, label="release")

            rows["1537.880000"]["codes"][1] = 63352
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "DAC候选值无效"):
                auto.require_retestable_repair_rows(
                    path, label="repair", certification_round=1,
                    target_power_mw=1.94,
                )

            rows["1537.880000"]["codes"][1] = 63351
            payload["neighbor_repair_summary"]["tested"] = 2000
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "完整2001点"):
                auto.require_retestable_repair_rows(
                    path, label="repair", certification_round=1,
                    target_power_mw=1.94,
                )

    def test_first_formal_checkpoint_skips_repeating_preconditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertIsNone(auto.completed_precondition_source(
                output, target_power_mw=1.94, power_relative_tolerance=0.05,
            ))
            source = output / "precondition_repaired_round4_avg3s1.json"
            source.write_text(json.dumps({
                "limits_ma": {"gain_ma": 145.0, "soa_ma": 145.0},
                "rows": self._rows(),
            }), encoding="utf-8")
            validation = {
                "source_table": source.name,
                "direction": "forward",
                "target_power_mw": 1.94,
                "power_relative_tolerance": 0.05,
            }
            first = output / "validation_forward_round1_avg3_postcondition.json"
            first.write_text(json.dumps(validation), encoding="utf-8")
            self.assertEqual(auto.completed_precondition_source(
                output, target_power_mw=1.94, power_relative_tolerance=0.05,
            ), source)
            validation["source_table"] = "unrelated.json"
            first.write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "不受信任"):
                auto.completed_precondition_source(
                    output, target_power_mw=1.94, power_relative_tolerance=0.05,
                )

    def test_warm_start_requires_a_complete_live_validated_grid(self):
        payload = {
            "limits_ma": {"gain_ma": 135.0, "soa_ma": 135.0},
            "rows": self._rows(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old-live-table.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            summary = auto.validate_warm_start_table(path)
            self.assertEqual(summary["rows"], 2001)
            self.assertEqual(summary["source_gain_ma"], 135.0)

            payload["rows"]["1545.000000"]["reading"][
                "side_mode_suppression_db"
            ] = 19.9
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SMSR"):
                auto.validate_warm_start_table(path)

    def test_from_zero_power_tuning_keeps_half_of_five_percent_as_margin(self):
        self.assertEqual(auto.scratch_power_tuning_tolerance(0.05), 0.025)
        self.assertEqual(auto.scratch_power_tuning_tolerance(0.03), 0.015)
        self.assertEqual(auto.scratch_power_tuning_tolerance(0.005), 0.005)

    def test_feedback_landscape_uses_only_explicit_live_maps(self):
        def measured_row(wavelength, power, pdt, pdr, single=True):
            return {
                "single_mode": single,
                "reading": {
                    "wavelength_nm": wavelength,
                    "power_mw": power,
                    "pdt_code": pdt,
                    "pdr_code": pdr,
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "machine2-live-map.json"
            unrelated = root / "machine1-old-map.json"
            report = root / "feedback.json"
            live.write_text(json.dumps({"rows": {
                "one": measured_row(1525.0, 1.0, 1800, 1700),
                "two": measured_row(1525.1, 1.1, 1750, 1650),
                "hop": measured_row(1526.0, 1.2, 1200, 900, False),
            }}), encoding="utf-8")
            unrelated.write_text(json.dumps({"rows": {
                "old": measured_row(1560.0, 9.9, 1, 1),
            }}), encoding="utf-8")

            result = auto.analyze_feedback_landscape([live], report)

            self.assertEqual(result["summary"]["observations"], 3)
            self.assertEqual(result["summary"]["mode_boundary_transitions"], 1)
            self.assertEqual(result["source_mode_maps"], [str(live)])
            self.assertNotIn(str(unrelated), report.read_text(encoding="utf-8"))
            self.assertEqual(
                result["pdt_pdr_role"],
                "branch continuity evidence only; not a hard acceptance gate",
            )

    def test_rework_seed_invalidates_only_certification_failures(self):
        payload = {
            "limits_ma": {"gain_ma": 145.0, "soa_ma": 145.0},
            "rows": self._rows(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.json", root / "rework.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            auto.write_rework_seed(source, output, {"1545.000000"}, 2)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(result["rows"]["1545.000000"]["success"])
            self.assertTrue(result["rows"]["1544.980000"]["success"])
            self.assertEqual(result["certification_rework_round"], 2)
            self.assertNotIn("certification_guard_band_expansion", result)
            self.assertNotIn("repair_guard_band_targets", result)

    def test_certification_rework_keeps_near_boundary_passes(self):
        rows = self._rows()
        rows["1525.000000"].update({
            "success": False,
            "wavelength_error_pm": 0.0,
            "power_relative_error": 0.0,
        })
        rows["1525.020000"].update({
            "wavelength_error_pm": 1.6,
            "power_relative_error": 0.0,
        })
        rows["1525.040000"].update({
            "wavelength_error_pm": 0.5,
            "power_relative_error": 0.021,
        })
        rows["1525.060000"].update({
            "wavelength_error_pm": 0.5,
            "power_relative_error": 0.01,
        })
        payload = {"rows": rows}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            targets = auto.validation_rework_targets(path, 0.03)

        self.assertIn("1525.000000", targets)
        self.assertNotIn("1525.020000", targets)
        self.assertNotIn("1525.040000", targets)
        self.assertNotIn("1525.060000", targets)

    def test_forward_rework_paths_preserve_completed_legacy_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old_seed = output / "certification_rework_seed_forward_round2_avg3_postcondition.json"
            old_result = output / "certification_repaired_forward_round2_avg3_postcondition.json"
            old_seed.touch()
            old_result.touch()
            self.assertEqual(auto.forward_rework_paths(output, 2), (old_seed, old_result))
            new_seed, new_result = auto.forward_rework_paths(output, 3)
            self.assertIn("failed_only_v1", new_seed.name)
            self.assertIn("failed_only_v1", new_result.name)
            self.assertNotEqual(new_seed, old_seed)

    def test_formal_pass_prevents_repairing_stale_inner_failures(self):
        rows = self._rows()
        rows["1525.000000"]["success"] = False
        rows["1525.020000"].update({
            "success": False,
            "certification_guard_band_rework_required": True,
            "certification_power_prediction_unverified": True,
        })
        validation_rows = self._rows()
        validation_rows["1525.000000"]["success"] = False
        payload = {"rows": rows}
        count = auto.retain_formally_passing_rows(
            payload, {"1525.000000"}, [{"rows": validation_rows}],
        )
        self.assertEqual(count, 1)
        self.assertTrue(rows["1525.020000"]["success"])
        self.assertNotIn(
            "certification_guard_band_rework_required", rows["1525.020000"],
        )
        self.assertFalse(rows["1525.000000"]["success"])

    def test_repair_subprocess_expands_guard_band_once(self):
        calibration_rows = self._rows()
        validation_rows = self._rows()
        for row in validation_rows.values():
            row["wavelength_error_pm"] = 0.5
            row["power_relative_error"] = 0.01
        validation_rows["1525.000000"]["success"] = False
        validation_rows["1525.000000"]["wavelength_error_pm"] = 5600.0
        validation_rows["1525.000000"]["reading"]["wavelength_nm"] = 1530.6
        validation_rows["1525.020000"]["wavelength_error_pm"] = 1.6
        validation_rows["1525.040000"]["power_relative_error"] = 0.021
        payload = {"rows": calibration_rows}

        with tempfile.TemporaryDirectory() as temporary:
            validation = Path(temporary) / "validation.json"
            validation.write_text(
                json.dumps({"rows": validation_rows}), encoding="utf-8",
            )
            first = model.expand_certification_guard_band_rows(
                payload, validation, 0.03,
            )
            second = model.expand_certification_guard_band_rows(
                payload, validation, 0.03,
            )

        self.assertEqual(first, {
            "release_failed": 1,
            "guard_band_added": 0,
            "total_rework_targets": 1,
            "sequence_deferred_predictions": 0,
        })
        self.assertEqual(second, first)
        self.assertFalse(payload["rows"]["1525.000000"]["success"])
        self.assertTrue(
            payload["rows"]["1525.000000"]["certification_guard_band_rework_required"]
        )
        self.assertTrue(payload["rows"]["1525.020000"]["success"])
        self.assertTrue(payload["rows"]["1525.040000"]["success"])
        self.assertTrue(payload["rows"]["1525.060000"]["success"])
        self.assertNotIn(
            "certification_sequence_wavelength_offset_pm",
            payload["rows"]["1525.020000"],
        )
        self.assertTrue(
            payload["rows"]["1525.000000"][
                "certification_forward_transition_confirmation_required"
            ]
        )

    def test_repair_preserves_forward_sweep_power_prediction(self):
        calibration_rows = self._rows()
        validation_rows = self._rows()
        for row in validation_rows.values():
            row.update({
                "wavelength_error_pm": 0.2,
                "power_relative_error": 0.0,
                "sample_count": 3,
                "mode_pass_count": 3,
            })

        power_only = "1525.000000"
        wavelength_failure = "1525.020000"
        mode_failure = "1525.040000"
        for key in (power_only, wavelength_failure, mode_failure):
            calibration_rows[key]["success"] = False
            calibration_rows[key]["certification_power_prediction_unverified"] = True
            validation_rows[key]["success"] = False
            validation_rows[key]["power_relative_error"] = 0.04
        validation_rows[wavelength_failure]["wavelength_error_pm"] = 2.2
        validation_rows[mode_failure]["mode_pass_count"] = 1
        validation_rows[mode_failure]["reading"]["peak_count"] = 2
        validation_rows[mode_failure]["reading"]["side_mode_suppression_db"] = 10.0

        payload = {"rows": calibration_rows}
        with tempfile.TemporaryDirectory() as temporary:
            validation = Path(temporary) / "validation.json"
            validation.write_text(
                json.dumps({"rows": validation_rows}), encoding="utf-8",
            )
            result = model.expand_certification_guard_band_rows(
                payload, validation, 0.03,
            )

        self.assertEqual(result, {
            "release_failed": 3,
            "guard_band_added": 0,
            "total_rework_targets": 2,
            "sequence_deferred_predictions": 1,
        })
        self.assertTrue(payload["rows"][power_only]["success"])
        self.assertTrue(
            payload["rows"][power_only][
                "certification_sequence_context_validation_pending"
            ]
        )
        self.assertFalse(payload["rows"][wavelength_failure]["success"])
        self.assertFalse(payload["rows"][mode_failure]["success"])
        self.assertEqual(
            payload["certification_guard_band_expansion"][
                "sequence_deferred_targets"
            ],
            [power_only],
        )

    def test_promotion_backs_up_and_atomically_replaces_all_runtime_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            runtime = root / "runtime"
            output.mkdir()
            runtime.mkdir()
            prefix = output / "operational_145_2001"
            feedback = output / "pdt_pdr_reference_145_2001.json"
            sources = [
                prefix.with_suffix(".json"),
                prefix.with_suffix(".csv"),
                prefix.with_suffix(".h"),
                prefix.with_name(prefix.name + "_transition_guards").with_suffix(".json"),
                feedback,
            ]
            targets = [
                runtime / "fullband_equal_power_operational_2001.json",
                runtime / "fullband_equal_power_operational_2001.csv",
                runtime / "fullband_equal_power_operational_2001.h",
                runtime / "fullband_equal_power_operational_2001_transition_guards.json",
                runtime / "fullband_pdt_pdr_reference_2001.json",
            ]
            for index, (source, target) in enumerate(zip(sources, targets)):
                source.write_text(f"new-{index}", encoding="utf-8")
                target.write_text(f"old-{index}", encoding="utf-8")

            result = auto.promote_runtime_table(
                output, prefix, feedback, runtime_dir=runtime
            )
            backup = Path(result["backup_dir"])
            for index, target in enumerate(targets):
                self.assertEqual(target.read_text(encoding="utf-8"), f"new-{index}")
                self.assertEqual(
                    (backup / target.name).read_text(encoding="utf-8"), f"old-{index}"
                )

    def test_runtime_directory_can_be_created_for_a_new_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            runtime = root / "device_profiles" / "machine_2" / "calibration"
            output.mkdir()
            prefix = output / "operational_145_2001"
            feedback = output / "pdt_pdr_reference_145_2001.json"
            sources = [
                prefix.with_suffix(".json"),
                prefix.with_suffix(".csv"),
                prefix.with_suffix(".h"),
                prefix.with_name(prefix.name + "_transition_guards").with_suffix(".json"),
                feedback,
            ]
            for index, source in enumerate(sources):
                source.write_text(f"machine-2-{index}", encoding="utf-8")

            result = auto.promote_runtime_table(
                output, prefix, feedback, runtime_dir=runtime
            )

            self.assertTrue(runtime.is_dir())
            self.assertEqual(len(result["promoted"]), 5)
            self.assertEqual(
                (runtime / "fullband_equal_power_operational_2001.csv").read_text(
                    encoding="utf-8"
                ),
                "machine-2-1",
            )

    def test_validation_resume_rejects_wrong_direction_before_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "table.json"
            output = root / "validation.json"
            table.write_text(
                json.dumps({"target_power_mw": 2.0, "rows": {}}),
                encoding="utf-8",
            )
            output.write_text(
                json.dumps({
                    "source_table": table.name,
                    "direction": "reverse",
                    "target_power_mw": 2.0,
                    "limits_ma": dict(zip(model.CONTROL_NAMES, model.LASER_MAX_MA)),
                    "rows": {},
                }),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input=table,
                output=output,
                target_power_mw=None,
                direction="forward",
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "validation direction"):
                model.run_validate_equal_power(args)

    def test_validation_resume_rejects_stale_partial_sweep_before_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "table.json"
            output = root / "validation.json"
            table.write_text(json.dumps({
                "target_power_mw": 2.0,
                "rows": {
                    "1525.000000": {"target_nm": 1525.0},
                    "1525.020000": {"target_nm": 1525.02},
                },
            }), encoding="utf-8")
            output.write_text(json.dumps({
                "source_table": table.name, "direction": "forward",
                "target_power_mw": 2.0,
                "limits_ma": dict(zip(model.CONTROL_NAMES, model.LASER_MAX_MA)),
                "rows": {"1525.000000": {"samples": [{}, {}, {}]}},
            }), encoding="utf-8")
            old = time.time() - model.MAX_PARTIAL_VALIDATION_RESUME_AGE_S - 60
            os.utime(table, (old - 60, old - 60))
            os.utime(output, (old, old))
            args = argparse.Namespace(
                input=table, output=output, target_power_mw=None,
                direction="forward", resume=True, samples_per_point=3,
            )
            with self.assertRaisesRegex(ValueError, "older session"):
                model.run_validate_equal_power(args)

    def test_validation_resume_rejects_changed_source_digest_before_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "table.json"
            output = root / "validation.json"
            table.write_text(
                json.dumps({"target_power_mw": 2.0, "rows": {}}),
                encoding="utf-8",
            )
            output.write_text(json.dumps({
                "source_table": table.name, "direction": "forward",
                "source_table_sha256": "0" * 64,
                "target_power_mw": 2.0,
                "limits_ma": dict(zip(model.CONTROL_NAMES, model.LASER_MAX_MA)),
                "rows": {},
            }), encoding="utf-8")
            args = argparse.Namespace(
                input=table, output=output, target_power_mw=None,
                direction="forward", resume=True,
            )
            with self.assertRaisesRegex(ValueError, "source table changed"):
                model.run_validate_equal_power(args)

    def test_validation_resume_reclassifies_retained_samples_at_three_percent(self):
        class DummyLaser:
            def close(self):
                return None

        class DummyMeter:
            identity = "offline test meter"
            measurement_profile = {"average_count": 3}

            def __init__(self, *_args, **_kwargs):
                pass

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = root / "table.json"
            output = root / "validation.json"
            target_nm = 1535.0
            reading = {
                "wavelength_nm": target_nm,
                "wavelength_error_pm": 0.0,
                "power_mw": 2.04,
                "power_dbm": 10.0 * math.log10(2.04),
                "peak_count": 1,
                "secondary_power_mw": None,
                "side_mode_suppression_db": None,
                "pdt_code": None,
                "pdr_code": None,
                "pdt_ma": None,
                "pdr_ma": None,
                "pdr_pdt_ratio": None,
            }
            codes = [63351, 63351, 1000, 2000, 3000]
            table.write_text(json.dumps({
                "target_power_mw": 2.0,
                "rows": {f"{target_nm:.6f}": {
                    "target_nm": target_nm,
                    "codes": codes,
                    "currents_ma": [145.0, 145.0, 0.3, 2.4, 3.7],
                    "reading": reading,
                }},
            }), encoding="utf-8")
            output.write_text(json.dumps({
                "source_table": table.name,
                "direction": "forward",
                "target_power_mw": 2.0,
                "limits_ma": dict(zip(model.CONTROL_NAMES, model.LASER_MAX_MA)),
                "power_relative_tolerance": 0.005,
                "rows": {f"{target_nm:.6f}": {
                    "success": False,
                    "target_nm": target_nm,
                    "codes": codes,
                    "currents_ma": [145.0, 145.0, 0.3, 2.4, 3.7],
                    "reading": reading,
                    "samples": [reading, reading, reading],
                    "sample_count": 3,
                    "mode_pass_count": 3,
                    "calibration_reading": reading,
                }},
            }), encoding="utf-8")
            args = argparse.Namespace(
                input=table, output=output, target_power_mw=None,
                direction="forward", resume=True,
                power_relative_tolerance=0.03, samples_per_point=3,
                repeat_settle_s=0.02, port="COM6", gpib="GPIB0::7::INSTR",
                settle_s=0.0, print_every=100, quiet_failures=True,
            )
            with mock.patch.object(model, "open_fullband_laser", return_value=DummyLaser()), \
                    mock.patch.object(model, "AQ6150B", DummyMeter):
                result = model.run_validate_equal_power(args)

            row = result["rows"][f"{target_nm:.6f}"]
            self.assertTrue(row["success"])
            self.assertEqual(result["power_relative_tolerance"], 0.03)
            audit = result["resume_tolerance_reclassification"][-1]
            self.assertEqual(audit["old_power_relative_tolerance"], 0.005)
            self.assertEqual(audit["new_power_relative_tolerance"], 0.03)
            self.assertTrue(audit["raw_measurements_reused"])

    def test_legacy_seed_wavelength_shift_is_learned_from_live_control_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live_path = root / "live145.json"
            old_path = root / "old135.json"
            live_rows = {}
            old_rows = {}
            for index in range(5):
                codes = [63351, 63351, 1000 + index, 2000 + index, 3000 + index]
                reading = {
                    "wavelength_nm": 1530.05 + index * 0.1,
                    "power_mw": 2.2,
                    "peak_count": 2,
                    "side_mode_suppression_db": 30.0,
                }
                live_rows[str(index)] = {
                    "codes": codes, "reading": reading, "single_mode": True
                }
                old_rows[str(index)] = {
                    "codes": [58982, 58982, *codes[2:]],
                    "reading": {**reading, "wavelength_nm": reading["wavelength_nm"] - 0.05},
                    "success": True,
                }
            live_path.write_text(json.dumps({
                "limits_ma": {"gain_ma": 145.0, "soa_ma": 145.0},
                "rows": live_rows,
            }), encoding="utf-8")
            old_path.write_text(json.dumps({
                "limits_ma": {"gain_ma": 135.0, "soa_ma": 135.0},
                "rows": old_rows,
            }), encoding="utf-8")
            rows = model.load_clean_phase_rows([live_path, old_path])
            old_loaded = [row for row in rows if row["source"] == old_path.name]
            self.assertEqual(len(old_loaded), 5)
            self.assertTrue(all(
                abs(row["warm_start_shift_nm"] - 0.05) < 1e-9
                for row in old_loaded
            ))
            self.assertTrue(all(
                abs(row["wavelength_nm"] - row["source_measured_wavelength_nm"] - 0.05)
                < 1e-9 for row in old_loaded
            ))

    def test_dense_repair_falls_back_to_phase_at_a_mirror_branch_boundary(self):
        def row(target: float, success: bool) -> dict:
            return {
                "success": success,
                "target_nm": target,
                "codes": [63351, 63351, 20000, 22000, 24575],
                "currents_ma": [145.0, 145.0, 6.1, 26.9, 30.0],
                "reading": {
                    "wavelength_nm": target if success else target + 0.006,
                    "power_mw": 2.1,
                    "peak_count": 2,
                    "side_mode_suppression_db": 30.0,
                },
                "source": "test",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "dense.json", root / "repaired.json"
            source.write_text(json.dumps({
                "rows": {
                    "1552.080000": row(1552.08, True),
                    "1552.100000": row(1552.10, False),
                    "1552.120000": row(1552.12, True),
                },
                "limits_ma": dict(zip(model.CONTROL_NAMES, model.LASER_MAX_MA)),
            }), encoding="utf-8")
            args = argparse.Namespace(
                resume=False, input=source, output=output, port="COM6",
                gpib="GPIB0::7::INSTR", passes=1, settle_s=0.0,
                max_measurements=10,
            )
            mirror_failure = {
                **row(1552.10, False), "history": [],
                "local_gradients_nm_per_ma": {},
            }
            phase_success = {
                **row(1552.10, True), "history": [],
                "phase_slope_nm_per_ma": -0.06,
            }
            laser = mock.Mock()
            meter = mock.Mock(identity="AQ6150B-test")
            with mock.patch.object(model, "open_fullband_laser", return_value=laser), \
                    mock.patch.object(model, "AQ6150B", return_value=meter), \
                    mock.patch.object(
                        model, "tune_local_three_controls", return_value=mirror_failure
                    ) as mirror_tune, \
                    mock.patch.object(
                        model, "tune_phase_on_branch", return_value=phase_success
                    ) as phase_tune:
                result = model.run_repair_dense(args)
            repaired = result["rows"]["1552.100000"]
            self.assertTrue(repaired["success"])
            self.assertIn("phase_from_low", repaired["source"])
            self.assertGreaterEqual(mirror_tune.call_count, 2)
            self.assertEqual(phase_tune.call_count, 1)

    def test_combined_wavelength_tuner_uses_phase_after_mirror_failure(self):
        mirror = {
            "success": False, "target_nm": 1525.08,
            "codes": [63351, 58000, 8192, 0, 5800],
            "currents_ma": [145.0, 132.75, 2.5, 0.0, 7.1],
            "reading": {
                "wavelength_nm": 1525.0745, "wavelength_error_pm": -5.5,
                "power_mw": 2.056, "power_dbm": 3.13, "peak_count": 2,
                "side_mode_suppression_db": 30.0,
            },
            "local_gradients_nm_per_ma": {}, "history": [],
        }
        phase = {
            **mirror, "success": True,
            "codes": [63351, 58000, 8400, 0, 5800],
            "reading": {
                **mirror["reading"], "wavelength_nm": 1525.0804,
                "wavelength_error_pm": 0.4,
            },
            "phase_slope_nm_per_ma": -0.075,
        }
        with mock.patch.object(
            model, "tune_local_three_controls", return_value=mirror
        ) as mirror_tune, mock.patch.object(
            model, "tune_phase_on_branch", return_value=phase
        ) as phase_tune:
            result, attempts = model.tune_wavelength_with_phase_fallback(
                mock.Mock(), mock.Mock(), mirror["codes"], 1525.08, 0.0, 20
            )
        self.assertTrue(result["success"])
        self.assertEqual(len(attempts), 2)
        mirror_tune.assert_called_once()
        phase_tune.assert_called_once()

    def test_repair_targets_inner_guard_band_without_tightening_release_gate(self):
        tuned = {
            "success": False,
            "target_nm": 1525.08,
            "codes": [63351, 58000, 8400, 0, 5800],
            "currents_ma": [145.0, 132.75, 2.56, 0.0, 7.1],
            "reading": {
                "wavelength_nm": 1525.0815,
                "wavelength_error_pm": 1.5,
                "power_mw": 2.056,
                "power_dbm": 3.13,
                "peak_count": 2,
                "side_mode_suppression_db": 30.0,
            },
            "local_gradients_nm_per_ma": {},
            "history": [],
        }
        with mock.patch.object(
            model, "tune_wavelength_with_phase_fallback",
            return_value=(tuned, []),
        ) as tuner:
            result = model.close_equal_power_from_seed(
                mock.Mock(), mock.Mock(), tuned["codes"], 1525.08,
                2.056, 2.5, [], 0.0, 0.03, 7, 24,
                closure_power_relative_tolerance=0.01,
                closure_wavelength_tolerance_pm=1.0,
            )

        # A repair candidate outside the requested inner +/-1 pm guard must
        # stay failed even though it still falls inside the public +/-2 pm
        # release limit.  Otherwise the next long forward sweep merely moves
        # the failure to another edge point.
        self.assertFalse(result["success"])
        self.assertAlmostEqual(
            result["closure_wavelength_error_pm"], 1.5, places=6,
        )
        self.assertEqual(
            tuner.call_args.kwargs["wavelength_tolerance_pm"], 1.0,
        )

    def test_repair_can_prebias_for_forward_sequence_wavelength_drift(self):
        closure_target = 1525.079
        tuned = {
            "success": True,
            "target_nm": closure_target,
            "codes": [63351, 58000, 8400, 0, 5800],
            "currents_ma": [145.0, 132.75, 2.56, 0.0, 7.1],
            "reading": {
                "wavelength_nm": closure_target,
                "wavelength_error_pm": 0.0,
                "power_mw": 2.056,
                "power_dbm": 3.13,
                "peak_count": 2,
                "side_mode_suppression_db": 30.0,
            },
            "local_gradients_nm_per_ma": {},
            "history": [],
        }
        reading = model.Reading(**tuned["reading"])
        with mock.patch.object(
            model, "tune_wavelength_with_phase_fallback",
            return_value=(tuned, []),
        ) as tuner, mock.patch.object(
            model, "probe", return_value=(tuned["codes"], reading),
        ):
            result = model.close_equal_power_from_seed(
                mock.Mock(), mock.Mock(), tuned["codes"], 1525.08,
                2.056, 2.5, [], 0.0, 0.03, 7, 24,
                closure_power_relative_tolerance=0.01,
                closure_wavelength_tolerance_pm=1.0,
                closure_wavelength_offset_pm=-1.0,
            )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["wavelength_error_pm"], -1.0, places=6)
        self.assertEqual(result["target_nm"], 1525.08)
        self.assertEqual(result["closure_wavelength_offset_pm"], -1.0)
        self.assertAlmostEqual(tuner.call_args.args[3], closure_target, places=9)

    def test_forward_transition_confirmation_rejects_sequential_mode_hop(self):
        predecessor = {
            "target_nm": 1525.0,
            "codes": [63351, 54000, 4000, 2000, 18000],
        }
        candidate = {
            "success": True,
            "target_nm": 1525.02,
            "codes": [63351, 54000, 5000, 2100, 18100],
            "reading": {},
        }

        def sequential_probe(_laser, _meter, codes, target_nm, _settle_s):
            measured_nm = 1525.0 if target_nm == 1525.0 else 1530.62
            reading = model.Reading(
                wavelength_nm=measured_nm,
                wavelength_error_pm=(measured_nm - target_nm) * 1000.0,
                power_mw=1.057,
                power_dbm=10.0 * math.log10(1.057),
                peak_count=1,
                side_mode_suppression_db=30.0,
            )
            return list(codes), reading

        with mock.patch.object(model, "probe", side_effect=sequential_probe) as probe:
            result = model.confirm_forward_transition_candidate(
                mock.Mock(), mock.Mock(), [predecessor], candidate,
                1525.02, 1.057, 0.0, 1.0, 0.025,
            )

        self.assertFalse(result["success"])
        self.assertGreater(abs(result["wavelength_error_pm"]), 100.0)
        self.assertEqual(probe.call_count, 6)
        self.assertEqual(
            result["forward_transition_confirmation"][
                "predecessor_targets_nm"
            ],
            [1525.0],
        )

    def test_adaptive_power_search_stops_at_a_searched_global_bottleneck(self):
        def row(target: float, power: float) -> dict:
            return {
                "success": True, "target_nm": target,
                "codes": [63351, 63351, 1000, 2000, 3000],
                "currents_ma": [145.0, 145.0, 0.3, 2.4, 3.7],
                "reading": {
                    "wavelength_nm": target, "power_mw": power,
                    "peak_count": 2, "side_mode_suppression_db": 30.0,
                },
                "source": "test",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source.json", root / "output.json"
            candidates = root / "candidates.json"
            source.write_text(json.dumps({
                "rows": {
                    "1550.000000": row(1550.00, 1.0),
                    "1550.020000": row(1550.02, 2.0),
                    "1550.040000": row(1550.04, 3.0),
                }
            }), encoding="utf-8")
            candidates.write_text(json.dumps({"rows": {
                key: {"candidates": [{
                    "codes": [63351, 63351, 1100 + index, 2100, 3100],
                    "predicted_power_mw": 3.5,
                }]}
                for index, key in enumerate(
                    ("1550.000000", "1550.020000", "1550.040000")
                )
            }}), encoding="utf-8")
            args = argparse.Namespace(
                resume=False, input=source, output=output,
                candidates=candidates, phase_maps=[], port="COM6",
                gpib="GPIB0::7::INSTR", adaptive_bottleneck=True,
                floor_mw=1.6, minimum_gain_mw=0.002,
                search_radius_nm=0.25, measured_seeds=4, screen_seeds=8,
                refine_seeds=2, max_measurements=14, settle_s=0.0,
            )
            powers = {1550.00: 2.5, 1550.02: 2.0, 1550.04: 3.2}

            def fake_probe(_laser, _meter, codes, target, _settle):
                power = powers[target]
                return list(codes), model.Reading(
                    wavelength_nm=target, wavelength_error_pm=0.0,
                    power_mw=power, power_dbm=0.0, peak_count=2,
                    side_mode_suppression_db=30.0,
                )

            laser = mock.Mock()
            meter = mock.Mock(identity="AQ6150B-test")
            with mock.patch.object(model, "open_fullband_laser", return_value=laser), \
                    mock.patch.object(model, "AQ6150B", return_value=meter), \
                    mock.patch.object(model, "load_clean_phase_rows", return_value=[]), \
                    mock.patch.object(model, "probe", side_effect=fake_probe) as probe:
                result = model.run_upgrade_low_power(args)
            summary = result["power_upgrade_summary"]
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(summary["searched_points"], 2)
            self.assertEqual(summary["bottleneck_target_nm"], 1550.02)
            self.assertTrue(summary["bottleneck_search_complete"])
            self.assertFalse(
                result["rows"]["1550.040000"].get(
                    "power_upgrade_searched_complete", False
                )
            )

    def test_ui_machine_event_parser_ignores_plain_or_malformed_lines(self):
        event = FullbandAutoCalibrationWorker._event(
            '@@AUTO@@ {"kind":"stage_started","index":1,"total":17}'
        )
        self.assertEqual(event["total"], auto.STAGE_COUNT)
        self.assertIsNone(FullbandAutoCalibrationWorker._event("ordinary log"))
        self.assertIsNone(FullbandAutoCalibrationWorker._event("@@AUTO@@ nope"))

    def test_orchestrator_can_accept_candidate_gap_exit_code_only_when_explicit(self):
        process = mock.Mock()
        process.stdout = []
        process.wait.return_value = 2
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            with mock.patch.object(auto.subprocess, "Popen", return_value=process):
                auto.run_command(
                    ["candidate-builder"], log_path, accepted_exit_codes=(0, 2)
                )
            with mock.patch.object(auto.subprocess, "Popen", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "退出码为2"):
                    auto.run_command(["strict-command"], log_path)

    def test_orchestrator_forces_utf8_child_output_and_escapes_console_text(self):
        process = mock.Mock()
        process.stdout = ["serial failure: \ufffd\n"]
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "run.log"
            with mock.patch.object(auto.subprocess, "Popen", return_value=process) as popen, \
                    mock.patch.object(auto, "_write_console_text") as write_console:
                auto.run_command(["serial-command"], log_path)
            self.assertEqual(
                popen.call_args.kwargs["env"]["PYTHONIOENCODING"],
                "utf-8:backslashreplace",
            )
            write_console.assert_called_once_with("serial failure: \ufffd\n")
            self.assertIn("serial failure: \ufffd", log_path.read_text(encoding="utf-8"))

        class GbkOnlyStream:
            encoding = "gbk"

            def __init__(self):
                self.text = ""

            def write(self, value):
                value.encode(self.encoding)
                self.text += value

            def flush(self):
                pass

        stream = GbkOnlyStream()
        with mock.patch.object(auto.sys, "stdout", stream):
            auto._write_console_text("failure: \ufffd")
        self.assertEqual(stream.text, r"failure: \ufffd")


if __name__ == "__main__":
    unittest.main()
