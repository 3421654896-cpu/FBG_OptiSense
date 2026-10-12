import copy
import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import mode_table_manager
from migrate_selected_ch1_candidate import (
    build_migrated_candidate,
    build_migrated_checkpoint,
    reconcile_calibration_reuse,
)
from prepare_ch1_selected_candidate import DEFAULT_GROUPS, ROLES, build_candidate
from selected_calibration_bridge import (
    build_certified_v4_report,
    sparse_stress_table_fingerprint,
)


def audited_codes(index):
    return [50000, 42000 + index % 100, 10000 + index % 100, 6000, 7000]


def final_codes(index):
    values = audited_codes(index)
    values[1] -= 10
    values[2] += 1
    return values


class BridgeFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.python = self.root / "Python"
        self.artifacts = self.python / "artifacts"
        self.artifacts.mkdir(parents=True)
        self._write_fullband_csv()
        self.operational = self.python / "fullband_equal_power_operational_2001.json"
        self._write_operational_json()
        self.candidate = self.artifacts / "candidate.json"
        self.candidate.write_text(
            json.dumps(build_candidate(self.operational), indent=2), encoding="utf-8"
        )
        self.calibration = self.artifacts / "calibration.json"
        self.calibration_payload = self._calibration_payload()
        self.write_calibration()
        self.selection = self._selection_payload()

    def _write_fullband_csv(self):
        path = self.python / "fullband_equal_power_operational_2001.csv"
        fields = [
            "index",
            "target_wavelength_nm",
            "measured_wavelength_nm",
            "single_mode_pass",
            "gain_code",
            "soa_code",
            "phase_code",
            "wavelength_a_code",
            "wavelength_b_code",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(2001):
                target = 1525.0 + 0.02 * index
                codes = audited_codes(index)
                writer.writerow(
                    {
                        "index": index,
                        "target_wavelength_nm": f"{target:.6f}",
                        "measured_wavelength_nm": f"{target:.6f}",
                        "single_mode_pass": 1,
                        "gain_code": codes[0],
                        "soa_code": codes[1],
                        "phase_code": codes[2],
                        "wavelength_a_code": codes[3],
                        "wavelength_b_code": codes[4],
                    }
                )

    def _write_operational_json(self):
        rows = {}
        for index in range(2001):
            target = 1525.0 + 0.02 * index
            rows[f"{target:.6f}"] = {
                "target_nm": target,
                "codes": audited_codes(index),
            }
        self.operational.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    def _calibration_payload(self):
        candidate_sha = hashlib.sha256(self.candidate.read_bytes()).hexdigest()
        sparse_session_id = "fresh-sparse-session"
        ceiling_rows = {}
        calibration_rows = {}
        ordered = [index for group in DEFAULT_GROUPS for index in group]
        for order, index in enumerate(ordered):
            target = 1525.0 + 0.02 * index
            ceiling_rows[str(order)] = {
                "order": order,
                "fullband_index": index,
                "target_wavelength_nm": target,
                "ceiling_codes": audited_codes(index),
                "clean_ceiling_power_mw": 1.72,
                "qualified": True,
            }
            confirmations = [
                {
                    "wavelength_nm": target - 0.0002,
                    "power_mw": 1.649,
                    "normalized_smsr_db": 31.0,
                },
                {
                    "wavelength_nm": target + 0.0002,
                    "power_mw": 1.651,
                    "normalized_smsr_db": 30.0,
                },
            ]
            calibration_rows[str(order)] = {
                "order": order,
                "fullband_index": index,
                "target_wavelength_nm": target,
                "original_codes": audited_codes(index),
                "codes": final_codes(index),
                "target_power_mw": 1.65,
                "power_tolerance_mw": 0.04,
                "confirmations": confirmations,
                "final_wavelength_nm": target,
                "final_wavelength_error_pm": 0.0,
                "final_wavelength_span_pm": 0.4,
                "final_power_mw": 1.65,
                "final_power_span_fraction": 0.002 / 1.65,
                "final_smsr_min_db": 30.0,
                "passed": True,
            }

        rounds = {}
        for round_index in range(2):
            rows = {}
            for order, index in enumerate(ordered):
                predecessor = (order - 1) % 45
                target = 1525.0 + 0.02 * index
                rows[str(order)] = {
                    "round": round_index,
                    "order": order,
                    "predecessor_order": predecessor,
                    "wraparound": order == 0,
                    "fullband_index": index,
                    "target_wavelength_nm": target,
                    "predecessor_codes": final_codes(ordered[predecessor]),
                    "codes": final_codes(index),
                    "reading": {
                        "wavelength_nm": target,
                        "power_mw": 1.65,
                        "normalized_smsr_db": 30.0,
                    },
                    "target_power_mw": 1.65,
                    "power_tolerance_mw": 0.04,
                    "power_ok": True,
                    "passed": True,
                    "runtime_path": "previous_sparse->target",
                    "used_hidden_predecessor": False,
                    "evidence_session_id": sparse_session_id,
                }
            rounds[str(round_index)] = rows
        return {
            "schema": "fbg_selected_ch1_calibration_v1",
            "candidate_source": {"sha256": candidate_sha, "point_count": 45},
            "settings": {
                "wavelength_tolerance_pm": 2.0,
                "minimum_smsr_db": 20.0,
                "probe_branch_tolerance_pm": 50.0,
                "confirmation_wavelength_span_pm": 2.0,
                "confirmation_power_span_fraction": 0.03,
                "sequence_rounds": 2,
            },
            "ceiling_probe": {
                "rows": ceiling_rows,
                "summary": {
                    "all_points_qualified": True,
                    "recommended_common_power_mw": 1.70,
                },
            },
            "calibration": {
                "target_power_mw": 1.65,
                "rows": calibration_rows,
                "summary": {"all_points_passed": True},
            },
            "sparse_sequence": {
                "code_source": "calibration",
                "hidden_predecessor_request": None,
                "rounds": rounds,
                "summary": {
                    "all_points_runtime_path_certified": True,
                    "hidden_predecessor_orders": [],
                    "hidden_predecessor_hold_us": None,
                },
            },
            "sessions": [
                {
                    "session_id": sparse_session_id,
                    "candidate_sha256": candidate_sha,
                    "mode": "sparse-sequence",
                    "status": "complete",
                    "shutter_confirmed": True,
                    "hidden_predecessor_request": None,
                }
            ],
            "shutter_confirmed": True,
            "writes_runtime_table": False,
            "writes_firmware": False,
            "final_runtime_certified": True,
        }

    def write_calibration(self):
        self.calibration.write_text(
            json.dumps(self.calibration_payload, indent=2), encoding="utf-8"
        )

    def enable_hidden_predecessor(self, order=2):
        ordered = [index for group in DEFAULT_GROUPS for index in group]
        fullband_index = ordered[order]
        request = {
            "mode_order": order,
            "fullband_index": fullband_index,
            "predecessor_fullband_index": fullband_index - 1,
            "predecessor_codes": audited_codes(fullband_index - 1),
            "hold_us": 5000,
            "runtime_path": (
                "previous_sparse->hidden_fullband_predecessor->target"
            ),
        }
        sequence = self.calibration_payload["sparse_sequence"]
        sequence["hidden_predecessor_request"] = copy.deepcopy(request)
        sequence["summary"]["hidden_predecessor_orders"] = [order]
        sequence["summary"]["hidden_predecessor_hold_us"] = 5000
        for rows in sequence["rounds"].values():
            row = rows[str(order)]
            row["runtime_path"] = request["runtime_path"]
            row["used_hidden_predecessor"] = True
            row["hidden_predecessor"] = {
                "fullband_index": fullband_index - 1,
                "codes": audited_codes(fullband_index - 1),
                "hold_us": 5000,
            }
        self.calibration_payload["sessions"][-1][
            "hidden_predecessor_request"
        ] = copy.deepcopy(request)
        return request

    def enable_candidate_seed(self, order=24):
        ordered = [index for group in DEFAULT_GROUPS for index in group]
        index = ordered[order]
        candidate_codes = audited_codes(index)
        ceiling_codes = list(candidate_codes)
        ceiling_codes[1] += 100
        ceiling_codes[2] += 5
        ceiling_codes[3] += 10
        ceiling_codes[4] -= 10
        self.calibration_payload["ceiling_probe"]["rows"][str(order)][
            "ceiling_codes"
        ] = ceiling_codes
        row = self.calibration_payload["calibration"]["rows"][str(order)]
        target = 1525.0 + 0.02 * index
        target_power = float(row["target_power_mw"])
        candidate_power = target_power + 0.01
        ceiling_power = target_power + 0.20
        row.update(
            {
                "candidate_codes": list(candidate_codes),
                "seed_source": "candidate",
                "selected_seed_codes": list(candidate_codes),
                "calibration_controller_revision": (
                    "dual_measured_seed_staged_slope_v3_internal_1pct_1pm_smsr_latch"
                ),
                "seed_probes": [
                    {
                        "action": "calibration_seed_probe",
                        "source": "candidate",
                        "codes": list(candidate_codes),
                        "reading": {
                            "wavelength_nm": target,
                            "wavelength_error_pm": 0.0,
                            "power_mw": candidate_power,
                            "normalized_smsr_db": 31.0,
                        },
                        "qualified": True,
                        "power_distance_to_target_mw": abs(
                            candidate_power - target_power
                        ),
                    },
                    {
                        "action": "calibration_seed_probe",
                        "source": "ceiling_probe",
                        "codes": list(ceiling_codes),
                        "reading": {
                            "wavelength_nm": target,
                            "wavelength_error_pm": 0.0,
                            "power_mw": ceiling_power,
                            "normalized_smsr_db": 32.0,
                        },
                        "qualified": True,
                        "power_distance_to_target_mw": abs(
                            ceiling_power - target_power
                        ),
                    },
                ],
            }
        )
        return row

    def _selection_payload(self):
        stress_rows = []
        peaks = []
        for peak_number, group in enumerate(DEFAULT_GROUPS, start=1):
            targets = [1525.0 + 0.02 * index for index in group]
            center = targets[2]
            dense_x = [
                targets[0] - 0.1 + 0.02 * i
                for i in range(1 + round((targets[-1] - targets[0] + 0.2) / 0.02))
            ]
            dense_y = [
                0.02 + math.exp(-0.5 * ((x - center) / 0.16) ** 2) for x in dense_x
            ]
            peaks.append(
                {
                    "peak_number": peak_number,
                    "source_channel": 1,
                    "center_nm": center,
                    "sigma_nm": 0.16,
                    "dense_wavelength_nm": dense_x,
                    "dense_template_signal_v": dense_y,
                    "selected_fullband_indices": list(group),
                    "selected_template_signal_v": [1.0] * 5,
                    "selected_reliability_weight": [1.0] * 5,
                }
            )
            for index in group:
                target = 1525.0 + 0.02 * index
                stress_rows.append(
                    {
                        "peak_number": peak_number,
                        "source_channel": 1,
                        "fullband_index": index,
                        "target_wavelength_nm": target,
                        "measured_wavelength_nm": target,
                        "codes": audited_codes(index),
                    }
                )
        mode = {
            "point_count": 45,
            "minimum_points_per_peak": 5,
            "points_per_peak": [5] * 9,
            "peaks": peaks,
            "rows": stress_rows,
        }
        return {
            "schema": "equal_interval_auto_mode_selection_v4",
            "source_point_count": 2001,
            "detected_peak_count": 9,
            "configured_peak_count": 9,
            "fbg_channels": [1],
            "settle_s": 0.2,
            "selection": {
                "method": "measured_template_fisher_v1",
                "dac_row_policy": "copy_exact_fullband_2001_audited_row",
            },
            "analog_feedback": {
                "ch0_selector": 1,
                "ch1_selector": 1,
                "ch0_kohm": 40,
                "ch1_kohm": 40,
            },
            "stress": copy.deepcopy(mode),
            "temperature": copy.deepcopy(mode),
            "sparse_sequence_validation": {"status": "pending"},
        }

    def build(self):
        return build_certified_v4_report(
            self.selection,
            candidate_path=self.candidate,
            calibration_path=self.calibration,
            repository=self.root,
        )


class SelectedCalibrationBridgeTests(unittest.TestCase):
    def test_candidate_generator_uses_exact_replacement_groups_and_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            payload = json.loads(fixture.candidate.read_text(encoding="utf-8"))
            groups = tuple(
                tuple(point["source_index"] for point in peak["points"])
                for peak in payload["peaks"]
            )
            self.assertEqual(groups, DEFAULT_GROUPS)
            self.assertEqual(
                tuple(point["role"] for point in payload["peaks"][0]["points"]),
                ROLES,
            )
            self.assertFalse(payload["writes_runtime_table"])
            self.assertEqual(payload["hardware_sparse_sequence_validation"], "pending")

    def test_complete_report_promotes_exact_final_codes_and_manager_accepts(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            result = fixture.build()
            rows = result["stress"]["rows"]
            expected_indices = [index for group in DEFAULT_GROUPS for index in group]
            self.assertEqual([row["fullband_index"] for row in rows], expected_indices)
            self.assertEqual(
                [row["codes"] for row in rows],
                [final_codes(i) for i in expected_indices],
            )
            self.assertEqual(result["stress"]["points_per_peak"], [5] * 9)
            validation = result["sparse_sequence_validation"]
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["validated_repeats"], 2)
            self.assertEqual(
                validation["stress_table_fingerprint_sha256"],
                sparse_stress_table_fingerprint(rows),
            )
            normalized = mode_table_manager.validate_candidate_report(
                result, fixture.root
            )
            self.assertEqual(
                normalized["stress"]["rows"][0]["codes"],
                final_codes(expected_indices[0]),
            )

    def test_current_controller_accepts_measured_qualified_candidate_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.enable_candidate_seed(order=24)
            fixture.write_calibration()
            result = fixture.build()
            self.assertEqual(result["sparse_sequence_validation"]["status"], "passed")

    def test_current_controller_rejects_unbound_selected_seed_codes(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            row = fixture.enable_candidate_seed(order=24)
            row["selected_seed_codes"][1] -= 1
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "选定起始DAC与认证分支"):
                fixture.build()

    def test_current_controller_rejects_unqualified_selected_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            row = fixture.enable_candidate_seed(order=24)
            row["seed_probes"][0]["qualified"] = False
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "探针合格状态不可重算"):
                fixture.build()

    def test_two_round_hidden_path_is_certified_and_manager_accepts(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            request = fixture.enable_hidden_predecessor(order=2)
            fixture.write_calibration()
            result = fixture.build()
            validation = result["sparse_sequence_validation"]
            self.assertTrue(
                validation["per_point_results"][2][
                    "requires_hidden_predecessor"
                ]
            )
            self.assertEqual(
                validation["certified_hidden_predecessors"],
                [
                    {
                        "mode_index": 2,
                        "fullband_index": request["fullband_index"],
                        "predecessor_fullband_index": request[
                            "predecessor_fullband_index"
                        ],
                        "predecessor_codes": request["predecessor_codes"],
                        "hold_us": 5000,
                        "runtime_path": request["runtime_path"],
                        "validated_repeats": 2,
                        "certified": True,
                    }
                ],
            )
            normalized = mode_table_manager.validate_candidate_report(
                result, fixture.root
            )
            index, codes = mode_table_manager._stress_path_precondition(
                normalized["stress"]["rows"],
                normalized["sparse_sequence_validation"],
            )
            self.assertEqual(index, 2)
            self.assertEqual(codes, request["predecessor_codes"])

    def test_hidden_path_requires_identical_metadata_in_both_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.enable_hidden_predecessor(order=2)
            fixture.calibration_payload["sparse_sequence"]["rounds"]["1"]["2"][
                "hidden_predecessor"
            ]["hold_us"] = 4999
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "逐轮隐藏前驱元数据"):
                fixture.build()

    def test_hidden_path_summary_hold_must_match_both_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.enable_hidden_predecessor(order=2)
            fixture.calibration_payload["sparse_sequence"]["summary"][
                "hidden_predecessor_hold_us"
            ] = 4999
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "摘要的隐藏前驱保持时间"):
                fixture.build()

    def test_sparse_bridge_rejects_mixed_old_session_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.enable_hidden_predecessor(order=2)
            fixture.calibration_payload["sparse_sequence"]["rounds"]["0"]["0"][
                "evidence_session_id"
            ] = "old-direct-session"
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "混用了旧硬件会话"):
                fixture.build()

    def test_incomplete_calibration_cannot_emit_v4(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.calibration_payload["final_runtime_certified"] = False
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "尚未通过最终稀疏路径"):
                fixture.build()

    def test_top_level_shutter_true_cannot_mask_failed_final_session(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.calibration_payload["shutter_confirmed"] = True
            fixture.calibration_payload["sessions"][-1]["shutter_confirmed"] = False
            fixture.calibration_payload["sessions"][-1]["status"] = "failed"
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "最后一次硬件会话"):
                fixture.build()

    def test_wrong_sparse_predecessor_codes_cannot_emit_v4(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            fixture.calibration_payload["sparse_sequence"]["rounds"]["1"]["0"][
                "predecessor_codes"
            ][1] -= 1
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "前驱DAC"):
                fixture.build()

    def test_manager_rejects_code_tamper_after_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            result = fixture.build()
            result["stress"]["rows"][0]["codes"][1] -= 1
            with self.assertRaisesRegex(ValueError, "5路DAC不是最终标定"):
                mode_table_manager.validate_candidate_report(result, fixture.root)

    def test_manager_rejects_calibrated_codes_if_certificate_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            result = fixture.build()
            del result["hardware_calibration_certificate"]
            with self.assertRaisesRegex(ValueError, "DAC码与2001点标定表不一致"):
                mode_table_manager.validate_candidate_report(result, fixture.root)

    def test_manager_rejects_evidence_file_changed_after_bridge(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = BridgeFixture(folder)
            result = fixture.build()
            fixture.calibration_payload["note"] = "changed after certification"
            fixture.write_calibration()
            with self.assertRaisesRegex(ValueError, "报告SHA256已失效"):
                mode_table_manager.validate_candidate_report(result, fixture.root)


class SelectedCalibrationMigrationTests(unittest.TestCase):
    def _prepare(self, folder):
        fixture = BridgeFixture(folder)
        fixture.calibration_payload["fullband_predecessor_sha256"] = "a" * 64
        fixture.calibration_payload["sessions"][-1]["finished"] = (
            "2026-09-05T05:00:00+08:00"
        )
        fixture.write_calibration()
        migrated_candidate = fixture.artifacts / "candidate_migrated.json"
        candidate_payload = build_migrated_candidate(
            base_candidate_path=fixture.candidate,
            base_report_path=fixture.calibration,
            replacements={24: 998},
            repository=fixture.root,
            reason="order24 low SMSR",
        )
        migrated_candidate.write_text(
            json.dumps(candidate_payload, indent=2), encoding="utf-8"
        )
        migrated_report = fixture.artifacts / "calibration_migrated.json"
        report_payload = build_migrated_checkpoint(
            migrated_candidate_path=migrated_candidate,
            base_report_path=fixture.calibration,
            repository=fixture.root,
        )
        migrated_report.write_text(
            json.dumps(report_payload, indent=2), encoding="utf-8"
        )
        return fixture, migrated_candidate, migrated_report, report_payload

    def _complete(self, fixture, candidate, report_path, report):
        candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        ordered = [index for group in DEFAULT_GROUPS for index in group]
        ordered[24] = 998
        target = 1525.0 + 0.02 * 998
        ceiling_session = "fresh-ceiling"
        calibration_session = "fresh-calibration"
        sparse_session = "fresh-sparse"
        report["ceiling_probe"]["rows"]["24"] = {
            "order": 24,
            "fullband_index": 998,
            "target_wavelength_nm": target,
            "ceiling_codes": audited_codes(998),
            "clean_ceiling_power_mw": 1.72,
            "qualified": True,
            "evidence_session_id": ceiling_session,
        }
        report["ceiling_probe"]["summary"] = {
            "all_points_qualified": True,
            "recommended_common_power_mw": 1.70,
        }
        reconcile_calibration_reuse(report, 1.65)
        report["calibration"]["rows"]["24"] = {
            "order": 24,
            "fullband_index": 998,
            "target_wavelength_nm": target,
            "original_codes": audited_codes(998),
            "codes": final_codes(998),
            "target_power_mw": 1.65,
            "power_tolerance_mw": 0.04,
            "confirmations": [
                {
                    "wavelength_nm": target - 0.0002,
                    "power_mw": 1.649,
                    "normalized_smsr_db": 31.0,
                },
                {
                    "wavelength_nm": target + 0.0002,
                    "power_mw": 1.651,
                    "normalized_smsr_db": 30.0,
                },
            ],
            "final_wavelength_nm": target,
            "final_wavelength_error_pm": 0.0,
            "final_wavelength_span_pm": 0.4,
            "final_power_mw": 1.65,
            "final_power_span_fraction": 0.002 / 1.65,
            "final_smsr_min_db": 30.0,
            "passed": True,
            "evidence_session_id": calibration_session,
        }
        report["calibration"]["target_power_mw"] = 1.65
        report["calibration"]["summary"] = {"all_points_passed": True}
        rounds = {}
        for round_index in range(2):
            rows = {}
            for order, index in enumerate(ordered):
                predecessor = (order - 1) % 45
                point_target = 1525.0 + 0.02 * index
                rows[str(order)] = {
                    "round": round_index,
                    "order": order,
                    "predecessor_order": predecessor,
                    "wraparound": order == 0,
                    "fullband_index": index,
                    "target_wavelength_nm": point_target,
                    "predecessor_codes": final_codes(ordered[predecessor]),
                    "codes": final_codes(index),
                    "reading": {
                        "wavelength_nm": point_target,
                        "power_mw": 1.65,
                        "normalized_smsr_db": 30.0,
                    },
                    "target_power_mw": 1.65,
                    "power_tolerance_mw": 0.04,
                    "power_ok": True,
                    "passed": True,
                    "runtime_path": "previous_sparse->target",
                    "used_hidden_predecessor": False,
                    "evidence_session_id": sparse_session,
                }
            rounds[str(round_index)] = rows
        report["sparse_sequence"] = {
            "code_source": "calibration",
            "hidden_predecessor_request": None,
            "rounds": rounds,
            "summary": {
                "all_points_runtime_path_certified": True,
                "hidden_predecessor_orders": [],
                "hidden_predecessor_hold_us": None,
            },
        }
        report["sessions"] = [
            {
                "session_id": ceiling_session,
                "candidate_sha256": candidate_sha,
                "mode": "ceiling-probe",
                "status": "complete",
                "shutter_confirmed": True,
                "finished": "2026-09-05T05:10:00+08:00",
            },
            {
                "session_id": calibration_session,
                "candidate_sha256": candidate_sha,
                "mode": "calibrate",
                "status": "complete",
                "shutter_confirmed": True,
                "finished": "2026-09-05T05:20:00+08:00",
            },
            {
                "session_id": sparse_session,
                "candidate_sha256": candidate_sha,
                "mode": "sparse-sequence",
                "status": "complete",
                "shutter_confirmed": True,
                "hidden_predecessor_request": None,
                "finished": "2026-09-05T05:30:00+08:00",
            },
        ]
        report["shutter_confirmed"] = True
        report["final_runtime_certified"] = True
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def test_migration_reuses_only_byte_identical_points_and_clears_sparse(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture, candidate, _, report = self._prepare(folder)
            candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
            self.assertNotEqual(
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
                hashlib.sha256(fixture.candidate.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                candidate_payload["candidate_migration"]["changed_orders"], [24]
            )
            self.assertNotIn("24", report["ceiling_probe"]["rows"])
            self.assertNotIn("24", report["calibration"]["rows"])
            self.assertEqual(len(report["ceiling_probe"]["rows"]), 44)
            self.assertEqual(len(report["calibration"]["rows"]), 44)
            self.assertEqual(report["sparse_sequence"]["rounds"], {})
            self.assertFalse(report["final_runtime_certified"])

    def test_changed_common_power_discards_every_reused_calibration_row(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, report = self._prepare(folder)
            reconcile_calibration_reuse(report, 1.64)
            self.assertEqual(report["calibration"]["rows"], {})
            self.assertEqual(
                report["evidence_migration"]["calibration_reuse_status"],
                "discarded_target_changed",
            )
            self.assertEqual(
                report["evidence_migration"]["reused_calibration_orders"], []
            )

    def test_complete_migration_requires_fresh_changed_rows_and_full_sparse(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture, candidate, report_path, report = self._prepare(folder)
            self._complete(fixture, candidate, report_path, report)
            result = build_certified_v4_report(
                fixture.selection,
                candidate_path=candidate,
                calibration_path=report_path,
                repository=fixture.root,
            )
            self.assertEqual(result["stress"]["rows"][24]["fullband_index"], 998)
            self.assertTrue(
                result["hardware_calibration_certificate"]["candidate_migration"][
                    "full_sparse_sequence_remeasured"
                ]
            )
            mode_table_manager.validate_candidate_report(result, fixture.root)

    def test_migration_rejects_sparse_row_without_new_candidate_session(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture, candidate, report_path, report = self._prepare(folder)
            self._complete(fixture, candidate, report_path, report)
            report["sparse_sequence"]["rounds"]["0"]["0"][
                "evidence_session_id"
            ] = "fresh-sparse-session"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "混用了旧硬件会话"):
                build_certified_v4_report(
                    fixture.selection,
                    candidate_path=candidate,
                    calibration_path=report_path,
                    repository=fixture.root,
                )


if __name__ == "__main__":
    unittest.main()
