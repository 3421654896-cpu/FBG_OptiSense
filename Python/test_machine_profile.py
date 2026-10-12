import unittest

from machine_profile import (
    MACHINE_ID,
    MACHINE_LABEL,
    PARAMETER_NOTE,
    ensure_machine_compatible,
    runtime_fullband_profile,
    runtime_parameter_note,
    set_runtime_machine_id,
    stamp_machine_metadata,
    verify_profile_artifacts,
)


class MachineProfileTests(unittest.TestCase):
    def setUp(self):
        set_runtime_machine_id("machine_1")

    def tearDown(self):
        set_runtime_machine_id("machine_1")

    def test_active_profile_is_machine_one(self):
        self.assertEqual(MACHINE_ID, "machine_1")
        self.assertEqual(MACHINE_LABEL, "一号机")
        self.assertIn("9峰", PARAMETER_NOTE)
        self.assertIn("2001点", PARAMETER_NOTE)
        self.assertIn("3峰", PARAMETER_NOTE)

    def test_all_pinned_machine_one_artifacts_match(self):
        results = verify_profile_artifacts()
        self.assertGreaterEqual(len(results), 8)
        self.assertTrue(all(item["exists"] for item in results), results)
        self.assertTrue(all(item["sha256_ok"] for item in results), results)
        self.assertTrue(all(item["size_ok"] for item in results), results)

    def test_new_payloads_are_marked_and_other_machine_is_rejected(self):
        payload = stamp_machine_metadata({"value": 1}, ("fullband_2001",))
        self.assertTrue(ensure_machine_compatible(payload))
        self.assertEqual(payload["parameter_owner"], "一号机")
        self.assertEqual(payload["parameter_sets"], ["fullband_2001"])
        with self.assertRaisesRegex(ValueError, "二号机"):
            ensure_machine_compatible({
                "machine_id": "machine_2",
                "machine_label": "二号机",
            })

    def test_pinned_legacy_records_are_accepted(self):
        self.assertFalse(ensure_machine_compatible({"schema": "legacy"}))

    def test_machine_two_uses_first_calibration_with_visible_warning(self):
        set_runtime_machine_id("machine_2")
        profile = runtime_fullband_profile()
        self.assertEqual(profile["label"], "2号解调仪第一次校准")
        self.assertEqual(profile["validation_passed"], 1716)
        self.assertEqual(profile["validation_failed"], 285)
        self.assertFalse(profile["certified"])
        self.assertIn("285点待修复", runtime_parameter_note())
        payload = stamp_machine_metadata({"value": 2}, ("fullband_2001",))
        self.assertEqual(payload["machine_id"], "machine_2")
        self.assertEqual(payload["parameter_owner"], "二号机")

    def test_machine_three_uses_provisional_table_with_partial_audit_warning(self):
        set_runtime_machine_id("machine_3")
        profile = runtime_fullband_profile()
        self.assertEqual(profile["source_format"], "calibration_json")
        self.assertEqual(profile["validation_tested"], 725)
        self.assertFalse(profile["certified"])
        note = runtime_parameter_note()
        self.assertIn("临时表", note)
        self.assertIn("703通过", note)
        self.assertIn("22未通过", note)
        self.assertIn("1276待测", note)


if __name__ == "__main__":
    unittest.main()
