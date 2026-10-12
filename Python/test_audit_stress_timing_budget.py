from pathlib import Path
import unittest

from audit_stress_timing_budget import (
    audit_report, read_source_model, replay_write_counts, timing_trace,
)
from frame_telemetry import sentinel_offsets_for_version
from test_frame_telemetry import _v4_schedule_extension


def wire(sequence=0, *, version=4, full=False):
    indices = tuple(range(45)) if full else tuple(
        segment*5 + offset for segment, offset in enumerate(sentinel_offsets_for_version(version)))
    extension = bytearray(_v4_schedule_extension(
        0 if full else 1, indices, version=version, map_age=0 if full else 2))
    extension[16] = 2
    extension[22:24] = bytes((2, 2))
    extension[24:28] = (0x01001F).to_bytes(4, "big")
    extension[28:32] = sequence.to_bytes(4, "big")
    frame = bytearray(b"\xee\xee\x00\x2d" + bytes(45*8))
    for point in range(45):
        frame[6 + point*8:8 + point*8] = (point + 20).to_bytes(2, "big")
    return bytes(frame) + b"\xab" + bytes(12) + bytes((0xB5, 0x4D, 1, len(extension))) + extension + b"\xff\xef"


def report(frames):
    return {"point_count": 45, "firmware_versions": ["1.0.31"], "feedback_selector": 2,
            "board_telemetry": {"available_frames": len(frames)},
            "fresh_value_trace": [timing_trace(frame) for frame in frames]}


class TimingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parent.parent
        cls.model = read_source_model(cls.repo)

    def test_current_source_nominal_wire_count_not_all_225(self):
        self.assertEqual(self.model["steady_register_write_count"], 177)
        self.assertEqual(self.model["cold_register_write_count"], 178)
        self.assertEqual(self.model["nominal_all_dac_wire_floor_us"], 15930)
        self.assertIsNone(self.model["wave_time_us"])

    def test_hidden_target_forces_ten_even_if_values_unchanged(self):
        rows = [(1, 2, 3, 4, 5)]*3
        self.assertEqual(replay_write_counts(rows, rows[0], 1), [0, 10, 0])
        self.assertEqual(replay_write_counts(rows, rows[0], 1, cold=True), [5, 10, 0])

    def test_unsafe_row_rejected_offline(self):
        with self.assertRaises(ValueError):
            replay_write_counts([(65535, 1, 1, 1, 1)], (1, 1, 1, 1, 1), 0)

    def test_trace_omits_cached_rows_and_keeps_original(self):
        for version in (4, 5):
            raw = wire(version=version)
            trace = timing_trace(raw)
            self.assertEqual(trace["wire_hex"], raw.hex())
            self.assertEqual(len(trace["fresh_ch1_codes"]), 9)
            self.assertNotIn("0", trace["fresh_ch1_codes"])
            self.assertEqual(trace["fresh_ch1_codes"]["3"], 23)
            self.assertEqual(trace["fresh_sample_completion_offset_us"]["3"], 20)
            self.assertEqual(trace["firmware_version"], 0x01001F)

    def test_wire_footer_and_extra_bytes_rejected(self):
        for raw in (wire()[:-1], wire() + b"extra", wire()[:-2] + b"\xff\xff"):
            with self.assertRaises(ValueError):
                timing_trace(raw)

    def test_offset_after_frame_duration_rejected(self):
        raw = bytearray(wire())
        extension_start = 4 + 45*8 + 1 + 12 + 4
        raw[extension_start+32:extension_start+36] = (1).to_bytes(4, "big")
        with self.assertRaises(ValueError):
            timing_trace(bytes(raw))

    def test_original_redecoded_and_seq_wrap_accepted(self):
        result = audit_report(report([wire(0xFFFFFFFF), wire(0)]), self.model)
        self.assertEqual(result["wire_redecoded_frames"], 2)
        self.assertFalse(result["physical_bandwidth_verified"])
        self.assertFalse(result["training_eligible"])
        self.assertAlmostEqual(result["profiles"]["SURVEY9"]["nominal_accounted_floor_mean_ms"], 26.98)
        first = result["sample_completion_intervals"][0]
        self.assertEqual(first["replayed_dac_rows"], 4)
        self.assertEqual(first["register_writes"], 22)
        self.assertEqual(first["nominal_accounted_floor_us"], 7830)

    def test_mutated_trace_not_trusted(self):
        row = report([wire()])
        row["fresh_value_trace"][0]["fresh_sample_completion_offset_us"]["3"] += 2
        with self.assertRaisesRegex(ValueError, "differs"):
            audit_report(row, self.model)

    def test_gaps_duplicates_missing_original_coverage_fail(self):
        for frames in ([wire(1), wire(3)], [wire(1), wire(1)]):
            with self.assertRaisesRegex(ValueError, "sequence"):
                audit_report(report(frames), self.model)
        row = report([wire()])
        del row["fresh_value_trace"][0]["wire_hex"]
        with self.assertRaisesRegex(ValueError, "original"):
            audit_report(row, self.model)
        row = report([wire()])
        row["board_telemetry"]["available_frames"] = 2
        with self.assertRaisesRegex(ValueError, "partial"):
            audit_report(row, self.model)

    def test_metadata_mismatch_rejected(self):
        for key, value in (("firmware_versions", ["1.0.30"]), ("feedback_selector", 1)):
            row = report([wire()])
            row[key] = value
            with self.assertRaises(ValueError):
                audit_report(row, self.model)

    def test_complete_map_cost_remains_full_wait(self):
        result = audit_report(report([wire(full=True)]), self.model)
        self.assertEqual(result["profiles"]["MAP45"]["nominal_pair_wait_floor_mean_ms"], 31.65)

    def test_source_dac_precedes_optional_adc_skip(self):
        source = (self.repo / "JDSU/Core/Src/ms5614t.c").read_text("utf-8")
        loop = source.split("void write_ms5614t_table(void)", 1)[1].split("void write_ms5614t_manual", 1)[0]
        self.assertLess(loop.index("else if(PI11210_ApplyStressRowCalibrationOrder()"),
                        loop.index("stressAppendCachedCh1(activeScanPointIndex)"))


if __name__ == "__main__":
    unittest.main()
