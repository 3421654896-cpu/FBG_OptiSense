import copy
import unittest

from evaluate_fresh_sentinel_position import compare


class IndependentPositionTests(unittest.TestCase):
    def capture(self):
        return {"source": "train", "source_sha256": {"frames.jsonl": "abc"},
                "protocol": "fresh_sentinel9_v1", "table_crc32": 123,
                "wavelengths_nm": [1525.0], "transimpedance_ohm": 5000, "depth_mm": 0.3,
                "adc_value_kind_counts": {"firmware_rc_estimate": 20},
                "records": [
                    {"point_index": 1, "model_xy_mm": [0, 0], "delta9": [1, 0], "hold_delta9": [[1, 0]]},
                    {"point_index": 2, "model_xy_mm": [4, 0], "delta9": [0, 1], "hold_delta9": [[0, 1]]}]}

    def test_reverse_route_uses_geometry_not_index(self):
        train = self.capture()
        test = copy.deepcopy(train)
        test.update(source="test", source_sha256={"frames.jsonl": "def"})
        test["records"].reverse()
        for index, record in enumerate(test["records"], 1):
            record["point_index"] = index
        result = compare(train, test)
        self.assertEqual(result["single_fresh_frame"]["exact_grid_point_percent"], 100)
        self.assertEqual(result["contact_median"]["max_error_mm"], 0)

    def test_duplicate_capture_rejected(self):
        train = self.capture()
        with self.assertRaisesRegex(ValueError, "Independent"):
            compare(train, copy.deepcopy(train))


if __name__ == "__main__":
    unittest.main()
