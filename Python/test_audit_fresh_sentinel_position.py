import unittest

from audit_fresh_sentinel_position import SENTINELS, fresh_vector


class FreshSentinelTests(unittest.TestCase):
    def frame(self):
        return {"schedule_version": 4, "bandwidth_discontinuity": False,
                "fresh_point_mask": [i in SENTINELS for i in range(45)],
                "point_age_frames": [0 if i in SENTINELS else 99 for i in range(45)],
                "sample_offset_us": [100 * i if i in SENTINELS else None for i in range(45)],
                "raw_adc_codes": {"1": list(range(45))}}

    def test_cached_non_sentinels_do_not_enter_feature(self):
        frame = self.frame()
        for i in range(45):
            if i not in SENTINELS:
                frame["raw_adc_codes"]["1"][i] = 4095
        self.assertEqual(fresh_vector(frame).tolist(), list(SENTINELS))

    def test_v5_requires_new_g8_point(self):
        frame = self.frame()
        frame["schedule_version"] = 5
        with self.assertRaisesRegex(ValueError, "stale_sentinel"):
            fresh_vector(frame)
        frame["fresh_point_mask"][36] = False
        frame["fresh_point_mask"][38] = True
        frame["point_age_frames"][38] = 0
        frame["sample_offset_us"][38] = 3800
        self.assertEqual(fresh_vector(frame)[7], 38)

    def test_any_stale_sentinel_blocks_feature(self):
        for i in SENTINELS:
            frame = self.frame()
            frame["fresh_point_mask"][i] = False
            with self.assertRaisesRegex(ValueError, "stale_sentinel"):
                fresh_vector(frame)

    def test_discontinuity_and_missing_time_block_feature(self):
        frame = self.frame()
        frame["bandwidth_discontinuity"] = True
        with self.assertRaises(ValueError):
            fresh_vector(frame)
        frame = self.frame()
        frame["sample_offset_us"][SENTINELS[0]] = None
        with self.assertRaises(ValueError):
            fresh_vector(frame)


if __name__ == "__main__":
    unittest.main()
