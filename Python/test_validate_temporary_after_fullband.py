import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from validate_temporary_after_fullband import (
    analyze_live_capture,
    validate_runtime_table_identity,
)


class TemporaryAfterFullbandTests(unittest.TestCase):
    @staticmethod
    def plan_and_capture():
        rows = []
        values = []
        for group in range(9):
            center = 1527.0 + group * 4.0
            x = center + np.linspace(-0.4, 0.4, 5)
            y = 40.0 + 700.0 * np.exp(-0.5 * ((x - center) / 0.2) ** 2)
            rows.extend({"measured_nm": float(value)} for value in x)
            values.extend(int(round(value)) for value in y)
        frames = []
        for sequence in range(16, 20):
            frames.append({
                "sequence": sequence,
                "records": [
                    {"first_code": value, "second_code": value}
                    for value in values
                ],
            })
        plan = {"rows": rows}
        capture = {
            "complete": True,
            "disarm_ack": True,
            "shutter_ack": True,
            "frames": frames,
        }
        return plan, capture

    def test_all_nine_peaks_must_fit_in_every_post_warmup_frame(self):
        plan, capture = self.plan_and_capture()
        summary, fitted = analyze_live_capture(plan, capture)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["all_nine_valid_frames"], 4)
        self.assertEqual(len(fitted), 4)
        self.assertTrue(all(row["valid_fraction"] == 1.0
                            for row in summary["peak_summaries"]))

        for record in capture["frames"][-1]["records"][10:15]:
            record["second_code"] = 20
        failed, _ = analyze_live_capture(plan, capture)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["peak_summaries"][2]["valid_frames"], 3)

    def test_runtime_json_and_hardware_csv_rows_must_be_identical(self):
        rows = {}
        points = []
        for index in range(2001):
            target = 1525.0 + index * 0.02
            measured = target + 0.0005
            codes = (60000, 50000, 1000 + index, 2000, 3000)
            rows[f"{target:.6f}"] = {
                "target_nm": target,
                "reading": {"wavelength_nm": measured},
                "codes": list(codes),
            }
            points.append(SimpleNamespace(
                index=index, target_nm=target, measured_nm=measured, codes=codes,
            ))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            result = validate_runtime_table_identity(path, points)
            self.assertEqual(result["point_count"], 2001)
            points[100].codes = (60000, 50000, 9999, 2000, 3000)
            with self.assertRaisesRegex(RuntimeError, "第101点不一致"):
                validate_runtime_table_identity(path, points)


if __name__ == "__main__":
    unittest.main()
