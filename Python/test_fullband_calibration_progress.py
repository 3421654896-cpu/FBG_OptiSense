import json
import tempfile
import unittest
from pathlib import Path

from fullband_calibration_progress import validation_file_progress


class CalibrationProgressTests(unittest.TestCase):
    def test_validation_checkpoint_reports_each_measured_point(self):
        payload = {
            "rows": {
                "1525.000000": {
                    "success": True,
                    "target_nm": 1525.0,
                    "wavelength_error_pm": 0.25,
                    "power_relative_error": -0.001,
                },
                "1525.020000": {
                    "success": False,
                    "target_nm": 1525.02,
                    "wavelength_error_pm": 2.25,
                    "power_relative_error": 0.006,
                },
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "validation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            current, total, line = validation_file_progress(path)
        self.assertEqual((current, total), (2, 2001))
        self.assertIn("target=1525.020000", line)
        self.assertIn("err=+2.25pm", line)
        self.assertIn("perr=+0.60%", line)
        self.assertIn("ok=False", line)

    def test_incomplete_checkpoint_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "validation.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(validation_file_progress(path), (0, 0, ""))


if __name__ == "__main__":
    unittest.main()
