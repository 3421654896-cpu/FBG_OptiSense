import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

import yaml

from fbg_remote_viewer import MODE_STRESS, MODE_TEMPERATURE, load_wave_tables


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "JDSU"
CERTIFIED_REPORT = (
    HERE / "artifacts/ch1_measured_template_selection_v4_certified_20260905.json"
)
EXPECTED_POINT_COUNTS = {"stress": 45, "temperature": 54}
EXPECTED_SEGMENT_COUNT = 9


def yaml_axis(path):
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))["Wave_DATA"]
    return [int(row[0]) * 1000 + int(row[1]) for row in rows]


def c_rows(path, width, count):
    text = path.read_text(encoding="utf-8")
    rows = []
    for match in re.finditer(r"\{\s*([^{}]+?)\s*\},?", text):
        fields = [field.strip() for field in match.group(1).split(",")]
        if len(fields) != width:
            continue
        try:
            row = tuple(int(field, 0) for field in fields)
        except ValueError:
            continue
        rows.append(row)
        if len(rows) == count:
            return rows
    raise AssertionError(f"{path.name} only has {len(rows)}/{count} rows")


class ModeTablesFromFullbandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(
            (HERE / "mode_tables_from_fullband_2001.json").read_text(
                encoding="utf-8"
            )
        )
        with (HERE / "fullband_equal_power_operational_2001.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            cls.fullband = list(csv.DictReader(stream))
        cls.certificate = json.loads(CERTIFIED_REPORT.read_text(encoding="utf-8"))

    def test_both_runtime_modes_use_only_unique_2001_rows(self):
        expected = EXPECTED_POINT_COUNTS
        self.assertTrue(self.report["all_runtime_points_from_fullband_2001"])
        self.assertEqual(self.report["detected_peak_count"], EXPECTED_SEGMENT_COUNT)
        self.assertEqual(
            self.report["source_selection_schema"], self.certificate["schema"]
        )
        for name, count in expected.items():
            rows = self.report["modes"][name]["rows"]
            self.assertEqual(len(self.certificate[name]["rows"]), count)
            indices = [int(row["fullband_index"]) for row in rows]
            self.assertEqual(len(rows), count)
            self.assertEqual(len(set(indices)), count)
            self.assertEqual(indices, sorted(indices))
            for row in rows:
                source = self.fullband[int(row["fullband_index"])]
                self.assertEqual(int(source["single_mode_pass"]), 1)

    def test_axes_and_firmware_dac_rows_match_report(self):
        stress = self.report["modes"]["stress"]["rows"]
        temperature = self.report["modes"]["temperature"]["rows"]
        expected_stress_axis = [
            round(float(row["measured_wavelength_nm"]) * 1000) for row in stress
        ]
        expected_temperature_axis = [
            round(float(row["measured_wavelength_nm"]) * 1000)
            for row in temperature
        ]
        self.assertEqual(yaml_axis(HERE / "stress_wave_const.yaml"), expected_stress_axis)
        self.assertEqual(yaml_axis(HERE / "wave_const.yaml"), expected_temperature_axis)

        stress_codes = c_rows(
            PROJECT / "Core" / "Src" / "stress_table.c",
            5,
            EXPECTED_POINT_COUNTS["stress"],
        )
        temperature_codes = c_rows(
            PROJECT / "Core" / "Src" / "dac_const.c",
            5,
            EXPECTED_POINT_COUNTS["temperature"],
        )
        self.assertEqual(stress_codes, [tuple(row["codes"]) for row in stress])
        self.assertEqual(
            temperature_codes, [tuple(row["codes"]) for row in temperature]
        )

    def test_current_nine_certified_segments_are_retained(self):
        for mode, filename in (
            ("stress", "stress_wave_const.yaml"),
            ("temperature", "wave_const.yaml"),
        ):
            rows = self.report["modes"][mode]["rows"]
            peak_numbers = sorted({int(row["peak_number"]) for row in rows})
            self.assertEqual(peak_numbers, list(range(1, EXPECTED_SEGMENT_COUNT + 1)))
            self.assertEqual(len(yaml_axis(HERE / filename)), len(rows))

    def test_demo_and_local_modes_load_their_own_derived_axes(self):
        tables = load_wave_tables(HERE / "wave_const.yaml")
        self.assertEqual(
            len(tables[MODE_STRESS]), EXPECTED_POINT_COUNTS["stress"]
        )
        self.assertEqual(
            len(tables[MODE_TEMPERATURE]), EXPECTED_POINT_COUNTS["temperature"]
        )
        self.assertEqual(
            [round(value * 1000) for value in tables[MODE_STRESS]],
            yaml_axis(HERE / "stress_wave_const.yaml"),
        )

    def test_remote_viewer_accepts_dynamic_safe_axis_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            temperature_path = root / "wave_const.yaml"
            stress_path = root / "stress_wave_const.yaml"
            temperature_rows = [[1525, index] for index in range(0, 198, 2)]
            stress_rows = [[1525, index] for index in range(0, 90, 2)]
            temperature_path.write_text(
                yaml.safe_dump({"Wave_DATA": temperature_rows}), encoding="utf-8"
            )
            stress_path.write_text(
                yaml.safe_dump({"Wave_DATA": stress_rows}), encoding="utf-8"
            )
            tables = load_wave_tables(temperature_path)
            self.assertEqual(len(tables[MODE_STRESS]), 45)
            self.assertEqual(len(tables[MODE_TEMPERATURE]), 99)


if __name__ == "__main__":
    unittest.main()
