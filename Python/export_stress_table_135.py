"""Export the validated 63-point 135 mA table to C and YAML."""

from __future__ import annotations

import json
from pathlib import Path

from laser_power_calibration import save_json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FINAL = HERE / "stress_high_power_135_final.json"
VALIDATION = HERE / "stress_high_power_135_validation.json"


def wave_pair(wave: float) -> tuple[int, int]:
    total_pm = int(round(wave * 1000.0))
    return total_pm // 1000, total_pm % 1000


def main() -> int:
    source = json.loads(FINAL.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    rows = source["rows"]
    results = validation.get("results", ())
    if len(results) != len(rows):
        raise ValueError("标定表与复测结果数量不一致")
    if validation["summary"]["passed"] != len(rows):
        raise RuntimeError("拒绝导出未全部通过复测的标定表")

    paired = []
    for row, result in zip(rows, results):
        samples = result["samples"]
        if not samples:
            raise ValueError("复测结果中存在空采样")
        row["final_wavelength_nm"] = sum(
            float(item["wavelength_nm"]) for item in samples
        ) / len(samples)
        row["final_power_mw"] = sum(
            float(item["power_mw"]) for item in samples
        ) / len(samples)
        finite_smsr = [
            float(item["side_mode_suppression_db"])
            for item in samples
            if item.get("side_mode_suppression_db") is not None
        ]
        row["final_smsr_db"] = min(finite_smsr) if finite_smsr else None
        row["single_peak_repeats"] = sum(
            int(item.get("peak_count", 0)) == 1 for item in samples
        )
        paired.append((row, result))

    # The desired-target order can differ from the measured branch order.  The
    # scan rows and their validation records must always be reordered together.
    paired.sort(
        key=lambda item: (
            int(item[0]["segment"]),
            float(item[0]["final_wavelength_nm"]),
        )
    )
    rows = []
    validation["results"] = []
    for index, (row, result) in enumerate(paired):
        row["index"] = index
        result["index"] = index
        rows.append(row)
        validation["results"].append(result)
    save_json(VALIDATION, validation)

    header = ROOT / "JDSU" / "Core" / "Inc" / "stress_table.h"
    header.write_text(
        """#ifndef __STRESS_TABLE_H
#define __STRESS_TABLE_H

#include <stdint.h>

#define STRESS_TABLE_POINT_COUNT 63U
#define STRESS_LAST_SECTION_START_INDEX 56U

extern const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5];
extern const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2];

#endif
""",
        encoding="utf-8",
    )

    c_lines = [
        '#include "stress_table.h"',
        "",
        "/* Fresh AQ6150B calibration on the current optical path, 2026-08-14.",
        " * GAIN/SOA were constrained to <=135 mA.  Every row passed two",
        " * independent replays; no double peak was accepted (SMSR >=20 dB,",
        " * with an absent secondary peak treated as a true single peak).",
        " */",
        "const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5] = {",
    ]
    for index in range(0, len(rows), 2):
        pieces = [
            "{" + ",".join(str(int(value)) for value in row["codes"]) + "},"
            for row in rows[index : index + 2]
        ]
        c_lines.append(" ".join(pieces))
    c_lines.extend(
        [
            "};",
            "",
            "/* Plot coordinates are the mean wavelengths from the final two",
            " * independent AQ6150B replays, rounded to 1 pm.",
            " */",
            "const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2] = {",
        ]
    )
    for index in range(0, len(rows), 7):
        c_lines.append(
            ",".join(
                "{%d,%d}" % wave_pair(row["final_wavelength_nm"])
                for row in rows[index : index + 7]
            )
            + ","
        )
    c_lines.extend(["};", ""])
    (ROOT / "JDSU" / "Core" / "Src" / "stress_table.c").write_text(
        "\n".join(c_lines), encoding="utf-8"
    )

    yaml_lines = ["Wave_DATA:"]
    for row in rows:
        integer, fraction = wave_pair(row["final_wavelength_nm"])
        yaml_lines.append(f"  - [{integer}, {fraction}]")
    (HERE / "stress_wave_const.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )

    source["rows"] = rows
    source["summary"] = {
        "point_count": len(rows),
        "segment_count": 9,
        "power_min_mw": min(row["final_power_mw"] for row in rows),
        "power_max_mw": max(row["final_power_mw"] for row in rows),
        "power_mean_mw": sum(row["final_power_mw"] for row in rows) / len(rows),
        "minimum_finite_smsr_db": min(
            row["final_smsr_db"]
            for row in rows
            if row["final_smsr_db"] is not None
        ),
        "maximum_wavelength_replay_span_pm": validation["summary"][
            "max_wavelength_span_pm"
        ],
    }
    save_json(FINAL, source)
    print(json.dumps(source["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
