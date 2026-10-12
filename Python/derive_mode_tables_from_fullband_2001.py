"""Derive the fast stress and precision temperature tables from the audited 2001 points.

The existing mode tables define the nine FBG wavelength windows and the point
count allocated to each window.  This exporter preserves that geometry and
chooses the nearest unique, strictly ordered wavelength-meter measurement from
``fullband_equal_power_operational_2001.csv``.  It updates both the firmware DAC
rows and the desktop wavelength axes together, then records complete provenance
in JSON so a plotted wavelength can always be traced back to its 2001-point row.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "JDSU"
FULLBAND_CSV = HERE / "fullband_equal_power_operational_2001.csv"
REPORT_JSON = HERE / "mode_tables_from_fullband_2001.json"
STRESS_YAML = HERE / "stress_wave_const.yaml"
TEMPERATURE_YAML = HERE / "wave_const.yaml"
STRESS_C = PROJECT / "Core" / "Src" / "stress_table.c"
STRESS_H = PROJECT / "Core" / "Inc" / "stress_table.h"
TEMPERATURE_DAC_C = PROJECT / "Core" / "Src" / "dac_const.c"
TEMPERATURE_WAVE_C = PROJECT / "Core" / "Src" / "wave_const.c"

CODE_FIELDS = (
    "gain_code",
    "soa_code",
    "phase_code",
    "wavelength_a_code",
    "wavelength_b_code",
)
CODE_LIMITS = (58981, 58981, 32767, 24575, 24575)
EXPECTED_COUNTS = {"stress": 63, "temperature": 100}


def _load_axis(path: Path) -> np.ndarray:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("Wave_DATA")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path.name} 没有可用的 Wave_DATA")
    axis = np.asarray(
        [float(row[0]) + float(row[1]) * 0.001 for row in rows], dtype=float
    )
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{path.name} 含非有限波长")
    return axis


def _load_fullband_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "index",
            "target_wavelength_nm",
            "measured_wavelength_nm",
            "measured_power_mw",
            "smsr_db",
            "single_mode_pass",
            *CODE_FIELDS,
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("2001点标定表字段不完整")
        for expected_index, raw in enumerate(reader):
            index = int(raw["index"])
            if index != expected_index:
                raise ValueError(f"2001点索引不连续：{index}")
            if int(raw["single_mode_pass"]) != 1:
                raise ValueError(f"2001点中的第{index + 1}点未通过单模门限")
            codes = []
            for field, limit in zip(CODE_FIELDS, CODE_LIMITS):
                code = int(raw[field])
                # The calibration export's mathematical 135 mA boundary is one
                # LSB above the MCU clamp.  The live calibration readback was
                # already clamped, so normalize that exact historical boundary.
                if code == limit + 1:
                    code = limit
                if not 0 <= code <= limit:
                    raise ValueError(f"2001点第{index + 1}点 {field} 超过安全上限")
                codes.append(code)
            measured = float(raw["measured_wavelength_nm"])
            if not math.isfinite(measured):
                raise ValueError(f"2001点第{index + 1}点实测波长无效")
            smsr_text = str(raw.get("smsr_db", "")).strip()
            smsr_db = float(smsr_text) if smsr_text else None
            rows.append(
                {
                    "fullband_index": index,
                    "target_wavelength_nm": float(raw["target_wavelength_nm"]),
                    "measured_wavelength_nm": measured,
                    "measured_power_mw": float(raw["measured_power_mw"]),
                    # An empty SMSR means the meter did not report a secondary
                    # peak; ``single_mode_pass`` remains the authoritative gate.
                    "smsr_db": smsr_db,
                    "codes": codes,
                }
            )
    if len(rows) != 2001:
        raise ValueError(f"2001点标定表实际为{len(rows)}点")
    measured = np.asarray(
        [row["measured_wavelength_nm"] for row in rows], dtype=float
    )
    if np.any(np.diff(measured) <= 0.0):
        raise ValueError("2001点实测波长不是严格递增")
    return rows


def _select_rows(axis: np.ndarray, candidates: list[dict]) -> list[dict]:
    measured = np.asarray(
        [row["measured_wavelength_nm"] for row in candidates], dtype=float
    )
    selected = []
    used = set()
    previous_index = -1
    for mode_index, requested_nm in enumerate(axis):
        ranked = np.argsort(np.abs(measured - requested_nm), kind="stable")
        candidate_index = next(
            (
                int(index)
                for index in ranked
                if int(index) not in used and int(index) > previous_index
            ),
            None,
        )
        if candidate_index is None:
            raise RuntimeError(f"无法为模式点{mode_index + 1}选择递增的2001点候选")
        source = candidates[candidate_index]
        selected.append(
            {
                "mode_index": mode_index,
                "requested_wavelength_nm": float(requested_nm),
                "selection_error_pm": (
                    float(source["measured_wavelength_nm"]) - float(requested_nm)
                )
                * 1000.0,
                **source,
            }
        )
        used.add(candidate_index)
        previous_index = candidate_index
    return selected


def _wave_pair(wavelength_nm: float) -> tuple[int, int]:
    total_pm = int(round(float(wavelength_nm) * 1000.0))
    return total_pm // 1000, total_pm % 1000


def _write_yaml(path: Path, rows: list[dict]) -> None:
    payload = {
        "Wave_DATA": [
            list(_wave_pair(row["measured_wavelength_nm"])) for row in rows
        ]
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _replace_first_numeric_rows(
    path: Path, width: int, replacements: list[tuple[int, ...]]
) -> None:
    text = path.read_text(encoding="utf-8")
    row_pattern = re.compile(r"\{\s*([^{}]+?)\s*\},?")
    replacement_index = 0

    def replace(match: re.Match) -> str:
        nonlocal replacement_index
        fields = [part.strip() for part in match.group(1).split(",")]
        if len(fields) != width:
            return match.group(0)
        try:
            [int(field, 0) for field in fields]
        except ValueError:
            return match.group(0)
        if replacement_index >= len(replacements):
            return match.group(0)
        row = replacements[replacement_index]
        replacement_index += 1
        return "{" + ", ".join(str(int(value)) for value in row) + "},"

    updated = row_pattern.sub(replace, text)
    if replacement_index != len(replacements):
        raise RuntimeError(
            f"{path.name} 只替换了{replacement_index}/{len(replacements)}行"
        )
    path.write_text(updated, encoding="utf-8")


def _segment_starts(rows: list[dict]) -> list[int]:
    starts = [0]
    wavelengths = [float(row["measured_wavelength_nm"]) for row in rows]
    for index, step in enumerate(np.diff(wavelengths), start=1):
        if step > 0.5:
            starts.append(index)
    return starts


def _write_stress_sources(
    rows: list[dict], temperature_rows: list[dict]
) -> None:
    segment_starts = _segment_starts(rows)
    temperature_segment_starts = _segment_starts(temperature_rows)
    if len(segment_starts) != 9:
        raise ValueError(f"应力表应保留9个波段，实际为{len(segment_starts)}个")
    previous_header = STRESS_H.read_text(encoding="utf-8") if STRESS_H.exists() else ""

    def retained_macro(name: str, default: int) -> int:
        match = re.search(rf"^#define\s+{re.escape(name)}\s+(\d+)U", previous_header, re.M)
        return int(match.group(1)) if match else int(default)

    selector_ch0 = retained_macro("TEMPERATURE_FEEDBACK_SELECTOR_CH0", 0)
    selector_ch1 = retained_macro("TEMPERATURE_FEEDBACK_SELECTOR_CH1", 0)
    settle_us = retained_macro("TEMPERATURE_ADC_SETTLE_US", 12000)
    path_index = min(
        range(len(rows)),
        key=lambda candidate: abs(
            float(rows[candidate]["measured_wavelength_nm"]) - 1538.620
        ),
    )
    if abs(float(rows[path_index]["measured_wavelength_nm"]) - 1538.620) > 0.011:
        path_index = len(rows)
        path_codes = [0, 0, 0, 0, 0]
    else:
        path_codes = list(rows[path_index]["precondition_codes"])
    STRESS_H.write_text(
        """#ifndef __STRESS_TABLE_H
#define __STRESS_TABLE_H

#include <stdint.h>

#define STRESS_TABLE_POINT_COUNT 63U
#define STRESS_SEGMENT_COUNT %dU
#define STRESS_LAST_SECTION_START_INDEX %dU
#define TEMPERATURE_TABLE_POINT_COUNT 100U
#define TEMPERATURE_SEGMENT_COUNT %dU
#define TEMPERATURE_FEEDBACK_SELECTOR_CH0 %dU
#define TEMPERATURE_FEEDBACK_SELECTOR_CH1 %dU
#define TEMPERATURE_ADC_SETTLE_US %dU
#define TEMPERATURE_PRECONDITION_HOLD_US 5000U
#define STRESS_PATH_PRECONDITION_POINT_INDEX %dU
#define STRESS_PATH_PRECONDITION_HOLD_US 5000U

extern const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5];
extern const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2];
extern const uint16_t Stress_Path_Precondition_DAC[5];
extern const uint16_t Temperature_Precondition_DAC[TEMPERATURE_TABLE_POINT_COUNT][5];

#endif
""" % (
            len(segment_starts),
            segment_starts[-1],
            len(temperature_segment_starts),
            selector_ch0,
            selector_ch1,
            settle_us,
            path_index,
        ),
        encoding="utf-8",
    )
    lines = [
        '#include "stress_table.h"',
        "",
        "/* Derived only from single-mode-pass rows in the audited 2001-point",
        " * wavelength-meter table.  The nine FBG windows and 63-point frame",
        " * budget are retained for transient stress response.",
        " */",
        "const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5] = {",
    ]
    lines.extend(
        "{" + ",".join(str(value) for value in row["codes"]) + "},"
        for row in rows
    )
    lines.extend(
        [
            "};",
            "",
            "const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2] = {",
        ]
    )
    lines.extend(
        "{%d,%d}," % _wave_pair(row["measured_wavelength_nm"]) for row in rows
    )
    lines.extend(
        [
            "};",
            "",
            "/* Immediate audited predecessor for the path-sensitive stress target.",
            " * This optical-state command is never sampled or plotted.",
            " */",
            "const uint16_t Stress_Path_Precondition_DAC[5] =",
            "{" + ",".join(str(value) for value in path_codes) + "};",
            "",
            "/* Immediate audited 2001-point predecessor for every sparse",
            " * temperature target.  These rows establish the same cavity-mode",
            " * path as a complete equal-interval scan; they are never sampled.",
            " */",
            "const uint16_t Temperature_Precondition_DAC[TEMPERATURE_TABLE_POINT_COUNT][5] = {",
        ]
    )
    lines.extend(
        "{" + ",".join(str(value) for value in row["precondition_codes"]) + "},"
        for row in temperature_rows
    )
    lines.extend(["};", ""])
    STRESS_C.write_text("\n".join(lines), encoding="utf-8")


def _summary(rows: list[dict]) -> dict:
    errors = np.abs([row["selection_error_pm"] for row in rows])
    powers = np.asarray([row["measured_power_mw"] for row in rows], dtype=float)
    return {
        "point_count": len(rows),
        "fullband_candidate_count": len({row["fullband_index"] for row in rows}),
        "selection_error_median_pm": float(np.median(errors)),
        "selection_error_p95_pm": float(np.percentile(errors, 95)),
        "selection_error_max_pm": float(np.max(errors)),
        "power_min_mw": float(np.min(powers)),
        "power_max_mw": float(np.max(powers)),
        "minimum_finite_smsr_db": float(
            min(row["smsr_db"] for row in rows if row["smsr_db"] is not None)
        ),
    }


def main() -> int:
    candidates = _load_fullband_rows(FULLBAND_CSV)
    stress = _select_rows(_load_axis(STRESS_YAML), candidates)
    temperature = _select_rows(_load_axis(TEMPERATURE_YAML), candidates)
    for row in stress + temperature:
        row["precondition_codes"] = list(
            candidates[max(0, int(row["fullband_index"]) - 1)]["codes"]
        )
    modes = {"stress": stress, "temperature": temperature}
    for name, rows in modes.items():
        if len(rows) != EXPECTED_COUNTS[name]:
            raise ValueError(
                f"{name}模式应为{EXPECTED_COUNTS[name]}点，实际为{len(rows)}点"
            )

    _write_stress_sources(stress, temperature)
    _write_yaml(STRESS_YAML, stress)
    _write_yaml(TEMPERATURE_YAML, temperature)
    _replace_first_numeric_rows(
        TEMPERATURE_DAC_C,
        5,
        [tuple(row["codes"]) for row in temperature],
    )
    _replace_first_numeric_rows(
        TEMPERATURE_WAVE_C,
        2,
        [_wave_pair(row["measured_wavelength_nm"]) for row in temperature],
    )

    report = {
        "source_table": FULLBAND_CSV.name,
        "selection_rule": (
            "preserve each mode's nine FBG windows and point budget; choose the "
            "nearest unique strictly increasing single-mode-pass 2001-point row"
        ),
        "all_runtime_points_from_fullband_2001": True,
        "modes": {
            name: {"summary": _summary(rows), "rows": rows}
            for name, rows in modes.items()
        },
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: _summary(rows) for name, rows in modes.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
