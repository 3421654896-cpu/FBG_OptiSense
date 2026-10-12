"""Assemble the final 100-point table from only current-session successes."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import yaml

from laser_power_calibration import code_to_current, save_json, write_proposed_c


HERE = Path(__file__).resolve().parent


def successful_replacements(mapping_path: Path, report_path: Path) -> dict[int, list[dict]]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output: dict[int, list[dict]] = {}
    for group in mapping["groups"]:
        hybrid_index = int(group["hybrid_index"])
        target = float(group["failed_target_nm"])
        rows = []
        for replacement_index in group["replacement_indices"]:
            result = report["results"].get(str(replacement_index))
            if not result or not result.get("success"):
                continue
            final = result["final"]
            rows.append(
                {
                    "hybrid_index": hybrid_index,
                    "failed_target_nm": target,
                    "replacement_index": int(replacement_index),
                    "result": result,
                    "distance_pm": abs(float(final["wavelength_nm"]) - target) * 1000.0,
                }
            )
        output.setdefault(hybrid_index, []).extend(rows)
    return output


def main() -> int:
    hybrid = json.loads(
        (HERE / "fbg_hybrid_calibration_report.json").read_text(encoding="utf-8")
    )
    replacements = successful_replacements(
        HERE / "fbg_replacement_candidates.json",
        HERE / "fbg_replacement_calibration_report.json",
    )
    second = successful_replacements(
        HERE / "fbg_replacement2_candidates.json",
        HERE / "fbg_replacement2_calibration_report.json",
    )
    for index, rows in second.items():
        replacements.setdefault(index, []).extend(rows)

    results = {}
    rows = []
    wavelengths = []
    replacement_choices = []
    for index in range(100):
        result = hybrid["results"][str(index)]
        provenance = "hybrid_current_session"
        if not result.get("success"):
            alternatives = replacements.get(index, [])
            if not alternatives:
                raise RuntimeError(f"No successful replacement for index {index}")
            choice = min(
                alternatives,
                key=lambda item: (
                    item["distance_pm"]
                    + 5.0 * max(float(item["result"].get("soa_ma", 999.0)) - 115.0, 0.0),
                    float(item["result"].get("soa_ma", 999.0)),
                ),
            )
            result = choice["result"]
            provenance = "replacement_candidate"
            replacement_choices.append(
                {
                    "final_index": index,
                    "failed_target_nm": choice["failed_target_nm"],
                    "replacement_index": choice["replacement_index"],
                    "replacement_wavelength_nm": result["final"]["wavelength_nm"],
                    "distance_pm": choice["distance_pm"],
                }
            )

        codes = [int(value) for value in result["calibrated_codes"]]
        final_wave = float(result["final"]["wavelength_nm"])
        rounded_wave = round(final_wave, 3)
        final = dict(result["final"])
        final["wavelength_error_pm"] = (final_wave - rounded_wave) * 1000.0
        rows.append(codes)
        wavelengths.append(rounded_wave)
        results[str(index)] = {
            "index": index,
            "target_nm": rounded_wave,
            "success": True,
            "original_codes": codes,
            "calibrated_codes": codes,
            "gain_ma": code_to_current(codes[0], 0),
            "soa_ma": code_to_current(codes[1], 1),
            "phase_ma": code_to_current(codes[2], 2),
            "final": final,
            "history": [],
            "failure": None,
            "validation": provenance,
        }

    segment_lengths = [11, 12, 11, 11, 11, 11, 11, 11, 11]
    position = 0
    for length in segment_lengths:
        section = wavelengths[position : position + length]
        if any(right <= left for left, right in zip(section, section[1:])):
            raise RuntimeError(f"Non-increasing final wavelength section at {position}")
        position += length

    wave_rows = []
    for wavelength in wavelengths:
        major = int(math.floor(wavelength))
        minor = int(round((wavelength - major) * 1000.0))
        if minor == 1000:
            major += 1
            minor = 0
        wave_rows.append([major, minor])
    (HERE / "wave_const_final_optimized.yaml").write_text(
        yaml.safe_dump({"Wave_DATA": wave_rows}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    wave_c_lines = [
        '#include "main.h"',
        "",
        "const uint16_t Wave_DATA[Number][2]  __at (Wave_Addr)",
        "=",
        "{",
    ]
    wave_c_lines.extend(f"{{{major},{minor}}}," for major, minor in wave_rows)
    wave_c_lines.append("};")
    (HERE / "wave_const_final_optimized.c").write_text(
        "\n".join(wave_c_lines) + "\n", encoding="utf-8"
    )
    write_proposed_c(
        HERE / "dac_const_hybrid_calibrated.c",
        rows,
        HERE / "dac_const_final_optimized.c",
    )

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "meter": hybrid.get("meter"),
        "target_power_mw": 0.5,
        "power_tolerance_mw": 0.006,
        "wavelength_tolerance_pm": 3.0,
        "results": results,
        "replacement_choices": replacement_choices,
    }
    save_json(HERE / "fbg_final_calibration_report.json", payload)
    print(
        f"assembled=100 replacements={len(replacement_choices)} "
        f"soa_max_ma={max(item['soa_ma'] for item in results.values()):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
