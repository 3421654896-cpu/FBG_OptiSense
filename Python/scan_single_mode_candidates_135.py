"""Expand the 135/135 mA map with local PHASE scans and select fresh points.

The old 69-point wavelengths are not used.  Existing DAC rows provide only
WAVE-A/WAVE-B/PHASE seed coordinates.  Every accepted wavelength and power is
measured again with the AQ6150B at the new GAIN/SOA operating point.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import (
    AQ6150B,
    DAC_SOURCE,
    LaserSerial,
    code_to_current,
    current_to_code,
    parse_dac_rows,
    save_json,
    take_reading,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "high_power_tuning_map_gain135_soa135.json"
OUTPUT = HERE / "single_mode_candidates_135.json"
SELECTED = HERE / "fresh_stress_points_135_selected.json"

# Physical FBG centers measured through CH1.  They define only the optical
# windows that must be sampled; no wavelength from the old scan table is used.
FBG_CENTERS_NM = (
    1527.941, 1532.238, 1536.326, 1540.220, 1544.456,
    1547.902, 1551.702, 1555.827, 1559.884,
)
WINDOW_HALF_WIDTH_NM = 0.58
TARGET_OFFSETS_NM = (-0.50, -0.32, -0.16, 0.0, 0.16, 0.32, 0.50)
PHASE_OFFSETS_MA = (-2.4, -1.6, -0.9, -0.45, 0.45, 0.9, 1.6, 2.4)
MIN_SMSR_DB = 20.0
MIN_POWER_MW = 1.35


def clean_reading(record: dict) -> bool:
    reading = record["reading"]
    smsr = reading.get("side_mode_suppression_db")
    return bool(
        math.isfinite(float(reading["wavelength_nm"]))
        and math.isfinite(float(reading["power_mw"]))
        and float(reading["power_mw"]) >= MIN_POWER_MW
        and smsr is not None
        and float(smsr) >= MIN_SMSR_DB
    )


def in_window(wavelength_nm: float, center_nm: float) -> bool:
    return abs(float(wavelength_nm) - center_nm) <= WINDOW_HALF_WIDTH_NM


def unique_candidates(records: list[dict]) -> list[dict]:
    """Remove exact DAC duplicates and nearly identical optical duplicates."""
    by_codes = {}
    for row in records:
        key = tuple(int(value) for value in row["codes"])
        previous = by_codes.get(key)
        if previous is None or float(row["reading"]["side_mode_suppression_db"]) > float(
            previous["reading"]["side_mode_suppression_db"]
        ):
            by_codes[key] = row
    ordered = sorted(by_codes.values(), key=lambda row: float(row["reading"]["wavelength_nm"]))
    result = []
    for row in ordered:
        if result and abs(float(row["reading"]["wavelength_nm"]) - float(
            result[-1]["reading"]["wavelength_nm"]
        )) < 0.008:
            old = result[-1]
            old_score = float(old["reading"]["power_mw"]) + 0.01 * float(
                old["reading"]["side_mode_suppression_db"]
            )
            new_score = float(row["reading"]["power_mw"]) + 0.01 * float(
                row["reading"]["side_mode_suppression_db"]
            )
            if new_score > old_score:
                result[-1] = row
        else:
            result.append(row)
    return result


def choose_anchors(rows: list[dict], center_nm: float) -> list[dict]:
    ordered = sorted(rows, key=lambda row: float(row["reading"]["wavelength_nm"]))
    if not ordered:
        return []
    picks = [ordered[0], min(ordered, key=lambda row: abs(
        float(row["reading"]["wavelength_nm"]) - center_nm)), ordered[-1]]
    output = []
    used = set()
    for row in picks:
        key = tuple(row["codes"])
        if key not in used:
            used.add(key)
            output.append(row)
    return output


def select_points(rows: list[dict], center_nm: float) -> list[dict]:
    pool = unique_candidates([
        row for row in rows
        if clean_reading(row) and in_window(row["reading"]["wavelength_nm"], center_nm)
    ])
    if len(pool) < len(TARGET_OFFSETS_NM):
        raise RuntimeError(f"Only {len(pool)} clean candidates around {center_nm:.3f} nm")
    chosen = []
    used = set()
    for offset in TARGET_OFFSETS_NM:
        target = center_nm + offset
        eligible = [row for row in pool if tuple(row["codes"]) not in used]
        # Wavelength placement dominates.  Power and SMSR break near ties.
        best = min(
            eligible,
            key=lambda row: (
                abs(float(row["reading"]["wavelength_nm"]) - target),
                -float(row["reading"]["power_mw"]),
                -float(row["reading"]["side_mode_suppression_db"]),
            ),
        )
        used.add(tuple(best["codes"]))
        chosen.append(best)
    return sorted(chosen, key=lambda row: float(row["reading"]["wavelength_nm"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--selected", type=Path, default=SELECTED)
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    base = []
    for item in source["results"].values():
        if clean_reading(item):
            base.append({
                "source": "135mA_base_map",
                "seed_index": int(item["seed_index"]),
                "codes": [int(value) for value in item["codes"]],
                "currents_ma": [float(value) for value in item["currents_ma"]],
                "reading": dict(item["reading"]),
            })

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": "fresh_135mA_base_map_plus_local_phase_scan",
        "limits_ma": {"gain": 135.0, "soa": 135.0, "phase": 10.0,
                      "wave_a": 30.0, "wave_b": 30.0},
        "minimum_smsr_db": MIN_SMSR_DB,
        "meter": source.get("meter"),
        "base_candidates": base,
        "phase_scans": {},
    }
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        payload["phase_scans"].update(old.get("phase_scans", {}))

    old_rows = parse_dac_rows(DAC_SOURCE)
    park = list(old_rows[min(44, len(old_rows) - 1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(90.0, code_to_current(park[1], 1)), 1)
    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    try:
        for segment, center_nm in enumerate(FBG_CENTERS_NM):
            base_segment = [row for row in base if in_window(
                row["reading"]["wavelength_nm"], center_nm
            )]
            anchors = choose_anchors(base_segment, center_nm)
            for anchor_number, anchor in enumerate(anchors):
                base_phase = code_to_current(anchor["codes"][2], 2)
                base_wave = float(anchor["reading"]["wavelength_nm"])
                for phase_offset in PHASE_OFFSETS_MA:
                    phase_ma = min(max(base_phase + phase_offset, 0.0), 10.0)
                    codes = list(anchor["codes"])
                    codes[2] = current_to_code(phase_ma, 2)
                    key = f"{segment}:{anchor_number}:{codes[2]}"
                    if args.resume and key in payload["phase_scans"]:
                        continue
                    reading = take_reading(
                        laser, meter, codes, center_nm,
                        settle_s=0.28, select_main_peak=True,
                    )
                    same_branch = abs(reading.wavelength_nm - base_wave) <= 0.75
                    accepted = bool(
                        same_branch and in_window(reading.wavelength_nm, center_nm)
                        and reading.power_mw >= MIN_POWER_MW
                        and reading.side_mode_suppression_db is not None
                        and reading.side_mode_suppression_db >= MIN_SMSR_DB
                    )
                    row = {
                        "segment": segment,
                        "center_nm": center_nm,
                        "anchor_seed_index": anchor["seed_index"],
                        "phase_offset_ma": phase_offset,
                        "codes": codes,
                        "currents_ma": [code_to_current(code, channel)
                                        for channel, code in enumerate(codes)],
                        "same_branch": same_branch,
                        "accepted": accepted,
                        "reading": asdict(reading),
                    }
                    payload["phase_scans"][key] = row
                    save_json(args.output, payload)
                    print(
                        f"seg={segment} anchor={anchor_number} phase={phase_ma:.3f} "
                        f"wave={reading.wavelength_nm:.6f} power={reading.power_mw:.3f} "
                        f"SMSR={reading.side_mode_suppression_db!s} accepted={accepted}",
                        flush=True,
                    )
    finally:
        try:
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    accepted_scans = [row for row in payload["phase_scans"].values()
                      if row.get("accepted")]
    selected_rows = []
    segments = []
    for segment, center_nm in enumerate(FBG_CENTERS_NM):
        segment_pool = [row for row in base if in_window(
            row["reading"]["wavelength_nm"], center_nm
        )]
        segment_pool.extend(row for row in accepted_scans if int(row["segment"]) == segment)
        chosen = select_points(segment_pool, center_nm)
        start_index = len(selected_rows)
        for row in chosen:
            selected_rows.append({
                "index": len(selected_rows),
                "segment": segment,
                "fbg_center_nm": center_nm,
                "selection_wavelength_nm": float(row["reading"]["wavelength_nm"]),
                "selection_power_mw": float(row["reading"]["power_mw"]),
                "selection_smsr_db": float(row["reading"]["side_mode_suppression_db"]),
                "codes": [int(value) for value in row["codes"]],
                "currents_ma": [float(value) for value in row["currents_ma"]],
                "source": row.get("source", "local_phase_scan"),
            })
        segments.append({
            "segment": segment,
            "center_nm": center_nm,
            "start_index": start_index,
            "stop_index": len(selected_rows),
        })
    powers = [row["selection_power_mw"] for row in selected_rows]
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "fresh_135mA_single_mode_stress_candidates",
        "old_69_wavelengths_used": False,
        "point_count": len(selected_rows),
        "segments": segments,
        "rows": selected_rows,
        "summary": {
            "power_min_mw": min(powers),
            "power_median_mw": statistics.median(powers),
            "power_max_mw": max(powers),
            "minimum_smsr_db": min(row["selection_smsr_db"] for row in selected_rows),
        },
    }
    save_json(args.selected, result)
    payload["summary"] = {
        "phase_measurements": len(payload["phase_scans"]),
        "accepted_phase_measurements": len(accepted_scans),
    }
    save_json(args.output, payload)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
