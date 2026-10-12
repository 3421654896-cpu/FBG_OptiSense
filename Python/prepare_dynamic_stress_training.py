"""Build/install a reproducible post-power stress candidate table.

The long table is used only while learning fast-scan transition behaviour.  A
selected JSON table can also be installed later with --table-source.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FINAL_V2 = HERE / "stress_shape_power_final_v2.json"
POOL_MAIN = HERE / "uniform_power_candidate_pool.json"
POOL_EXTRA = HERE / "uniform_power_candidate_pool_extra.json"
DEFAULT_OUTPUT = HERE / "stress_dynamic_training_pool.json"
TABLE_C = ROOT / "JDSU" / "Core" / "Src" / "stress_table.c"
TABLE_H = ROOT / "JDSU" / "Core" / "Inc" / "stress_table.h"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pool() -> dict:
    final_rows = _load(FINAL_V2)["rows"]
    main_rows = _load(POOL_MAIN)["rows"]
    extra_rows = _load(POOL_EXTRA)["rows"]
    rows: list[dict] = []
    for segment in range(9):
        if segment in (0, 1, 5):
            source = [row for row in final_rows if int(row["segment"]) == segment]
            source_name = FINAL_V2.name
        else:
            pool = extra_rows if segment in (6, 8) else main_rows
            source = [
                row for row in pool
                if int(row["segment"]) == segment and bool(row["accepted"])
            ]
            source_name = POOL_EXTRA.name if segment in (6, 8) else POOL_MAIN.name
        for row in sorted(source, key=lambda item: float(item["wavelength_nm"])):
            currents = [float(value) for value in row["currents_ma"]]
            limits = (135.0, 135.0, 10.0, 30.0, 30.0)
            if any(value > limit + 0.002 for value, limit in zip(currents, limits)):
                raise RuntimeError(f"current limit exceeded: segment {segment}")
            rows.append({
                "index": len(rows),
                "segment": segment,
                "wavelength_nm": float(row["wavelength_nm"]),
                "power_mw": float(row["power_mw"]),
                "ch1_v": float(row["ch1_v"]),
                "codes": [int(value) for value in row["codes"]],
                "currents_ma": currents,
                "smsr_min_db": float(row.get("smsr_min_db", 99.0)),
                "wavelength_span_pm": float(row.get("wavelength_span_pm", 0.0)),
                "source_pool": source_name,
            })
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "temporary_dynamic_transition_training_pool",
        "point_count": len(rows),
        "rows": rows,
    }


def install_table(payload: dict, source_name: str) -> None:
    rows = payload["rows"]
    if not rows:
        raise RuntimeError("empty stress table")
    segments = [int(row["segment"]) for row in rows]
    if segments != sorted(segments):
        raise RuntimeError("rows must be sorted by segment")
    last_start = segments.index(max(segments))
    header = f"""#ifndef __STRESS_TABLE_H
#define __STRESS_TABLE_H

#include <stdint.h>

#define STRESS_TABLE_POINT_COUNT {len(rows)}U
#define STRESS_LAST_SECTION_START_INDEX {last_start}U

extern const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5];
extern const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2];

#endif
"""
    dac_lines = []
    wave_lines = []
    for row in rows:
        codes = [int(value) for value in row["codes"]]
        if len(codes) != 5 or any(not 0 <= value <= 65535 for value in codes):
            raise RuntimeError("invalid DAC row")
        dac_lines.append("{" + ",".join(str(value) for value in codes) + "},")
        wavelength_pm = int(round(float(row["wavelength_nm"]) * 1000.0))
        wave_lines.append(f"{{{wavelength_pm // 1000},{wavelength_pm % 1000}}},")
    source = (
        '#include "stress_table.h"\n\n'
        f"/* Generated from {source_name}; {len(rows)} calibrated points. */\n"
        "const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5] = {\n"
        + "\n".join(dac_lines)
        + "\n};\n\n"
        "const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2] = {\n"
        + "\n".join(wave_lines)
        + "\n};\n"
    )
    TABLE_H.write_text(header, encoding="utf-8", newline="\n")
    TABLE_C.write_text(source, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--table-source", type=Path,
        help="install rows from an existing JSON instead of building the training pool",
    )
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    if args.table_source:
        payload = _load(args.table_source)
        source_name = args.table_source.name
    else:
        payload = build_pool()
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        source_name = args.output.name
    if args.install:
        install_table(payload, source_name)
    counts = {
        segment: sum(int(row["segment"]) == segment for row in payload["rows"])
        for segment in range(9)
    }
    print(json.dumps({"point_count": len(payload["rows"]), "segments": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
