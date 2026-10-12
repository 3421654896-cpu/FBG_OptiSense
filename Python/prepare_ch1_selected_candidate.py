"""Build the strict nine-FBG candidate used by the final hardware calibration.

The default selection keeps the measured-template 45-point plan and replaces
only points that the AQ6150 hardware ceiling audit proved unsuitable.  DAC rows
are copied from the audited 2001-point table; this module never interpolates a
laser command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from laser_dac_safety import validate_pi11210_codes


DEFAULT_GROUPS = (
    (118, 122, 141, 170, 178),
    (333, 341, 357, 365, 386),
    (538, 551, 567, 584, 589),
    (724, 750, 752, 764, 784),
    (943, 957, 960, 961, 997),
    (1110, 1123, 1137, 1158, 1174),
    (1307, 1313, 1334, 1356, 1362),
    (1503, 1524, 1539, 1547, 1569),
    (1721, 1735, 1744, 1757, 1776),
)
ROLES = (
    "left_baseline",
    "left_shift_slope",
    "amplitude_peak",
    "right_shift_slope",
    "right_baseline",
)
ACTUAL_MAX_CODES = (58981, 58981, 32767, 24575, 24575)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_for_index(rows: dict[str, Any], index: int) -> dict[str, Any]:
    key = f"{1525.0 + 0.02 * index:.6f}"
    try:
        return rows[key]
    except KeyError as exc:
        raise ValueError(f"2001点审计表缺少索引{index} / {key} nm") from exc


def build_candidate(
    operational_path: Path,
    *,
    groups: tuple[tuple[int, ...], ...] = DEFAULT_GROUPS,
) -> dict[str, Any]:
    source = json.loads(operational_path.read_text(encoding="utf-8"))
    rows = source.get("rows")
    if not isinstance(rows, dict) or len(rows) != 2001:
        raise ValueError("只接受完整的2001点审计表")
    if len(groups) != 9 or any(len(group) != 5 for group in groups):
        raise ValueError("CH1实时计划必须恰好为9峰×5点")

    flattened = tuple(index for group in groups for index in group)
    if len(flattened) != 45 or len(set(flattened)) != 45:
        raise ValueError("45点索引必须完整且不重复")
    if tuple(sorted(flattened)) != flattened:
        raise ValueError("45点必须按波长严格递增")

    peaks: list[dict[str, Any]] = []
    for peak_number, indices in enumerate(groups, start=1):
        points: list[dict[str, Any]] = []
        for within_peak_order, (index, role) in enumerate(
            zip(indices, ROLES, strict=True)
        ):
            source_row = _row_for_index(rows, index)
            raw_codes = tuple(int(value) for value in source_row["codes"])
            if len(raw_codes) != 5:
                raise ValueError(f"索引{index}的DAC行不是5路")
            codes = tuple(
                min(code, ACTUAL_MAX_CODES[channel])
                for channel, code in enumerate(raw_codes)
            )
            validate_pi11210_codes(codes)
            points.append(
                {
                    "plan_order": (peak_number - 1) * 5 + within_peak_order,
                    "within_peak_order": within_peak_order,
                    "source_index": index,
                    "role": role,
                    "target_wavelength_nm": float(source_row["target_nm"]),
                    "dac_codes": list(codes),
                }
            )
        peaks.append({"peak_number": peak_number, "fbg": f"G{peak_number}", "points": points})

    return {
        "schema": "ch1_hardware_ceiling_constrained_candidate/v1",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "channel": 1,
        "expected_fbg_count": 9,
        "points_per_peak": 5,
        "total_points": 45,
        "selection_method": (
            "measured-template anchors with AQ6150 clean-ceiling constrained replacements"
        ),
        "source": {
            "operational_table": str(operational_path),
            "operational_table_sha256": _sha256(operational_path),
        },
        "replacements": [
            {
                "removed_fullband_index": 163,
                "replacement_fullband_index": 178,
                "reason": "raise common clean power while retaining index170 as right slope",
            },
            {
                "removed_fullband_index": 1770,
                "replacement_fullband_index": 1776,
                "reason": "removed point requires PHASE above the 10mA hard limit",
            },
        ],
        "selected_source_indices": list(flattened),
        "peaks": peaks,
        "writes_runtime_table": False,
        "writes_firmware": False,
        "hardware_sparse_sequence_validation": "pending",
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operational-json",
        type=Path,
        default=Path("fullband_equal_power_operational_2001.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ch1_selected_candidate_45_v2_20260905.json"),
    )
    args = parser.parse_args()
    payload = build_candidate(args.operational_json.resolve())
    _atomic_write(args.output.resolve(), payload)
    print(
        f"candidate={args.output.resolve()} points={payload['total_points']} "
        f"peaks={payload['expected_fbg_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
