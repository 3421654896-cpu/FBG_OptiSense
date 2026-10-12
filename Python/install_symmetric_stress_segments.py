"""Install the replay-qualified symmetric FBG-3/FBG-7 replacements.

The two proposal files are kept as immutable calibration evidence.  This tool
backs up the currently installed JSON/YAML tables, creates a versioned merged
table, updates the two GUI wavelength files, and generates the matching C
table.  It refuses installation if optical or replay validation is missing.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from prepare_dynamic_stress_training import install_table


HERE = Path(__file__).resolve().parent
CURRENT = HERE / "stress_shape_power_dynamic_v3.json"
S3_PROPOSAL = HERE / "stress_symmetric_fbg3_wide_proposal.json"
S7_PROPOSAL = HERE / "stress_symmetric_segments_3_7_proposal.json"
S3_REPLAY = HERE / "stress_symmetric_fbg3_wide_replay_validation.json"
S7_REPLAY = HERE / "stress_symmetric_fbg7_replay_validation.json"
VERSIONED = HERE / "stress_shape_power_symmetric_v4.json"
YAML_FILES = (
    HERE / "stress_wave_const.yaml",
    HERE / "stress_wave_const_v3.yaml",
)
BACKUP_DIR = HERE / "calibration_backups"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validated_rows(proposal_path: Path, replay_path: Path, segment: int) -> list[dict]:
    proposal = load(proposal_path)
    replay = load(replay_path)
    rows = [row for row in proposal["rows"] if int(row["segment"]) == segment]
    if len(rows) != 7 or int(replay.get("summary", {}).get("valid_count", 0)) != 10:
        raise RuntimeError(f"FBG {segment + 1} lacks ten valid replay fits")
    if float(replay["summary"]["minimum_r_squared"]) < 0.88:
        raise RuntimeError(f"FBG {segment + 1} replay R2 is below 0.88")
    ch1 = np.median(np.asarray(replay["traces_ch1_v"], dtype=float), axis=0)
    converted = []
    for local_index, (row, ch1_v) in enumerate(zip(rows, ch1)):
        currents = [float(value) for value in row["currents_ma"]]
        limits = (135.0, 135.0, 10.0, 30.0, 30.0)
        if any(value > limit + 0.002 for value, limit in zip(currents, limits)):
            raise RuntimeError(f"FBG {segment + 1} point {local_index + 1} exceeds current limit")
        if not bool(row["passed"]):
            raise RuntimeError(f"FBG {segment + 1} point {local_index + 1} did not pass")
        if float(row["final_smsr_min_db"]) < 25.0:
            raise RuntimeError(f"FBG {segment + 1} point {local_index + 1} SMSR below 25 dB")
        if not 1.44 <= float(row["final_power_mw"]) <= 1.56:
            raise RuntimeError(f"FBG {segment + 1} point {local_index + 1} power out of range")
        converted.append({
            "segment": segment,
            "wavelength_nm": float(row["final_wavelength_nm"]),
            "power_mw": float(row["final_power_mw"]),
            "ch1_v": float(ch1_v),
            "training_dynamic_ch1_v": float(ch1_v),
            "training_dynamic_std_v": float(np.std(
                np.asarray(replay["traces_ch1_v"], dtype=float)[:, local_index]
            )),
            "training_abs_error_v": 0.0,
            "codes": [int(value) for value in row["codes"]],
            "currents_ma": currents,
            "smsr_min_db": float(row["final_smsr_min_db"]),
            "wavelength_span_pm": float(row["final_wavelength_span_pm"]),
            "source_pool": proposal_path.name,
            "symmetry_target_nm": float(row["target_nm"]),
        })
    return converted


def write_yaml(path: Path, rows: list[dict]) -> None:
    lines = ["Wave_DATA:"]
    for row in rows:
        wavelength = float(row["wavelength_nm"])
        integer = int(np.floor(wavelength))
        fractional_pm = (wavelength - integer) * 1000.0
        lines.append(f"  - [{integer}, {fractional_pm:.3f}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    current = load(CURRENT)
    s3 = validated_rows(S3_PROPOSAL, S3_REPLAY, 2)
    s7 = validated_rows(S7_PROPOSAL, S7_REPLAY, 6)
    replacements = {2: s3, 6: s7}
    merged = []
    for segment in range(9):
        section = replacements.get(segment)
        if section is None:
            section = [
                dict(row) for row in current["rows"]
                if int(row["segment"]) == segment
            ]
        if len(section) != 7:
            raise RuntimeError(f"segment {segment + 1} does not contain seven rows")
        section.sort(key=lambda row: float(row["wavelength_nm"]))
        for row in section:
            row["index"] = len(merged)
            row["segment"] = segment
            merged.append(row)
    if len(merged) != 63:
        raise RuntimeError(f"expected 63 rows, got {len(merged)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for path in (CURRENT, *YAML_FILES):
        shutil.copy2(path, BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}")

    powers = np.asarray([float(row["power_mw"]) for row in merged])
    payload = dict(current)
    payload.update({
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "63-point stress table with symmetric replay-qualified FBG 3 and 7",
        "point_count": 63,
        "rows": merged,
        "symmetric_replacements": {
            "2": {"proposal": S3_PROPOSAL.name, "replay": S3_REPLAY.name},
            "6": {"proposal": S7_PROPOSAL.name, "replay": S7_REPLAY.name},
        },
        "power_summary_mw": {
            "minimum": float(np.min(powers)),
            "mean": float(np.mean(powers)),
            "maximum": float(np.max(powers)),
            "coefficient_of_variation_percent": float(
                np.std(powers) / np.mean(powers) * 100.0
            ),
        },
    })
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    VERSIONED.write_text(text, encoding="utf-8", newline="\n")
    CURRENT.write_text(text, encoding="utf-8", newline="\n")
    for yaml_path in YAML_FILES:
        write_yaml(yaml_path, merged)
    install_table(payload, VERSIONED.name)
    print(json.dumps({
        "installed": True,
        "point_count": len(merged),
        "FBG3_nm": [row["wavelength_nm"] for row in s3],
        "FBG7_nm": [row["wavelength_nm"] for row in s7],
        "backup_stamp": stamp,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
