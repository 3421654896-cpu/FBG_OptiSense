"""Select a 100-point, FBG-fit-oriented PI11210 table from extracted total-table data.

The source workbook is read with the spreadsheet artifact runtime and exported
to JSON before this script is called.  Selection uses wavelength-meter repeat
agreement, single-mode quality, wavelength accuracy, and within-segment DAC
continuity.  It never overwrites the active firmware or YAML table.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "JDSU"
ACTIVE_DAC_C = PROJECT / "Core" / "Src" / "dac_const.c"
ACTIVE_WAVE_YAML = HERE / "wave_const.yaml"
DEFAULT_VALIDATION = HERE / "fbg_monotonic_gaussian_validation.json"
DEFAULT_OUTPUT_JSON = HERE / "fbg_sampling_selection.json"
DEFAULT_OUTPUT_YAML = HERE / "wave_const_candidate.yaml"
DEFAULT_OUTPUT_C = HERE / "dac_const_candidate_uncalibrated.c"

PI11210_FULL_SCALE_MA = (150.0, 150.0, 20.0, 80.0, 80.0)
LASER_MAX_MA = (125.0, 125.0, 10.0, 30.0, 30.0)


@dataclass(frozen=True)
class Candidate:
    source_excel_row: int
    wavelength_nm: float
    repeat_wavelength_nm: float
    repeat_error_pm: float
    historical_power_mw: float
    secondary_power_mw: float
    currents_ma: tuple[float, float, float, float, float]
    codes: tuple[int, int, int, int, int]


def finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_candidates(source_json: Path) -> list[Candidate]:
    rows = json.loads(source_json.read_text(encoding="utf-8"))
    candidates: list[Candidate] = []
    for excel_row, row in enumerate(rows[1:], start=2):
        if len(row) < 22:
            continue
        repeat_wave = finite(row[12])
        wave = finite(row[18])
        power = finite(row[19])
        secondary_power = finite(row[21]) or 0.0
        currents = tuple(finite(value) for value in row[1:6])
        code_values = tuple(finite(value) for value in row[6:11])
        if (
            wave is None
            or repeat_wave is None
            or power is None
            or any(value is None for value in currents)
            or any(value is None for value in code_values)
        ):
            continue
        currents = tuple(float(value) for value in currents)
        if any(
            current < 0.0 or current > maximum + 1e-6
            for current, maximum in zip(currents, LASER_MAX_MA)
        ):
            continue
        repeat_error_pm = abs(wave - repeat_wave) * 1000.0
        if not 1526.5 <= wave <= 1561.0 or repeat_error_pm > 20.0 or power <= 0.02:
            continue
        codes = tuple(int(round(float(value))) for value in code_values)
        candidates.append(
            Candidate(
                source_excel_row=excel_row,
                wavelength_nm=wave,
                repeat_wavelength_nm=repeat_wave,
                repeat_error_pm=repeat_error_pm,
                historical_power_mw=power,
                secondary_power_mw=secondary_power,
                currents_ma=currents,
                codes=codes,
            )
        )
    return sorted(candidates, key=lambda item: item.wavelength_nm)


def current_peak_centers(validation_path: Path) -> list[float]:
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    centers: list[float] = []
    for segment in range(9):
        channel = 2 if segment < 3 else (0 if segment < 6 else 1)
        peak = report["channels"][channel]["peaks"][segment]
        center = peak.get("stabilized", {}).get("median_nm")
        if center is None:
            section = report["segments"][segment]
            center = 0.5 * (section["start_nm"] + section["stop_nm"])
        centers.append(float(center))
    return centers


def desired_offsets(point_count: int) -> np.ndarray:
    if point_count == 12:
        return np.array(
            [-0.52, -0.42, -0.33, -0.25, -0.17, -0.08,
              0.08, 0.17, 0.25, 0.33, 0.42, 0.52],
            dtype=float,
        )
    return np.array(
        [-0.52, -0.40, -0.31, -0.23, -0.14, 0.0,
          0.14, 0.23, 0.31, 0.40, 0.52],
        dtype=float,
    )


def point_cost(candidate: Candidate, target_nm: float) -> float:
    wavelength_cost = ((candidate.wavelength_nm - target_nm) / 0.010) ** 2
    repeat_cost = 0.20 * (candidate.repeat_error_pm / 5.0) ** 2
    secondary_ratio = candidate.secondary_power_mw / max(
        candidate.historical_power_mw, 1e-6
    )
    secondary_cost = 20.0 * max(secondary_ratio - 0.01, 0.0) ** 2
    return wavelength_cost + repeat_cost + secondary_cost


def transition_cost(previous: Candidate, current: Candidate) -> float:
    # GAIN/SOA will be recalibrated later.  Penalize only tuning-section jumps.
    normalized = [
        abs(current.codes[2] - previous.codes[2]) / 32768.0,
        abs(current.codes[3] - previous.codes[3]) / 24576.0,
        abs(current.codes[4] - previous.codes[4]) / 24576.0,
    ]
    return 0.35 * sum(value * value for value in normalized)


def select_segment(candidates: list[Candidate], targets: np.ndarray) -> list[Candidate]:
    layers: list[list[Candidate]] = []
    for target in targets:
        nearby = [item for item in candidates if abs(item.wavelength_nm - target) <= 0.055]
        if len(nearby) < 4:
            nearby = sorted(candidates, key=lambda item: abs(item.wavelength_nm - target))[:12]
        layers.append(sorted(nearby, key=lambda item: item.wavelength_nm))

    costs: list[dict[int, tuple[float, int | None]]] = []
    first = {
        index: (point_cost(candidate, float(targets[0])), None)
        for index, candidate in enumerate(layers[0])
    }
    costs.append(first)
    for layer_index in range(1, len(layers)):
        layer_costs: dict[int, tuple[float, int | None]] = {}
        for current_index, current in enumerate(layers[layer_index]):
            best_cost = math.inf
            best_previous = None
            for previous_index, previous in enumerate(layers[layer_index - 1]):
                if previous_index not in costs[-1]:
                    continue
                if current.wavelength_nm <= previous.wavelength_nm + 0.004:
                    continue
                candidate_cost = (
                    costs[-1][previous_index][0]
                    + point_cost(current, float(targets[layer_index]))
                    + transition_cost(previous, current)
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_previous = previous_index
            if best_previous is not None:
                layer_costs[current_index] = (best_cost, best_previous)
        if not layer_costs:
            raise RuntimeError(f"No increasing candidate path at target {targets[layer_index]:.3f} nm")
        costs.append(layer_costs)

    final_index = min(costs[-1], key=lambda index: costs[-1][index][0])
    selected = [layers[-1][final_index]]
    for layer_index in range(len(layers) - 1, 0, -1):
        previous_index = costs[layer_index][final_index][1]
        assert previous_index is not None
        final_index = previous_index
        selected.append(layers[layer_index - 1][final_index])
    selected.reverse()
    return selected


def write_candidate_c(template: Path, rows: list[list[int]], destination: Path) -> None:
    text = template.read_text(encoding="utf-8")
    row_re = re.compile(r"\{\s*([^{}]+?)\s*\},?")
    valid_index = 0

    def replace_row(match: re.Match) -> str:
        nonlocal valid_index
        fields = [part.strip() for part in match.group(1).split(",")]
        try:
            values = [int(field, 0) for field in fields]
        except ValueError:
            return match.group(0)
        if len(values) != 5 or values[:3] == [0xFFFF, 0xFFFF, 0xFFFF]:
            return match.group(0)
        if valid_index >= len(rows):
            return match.group(0)
        row = rows[valid_index]
        valid_index += 1
        return "{" + ", ".join(str(value) for value in row) + "},"

    destination.write_text(row_re.sub(replace_row, text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT_YAML)
    parser.add_argument("--output-c", type=Path, default=DEFAULT_OUTPUT_C)
    args = parser.parse_args()

    candidates = load_candidates(args.source_json)
    centers = current_peak_centers(args.validation)
    all_selected: list[Candidate] = []
    selections = []
    for segment, center in enumerate(centers):
        point_count = 12 if segment == 1 else 11
        targets = center + desired_offsets(point_count)
        selected = select_segment(candidates, targets)
        all_selected.extend(selected)
        selections.append(
            {
                "segment": segment,
                "current_center_nm": center,
                "point_count": point_count,
                "points": [
                    {
                        "desired_nm": float(target),
                        "selection_error_pm": (item.wavelength_nm - target) * 1000.0,
                        **asdict(item),
                    }
                    for target, item in zip(targets, selected)
                ],
            }
        )

    if len(all_selected) != 100:
        raise RuntimeError(f"Expected 100 selected rows, got {len(all_selected)}")
    wavelengths = [round(item.wavelength_nm, 3) for item in all_selected]
    if any(right <= left for segment in selections for left, right in zip(
        [point["wavelength_nm"] for point in segment["points"]][:-1],
        [point["wavelength_nm"] for point in segment["points"]][1:],
    )):
        raise RuntimeError("Selected wavelength path is not strictly increasing within a segment")

    output = {
        "source_workbook": "JDSU_V1.2_总表_带公式_全值_filtered.xlsx",
        "candidate_count_after_safety_and_repeat_filters": len(candidates),
        "selection_rules": {
            "repeat_error_max_pm": 20.0,
            "target_power_mw_after_recalibration": 0.5,
            "segment_1_extra_point": True,
            "monotonic_single_peak_fit": True,
        },
        "segments": selections,
    }
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    yaml_rows = [[int(math.floor(wave)), int(round((wave % 1.0) * 1000.0))]
                 for wave in wavelengths]
    args.output_yaml.write_text(
        yaml.safe_dump({"Wave_DATA": yaml_rows}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_candidate_c(
        ACTIVE_DAC_C,
        [list(item.codes) for item in all_selected],
        args.output_c,
    )

    errors = [
        abs(point["selection_error_pm"])
        for segment in selections
        for point in segment["points"]
    ]
    repeats = [item.repeat_error_pm for item in all_selected]
    print(
        f"candidates={len(candidates)} selected={len(all_selected)} "
        f"selection_p95_pm={np.percentile(errors, 95):.3f} "
        f"repeat_p95_pm={np.percentile(repeats, 95):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
