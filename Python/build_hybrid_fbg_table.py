"""Build a 100-point table only from wavelength/power-validated rows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from laser_power_calibration import write_proposed_c


HERE = Path(__file__).resolve().parent
ACTIVE_DAC = HERE.parent / "JDSU" / "Core" / "Src" / "dac_const.c"


@dataclass(frozen=True)
class VerifiedPoint:
    key: str
    source: str
    source_index: int
    wavelength_nm: float
    target_nm: float
    power_mw: float
    wavelength_error_pm: float
    codes: tuple[int, int, int, int, int]


def load_verified(report_path: Path, source: str) -> list[VerifiedPoint]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    points = []
    for key, result in report.get("results", {}).items():
        if not result.get("success"):
            continue
        final = result.get("final", {})
        wave = float(final["wavelength_nm"])
        power = float(final["power_mw"])
        error = float(final["wavelength_error_pm"])
        codes = tuple(int(value) for value in result["calibrated_codes"])
        if (
            abs(power - 0.5) <= 0.0065
            and abs(error) <= 5.0
            and len(codes) == 5
        ):
            points.append(
                VerifiedPoint(
                    key=f"{source}:{key}",
                    source=source,
                    source_index=int(result["index"]),
                    wavelength_nm=wave,
                    target_nm=float(result["target_nm"]),
                    power_mw=power,
                    wavelength_error_pm=error,
                    codes=codes,
                )
            )
    return points


def transition_cost(left: VerifiedPoint, right: VerifiedPoint) -> float:
    tuning_cost = (
        ((right.codes[2] - left.codes[2]) / 32768.0) ** 2
        + ((right.codes[3] - left.codes[3]) / 24576.0) ** 2
        + ((right.codes[4] - left.codes[4]) / 24576.0) ** 2
    )
    source_switch = 0.04 if left.source != right.source else 0.0
    return 0.30 * tuning_cost + source_switch


def point_cost(point: VerifiedPoint, target_nm: float) -> float:
    # Prefer rows calibrated in the current session.  Existing-table rows are
    # reliable wavelength fallbacks, but their old 0.5 mW power state must be
    # recalibrated under the present optical setup.
    existing_power_recalibration_cost = 100.0 if point.source == "existing_table" else 0.0
    return ((point.wavelength_nm - target_nm) / 0.010) ** 2 + existing_power_recalibration_cost


def choose_segment(pool: list[VerifiedPoint], targets: list[float]) -> list[VerifiedPoint]:
    layers: list[list[VerifiedPoint]] = []
    for target in targets:
        nearby = [point for point in pool if abs(point.wavelength_nm - target) <= 0.070]
        if len(nearby) < 4:
            nearby = sorted(pool, key=lambda point: abs(point.wavelength_nm - target))[:16]
        layers.append(sorted(nearby, key=lambda point: point.wavelength_nm))

    costs: list[dict[int, tuple[float, int | None]]] = []
    costs.append(
        {
            index: (point_cost(point, targets[0]), None)
            for index, point in enumerate(layers[0])
        }
    )
    for layer_index in range(1, len(layers)):
        layer_costs: dict[int, tuple[float, int | None]] = {}
        for current_index, current in enumerate(layers[layer_index]):
            best = (math.inf, None)
            for previous_index, previous in enumerate(layers[layer_index - 1]):
                if previous_index not in costs[-1]:
                    continue
                if current.wavelength_nm <= previous.wavelength_nm + 0.004:
                    continue
                score = (
                    costs[-1][previous_index][0]
                    + point_cost(current, targets[layer_index])
                    + transition_cost(previous, current)
                )
                if score < best[0]:
                    best = (score, previous_index)
            if best[1] is not None:
                layer_costs[current_index] = best
        if not layer_costs:
            raise RuntimeError(f"No verified path around {targets[layer_index]:.3f} nm")
        costs.append(layer_costs)

    current_index = min(costs[-1], key=lambda index: costs[-1][index][0])
    chosen = [layers[-1][current_index]]
    for layer_index in range(len(layers) - 1, 0, -1):
        previous_index = costs[layer_index][current_index][1]
        assert previous_index is not None
        current_index = previous_index
        chosen.append(layers[layer_index - 1][current_index])
    chosen.reverse()
    return chosen


def main() -> int:
    candidate_report = HERE / "fbg_candidate_calibration_report.json"
    existing_report = HERE / "laser_calibration_report.json"
    design = json.loads((HERE / "fbg_sampling_selection.json").read_text(encoding="utf-8"))
    pool = load_verified(candidate_report, "new_candidate")
    pool.extend(load_verified(existing_report, "existing_table"))

    selected: list[VerifiedPoint] = []
    segments = []
    for segment in design["segments"]:
        targets = [float(point["desired_nm"]) for point in segment["points"]]
        center = float(segment["current_center_nm"])
        local_pool = [point for point in pool if abs(point.wavelength_nm - center) <= 0.75]
        chosen = choose_segment(local_pool, targets)
        selected.extend(chosen)
        segments.append(
            {
                "segment": int(segment["segment"]),
                "current_center_nm": center,
                "points": [
                    {
                        "desired_nm": target,
                        "selection_error_pm": (point.wavelength_nm - target) * 1000.0,
                        "source": point.source,
                        "source_index": point.source_index,
                        "wavelength_nm": point.wavelength_nm,
                        "power_mw": point.power_mw,
                        "wavelength_error_pm": point.wavelength_error_pm,
                        "codes": list(point.codes),
                    }
                    for target, point in zip(targets, chosen)
                ],
            }
        )

    if len(selected) != 100 or len({point.key for point in selected}) != 100:
        raise RuntimeError("Hybrid selection did not produce 100 unique verified points")

    output = {
        "point_count": len(selected),
        "verified_pool_count": len(pool),
        "target_power_mw": 0.5,
        "segments": segments,
    }
    (HERE / "fbg_hybrid_selection.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    yaml_rows = []
    for point in selected:
        rounded = round(point.wavelength_nm, 3)
        integer = int(math.floor(rounded))
        fraction = int(round((rounded - integer) * 1000.0))
        if fraction == 1000:
            integer += 1
            fraction = 0
        yaml_rows.append([integer, fraction])
    (HERE / "wave_const_hybrid.yaml").write_text(
        yaml.safe_dump({"Wave_DATA": yaml_rows}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_proposed_c(
        ACTIVE_DAC,
        [list(point.codes) for point in selected],
        HERE / "dac_const_hybrid_calibrated.c",
    )

    errors = [
        abs(point["selection_error_pm"])
        for segment in segments
        for point in segment["points"]
    ]
    sources = {name: sum(point.source == name for point in selected)
               for name in ("new_candidate", "existing_table")}
    print(
        f"selected=100 new={sources['new_candidate']} existing={sources['existing_table']} "
        f"target_error_p95_pm={np.percentile(errors, 95):.3f} "
        f"target_error_max_pm={max(errors):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
