"""Select a monotone F45 indentation-depth proxy using training rounds only.

The current hardware has no force reference.  This module therefore estimates
the commanded CNC indentation depth, not force.  Candidate scalar responses
are evaluated by holding out each complete 0.30 mm acquisition round.  The
0.10 and 0.80 mm endpoint rounds define the interpolation range; later 0.20
and 0.60 mm rounds remain untouched for locked validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from evaluate_f45_cross_session_models import load_rounds
from select_f45_multidepth_position_candidate import load_depth_manifest


DEPTHS = (0.10, 0.30, 0.80)


def response(record: dict, name: str) -> float:
    blocks = np.asarray(
        record["features"]["pair_transition_delta45"], dtype=float
    ).reshape(3, 45)
    first, second, transition = blocks
    if name == "first_rms":
        return float(np.sqrt(np.mean(first**2)))
    if name == "second_rms":
        return float(np.sqrt(np.mean(second**2)))
    if name == "pair_rms":
        return float(np.sqrt(np.mean(np.concatenate((first, second)) ** 2)))
    if name == "first_second_mean_rms":
        return float((np.sqrt(np.mean(first**2)) + np.sqrt(np.mean(second**2))) / 2.0)
    if name == "all_block_rms":
        return float(np.sqrt(np.mean(blocks**2)))
    if name == "first_median_abs":
        return float(np.median(np.abs(first)))
    if name == "second_median_abs":
        return float(np.median(np.abs(second)))
    raise ValueError(name)


def fit_curve(records: list[dict], point: int, name: str) -> dict:
    grouped = {}
    for depth in DEPTHS:
        values = [
            response(row, name)
            for row in records
            if row["point_index"] == point
            and abs(float(row["depth_mm"]) - depth) <= 1e-9
        ]
        if not values:
            raise ValueError(f"P{point}: no {depth:.2f} mm training response")
        grouped[depth] = values
    raw = np.asarray([np.median(grouped[depth]) for depth in DEPTHS], dtype=float)
    weights = np.asarray([len(grouped[depth]) for depth in DEPTHS], dtype=float)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        np.asarray(DEPTHS, dtype=float), raw, sample_weight=weights
    )
    fitted = np.asarray(model.predict(np.asarray(DEPTHS, dtype=float)), dtype=float)
    return {
        "point": point,
        "depth_knots_mm": list(DEPTHS),
        "raw_response_knots": raw.tolist(),
        "isotonic_response_knots": fitted.tolist(),
        "replicates_per_knot": weights.astype(int).tolist(),
        "monotonic_adjustment_rms": float(np.sqrt(np.mean((fitted - raw) ** 2))),
    }


def invert_curve(curve: dict, value: float) -> float:
    depths = np.asarray(curve["depth_knots_mm"], dtype=float)
    responses = np.asarray(curve["isotonic_response_knots"], dtype=float)
    # Isotonic pooling can create flat adjacent knots.  Collapse each plateau
    # to its mean depth so inverse interpolation is deterministic.
    unique = []
    for level in np.unique(responses):
        mask = np.isclose(responses, level, rtol=0.0, atol=1e-12)
        unique.append((float(level), float(np.mean(depths[mask]))))
    xp = np.asarray([item[0] for item in unique], dtype=float)
    fp = np.asarray([item[1] for item in unique], dtype=float)
    if len(xp) == 1:
        return float(fp[0])
    return float(np.interp(float(value), xp, fp, left=fp[0], right=fp[-1]))


def evaluate_variant(rounds: dict[str, list[dict]], name: str) -> dict:
    fold_rows = []
    selection_names = [key for key in rounds if key.startswith("old_")]
    for held_name in selection_names:
        training = [
            row for round_name, values in rounds.items()
            if round_name != held_name for row in values
        ]
        curves = {point: fit_curve(training, point, name) for point in range(1, 18)}
        predictions = []
        for row in rounds[held_name]:
            value = response(row, name)
            predicted = invert_curve(curves[int(row["point_index"])], value)
            predictions.append({
                "point": int(row["point_index"]),
                "truth_depth_mm": float(row["depth_mm"]),
                "response": value,
                "predicted_depth_mm": predicted,
                "absolute_error_mm": abs(predicted - float(row["depth_mm"])),
            })
        errors = np.asarray([row["absolute_error_mm"] for row in predictions])
        fold_rows.append({
            "held_round": held_name,
            "mean_absolute_error_mm": float(np.mean(errors)),
            "p95_absolute_error_mm": float(np.percentile(errors, 95)),
            "maximum_absolute_error_mm": float(np.max(errors)),
            "within_0p10mm_percent": float(100.0 * np.mean(errors <= 0.10 + 1e-12)),
            "predictions": predictions,
        })

    all_rows = [row for values in rounds.values() for row in values]
    curves = {point: fit_curve(all_rows, point, name) for point in range(1, 18)}
    endpoint_separation = []
    for point, curve in curves.items():
        low, _, high = curve["isotonic_response_knots"]
        endpoint_separation.append(float(high - low))
    return {
        "response": name,
        "folds": fold_rows,
        "mean_loro_mae_mm": float(np.mean([row["mean_absolute_error_mm"] for row in fold_rows])),
        "worst_loro_mae_mm": float(max(row["mean_absolute_error_mm"] for row in fold_rows)),
        "worst_loro_p95_mm": float(max(row["p95_absolute_error_mm"] for row in fold_rows)),
        "worst_loro_maximum_mm": float(max(row["maximum_absolute_error_mm"] for row in fold_rows)),
        "worst_loro_within_0p10mm_percent": float(min(row["within_0p10mm_percent"] for row in fold_rows)),
        "strict_raw_monotone_points": int(sum(
            np.all(np.diff(np.asarray(curve["raw_response_knots"], dtype=float)) > 0)
            for curve in curves.values()
        )),
        "minimum_isotonic_endpoint_separation": float(min(endpoint_separation)),
        "curves": [curves[point] for point in range(1, 18)],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_0p10_manifest", type=Path)
    parser.add_argument("depth_0p80_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    here = Path(__file__).resolve().parent
    old = load_rounds(
        (here.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve(),
        [
            (here.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
            here / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
            here / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
        ],
    )
    rounds = {
        "depth_0p10_r1": load_depth_manifest(args.depth_0p10_manifest),
        **old,
        "depth_0p80_r1": load_depth_manifest(args.depth_0p80_manifest),
    }
    ranked = [
        evaluate_variant(rounds, name)
        for name in (
            "first_rms", "second_rms", "pair_rms", "first_second_mean_rms",
            "all_block_rms", "first_median_abs", "second_median_abs",
        )
    ]
    ranked.sort(key=lambda row: (
        row["worst_loro_p95_mm"], row["worst_loro_maximum_mm"],
        row["mean_loro_mae_mm"], -row["strict_raw_monotone_points"],
        -row["minimum_isotonic_endpoint_separation"],
    ))
    result = {
        "schema": "f45_monotone_depth_proxy_candidate_selection_v1",
        "target": "commanded_cnc_indentation_depth_mm_not_force",
        "training_depths_mm": list(DEPTHS),
        "training_contacts": sum(len(values) for values in rounds.values()),
        "random_frame_split_used": False,
        "selection_protocol": "leave one complete 0.30 mm acquisition round out",
        "untouched_validation_required": "complete 17-point rounds at 0.20 and 0.60 mm captured after lock",
        "force_ground_truth_present": False,
        "deployment_eligible": False,
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    summary = [{key: value for key, value in row.items() if key not in {"folds", "curves"}} for row in ranked]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
