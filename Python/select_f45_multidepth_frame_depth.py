"""Select a causal F45 commanded-indentation proxy with whole rounds held out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from select_f45_multidepth_position_candidate import load_depth_manifest


FEATURE = "pair_transition_delta45"


def transform_depth(values: np.ndarray, points: np.ndarray, variant: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    blocks = values.reshape(-1, 3, 45)
    norms = np.sqrt(np.mean(blocks**2, axis=2))
    segment = np.sqrt(np.mean(blocks.reshape(-1, 3, 9, 5) ** 2, axis=3)).reshape(-1, 27)
    if variant.startswith("raw"):
        result = values
    elif variant.startswith("signed_sqrt"):
        result = np.sign(values) * np.sqrt(np.abs(values))
    elif variant.startswith("engineered"):
        result = np.concatenate((values, norms, segment), axis=1)
    elif variant.startswith("shape_amplitude"):
        unit = blocks / np.maximum(np.linalg.norm(blocks, axis=2, keepdims=True), 1e-12)
        result = np.concatenate((unit.reshape(-1, 135), norms, segment), axis=1)
    else:
        raise ValueError(variant)
    if variant.endswith("_point"):
        one_hot = np.zeros((len(values), 17), dtype=float)
        one_hot[np.arange(len(values)), points.astype(int) - 1] = 1.0
        result = np.concatenate((result, one_hot), axis=1)
    return result


def candidates():
    variants = ("raw", "raw_point", "shape_amplitude", "shape_amplitude_point")
    for variant in variants:
        for leaf in (2, 4):
            yield variant, f"extra_trees_500_leaf{leaf}_sqrt", ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=leaf, max_features="sqrt",
                random_state=20260908, n_jobs=-1,
            )
        for leaf in (2, 4):
            yield variant, f"extra_trees_500_leaf{leaf}_half", ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=leaf, max_features=0.5,
                random_state=20260908, n_jobs=-1,
            )
        for alpha in (100.0, 1000.0):
            yield variant, f"ridge_a{alpha:g}", make_pipeline(
                StandardScaler(), Ridge(alpha=alpha)
            )
    # Histogram boosting is useful on the compact shape/amplitude form but is
    # expensive enough that it is not repeated for every representation.
    for variant in ("shape_amplitude", "shape_amplitude_point"):
        yield variant, "hist_gradient_leaf20", HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=20,
            min_samples_leaf=20, l2_regularization=1.0, random_state=20260908,
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-0p10", required=True, type=Path)
    parser.add_argument("--depth-0p20", required=True, type=Path)
    parser.add_argument("--depth-0p60", required=True, type=Path)
    parser.add_argument("--depth-0p80", required=True, type=Path)
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
        "depth_0p10_r1": load_depth_manifest(args.depth_0p10),
        "depth_0p20_r1": load_depth_manifest(args.depth_0p20),
        **old,
        "depth_0p60_r1": load_depth_manifest(args.depth_0p60),
        "depth_0p80_r1": load_depth_manifest(args.depth_0p80),
    }
    causal = {
        (name, int(row["point_index"])): causal_held_features(Path(row["path"]))[FEATURE]
        for name, records in rounds.items() for row in records
    }
    selection_names = [
        name for name in rounds
        if name not in {"depth_0p10_r1", "depth_0p80_r1"}
    ]
    ranked = []
    for variant, estimator_name, template in candidates():
        folds = []
        for held_name in selection_names:
            train_parts = []
            train_points = []
            train_depths = []
            for name, records in rounds.items():
                if name == held_name:
                    continue
                for row in records:
                    point = int(row["point_index"])
                    matrix = causal[(name, point)]
                    train_parts.append(matrix)
                    train_points.append(np.full(len(matrix), point, dtype=int))
                    train_depths.append(np.full(len(matrix), float(row["depth_mm"])))
            raw_train = np.concatenate(train_parts)
            point_train = np.concatenate(train_points)
            y_train = np.concatenate(train_depths)
            model = clone(template).fit(
                transform_depth(raw_train, point_train, variant), y_train
            )
            frame_truth = []
            frame_predicted = []
            trials = []
            for row in rounds[held_name]:
                point = int(row["point_index"])
                matrix = causal[(held_name, point)]
                prediction = np.clip(
                    np.asarray(model.predict(transform_depth(
                        matrix, np.full(len(matrix), point), variant
                    )), dtype=float),
                    0.10, 0.80,
                )
                truth = float(row["depth_mm"])
                frame_truth.extend([truth] * len(prediction))
                frame_predicted.extend(prediction.tolist())
                median = float(np.median(prediction))
                trials.append({
                    "point": point,
                    "truth_depth_mm": truth,
                    "median_predicted_depth_mm": median,
                    "median_absolute_error_mm": abs(median - truth),
                    "frame_mae_mm": float(np.mean(np.abs(prediction - truth))),
                    "frame_p95_absolute_error_mm": float(np.percentile(np.abs(prediction - truth), 95)),
                })
            truth = np.asarray(frame_truth)
            predicted = np.asarray(frame_predicted)
            trial_errors = np.asarray([row["median_absolute_error_mm"] for row in trials])
            folds.append({
                "held_round": held_name,
                "truth_depth_mm": float(rounds[held_name][0]["depth_mm"]),
                "trial_mae_mm": float(np.mean(trial_errors)),
                "trial_p95_absolute_error_mm": float(np.percentile(trial_errors, 95)),
                "trial_maximum_absolute_error_mm": float(np.max(trial_errors)),
                "frame_mae_mm": float(np.mean(np.abs(predicted - truth))),
                "frame_p95_absolute_error_mm": float(np.percentile(np.abs(predicted - truth), 95)),
                "trials": trials,
            })
        interpolation = [row for row in folds if row["held_round"] in {"depth_0p20_r1", "depth_0p60_r1"}]
        ranked.append({
            "feature": FEATURE,
            "preprocessing": variant,
            "estimator": estimator_name,
            "selection_folds": folds,
            "interpolation_trial_mae_mm_max": max(row["trial_mae_mm"] for row in interpolation),
            "interpolation_trial_p95_mm_max": max(row["trial_p95_absolute_error_mm"] for row in interpolation),
            "interpolation_frame_mae_mm_max": max(row["frame_mae_mm"] for row in interpolation),
            "all_fold_trial_mae_mm_max": max(row["trial_mae_mm"] for row in folds),
            "mean_fold_trial_mae_mm": float(np.mean([row["trial_mae_mm"] for row in folds])),
        })
    ranked.sort(key=lambda row: (
        row["interpolation_trial_mae_mm_max"],
        row["interpolation_trial_p95_mm_max"],
        row["interpolation_frame_mae_mm_max"],
        row["all_fold_trial_mae_mm_max"],
        row["mean_fold_trial_mae_mm"],
    ))
    result = {
        "schema": "f45_multidepth_causal_frame_depth_selection_v1",
        "target": "commanded_cnc_indentation_depth_mm_not_force",
        "training_rounds": list(rounds),
        "training_depths_mm": [0.10, 0.20, 0.30, 0.60, 0.80],
        "selection_folds": selection_names,
        "interpolation_selection_folds": ["depth_0p20_r1", "depth_0p60_r1"],
        "endpoint_rounds_always_in_training": ["depth_0p10_r1", "depth_0p80_r1"],
        "random_frame_split_used": False,
        "force_ground_truth_present": False,
        "deployment_eligible": False,
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps([
        {key: value for key, value in row.items() if key != "selection_folds"}
        for row in ranked[:15]
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
