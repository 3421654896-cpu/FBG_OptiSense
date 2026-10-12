"""Select a causal F45 position model after the first locked candidate failed.

The post-lock 0.40/0.70 mm rounds are deliberately treated as development
data by this script.  Candidate ranking holds out complete acquisition rounds;
no random frame split is used.  Endpoint depths 0.10/0.80 mm remain in every
selection fit, while all intermediate-depth rounds are used as selection folds.
A new algorithm lock and newly acquired depths are required before deployment.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from select_f45_multidepth_position_candidate import load_depth_manifest, transform


FEATURE = "pair_transition_delta45"


def candidate_models():
    """Yield a focused search space suitable for whole-round CV."""
    for variant in ("pair_unit90",):
        for leaf, max_features in ((1, "sqrt"), (2, "sqrt"), (4, "sqrt"), (2, 0.5)):
            suffix = str(max_features).replace(".", "p")
            yield variant, f"et_classifier_leaf{leaf}_mf{suffix}", "classifier", ExtraTreesClassifier(
                n_estimators=600,
                min_samples_leaf=leaf,
                max_features=max_features,
                class_weight="balanced",
                random_state=20260908,
                n_jobs=-1,
            )
        yield variant, "rf_classifier_leaf1_sqrt", "classifier", RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=20260908,
            n_jobs=-1,
        )
        yield variant, "lda_shrinkage", "classifier", make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        for neighbors in (1, 3, 5):
            yield variant, f"knn_{neighbors}_distance", "classifier", KNeighborsClassifier(
                n_neighbors=neighbors, weights="distance", metric="euclidean", n_jobs=-1
            )
        for c_value in (1.0, 10.0):
            yield variant, f"rbf_svc_c{c_value:g}", "classifier", make_pipeline(
                StandardScaler(), SVC(C=c_value, gamma="scale", class_weight="balanced")
            )
        yield variant, "grid_lda_shrinkage", "grid_classifier", MultiOutputClassifier(
            make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            )
        )
        for leaf in (2, 4):
            yield variant, f"grid_et_leaf{leaf}_sqrt", "grid_classifier", ExtraTreesClassifier(
                n_estimators=600,
                min_samples_leaf=leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=20260908,
                n_jobs=-1,
            )
        for leaf, max_features in ((1, "sqrt"), (2, "sqrt"), (4, "sqrt"), (2, 0.5)):
            suffix = str(max_features).replace(".", "p")
            yield variant, f"et_xy_leaf{leaf}_mf{suffix}", "xy_regressor", ExtraTreesRegressor(
                n_estimators=600,
                min_samples_leaf=leaf,
                max_features=max_features,
                random_state=20260908,
                n_jobs=-1,
            )
        for alpha in (1.0, 10.0, 100.0):
            yield variant, f"ridge_xy_a{alpha:g}", "xy_regressor", make_pipeline(
                StandardScaler(), Ridge(alpha=alpha)
            )
    # A small ablation checks whether retaining transition amplitude or
    # normalising each temporal block improves the weakest complete round.
    for variant in ("unit135", "block_unit135"):
        yield variant, "et_classifier_leaf2_sqrt", "classifier", ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=20260908,
            n_jobs=-1,
        )
        yield variant, "et_xy_leaf2_sqrt", "xy_regressor", ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=20260908,
            n_jobs=-1,
        )


def nearest_labels(predicted_xy: np.ndarray, labels: np.ndarray, label_xy: np.ndarray) -> np.ndarray:
    distance2 = np.sum((predicted_xy[:, None, :] - label_xy[None, :, :]) ** 2, axis=2)
    return labels[np.argmin(distance2, axis=1)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-0p10", required=True, type=Path)
    parser.add_argument("--depth-0p20", required=True, type=Path)
    parser.add_argument("--depth-0p40", required=True, type=Path)
    parser.add_argument("--depth-0p60", required=True, type=Path)
    parser.add_argument("--depth-0p70", required=True, type=Path)
    parser.add_argument("--depth-0p80", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only-grid", action="store_true")
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
        "depth_0p40_r1": load_depth_manifest(args.depth_0p40),
        "depth_0p60_r1": load_depth_manifest(args.depth_0p60),
        "depth_0p70_r1": load_depth_manifest(args.depth_0p70),
        "depth_0p80_r1": load_depth_manifest(args.depth_0p80),
    }
    causal = {
        (round_name, int(row["point_index"])): causal_held_features(Path(row["path"]))[FEATURE]
        for round_name, records in rounds.items()
        for row in records
    }
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in old["old_r1"]
    }
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])
    x_values = np.unique(label_xy[:, 0])
    y_values = np.unique(label_xy[:, 1])
    selection_names = [
        name for name in rounds if name not in {"depth_0p10_r1", "depth_0p80_r1"}
    ]

    ranked = []
    for variant, estimator_name, kind, template in candidate_models():
        if args.only_grid and kind != "grid_classifier":
            continue
        folds = []
        for held_name in selection_names:
            x_train = np.concatenate([
                transform(causal[(name, int(row["point_index"]))], variant)
                for name, records in rounds.items() if name != held_name
                for row in records
            ])
            label_train = np.concatenate([
                np.full(
                    len(causal[(name, int(row["point_index"]))]),
                    int(row["point_index"]),
                    dtype=int,
                )
                for name, records in rounds.items() if name != held_name
                for row in records
            ])
            if kind == "classifier":
                target_train = label_train
            elif kind == "grid_classifier":
                train_xy = np.stack([coordinates[int(label)] for label in label_train])
                target_train = np.column_stack((
                    np.argmin(np.abs(train_xy[:, 0, None] - x_values[None, :]), axis=1),
                    np.argmin(np.abs(train_xy[:, 1, None] - y_values[None, :]), axis=1),
                ))
            else:
                target_train = np.stack([coordinates[int(label)] for label in label_train])
            model = clone(template).fit(x_train, target_train)

            fold_total = fold_exact = fold_within = trial_exact = 0
            continuous_errors: list[float] = []
            per_point = []
            for row in rounds[held_name]:
                truth = int(row["point_index"])
                vectors = transform(causal[(held_name, truth)], variant)
                if kind == "classifier":
                    prediction = np.asarray(model.predict(vectors), dtype=int)
                    predicted_xy = np.stack([coordinates[int(label)] for label in prediction])
                elif kind == "grid_classifier":
                    grid_prediction = np.asarray(model.predict(vectors), dtype=int)
                    predicted_xy = np.column_stack((
                        x_values[grid_prediction[:, 0]],
                        y_values[grid_prediction[:, 1]],
                    ))
                    prediction = nearest_labels(predicted_xy, labels, label_xy)
                else:
                    predicted_xy = np.asarray(model.predict(vectors), dtype=float)
                    prediction = nearest_labels(predicted_xy, labels, label_xy)
                counts = Counter(prediction.tolist())
                majority = min(counts, key=lambda label: (-counts[label], label))
                errors = np.asarray([
                    np.linalg.norm(coordinates[int(label)] - coordinates[truth])
                    for label in prediction
                ])
                continuous = np.linalg.norm(predicted_xy - coordinates[truth], axis=1)
                continuous_errors.extend(float(value) for value in continuous)
                trial_exact += int(majority == truth)
                fold_total += len(prediction)
                fold_exact += int(np.sum(prediction == truth))
                fold_within += int(np.sum(errors <= 4.0 + 1e-6))
                per_point.append({
                    "truth": truth,
                    "frames": int(len(prediction)),
                    "majority": int(majority),
                    "exact_frames": int(np.sum(prediction == truth)),
                    "within_4mm_frames": int(np.sum(errors <= 4.0 + 1e-6)),
                    "median_continuous_error_mm": float(np.median(continuous)),
                    "prediction_counts": {str(k): int(v) for k, v in sorted(counts.items())},
                })
            folds.append({
                "held_round": held_name,
                "trial_majority_exact": trial_exact,
                "trial_majority_exact_percent": 100.0 * trial_exact / 17.0,
                "frames": fold_total,
                "frame_exact_percent": 100.0 * fold_exact / fold_total,
                "frame_within_4mm_percent": 100.0 * fold_within / fold_total,
                "continuous_position_mae_mm": float(np.mean(continuous_errors)),
                "continuous_position_p95_mm": float(np.percentile(continuous_errors, 95)),
                "per_point": per_point,
            })
        ranked.append({
            "feature": FEATURE,
            "preprocessing": variant,
            "estimator": estimator_name,
            "kind": kind,
            "selection_folds": folds,
            "worst_trial_majority_exact": min(row["trial_majority_exact"] for row in folds),
            "worst_frame_exact_percent": min(row["frame_exact_percent"] for row in folds),
            "worst_frame_within_4mm_percent": min(row["frame_within_4mm_percent"] for row in folds),
            "worst_continuous_position_p95_mm": max(row["continuous_position_p95_mm"] for row in folds),
            "mean_frame_exact_percent": float(np.mean([row["frame_exact_percent"] for row in folds])),
        })

    ranked.sort(key=lambda row: (
        -row["worst_trial_majority_exact"],
        -row["worst_frame_within_4mm_percent"],
        -row["worst_frame_exact_percent"],
        -row["mean_frame_exact_percent"],
        row["worst_continuous_position_p95_mm"],
    ))
    result = {
        "schema": "f45_multidepth_causal_frame_position_selection_v2",
        "training_rounds": list(rounds),
        "training_depths_mm": [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80],
        "selection_folds": selection_names,
        "endpoint_rounds_always_in_training": ["depth_0p10_r1", "depth_0p80_r1"],
        "training_unit": "causal held frame with capture-local frozen 64-frame baseline",
        "random_frame_split_used": False,
        "first_locked_candidate_validation_reused_for_selection": True,
        "untouched_validation_required": "new complete 17-point rounds at untrained depths",
        "deployment_eligible": False,
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps([
        {key: value for key, value in row.items() if key != "selection_folds"}
        for row in ranked[:16]
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
