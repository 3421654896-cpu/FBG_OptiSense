"""Evaluate continuous XY regressors for the six complete F18 contact rounds.

All validation folds exclude one complete physical round.  Training uses one
median vector per contact so correlated frames from a contact never leak into
the held round.  Each held causal frame is predicted independently, then the
continuous result is also mapped to the nearest commissioned grid position for
an exact-position diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from evaluate_f18_cross_session_models import load_rounds
from evaluate_f18_multidepth_sparse_holdout import load_manifest


FEATURES = (
    "second_delta18",
    "pair_delta36",
    "second_flank_shape18",
    "pair_flank_shape36",
    "second_delta_plus_flank36",
    "pair_delta_plus_flank72",
)


def estimators() -> list[tuple[str, object]]:
    return [
        ("ridge_a0p01", make_pipeline(StandardScaler(), Ridge(alpha=0.01))),
        ("ridge_a0p1", make_pipeline(StandardScaler(), Ridge(alpha=0.1))),
        ("ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("pls_2", make_pipeline(StandardScaler(), PLSRegression(n_components=2))),
        ("pls_4", make_pipeline(StandardScaler(), PLSRegression(n_components=4))),
        ("pls_8", make_pipeline(StandardScaler(), PLSRegression(n_components=8))),
        (
            "knn_3_distance",
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3, weights="distance")),
        ),
        (
            "knn_5_distance",
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
        ),
        (
            "rbf_svr_c10",
            make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.05)),
            ),
        ),
        (
            "rbf_svr_c100",
            make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(SVR(kernel="rbf", C=100.0, epsilon=0.05)),
            ),
        ),
        (
            "kernel_ridge_rbf_a0p1",
            make_pipeline(StandardScaler(), KernelRidge(alpha=0.1, kernel="rbf")),
        ),
        (
            "extra_trees_400",
            ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest_400",
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
    ]


def nearest_labels(
    prediction_xy: np.ndarray, coordinates: dict[int, np.ndarray]
) -> np.ndarray:
    labels = np.asarray(sorted(coordinates), dtype=int)
    centers = np.stack([coordinates[int(label)] for label in labels])
    distances = np.linalg.norm(
        prediction_xy[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
    )
    return labels[np.argmin(distances, axis=1)]


def evaluate(rounds: dict[str, list[dict]]) -> dict:
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in rounds["deep_r1"]
    }
    ranked = []
    for feature in FEATURES:
        for estimator_name, template in estimators():
            folds = []
            failed = None
            for held_name, test in rounds.items():
                train = [
                    row
                    for name, records in rounds.items()
                    if name != held_name
                    for row in records
                ]
                train_x = np.stack([row["median_features"][feature] for row in train])
                train_y = np.stack([coordinates[int(row["point_index"])] for row in train])
                try:
                    model = clone(template).fit(train_x, train_y)
                except Exception as exc:
                    failed = f"{type(exc).__name__}: {exc}"
                    break
                all_errors = []
                total = exact = within2 = within4 = trial_exact = 0
                per_point = []
                for row in test:
                    truth = int(row["point_index"])
                    truth_xy = coordinates[truth]
                    predicted_xy = np.asarray(
                        model.predict(row["frame_features"][feature]), dtype=float
                    )
                    labels = nearest_labels(predicted_xy, coordinates)
                    errors = np.linalg.norm(predicted_xy - truth_xy, axis=1)
                    median_xy = np.median(predicted_xy, axis=0)
                    median_label = int(nearest_labels(median_xy[np.newaxis, :], coordinates)[0])
                    total += len(labels)
                    exact += int(np.sum(labels == truth))
                    within2 += int(np.sum(errors <= 2.0))
                    within4 += int(np.sum(errors <= 4.0))
                    trial_exact += int(median_label == truth)
                    all_errors.extend(errors.tolist())
                    values, counts = np.unique(labels, return_counts=True)
                    per_point.append(
                        {
                            "truth": truth,
                            "median_label": median_label,
                            "median_xy": median_xy.tolist(),
                            "median_error_mm": float(np.linalg.norm(median_xy - truth_xy)),
                            "frame_nearest_exact_percent": float(100.0 * np.mean(labels == truth)),
                            "prediction_counts": {
                                str(int(value)): int(count)
                                for value, count in zip(values, counts)
                            },
                        }
                    )
                errors = np.asarray(all_errors)
                folds.append(
                    {
                        "held_round": held_name,
                        "depth_mm": float(test[0]["depth_mm"]),
                        "trial_median_nearest_exact": trial_exact,
                        "trial_median_nearest_exact_percent": 100.0 * trial_exact / 17.0,
                        "frames": total,
                        "frame_nearest_exact_percent": 100.0 * exact / total,
                        "continuous_within_2mm_percent": 100.0 * within2 / total,
                        "continuous_within_4mm_percent": 100.0 * within4 / total,
                        "continuous_median_error_mm": float(np.median(errors)),
                        "continuous_p95_error_mm": float(np.percentile(errors, 95)),
                        "continuous_maximum_error_mm": float(np.max(errors)),
                        "per_point": per_point,
                    }
                )
            if failed is not None:
                ranked.append({"feature": feature, "estimator": estimator_name, "error": failed})
                continue
            ranked.append(
                {
                    "feature": feature,
                    "estimator": estimator_name,
                    "worst_trial_median_nearest_exact_percent": min(
                        fold["trial_median_nearest_exact_percent"] for fold in folds
                    ),
                    "worst_frame_nearest_exact_percent": min(
                        fold["frame_nearest_exact_percent"] for fold in folds
                    ),
                    "worst_continuous_within_2mm_percent": min(
                        fold["continuous_within_2mm_percent"] for fold in folds
                    ),
                    "worst_continuous_within_4mm_percent": min(
                        fold["continuous_within_4mm_percent"] for fold in folds
                    ),
                    "worst_continuous_p95_error_mm": max(
                        fold["continuous_p95_error_mm"] for fold in folds
                    ),
                    "folds": folds,
                }
            )
    successful = [row for row in ranked if "error" not in row]
    successful.sort(
        key=lambda row: (
            -row["worst_trial_median_nearest_exact_percent"],
            -row["worst_continuous_within_2mm_percent"],
            row["worst_continuous_p95_error_mm"],
            -row["worst_frame_nearest_exact_percent"],
        )
    )
    return {
        "schema": "f18_multidepth_spatial_regression_loso_v1",
        "rounds": [
            {"name": name, "depth_mm": float(records[0]["depth_mm"])}
            for name, records in rounds.items()
        ],
        "contacts": sum(len(records) for records in rounds.values()),
        "random_frame_split_used": False,
        "best": successful[0],
        "ranked": successful + [row for row in ranked if "error" in row],
        "physical_15hz_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--deep-round", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if len(args.deep_round) != 2:
        parser.error("exactly two complete 0.30 mm development rounds are required")
    rounds = load_rounds(args.capture_directory)
    for index, path in enumerate(args.deep_round, start=1):
        rounds[f"deep_r{index}"] = load_manifest(path)
    result = evaluate(rounds)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    best = result["best"]
    print(
        json.dumps(
            {key: value for key, value in best.items() if key != "folds"},
            ensure_ascii=False,
            indent=2,
        )
    )
    for fold in best["folds"]:
        bad = [
            [row["truth"], row["median_label"], round(row["median_error_mm"], 3)]
            for row in fold["per_point"]
            if row["truth"] != row["median_label"]
        ]
        print(fold["held_round"], round(fold["continuous_p95_error_mm"], 3), bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
