"""Evaluate F45 position estimators with whole acquisition rounds held out.

Every contact run is reduced to one baseline-relative feature vector.  Frames
from one physical contact are never split between train and test, and an
entire 17-position round is held out at a time.  This is an analysis tool: it
does not access the optical board or command the machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from evaluate_fast_fullmap_position import load_manifest
from validate_fast_fullmap_position_locked import load_validation_round


def load_rounds(training_manifest: Path, validation_manifests: list[Path]) -> dict[str, list[dict]]:
    old = load_manifest(training_manifest)
    rounds: dict[str, list[dict]] = {"old_r1": old["r1"], "old_r2": old["r2"]}
    for index, manifest in enumerate(validation_manifests, start=3):
        rounds[f"round_{index}"] = load_validation_round(manifest)
    for name, records in rounds.items():
        if [int(row["point_index"]) for row in records] != list(range(1, 18)):
            raise ValueError(f"{name}: point labels are not exactly 1..17")
        if any(abs(float(row["depth_mm"]) - 0.30) > 1e-9 for row in records):
            raise ValueError(f"{name}: expected 0.30 mm indentation")
    return rounds


def _xy_by_point(rounds: dict[str, list[dict]]) -> dict[int, np.ndarray]:
    rows = next(iter(rounds.values()))
    result = {int(row["point_index"]): np.asarray(row["xy"], dtype=float) for row in rows}
    if set(result) != set(range(1, 18)):
        raise ValueError("missing point coordinate")
    return result


def _nearest_label(xy: np.ndarray, coordinates: dict[int, np.ndarray]) -> np.ndarray:
    labels = np.asarray(sorted(coordinates), dtype=int)
    matrix = np.stack([coordinates[int(label)] for label in labels])
    distance = np.linalg.norm(xy[:, None, :] - matrix[None, :, :], axis=2)
    return labels[np.argmin(distance, axis=1)]


def _metrics(truth: np.ndarray, predicted: np.ndarray, truth_xy: np.ndarray, predicted_xy: np.ndarray) -> dict:
    errors = np.linalg.norm(predicted_xy - truth_xy, axis=1)
    return {
        "samples": int(len(truth)),
        "exact_percent": float(100.0 * np.mean(predicted == truth)),
        "within_4mm_percent": float(100.0 * np.mean(errors <= 4.0 + 1e-6)),
        "median_error_mm": float(np.median(errors)),
        "p95_error_mm": float(np.percentile(errors, 95)),
        "maximum_error_mm": float(np.max(errors)),
    }


def estimators() -> list[tuple[str, str, object]]:
    classifiers: list[tuple[str, object]] = []
    for scaler_name, scaler in (("standard", StandardScaler()), ("robust", RobustScaler())):
        classifiers.extend(
            [
                (f"nearest_centroid_{scaler_name}", make_pipeline(clone(scaler), NearestCentroid())),
                (f"knn3_distance_{scaler_name}", make_pipeline(clone(scaler), KNeighborsClassifier(3, weights="distance"))),
                (f"knn5_distance_{scaler_name}", make_pipeline(clone(scaler), KNeighborsClassifier(5, weights="distance"))),
                (f"logistic_c0p1_{scaler_name}", make_pipeline(clone(scaler), LogisticRegression(C=0.1, max_iter=5000))),
                (f"logistic_c1_{scaler_name}", make_pipeline(clone(scaler), LogisticRegression(C=1.0, max_iter=5000))),
                (f"linear_svc_c0p1_{scaler_name}", make_pipeline(clone(scaler), SVC(kernel="linear", C=0.1))),
                (f"linear_svc_c1_{scaler_name}", make_pipeline(clone(scaler), SVC(kernel="linear", C=1.0))),
            ]
        )
    classifiers.extend(
        [
            ("lda_shrinkage", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
            ("extra_trees_400", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1, max_features="sqrt", random_state=20260908, n_jobs=-1)),
        ]
    )

    regressors: list[tuple[str, object]] = []
    for scaler_name, scaler in (("standard", StandardScaler()), ("robust", RobustScaler())):
        for alpha in (0.1, 1.0, 10.0, 100.0):
            regressors.append((f"ridge_a{alpha:g}_{scaler_name}", make_pipeline(clone(scaler), Ridge(alpha=alpha))))
        for count in (3, 5, 9):
            regressors.append((f"knn_reg{count}_{scaler_name}", make_pipeline(clone(scaler), KNeighborsRegressor(count, weights="distance"))))
    regressors.append(("extra_trees_reg400", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1, max_features="sqrt", random_state=20260908, n_jobs=-1)))
    return [("classifier", name, model) for name, model in classifiers] + [
        ("regressor", name, model) for name, model in regressors
    ]


def evaluate(rounds: dict[str, list[dict]]) -> dict:
    coordinates = _xy_by_point(rounds)
    features = tuple(next(iter(rounds.values()))[0]["features"])
    rows = []
    for feature in features:
        for kind, estimator_name, template in estimators():
            fold_results = []
            all_truth: list[int] = []
            all_prediction: list[int] = []
            all_truth_xy: list[np.ndarray] = []
            all_prediction_xy: list[np.ndarray] = []
            failed = None
            for held_name, held_records in rounds.items():
                training_records = [
                    row for name, records in rounds.items() if name != held_name for row in records
                ]
                x_train = np.stack([row["features"][feature] for row in training_records])
                y_label = np.asarray([row["point_index"] for row in training_records], dtype=int)
                y_xy = np.stack([row["xy"] for row in training_records])
                x_test = np.stack([row["features"][feature] for row in held_records])
                truth = np.asarray([row["point_index"] for row in held_records], dtype=int)
                truth_xy = np.stack([row["xy"] for row in held_records])
                try:
                    model = clone(template)
                    if kind == "classifier":
                        model.fit(x_train, y_label)
                        prediction = np.asarray(model.predict(x_test), dtype=int)
                        prediction_xy = np.stack([coordinates[int(label)] for label in prediction])
                    else:
                        model.fit(x_train, y_xy)
                        prediction_xy = np.asarray(model.predict(x_test), dtype=float)
                        prediction = _nearest_label(prediction_xy, coordinates)
                except Exception as exc:  # keep the audit complete when one candidate is singular
                    failed = f"{type(exc).__name__}: {exc}"
                    break
                fold_results.append({"held_round": held_name, **_metrics(truth, prediction, truth_xy, prediction_xy)})
                all_truth.extend(truth.tolist())
                all_prediction.extend(prediction.tolist())
                all_truth_xy.extend(truth_xy)
                all_prediction_xy.extend(prediction_xy)
            if failed:
                rows.append({"feature": feature, "kind": kind, "estimator": estimator_name, "failed": failed})
                continue
            truth = np.asarray(all_truth, dtype=int)
            prediction = np.asarray(all_prediction, dtype=int)
            item = {
                "feature": feature,
                "kind": kind,
                "estimator": estimator_name,
                "whole_round_loso": _metrics(
                    truth,
                    prediction,
                    np.asarray(all_truth_xy),
                    np.asarray(all_prediction_xy),
                ),
                "worst_fold_exact_percent": float(min(row["exact_percent"] for row in fold_results)),
                "worst_fold_within_4mm_percent": float(min(row["within_4mm_percent"] for row in fold_results)),
                "folds": fold_results,
                "confusion": confusion_matrix(truth, prediction, labels=list(range(1, 18))).tolist(),
            }
            rows.append(item)
    successful = [row for row in rows if not row.get("failed")]
    successful.sort(
        key=lambda row: (
            -row["worst_fold_within_4mm_percent"],
            -row["worst_fold_exact_percent"],
            row["whole_round_loso"]["p95_error_mm"],
            -row["whole_round_loso"]["exact_percent"],
            row["whole_round_loso"]["maximum_error_mm"],
        )
    )
    return {
        "schema": "f45_whole_round_loso_model_audit_v1",
        "rounds": list(rounds),
        "round_count": len(rounds),
        "contacts": sum(len(records) for records in rounds.values()),
        "selection_unit": "whole_17_position_physical_round",
        "random_frame_split_used": False,
        "best": successful[0],
        "ranked": successful,
        "failed": [row for row in rows if row.get("failed")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("validation_manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    report = evaluate(load_rounds(args.training_manifest, args.validation_manifests))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
