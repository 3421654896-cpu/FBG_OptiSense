"""Select an F18 multi-depth model using old, independent 0.30 mm probes.

The new complete 17-position 0.30 mm round and the four complete 0.10 mm
development rounds are eligible training sources.  Ten older 0.30 mm probes
at five positions remain holdouts, split by physical repeat.  Every held frame
is scored; adjacent frames are never randomly split across train and test.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_f18_cross_session_models import load_contact, load_rounds


PROBE_POINTS = (1, 5, 9, 13, 17)


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema") != "f18_guarded_position_round_manifest_v1":
        raise ValueError("unsupported F18 round manifest")
    if abs(float(payload.get("depth_mm", -1.0)) - 0.30) > 1e-9:
        raise ValueError("complete round must be 0.30 mm")
    rows = [
        load_contact(Path(item["path"]), int(item["point_index"]), expected_depth_mm=0.30)
        for item in payload["captures"]
    ]
    if [row["point_index"] for row in rows] != list(range(1, 18)):
        raise ValueError("manifest positions are not exactly P1..P17")
    return rows


def load_probe_folds(directory: Path) -> dict[str, list[dict]]:
    return {
        f"historical_probe_r{repeat}": [
            load_contact(
                directory
                / f"ch1_fast_flank_contact_p{point:02d}_0p30mm_probe_r{repeat}_20260906.json",
                point,
                expected_depth_mm=0.30,
            )
            for point in PROBE_POINTS
        ]
        for repeat in (1, 2)
    }


def estimators() -> list[tuple[str, object]]:
    return [
        (
            "extra_trees_400_leaf1",
            ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees_400_leaf2",
            ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest_400_leaf1",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
        (
            "lda_shrinkage",
            make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ),
        (
            "logistic_c0p1",
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        ),
        (
            "logistic_c1",
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        ),
        ("nearest_centroid", make_pipeline(StandardScaler(), NearestCentroid())),
        ("linear_svc_c0p1", make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.1))),
        ("rbf_svc_c1", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))),
        ("rbf_svc_c10", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))),
    ]


def score_fold(model, feature: str, records: list[dict], coordinates: dict[int, np.ndarray]) -> dict:
    total = exact = within = majority_exact = 0
    errors: list[float] = []
    per_point = []
    for row in records:
        truth = int(row["point_index"])
        predictions = np.asarray(model.predict(row["frame_features"][feature]), dtype=int)
        counts = Counter(predictions.tolist())
        majority = min(counts, key=lambda value: (-counts[value], value))
        point_errors = np.asarray(
            [np.linalg.norm(coordinates[int(value)] - coordinates[truth]) for value in predictions]
        )
        total += len(predictions)
        exact += int(np.sum(predictions == truth))
        within += int(np.sum(point_errors <= 4.0 + 1e-6))
        majority_exact += int(majority == truth)
        errors.extend(point_errors.tolist())
        per_point.append(
            {
                "truth": truth,
                "frames": len(predictions),
                "majority": int(majority),
                "exact_percent": float(100.0 * np.mean(predictions == truth)),
                "prediction_counts": {
                    str(label): int(count) for label, count in sorted(counts.items())
                },
            }
        )
    error = np.asarray(errors)
    return {
        "trials": len(records),
        "trial_majority_exact": majority_exact,
        "trial_majority_exact_percent": 100.0 * majority_exact / len(records),
        "frames": total,
        "frame_exact_percent": 100.0 * exact / total,
        "frame_within_4mm_percent": 100.0 * within / total,
        "p95_error_mm": float(np.percentile(error, 95)),
        "maximum_error_mm": float(np.max(error)),
        "per_point": per_point,
    }


def evaluate(complete: list[dict], shallow: dict[str, list[dict]], probes: dict[str, list[dict]]) -> dict:
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float) for row in complete
    }
    training_sets = {
        "complete_0p30_only": complete,
        "complete_0p30_plus_four_0p10_rounds": complete
        + [row for records in shallow.values() for row in records],
    }
    feature_names = tuple(complete[0]["median_features"])
    ranked = []
    for training_name, training in training_sets.items():
        for feature in feature_names:
            train_x = np.stack([row["median_features"][feature] for row in training])
            train_y = np.asarray([row["point_index"] for row in training], dtype=int)
            for estimator_name, template in estimators():
                try:
                    model = clone(template).fit(train_x, train_y)
                    folds = [
                        {"held_round": name, **score_fold(model, feature, rows, coordinates)}
                        for name, rows in probes.items()
                    ]
                except Exception as exc:
                    ranked.append(
                        {
                            "training": training_name,
                            "feature": feature,
                            "estimator": estimator_name,
                            "failed": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                ranked.append(
                    {
                        "training": training_name,
                        "feature": feature,
                        "estimator": estimator_name,
                        "training_contacts": len(training),
                        "worst_fold_trial_majority_exact_percent": min(
                            fold["trial_majority_exact_percent"] for fold in folds
                        ),
                        "worst_fold_frame_exact_percent": min(
                            fold["frame_exact_percent"] for fold in folds
                        ),
                        "worst_fold_frame_within_4mm_percent": min(
                            fold["frame_within_4mm_percent"] for fold in folds
                        ),
                        "worst_fold_p95_error_mm": max(fold["p95_error_mm"] for fold in folds),
                        "folds": folds,
                    }
                )
    successful = [row for row in ranked if not row.get("failed")]
    successful.sort(
        key=lambda row: (
            -row["worst_fold_trial_majority_exact_percent"],
            -row["worst_fold_frame_exact_percent"],
            -row["worst_fold_frame_within_4mm_percent"],
            row["worst_fold_p95_error_mm"],
        )
    )
    return {
        "schema": "f18_multidepth_sparse_holdout_model_selection_v1",
        "selection_holdout": "two historical 0.30 mm physical repeats at P1/P5/P9/P13/P17",
        "random_frame_split_used": False,
        "best": successful[0],
        "ranked": successful,
        "failed": [row for row in ranked if row.get("failed")],
        "new_untouched_full_round_required": True,
        "physical_15hz_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-round", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = evaluate(
        load_manifest(args.complete_round),
        load_rounds(args.capture_directory),
        load_probe_folds(args.capture_directory),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
