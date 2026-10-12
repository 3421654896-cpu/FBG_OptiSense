"""Whole-round LOSO selection across four 0.10 and two 0.30 mm F18 rounds.

The failed first candidate validation is now development data.  Selection is
still performed with entire physical rounds held out, and every causal held
frame is scored.  No hardware is accessed.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f18_cross_session_models import load_rounds
from evaluate_f18_multidepth_sparse_holdout import load_manifest


def estimators() -> list[tuple[str, object]]:
    return [
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
            "random_forest_400_leaf1",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
    ]


def evaluate(rounds: dict[str, list[dict]]) -> dict:
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in next(records for name, records in rounds.items() if name.startswith("deep"))
    }
    features = tuple(next(iter(rounds.values()))[0]["median_features"])
    ranked = []
    for feature in features:
        for estimator_name, template in estimators():
            folds = []
            for held_name, test in rounds.items():
                train = [
                    row for name, records in rounds.items() if name != held_name for row in records
                ]
                model = clone(template).fit(
                    np.stack([row["median_features"][feature] for row in train]),
                    np.asarray([row["point_index"] for row in train], dtype=int),
                )
                total = exact = within = majority_exact = 0
                errors: list[float] = []
                per_point = []
                for row in test:
                    truth = int(row["point_index"])
                    prediction = np.asarray(
                        model.predict(row["frame_features"][feature]), dtype=int
                    )
                    counts = Counter(prediction.tolist())
                    majority = min(counts, key=lambda value: (-counts[value], value))
                    point_errors = np.asarray(
                        [
                            np.linalg.norm(coordinates[int(value)] - coordinates[truth])
                            for value in prediction
                        ]
                    )
                    total += len(prediction)
                    exact += int(np.sum(prediction == truth))
                    within += int(np.sum(point_errors <= 4.0 + 1e-6))
                    majority_exact += int(majority == truth)
                    errors.extend(point_errors.tolist())
                    per_point.append(
                        {
                            "truth": truth,
                            "majority": int(majority),
                            "frames": len(prediction),
                            "exact_percent": float(100.0 * np.mean(prediction == truth)),
                            "prediction_counts": {
                                str(label): int(count)
                                for label, count in sorted(counts.items())
                            },
                        }
                    )
                error = np.asarray(errors)
                folds.append(
                    {
                        "held_round": held_name,
                        "depth_mm": float(test[0]["depth_mm"]),
                        "trial_majority_exact": majority_exact,
                        "trial_majority_exact_percent": 100.0 * majority_exact / 17.0,
                        "frames": total,
                        "frame_exact_percent": 100.0 * exact / total,
                        "frame_within_4mm_percent": 100.0 * within / total,
                        "p95_error_mm": float(np.percentile(error, 95)),
                        "maximum_error_mm": float(np.max(error)),
                        "per_point": per_point,
                    }
                )
            ranked.append(
                {
                    "feature": feature,
                    "estimator": estimator_name,
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
    ranked.sort(
        key=lambda row: (
            -row["worst_fold_trial_majority_exact_percent"],
            -row["worst_fold_frame_exact_percent"],
            -row["worst_fold_frame_within_4mm_percent"],
            row["worst_fold_p95_error_mm"],
        )
    )
    return {
        "schema": "f18_multidepth_whole_round_loso_v1",
        "rounds": [
            {"name": name, "depth_mm": float(records[0]["depth_mm"])}
            for name, records in rounds.items()
        ],
        "contacts": sum(len(records) for records in rounds.values()),
        "test_unit": "every causal held frame from one excluded complete physical round",
        "random_frame_split_used": False,
        "best": ranked[0],
        "ranked": ranked,
        "new_untouched_full_round_required": True,
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
    rounds = load_rounds(args.capture_directory)
    for index, path in enumerate(args.deep_round, start=1):
        rounds[f"deep_r{index}"] = load_manifest(path)
    if len(args.deep_round) != 2:
        parser.error("exactly two complete 0.30 mm development rounds are required")
    result = evaluate(rounds)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
