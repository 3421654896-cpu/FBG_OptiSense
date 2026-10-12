"""Select a depth-robust F45 position model from real causal held frames.

The 0.20/0.60 mm rounds are development data after the first median-contact
candidate failed.  Candidate selection holds out complete physical rounds;
frames from a held contact never enter training.  New depths must be captured
after the resulting algorithm is locked.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from select_f45_multidepth_position_candidate import load_depth_manifest, transform


FEATURE = "pair_transition_delta45"


def candidates():
    for variant in ("raw135", "unit135", "block_unit135", "pair_unit90"):
        for leaf in (1, 2, 4):
            yield variant, f"extra_trees_600_leaf{leaf}_sqrt", ExtraTreesClassifier(
                n_estimators=600, min_samples_leaf=leaf, max_features="sqrt",
                random_state=20260908, n_jobs=-1,
            )
        for leaf in (1, 2):
            yield variant, f"extra_trees_600_leaf{leaf}_half", ExtraTreesClassifier(
                n_estimators=600, min_samples_leaf=leaf, max_features=0.5,
                random_state=20260908, n_jobs=-1,
            )
        yield variant, "random_forest_600_leaf1_sqrt", RandomForestClassifier(
            n_estimators=600, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=20260908, n_jobs=-1,
        )
        yield variant, "lda_shrinkage", make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
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
        (round_name, int(row["point_index"])): causal_held_features(Path(row["path"]))[FEATURE]
        for round_name, records in rounds.items() for row in records
    }
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in old["old_r1"]
    }
    selection_names = [
        name for name in rounds
        if name not in {"depth_0p10_r1", "depth_0p80_r1"}
    ]
    ranked = []
    for variant, estimator_name, template in candidates():
        folds = []
        for held_name in selection_names:
            x_train = np.concatenate([
                transform(causal[(name, int(row["point_index"]))], variant)
                for name, records in rounds.items() if name != held_name
                for row in records
            ])
            y_train = np.concatenate([
                np.full(len(causal[(name, int(row["point_index"]))]), int(row["point_index"]), dtype=int)
                for name, records in rounds.items() if name != held_name
                for row in records
            ])
            model = clone(template).fit(x_train, y_train)
            fold_total = fold_exact = fold_within = trial_exact = 0
            per_point = []
            for row in rounds[held_name]:
                truth = int(row["point_index"])
                vectors = transform(causal[(held_name, truth)], variant)
                prediction = np.asarray(model.predict(vectors), dtype=int)
                counts = Counter(prediction.tolist())
                majority = min(counts, key=lambda label: (-counts[label], label))
                errors = np.asarray([
                    np.linalg.norm(coordinates[int(label)] - coordinates[truth])
                    for label in prediction
                ])
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
                    "prediction_counts": {str(k): int(v) for k, v in sorted(counts.items())},
                })
            folds.append({
                "held_round": held_name,
                "trial_majority_exact": trial_exact,
                "trial_majority_exact_percent": 100.0 * trial_exact / 17.0,
                "frames": fold_total,
                "frame_exact_percent": 100.0 * fold_exact / fold_total,
                "frame_within_4mm_percent": 100.0 * fold_within / fold_total,
                "per_point": per_point,
            })
        ranked.append({
            "feature": FEATURE,
            "preprocessing": variant,
            "estimator": estimator_name,
            "selection_folds": folds,
            "worst_trial_majority_exact": min(row["trial_majority_exact"] for row in folds),
            "worst_frame_exact_percent": min(row["frame_exact_percent"] for row in folds),
            "worst_frame_within_4mm_percent": min(row["frame_within_4mm_percent"] for row in folds),
            "mean_frame_exact_percent": float(np.mean([row["frame_exact_percent"] for row in folds])),
        })
    ranked.sort(key=lambda row: (
        -row["worst_trial_majority_exact"],
        -row["worst_frame_within_4mm_percent"],
        -row["worst_frame_exact_percent"],
        -row["mean_frame_exact_percent"],
    ))
    result = {
        "schema": "f45_multidepth_causal_frame_position_selection_v1",
        "training_rounds": list(rounds),
        "training_depths_mm": [0.10, 0.20, 0.30, 0.60, 0.80],
        "selection_folds": selection_names,
        "endpoint_rounds_always_in_training": ["depth_0p10_r1", "depth_0p80_r1"],
        "training_unit": "causal held frame with capture-local frozen 64-frame baseline",
        "random_frame_split_used": False,
        "untouched_validation_required": "new complete 17-point rounds at untrained depths",
        "deployment_eligible": False,
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps([
        {key: value for key, value in row.items() if key != "selection_folds"}
        for row in ranked[:12]
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
