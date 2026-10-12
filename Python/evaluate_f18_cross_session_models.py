"""Evaluate causal one-frame F18 position models with whole rounds held out.

The four existing 0.10 mm rounds were acquired using two commissioning
baselines.  Each physical contact contributes one median training vector;
every held contact frame in an excluded round is tested individually.  The
script is offline and never accesses the optical board or CNC controller.
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

from capture_fast_flank_stream import ROWS


ROUND_PATTERNS = {
    "physical_r1": "ch1_fast_flank_contact_p{point:02d}_0p10mm_physical_r1_20260906.json",
    "physical_r2": "ch1_fast_flank_contact_p{point:02d}_0p10mm_physical_r2_20260906.json",
    "localbaseline_r1": "ch1_fast_flank_contact_p{point:02d}_0p10mm_localbaseline_r1_20260906.json",
    "localbaseline_r2": "ch1_fast_flank_contact_p{point:02d}_0p10mm_localbaseline_r2_20260906.json",
}


def _matrix(frames: list[dict], field: str) -> np.ndarray:
    matrix = np.asarray(
        [[record[field] for record in frame["records"]] for frame in frames],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(ROWS) or not np.isfinite(matrix).all():
        raise ValueError(f"malformed F18 {field} matrix")
    return matrix


def _unit_amplitude(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.concatenate((centered / np.maximum(norms, 1e-9), np.log1p(norms)), axis=1)


def _flank_shape(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return nine differential-shift and nine common-mode flank responses.

    Every FBG contributes a left/right sample pair.  The normalized
    differential suppresses common optical-power changes while retaining the
    signed peak shift; the normalized pair sum retains local amplitude/shape
    changes which can still carry useful position information.  Both terms
    are differences from the same capture-local no-contact baseline.
    """

    left = values[:, 0::2]
    right = values[:, 1::2]
    base_left = baseline[0::2]
    base_right = baseline[1::2]
    current_sum = np.maximum(left + right, 1.0)
    baseline_sum = np.maximum(base_left + base_right, 1.0)
    differential = (right - left) / current_sum - (
        (base_right - base_left) / baseline_sum
    )
    common_mode = current_sum / baseline_sum - 1.0
    return np.concatenate((differential, common_mode), axis=1)


def _features(
    first: np.ndarray,
    second: np.ndarray,
    baseline_first: np.ndarray,
    baseline_second: np.ndarray,
) -> dict[str, np.ndarray]:
    first_delta = first - baseline_first
    second_delta = second - baseline_second
    mean_delta = 0.5 * (first + second) - 0.5 * (
        baseline_first + baseline_second
    )
    transition = (second - first) - (baseline_second - baseline_first)
    first_left = np.maximum(first[:, 0::2], 1.0)
    first_right = np.maximum(first[:, 1::2], 1.0)
    second_left = np.maximum(second[:, 0::2], 1.0)
    second_right = np.maximum(second[:, 1::2], 1.0)
    base_first_ratio = np.log(
        np.maximum(baseline_first[1::2], 1.0)
        / np.maximum(baseline_first[0::2], 1.0)
    )
    base_second_ratio = np.log(
        np.maximum(baseline_second[1::2], 1.0)
        / np.maximum(baseline_second[0::2], 1.0)
    )
    first_ratio = np.log(first_right / first_left) - base_first_ratio
    second_ratio = np.log(second_right / second_left) - base_second_ratio
    pair = np.concatenate((first_delta, second_delta), axis=1)
    pair_transition = np.concatenate((first_delta, second_delta, transition), axis=1)
    ratio_pair = np.concatenate((first_ratio, second_ratio), axis=1)
    first_flank_shape = _flank_shape(first, baseline_first)
    second_flank_shape = _flank_shape(second, baseline_second)
    mean_values = 0.5 * (first + second)
    mean_baseline = 0.5 * (baseline_first + baseline_second)
    mean_flank_shape = _flank_shape(mean_values, mean_baseline)
    # The two-point flank response changes gain when the unpressed peak drifts
    # relative to either laser sample.  Baseline samples are already available
    # before contact, so including their state is causal and lets a model
    # compensate that sensitivity instead of confusing it with contact XY.
    count = first.shape[0]
    baseline_first_rows = np.broadcast_to(baseline_first, (count, len(baseline_first)))
    baseline_second_rows = np.broadcast_to(
        baseline_second, (count, len(baseline_second))
    )
    baseline_pair = np.concatenate((baseline_first_rows, baseline_second_rows), axis=1)
    result = {
        "first_delta18": first_delta,
        "second_delta18": second_delta,
        "mean_delta18": mean_delta,
        "transition_delta18": transition,
        "pair_delta36": pair,
        "pair_transition_delta54": pair_transition,
        "ratio_pair18": ratio_pair,
        "first_flank_shape18": first_flank_shape,
        "second_flank_shape18": second_flank_shape,
        "mean_flank_shape18": mean_flank_shape,
        "pair_flank_shape36": np.concatenate(
            (first_flank_shape, second_flank_shape), axis=1
        ),
        "second_delta_baseline36": np.concatenate(
            (second_delta, baseline_second_rows), axis=1
        ),
        "pair_delta_baseline72": np.concatenate((pair, baseline_pair), axis=1),
    }
    result["second_delta_plus_flank36"] = np.concatenate(
        (second_delta, second_flank_shape), axis=1
    )
    result["pair_delta_plus_flank72"] = np.concatenate(
        (pair, result["pair_flank_shape36"]), axis=1
    )
    result["pair_transition_plus_flank90"] = np.concatenate(
        (pair_transition, result["pair_flank_shape36"]), axis=1
    )
    result["pair_flank_baseline72"] = np.concatenate(
        (result["pair_flank_shape36"], baseline_pair), axis=1
    )
    result["pair_delta_flank_baseline108"] = np.concatenate(
        (pair, result["pair_flank_shape36"], baseline_pair), axis=1
    )
    for name, values in tuple(result.items()):
        result[f"{name}_unit_amp"] = _unit_amplitude(values)
    return result


def load_contact(path: Path, point: int, *, expected_depth_mm: float = 0.10) -> dict:
    payload = json.loads(path.read_text("utf-8"))
    commission = payload["commissioning"]
    if abs(float(commission["target"]["depth_mm"]) - expected_depth_mm) > 1e-9:
        raise ValueError(f"{path.name}: wrong indentation depth")
    frames = {int(frame["sequence"]): frame for frame in payload["frames"]}
    trigger = int(commission.get("trigger_sequence", 96))
    baseline_frames = [frames[sequence] for sequence in range(trigger - 64, trigger)]
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commission["frame_receipts"]
    }
    events = {item["kind"]: int(item["host_monotonic_ns"]) for item in commission["events"]}
    held_frames = [
        frame
        for sequence, frame in sorted(frames.items())
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= receipts[sequence]
        < events["release_started"]
    ]
    if len(baseline_frames) != 64 or len(held_frames) < 25:
        raise ValueError(f"{path.name}: insufficient causal baseline/hold frames")
    baseline_first_matrix = _matrix(baseline_frames, "first_code")
    baseline_second_matrix = _matrix(baseline_frames, "second_code")
    held_first = _matrix(held_frames, "first_code")
    held_second = _matrix(held_frames, "second_code")
    baseline_first = np.median(baseline_first_matrix, axis=0)
    baseline_second = np.median(baseline_second_matrix, axis=0)
    frame_features = _features(
        held_first, held_second, baseline_first, baseline_second
    )
    median_features = {
        name: np.median(values, axis=0) for name, values in frame_features.items()
    }
    target = commission["target"]
    # Keep one calibration coordinate frame even when Mach3's live work zero
    # is changed between sessions.  Relative distances are unchanged by a
    # pure re-zero, but the stored runtime coordinates must not jump with it.
    xy = [
        float(value)
        for value in target.get(
            "calibration_work_xy_mm", [target["work_x"], target["work_y"]]
        )
    ]
    return {
        "path": str(path.resolve()),
        "point_index": int(point),
        "depth_mm": float(commission["target"]["depth_mm"]),
        "xy": xy,
        "frames": len(held_frames),
        "median_features": median_features,
        "frame_features": frame_features,
    }


def load_rounds(directory: Path) -> dict[str, list[dict]]:
    rounds = {}
    for name, template in ROUND_PATTERNS.items():
        rounds[name] = [
            load_contact(directory / template.format(point=point), point)
            for point in range(1, 18)
        ]
    return rounds


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
            "logistic_c1",
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        ),
    ]


def evaluate(rounds: dict[str, list[dict]]) -> dict:
    reference = next(iter(rounds.values()))
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float) for row in reference
    }
    feature_names = tuple(reference[0]["median_features"])
    ranked = []
    for feature in feature_names:
        for model_name, template in estimators():
            folds = []
            for held_name, test in rounds.items():
                train = [
                    row
                    for name, records in rounds.items()
                    if name != held_name
                    for row in records
                ]
                model = clone(template).fit(
                    np.stack([row["median_features"][feature] for row in train]),
                    np.asarray([row["point_index"] for row in train], dtype=int),
                )
                total = exact = within = majority_exact = 0
                errors = []
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
                        ],
                        dtype=float,
                    )
                    total += len(prediction)
                    exact += int(np.sum(prediction == truth))
                    within += int(np.sum(point_errors <= 4.0 + 1e-6))
                    majority_exact += int(majority == truth)
                    errors.extend(point_errors.tolist())
                    per_point.append(
                        {
                            "truth": truth,
                            "frames": len(prediction),
                            "majority": int(majority),
                            "exact_percent": float(100.0 * np.mean(prediction == truth)),
                            "prediction_counts": {
                                str(label): int(count) for label, count in sorted(counts.items())
                            },
                        }
                    )
                errors_array = np.asarray(errors)
                folds.append(
                    {
                        "held_round": held_name,
                        "trial_majority_exact": majority_exact,
                        "trial_majority_exact_percent": 100.0 * majority_exact / 17.0,
                        "frames": total,
                        "frame_exact_percent": 100.0 * exact / total,
                        "frame_within_4mm_percent": 100.0 * within / total,
                        "median_error_mm": float(np.median(errors_array)),
                        "p95_error_mm": float(np.percentile(errors_array, 95)),
                        "maximum_error_mm": float(np.max(errors_array)),
                        "per_point": per_point,
                    }
                )
            ranked.append(
                {
                    "feature": feature,
                    "estimator": model_name,
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
        "schema": "f18_causal_frame_whole_round_loso_v1",
        "rounds": list(rounds),
        "contacts": 68,
        "depth_mm": 0.10,
        "training_unit": "one median vector per physical contact",
        "test_unit": "every causal held frame from an excluded physical round",
        "temporal_window_frames": 1,
        "random_frame_split_used": False,
        "best": ranked[0],
        "ranked": ranked,
        "physical_15hz_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = evaluate(load_rounds(args.capture_directory))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
