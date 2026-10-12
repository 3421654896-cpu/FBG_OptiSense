"""Select realtime F45 classifiers using causal frames and held-out rounds.

Training observations are one median contact per point and physical round.
Test observations are every held frame from an entirely excluded round.  Each
test contact uses only its first 64 post-warmup, pre-motion frames to construct
the frozen local baseline.  The script is offline and never commands hardware.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.base import clone

from evaluate_coverage_position import _feature_variants
from evaluate_f45_cross_session_models import estimators, load_rounds


MIN_HELD_FRAMES = 20


def causal_held_features(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text("utf-8"))
    commission = payload["commissioning"]
    frames = {int(frame["sequence"]): frame for frame in payload["frames"]}
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commission["frame_receipts"]
    }
    events = {
        item["kind"]: int(item["host_monotonic_ns"])
        for item in commission["events"]
    }
    baseline_frames = [frames[sequence] for sequence in range(16, 80)]
    held_frames = [
        frame
        for sequence, frame in sorted(frames.items())
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= receipts[sequence]
        < events["release_started"]
    ]
    if len(baseline_frames) != 64 or len(held_frames) < MIN_HELD_FRAMES:
        raise ValueError(f"{path.name}: insufficient causal baseline or held frames")

    def matrix(field: str, selected: list[dict]) -> np.ndarray:
        values = np.asarray(
            [[record[field] for record in frame["records"]] for frame in selected],
            dtype=float,
        )
        if values.shape[1:] != (45,) or not np.isfinite(values).all():
            raise ValueError(f"{path.name}: malformed {field} values")
        return values

    baseline_first = matrix("first_code", baseline_frames)
    baseline_second = matrix("second_code", baseline_frames)
    held_first = matrix("first_code", held_frames)
    held_second = matrix("second_code", held_frames)
    bases = {
        "first": np.median(baseline_first, axis=0),
        "second": np.median(baseline_second, axis=0),
        "mean": np.median((baseline_first + baseline_second) / 2.0, axis=0),
    }
    sigma = 1.4826 * np.median(
        np.abs(baseline_second - np.median(baseline_second, axis=0)), axis=0
    )
    rows: dict[str, list[np.ndarray]] = {}
    for first, second in zip(held_first, held_second, strict=True):
        samples = {"first": first, "second": second, "mean": (first + second) / 2.0}
        families = {
            name: _feature_variants(bases[name], samples[name], sigma)
            for name in ("first", "second", "mean")
        }
        features: dict[str, np.ndarray] = {}
        for sample_name, variants in families.items():
            features.update({f"{sample_name}_{name}": value for name, value in variants.items()})
        for name in families["first"]:
            features[f"pair_{name}"] = np.concatenate(
                (families["first"][name], families["second"][name])
            )
        transition = (second - first) - (bases["second"] - bases["first"])
        features["pair_transition_delta45"] = np.concatenate(
            (families["first"]["delta45"], families["second"]["delta45"], transition)
        )
        for name, value in features.items():
            rows.setdefault(name, []).append(np.asarray(value, dtype=float))
    return {name: np.stack(values) for name, values in rows.items()}


def evaluate(rounds: dict[str, list[dict]]) -> dict:
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in next(iter(rounds.values()))
    }
    classifiers = [item for item in estimators() if item[0] == "classifier"]
    causal = {
        (round_name, int(row["point_index"])): causal_held_features(Path(row["path"]))
        for round_name, records in rounds.items()
        for row in records
    }
    feature_names = tuple(next(iter(causal.values())))
    ranked = []
    for feature in feature_names:
        for _kind, estimator_name, template in classifiers:
            folds = []
            failed = None
            for held_name, held_records in rounds.items():
                train_records = [
                    row for name, records in rounds.items() if name != held_name for row in records
                ]
                x_train = np.stack([row["features"][feature] for row in train_records])
                y_train = np.asarray([row["point_index"] for row in train_records], dtype=int)
                try:
                    model = clone(template).fit(x_train, y_train)
                    trial_rows = []
                    fold_exact = 0
                    fold_total = 0
                    fold_within = 0
                    trial_majority_exact = 0
                    for truth in held_records:
                        point = int(truth["point_index"])
                        vectors = causal[(held_name, point)][feature]
                        prediction = np.asarray(model.predict(vectors), dtype=int)
                        count = Counter(prediction.tolist())
                        majority = min(count, key=lambda label: (-count[label], label))
                        trial_majority_exact += int(majority == point)
                        errors = np.asarray(
                            [np.linalg.norm(coordinates[int(label)] - coordinates[point]) for label in prediction],
                            dtype=float,
                        )
                        exact = int(np.sum(prediction == point))
                        fold_exact += exact
                        fold_total += len(prediction)
                        fold_within += int(np.sum(errors <= 4.0 + 1e-6))
                        trial_rows.append(
                            {
                                "truth": point,
                                "frames": int(len(prediction)),
                                "exact_frames": exact,
                                "majority": int(majority),
                                "prediction_counts": {str(k): int(v) for k, v in sorted(count.items())},
                                "p95_error_mm": float(np.percentile(errors, 95)),
                            }
                        )
                except Exception as exc:
                    failed = f"{type(exc).__name__}: {exc}"
                    break
                folds.append(
                    {
                        "held_round": held_name,
                        "trials": 17,
                        "trial_majority_exact": trial_majority_exact,
                        "trial_majority_exact_percent": 100.0 * trial_majority_exact / 17.0,
                        "frames": fold_total,
                        "frame_exact_percent": 100.0 * fold_exact / fold_total,
                        "frame_within_4mm_percent": 100.0 * fold_within / fold_total,
                        "per_point": trial_rows,
                    }
                )
            if failed:
                continue
            ranked.append(
                {
                    "feature": feature,
                    "estimator": estimator_name,
                    "worst_fold_trial_majority_exact_percent": min(
                        fold["trial_majority_exact_percent"] for fold in folds
                    ),
                    "worst_fold_frame_exact_percent": min(fold["frame_exact_percent"] for fold in folds),
                    "worst_fold_frame_within_4mm_percent": min(
                        fold["frame_within_4mm_percent"] for fold in folds
                    ),
                    "mean_frame_exact_percent": float(
                        np.mean([fold["frame_exact_percent"] for fold in folds])
                    ),
                    "folds": folds,
                }
            )
    ranked.sort(
        key=lambda row: (
            -row["worst_fold_trial_majority_exact_percent"],
            -row["worst_fold_frame_exact_percent"],
            -row["worst_fold_frame_within_4mm_percent"],
            -row["mean_frame_exact_percent"],
        )
    )
    return {
        "schema": "f45_causal_frame_whole_round_loso_audit_v1",
        "rounds": list(rounds),
        "random_frame_split_used": False,
        "causal_baseline_frames": 64,
        "best": ranked[0],
        "ranked": ranked,
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
