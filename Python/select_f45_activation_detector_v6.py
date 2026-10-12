"""Select a per-frame F45 contact detector with whole-round holdout.

Unlike the legacy scalar RMS threshold, these candidates use the complete
135-value causal response shape.  No temporal averaging is applied, so the
pressure/amplitude path remains available at the full board frame rate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_coverage_position import _feature_variants
from select_f45_activation_threshold_v6 import load_manifest, sha256
from select_f45_multidepth_position_candidate import transform


FEATURE = "pair_transition_delta45"


def phase_features(
    path: Path,
    *,
    minimum_held_frames: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text("utf-8"))
    commissioning = payload["commissioning"]
    frames = {int(frame["sequence"]): frame for frame in payload["frames"]}
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commissioning["frame_receipts"]
    }
    events = {
        item["kind"]: int(item["host_monotonic_ns"])
        for item in commissioning["events"]
    }
    baseline_frames = [frames[sequence] for sequence in range(16, 80)]
    held_frames = [
        frame
        for sequence, frame in sorted(frames.items())
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= receipts[sequence]
        < events["release_started"]
    ]
    if len(baseline_frames) != 64 or len(held_frames) < minimum_held_frames:
        raise ValueError(f"{path}: insufficient causal baseline or held frames")

    def matrix(field: str, selected: list[dict]) -> np.ndarray:
        values = np.asarray(
            [[record[field] for record in frame["records"]] for frame in selected],
            dtype=float,
        )
        if values.shape[1:] != (45,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: malformed {field} values")
        return values

    baseline_first = matrix("first_code", baseline_frames)
    baseline_second = matrix("second_code", baseline_frames)
    held_first = matrix("first_code", held_frames)
    held_second = matrix("second_code", held_frames)
    first_base = np.median(baseline_first, axis=0)
    second_base = np.median(baseline_second, axis=0)
    sigma = 1.4826 * np.median(
        np.abs(baseline_second - np.median(baseline_second, axis=0)), axis=0
    )

    def build(first_rows: np.ndarray, second_rows: np.ndarray) -> np.ndarray:
        result = []
        for first, second in zip(first_rows, second_rows, strict=True):
            first_delta = _feature_variants(first_base, first, sigma)["delta45"]
            second_delta = _feature_variants(second_base, second, sigma)["delta45"]
            transition = (second - first) - (second_base - first_base)
            result.append(np.concatenate((first_delta, second_delta, transition)))
        return np.stack(result)

    return build(baseline_first, baseline_second), build(held_first, held_second)


def candidates():
    for representation in ("raw135", "signed_sqrt_unit135", "block_unit135", "segment_energy9"):
        yield representation, "lda_shrinkage_balanced", make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]),
        )
        for c_value in (0.1, 1.0, 10.0):
            yield representation, f"logistic_balanced_c{c_value:g}", make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=20260908,
                ),
            )
        for leaf in (1, 3, 10):
            yield representation, f"extra_trees_300_leaf{leaf}", ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=20260908,
                n_jobs=-1,
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--development-round", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    lock = json.loads(args.lock.read_text("utf-8"))
    captures = [dict(item) for item in lock["training"]["captures"]]
    for manifest in args.development_round:
        captures.extend(load_manifest(manifest))
    unique = {}
    for item in captures:
        path = Path(item["path"])
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"{path}: capture changed")
        unique[digest] = item

    by_round: dict[str, list[dict]] = {}
    for item in unique.values():
        baseline, held = phase_features(Path(item["path"]))
        by_round.setdefault(str(item["round"]), []).append(
            {**item, "baseline": baseline, "held": held}
        )

    ranked = []
    for representation, name, template in candidates():
        folds = []
        for held_name, test_records in by_round.items():
            train_records = [
                row for round_name, records in by_round.items()
                if round_name != held_name for row in records
            ]
            train_baseline = np.concatenate([row["baseline"] for row in train_records])
            train_held = np.concatenate([row["held"] for row in train_records])
            train_x = np.concatenate((train_baseline, train_held))
            train_y = np.concatenate((
                np.zeros(len(train_baseline), dtype=int),
                np.ones(len(train_held), dtype=int),
            ))
            model = clone(template).fit(transform(train_x, representation), train_y)
            test_baseline = np.concatenate([row["baseline"] for row in test_records])
            test_held = np.concatenate([row["held"] for row in test_records])
            baseline_prediction = model.predict(transform(test_baseline, representation))
            held_prediction = model.predict(transform(test_held, representation))
            per_capture = []
            for row in test_records:
                predicted = model.predict(transform(row["held"], representation))
                per_capture.append(
                    {
                        "point": int(row["point"]),
                        "held_frames": int(len(predicted)),
                        "detected_frames": int(np.sum(predicted == 1)),
                        "detected_percent": float(100.0 * np.mean(predicted == 1)),
                    }
                )
            folds.append(
                {
                    "held_round": held_name,
                    "depth_mm": float(np.median([float(row["depth_mm"]) for row in test_records])),
                    "captures": len(test_records),
                    "baseline_frames": int(len(baseline_prediction)),
                    "baseline_false_positive_frames": int(np.sum(baseline_prediction == 1)),
                    "baseline_false_positive_percent": float(100.0 * np.mean(baseline_prediction == 1)),
                    "held_frames": int(len(held_prediction)),
                    "held_detected_frames": int(np.sum(held_prediction == 1)),
                    "held_detected_percent": float(100.0 * np.mean(held_prediction == 1)),
                    "minimum_capture_held_detected_percent": float(
                        min(row["detected_percent"] for row in per_capture)
                    ),
                    "per_capture": per_capture,
                }
            )
        ranked.append(
            {
                "feature": FEATURE,
                "representation": representation,
                "estimator": name,
                "pressure_path": "unsmoothed_full_rate_feature_energy",
                "contact_detection_path": "unsmoothed_full_rate_binary_shape_classifier",
                "worst_round_held_detected_percent": float(min(row["held_detected_percent"] for row in folds)),
                "worst_capture_held_detected_percent": float(min(row["minimum_capture_held_detected_percent"] for row in folds)),
                "maximum_round_baseline_false_positive_percent": float(max(row["baseline_false_positive_percent"] for row in folds)),
                "total_baseline_false_positive_frames": int(sum(row["baseline_false_positive_frames"] for row in folds)),
                "total_baseline_frames": int(sum(row["baseline_frames"] for row in folds)),
                "total_held_missed_frames": int(sum(row["held_frames"] - row["held_detected_frames"] for row in folds)),
                "total_held_frames": int(sum(row["held_frames"] for row in folds)),
                "folds": folds,
            }
        )

    ranked.sort(
        key=lambda row: (
            -row["worst_capture_held_detected_percent"],
            -row["worst_round_held_detected_percent"],
            row["maximum_round_baseline_false_positive_percent"],
            row["total_baseline_false_positive_frames"],
            row["total_held_missed_frames"],
        )
    )
    result = {
        "schema": "f45_activation_detector_selection_v6",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_lock": {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)},
        "random_frame_split_used": False,
        "temporal_smoothing_used": False,
        "development_rounds": len(by_round),
        "development_captures": len(unique),
        "best": ranked[0],
        "ranked": ranked,
        "deployment_eligible": False,
        "physical_15hz_verified": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps([{key: value for key, value in row.items() if key != "folds"} for row in ranked[:12]], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
