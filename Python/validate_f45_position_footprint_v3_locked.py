"""Validate a frozen F45 position/footprint model on untouched depth rounds."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from select_f45_multidepth_position_candidate import transform
from select_f45_activation_detector_v6 import phase_features


FEATURE = "pair_transition_delta45"
TOOL_RADIUS_MM = 4.0
VALIDATION_MIN_HELD_FRAMES = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_validation_manifest(path: Path, depth: float, lock_time: datetime) -> dict:
    value = json.loads(path.read_text("utf-8"))
    if (
        value.get("schema") != "f45_visible_depth_round_v1"
        or not value.get("complete")
        or set(value.get("completed_points") or ()) != set(range(1, 18))
        or abs(float(value.get("depth_mm", -1.0)) - depth) > 1e-9
        or parse_time(value["created_utc"]) <= lock_time
    ):
        raise ValueError(f"{path}: validation round is incomplete, mismatched, or pre-lock")
    for point in range(1, 18):
        entry = value["captures"][str(point)]
        capture = Path(entry["capture"])
        if sha256(capture) != entry["capture_sha256"] or not all(entry["checks"].values()):
            raise ValueError(f"{capture}: guarded capture hash/check mismatch")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--validation-0p35", type=Path)
    parser.add_argument("--validation-0p65", type=Path)
    parser.add_argument(
        "--validation",
        action="append",
        default=[],
        type=Path,
        help="validation manifest in the same order as the lock's untouched depths",
    )
    parser.add_argument("--activation-model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    lock = json.loads(args.lock.read_text("utf-8"))
    lock_schema = lock.get("schema")
    if lock_schema not in (
        "f45_position_footprint_lock_v3",
        "f45_position_footprint_lock_v4",
        "f45_position_footprint_lock_v5",
        "f45_position_footprint_lock_v6",
        "f45_position_footprint_lock_v7",
    ):
        raise ValueError("unexpected model lock")
    v4 = lock_schema == "f45_position_footprint_lock_v4"
    v5 = lock_schema == "f45_position_footprint_lock_v5"
    v6 = lock_schema == "f45_position_footprint_lock_v6"
    v7 = lock_schema == "f45_position_footprint_lock_v7"
    expected_estimator = (
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "posterior_probability_xy_causal_ema",
            "position_probability_temperature": 3.0,
            "position_ema_alpha": 0.25,
            "reset_position_state_on_no_contact": True,
            "pressure_path": "unsmoothed_full_rate_feature_energy",
        }
        if v7
        else
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "hard_class_xy_causal_mean",
            "position_mean_window_frames": 5,
            "reset_position_state_on_no_contact": True,
            "pressure_path": "unsmoothed_full_rate_feature_energy",
        }
        if v6
        else
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "hard_class_xy_causal_ema",
            "position_ema_alpha": 0.25,
            "reset_position_state_on_no_contact": True,
            "pressure_path": "unsmoothed_full_rate_feature_energy",
        }
        if v5
        else
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "hard_class_to_calibrated_xy",
        }
        if v4
        else {
            "class": "sklearn.ensemble.ExtraTreesClassifier",
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 20260908,
            "n_jobs_fit": -1,
            "decoder": "hard_class_to_calibrated_xy",
        }
    )
    if (
        lock["position"]["feature"] != FEATURE
        or lock["position"]["preprocessing"]
        != ("pair_unit90" if (v6 or v7) else ("block_unit135" if v4 else "unit135"))
        or lock["position"]["estimator"] != expected_estimator
    ):
        raise ValueError("position algorithm differs from locked specification")
    canonical = json.dumps(
        lock["training"]["captures"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != lock["training"]["capture_manifest_sha256"]:
        raise ValueError("training capture manifest digest differs from lock")

    raw_parts = []
    label_parts = []
    for item in lock["training"]["captures"]:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"{path}: training capture changed after lock")
        matrix = causal_held_features(path)[FEATURE]
        raw_parts.append(matrix)
        label_parts.append(np.full(len(matrix), int(item["point"]), dtype=int))
    raw_train = np.concatenate(raw_parts)
    label_train = np.concatenate(label_parts)
    if v4 or v5 or v6 or v7:
        model = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ).fit(transform(raw_train, lock["position"]["preprocessing"]), label_train)
    else:
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=20260908,
            n_jobs=-1,
        ).fit(transform(raw_train, "unit135"), label_train)
    coordinates = {
        int(point): np.asarray(xy, dtype=float)
        for point, xy in lock["position"]["coordinates_by_point"].items()
    }
    if v6 or v7:
        expected_activation = {
            "feature": FEATURE,
            "preprocessing": "raw135",
            "estimator": {
                "class": "sklearn.ensemble.ExtraTreesClassifier",
                "n_estimators": 300,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": 20260908,
                "n_jobs_fit": -1,
            },
            "temporal_smoothing_used": False,
            "development": lock["activation"]["development"],
        }
        if lock.get("activation") != expected_activation:
            raise ValueError("activation algorithm differs from locked specification")
        activation_baseline_parts = []
        activation_held_parts = []
        for item in lock["training"]["captures"]:
            baseline_values, held_values = phase_features(Path(item["path"]))
            activation_baseline_parts.append(baseline_values)
            activation_held_parts.append(held_values)
        activation_baseline_train = np.concatenate(activation_baseline_parts)
        activation_held_train = np.concatenate(activation_held_parts)
        activation_x = np.concatenate((activation_baseline_train, activation_held_train))
        activation_y = np.concatenate((
            np.zeros(len(activation_baseline_train), dtype=int),
            np.ones(len(activation_held_train), dtype=int),
        ))
        activation_detector = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=20260908,
            n_jobs=-1,
        ).fit(activation_x, activation_y)
        threshold = None
        activation = None
    else:
        if args.activation_model is None:
            parser.error("legacy locks require --activation-model")
        activation = joblib.load(args.activation_model)
        threshold = float(activation["activation"]["threshold_codes"])
    lock_time = parse_time(lock["locked_utc"])
    validation_depths = [float(value) for value in lock["validation_protocol"]["untouched_depths_mm"]]
    if args.validation:
        validation_paths = args.validation
    else:
        validation_paths = [args.validation_0p35, args.validation_0p65]
    if len(validation_paths) != len(validation_depths) or any(path is None for path in validation_paths):
        parser.error("provide one --validation per precommitted depth")
    manifests = {
        depth: load_validation_manifest(path, depth, lock_time)
        for depth, path in zip(validation_depths, validation_paths, strict=True)
    }

    summaries = {}
    details = {}
    for depth, manifest in manifests.items():
        trials = []
        total = detected_total = exact_total = within_total = 0
        baseline_total = baseline_false_positive_total = 0
        sample_gate = True
        for point in range(1, 18):
            entry = manifest["captures"][str(point)]
            path = Path(entry["capture"])
            if v6 or v7:
                # The commanded one-second hold yields 24--28 usable frames at
                # the measured ~31 Hz because motion/event timestamps and frame
                # receipts are asynchronous.  Twenty frames is an ingestion
                # integrity floor, not a scoring/model threshold; every usable
                # held frame is still included in the locked validation.
                baseline_raw, raw = phase_features(
                    path,
                    minimum_held_frames=VALIDATION_MIN_HELD_FRAMES,
                )
                baseline_detected = np.asarray(
                    activation_detector.predict(baseline_raw), dtype=int
                ) == 1
                detected = np.asarray(activation_detector.predict(raw), dtype=int) == 1
                baseline_total += len(baseline_detected)
                baseline_false_positive_total += int(np.sum(baseline_detected))
            else:
                raw = causal_held_features(path)[FEATURE]
                detected = np.sqrt(np.mean(raw[:, :45] ** 2, axis=1)) >= threshold
            position_values = transform(raw, lock["position"]["preprocessing"])
            prediction = np.asarray(model.predict(position_values), dtype=int)
            posterior_xy = None
            if v7:
                probability = np.asarray(model.predict_proba(position_values), dtype=float)
                logits = np.log(np.clip(probability, 1e-300, 1.0)) / 3.0
                logits -= np.max(logits, axis=1, keepdims=True)
                probability = np.exp(logits)
                probability /= np.sum(probability, axis=1, keepdims=True)
                class_xy = np.stack([coordinates[int(label)] for label in model.classes_])
                posterior_xy = probability @ class_xy
            if v5 or v6 or v7:
                # Pressure/amplitude remains an unsmoothed per-frame value.
                # Only the contact centre is filtered, and its state is reset
                # for every independent contact (and after any no-contact
                # frame in a live implementation).
                filtered_xy = []
                state = None
                history = []
                for frame_index, (is_detected, label) in enumerate(
                    zip(detected, prediction, strict=True)
                ):
                    if not is_detected:
                        state = None
                        history.clear()
                        continue
                    value = posterior_xy[frame_index] if v7 else coordinates[int(label)]
                    if v6:
                        history.append(value.copy())
                        if len(history) > 5:
                            history.pop(0)
                        state = np.mean(history, axis=0)
                    else:
                        state = value.copy() if state is None else 0.25 * value + 0.75 * state
                    filtered_xy.append(state.copy())
                held_xy = np.asarray(filtered_xy, dtype=float)
                if len(held_xy):
                    label_values = np.asarray(sorted(coordinates), dtype=int)
                    label_xy = np.stack([coordinates[int(label)] for label in label_values])
                    held_prediction = label_values[np.argmin(
                        np.linalg.norm(held_xy[:, None, :] - label_xy[None, :, :], axis=2),
                        axis=1,
                    )]
                    errors = np.linalg.norm(held_xy - coordinates[point], axis=1)
                    trial_xy = np.mean(held_xy, axis=0)
                    majority = int(label_values[np.argmin(np.linalg.norm(label_xy - trial_xy, axis=1))])
                    majority_error = float(np.linalg.norm(trial_xy - coordinates[point]))
                else:
                    held_prediction = np.asarray([], dtype=int)
                    errors = np.asarray([], dtype=float)
                    majority = None
                    majority_error = None
            else:
                held_prediction = prediction[detected]
                errors = np.asarray(
                    [np.linalg.norm(coordinates[int(label)] - coordinates[point]) for label in held_prediction],
                    dtype=float,
                )
                counts = Counter(held_prediction.tolist())
                majority = min(counts, key=lambda label: (-counts[label], label)) if counts else None
                majority_error = (
                    float(np.linalg.norm(coordinates[int(majority)] - coordinates[point]))
                    if majority is not None
                    else None
                )
            counts = Counter(held_prediction.tolist())
            sample_gate = sample_gate and bool(entry["checks"]["sampling_rate_gate"])
            trials.append(
                {
                    "point": point,
                    "capture": str(path.resolve()),
                    "capture_sha256": sha256(path),
                    "held_frames": int(len(raw)),
                    "detected_frames": int(np.sum(detected)),
                    "majority_prediction": majority,
                    "majority_exact": majority == point,
                    "majority_error_mm": majority_error,
                    "majority_within_4mm": bool(
                        majority_error is not None and majority_error <= TOOL_RADIUS_MM + 1e-6
                    ),
                    "prediction_counts": {str(key): int(value) for key, value in sorted(counts.items())},
                    "frame_exact_percent": (
                        float(100.0 * np.mean(held_prediction == point)) if len(held_prediction) else None
                    ),
                    "frame_within_4mm_percent": (
                        float(100.0 * np.mean(errors <= TOOL_RADIUS_MM + 1e-6)) if len(errors) else None
                    ),
                }
            )
            total += len(raw)
            detected_total += int(np.sum(detected))
            exact_total += int(np.sum(held_prediction == point))
            within_total += int(np.sum(errors <= TOOL_RADIUS_MM + 1e-6))
        summaries[f"{depth:.2f}"] = {
            "truth_depth_mm": depth,
            "trial_nearest_label_exact": int(sum(row["majority_exact"] for row in trials)),
            "trial_within_4mm": int(sum(row["majority_within_4mm"] for row in trials)),
            "held_frames": total,
            "detected_frames": detected_total,
            "detected_held_frame_percent": 100.0 * detected_total / total,
            **(
                {
                    "baseline_frames": baseline_total,
                    "baseline_false_positive_frames": baseline_false_positive_total,
                    "baseline_false_positive_percent": 100.0 * baseline_false_positive_total / baseline_total,
                }
                if (v6 or v7)
                else {}
            ),
            "frame_exact_percent": 100.0 * exact_total / detected_total if detected_total else None,
            "frame_within_4mm_percent": 100.0 * within_total / detected_total if detected_total else None,
            "board_sampling_rate_gate_for_15hz": sample_gate,
        }
        details[f"{depth:.2f}"] = trials

    gates = lock["validation_protocol"]["gates"]
    exact_total_trials = sum(item["trial_nearest_label_exact"] for item in summaries.values())
    passed = (
        exact_total_trials >= int(gates["trial_nearest_label_exact_min_total"])
        and all(
            item["trial_nearest_label_exact"] >= int(gates["trial_nearest_label_exact_min_each_depth"])
            and item["trial_within_4mm"] == 17
            and item["frame_within_4mm_percent"] >= float(gates["frame_within_4mm_percent_min_each_depth"])
            and item["detected_held_frame_percent"] >= float(gates["detected_held_frame_percent_min_each_depth"])
            and (
                not (v6 or v7)
                or item["baseline_false_positive_percent"]
                <= float(gates["baseline_false_positive_percent_max_each_depth"])
            )
            and item["board_sampling_rate_gate_for_15hz"]
            for item in summaries.values()
        )
    )
    result = {
        "schema": (
            "f45_position_footprint_validation_v7"
            if v7
            else "f45_position_footprint_validation_v6"
            if v6
            else "f45_position_footprint_validation_v5"
            if v5
            else ("f45_position_footprint_validation_v4" if v4 else "f45_position_footprint_validation_v3")
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lock": {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)},
        "validation_captured_after_lock": True,
        "random_frame_split_used": False,
        "training_frames": int(len(raw_train)),
        "activation": (
            {
                "feature": FEATURE,
                "preprocessing": "raw135",
                "estimator": "extra_trees_300_leaf10",
                "training_baseline_frames": int(len(activation_baseline_train)),
                "training_held_frames": int(len(activation_held_train)),
                "temporal_smoothing_used": False,
            }
            if (v6 or v7)
            else {
                "threshold_codes": threshold,
                "source_model": str(args.activation_model.resolve()),
                "source_model_sha256": sha256(args.activation_model),
            }
        ),
        "deployment_gate_passed": passed,
        "trial_nearest_label_exact_total": exact_total_trials,
        "summaries": summaries,
        "per_point": details,
        "claims": {
            "position_tolerance_basis": "4 mm radius of fixed 8 mm press face",
            "nominal_8mm_tool_face_area_mm2": 50.26548245743669,
            "physical_contact_area_verified": False,
            "force_measured": False,
            "physical_15hz_pressure_response_verified": False,
            "sampling_rate_only_gate_for_15hz": True,
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_point"}, ensure_ascii=False, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
