"""Validate the second locked causal F45 models on post-lock depth rounds."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from select_f45_multidepth_frame_depth import transform_depth
from select_f45_multidepth_position_candidate import transform as transform_position


FEATURE = "pair_transition_delta45"
LOW_DEPTH = 0.25
HIGH_DEPTH = 0.55


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
        or abs(float(value.get("depth_mm", -1)) - depth) > 1e-9
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
    parser.add_argument("--validation-0p25", required=True, type=Path)
    parser.add_argument("--validation-0p55", required=True, type=Path)
    parser.add_argument("--activation-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    lock = json.loads(args.lock.read_text("utf-8"))
    if lock.get("schema") != "f45_multidepth_causal_frame_models_lock_v2":
        raise ValueError("unexpected model lock")
    expected_position = {
        "feature": FEATURE,
        "preprocessing": "pair_unit90",
        "estimator": "lda_shrinkage",
        "kind": "classifier",
        "classifier": {
            "class": "sklearn.pipeline(StandardScaler,LinearDiscriminantAnalysis)",
            "lda_solver": "lsqr",
            "lda_shrinkage": "auto",
        },
        "development_worst_trial_majority_exact": 14,
        "development_worst_frame_exact_percent": lock["position"]["development_worst_frame_exact_percent"],
        "development_worst_frame_within_4mm_percent": 100.0,
    }
    if lock["position"] != expected_position:
        raise ValueError("position algorithm differs from locked specification")
    if (
        lock["depth_proxy"]["preprocessing"] != "shape_amplitude_point"
        or lock["depth_proxy"]["estimator"] != "ridge_a100"
    ):
        raise ValueError("depth algorithm differs from locked specification")

    capture_manifest = lock["training"]["captures"]
    canonical = json.dumps(
        capture_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != lock["training"]["capture_manifest_sha256"]:
        raise ValueError("training capture manifest digest differs from lock")

    raw_parts = []
    point_parts = []
    depth_parts = []
    coordinates = {}
    for item in capture_manifest:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"{path}: training capture changed after lock")
        matrix = causal_held_features(path)[FEATURE]
        point = int(item["point"])
        raw_parts.append(matrix)
        point_parts.append(np.full(len(matrix), point, dtype=int))
        depth_parts.append(np.full(len(matrix), float(item["depth_mm"])))
        if point not in coordinates:
            payload = json.loads(path.read_text("utf-8"))
            target = payload["commissioning"]["target"]
            coordinates[point] = np.asarray(
                target.get("calibration_work_xy_mm", [target["work_x"], target["work_y"]]),
                dtype=float,
            )
    if len(capture_manifest) != 187 or set(coordinates) != set(range(1, 18)):
        raise ValueError("locked training corpus is incomplete")

    raw_train = np.concatenate(raw_parts)
    point_train = np.concatenate(point_parts)
    depth_train = np.concatenate(depth_parts)
    position_model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ).fit(transform_position(raw_train, "pair_unit90"), point_train)
    depth_model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(
        transform_depth(raw_train, point_train, "shape_amplitude_point"), depth_train
    )

    activation = joblib.load(args.activation_model)
    threshold = float(activation["activation"]["threshold_codes"])
    lock_time = parse_time(lock["locked_utc"])
    manifests = {
        LOW_DEPTH: load_validation_manifest(args.validation_0p25, LOW_DEPTH, lock_time),
        HIGH_DEPTH: load_validation_manifest(args.validation_0p55, HIGH_DEPTH, lock_time),
    }
    summaries = {}
    details = {}
    for depth, manifest in manifests.items():
        trials = []
        total = detected_total = exact_total = within_total = 0
        for point in range(1, 18):
            path = Path(manifest["captures"][str(point)]["capture"])
            raw = causal_held_features(path)[FEATURE]
            first_response = np.sqrt(np.mean(raw[:, :45] ** 2, axis=1))
            detected = first_response >= threshold
            position_prediction_all = np.asarray(
                position_model.predict(transform_position(raw, "pair_unit90")), dtype=int
            )
            position_prediction = position_prediction_all[detected]
            counts = Counter(position_prediction.tolist())
            majority = min(counts, key=lambda label: (-counts[label], label)) if counts else None
            errors = np.asarray([
                np.linalg.norm(coordinates[int(label)] - coordinates[point])
                for label in position_prediction
            ])
            majority_error = (
                float(np.linalg.norm(coordinates[int(majority)] - coordinates[point]))
                if majority is not None else None
            )
            depth_prediction_all = np.clip(
                np.asarray(depth_model.predict(transform_depth(
                    raw, position_prediction_all, "shape_amplitude_point"
                )), dtype=float),
                0.10,
                0.80,
            )
            depth_prediction = depth_prediction_all[detected]
            median_depth = float(np.median(depth_prediction)) if len(depth_prediction) else None
            trials.append({
                "point": point,
                "capture": str(path.resolve()),
                "capture_sha256": sha256(path),
                "held_frames": int(len(raw)),
                "detected_frames": int(np.sum(detected)),
                "majority_prediction": majority,
                "majority_exact": majority == point,
                "majority_error_mm": majority_error,
                "majority_within_tool_radius": (
                    majority_error <= 4.0 + 1e-6 if majority_error is not None else False
                ),
                "prediction_counts": {str(k): int(v) for k, v in sorted(counts.items())},
                "frame_exact_percent": (
                    float(100.0 * np.mean(position_prediction == point))
                    if len(position_prediction) else None
                ),
                "frame_within_4mm_percent": (
                    float(100.0 * np.mean(errors <= 4.0 + 1e-6)) if len(errors) else None
                ),
                "median_predicted_depth_mm": median_depth,
                "median_depth_absolute_error_mm": (
                    abs(median_depth - depth) if median_depth is not None else None
                ),
                "frame_depth_mae_mm": (
                    float(np.mean(np.abs(depth_prediction - depth))) if len(depth_prediction) else None
                ),
            })
            total += len(raw)
            detected_total += int(np.sum(detected))
            exact_total += int(np.sum(position_prediction == point))
            within_total += int(np.sum(errors <= 4.0 + 1e-6))
        trial_depth_errors = np.asarray([row["median_depth_absolute_error_mm"] for row in trials])
        summaries[f"{depth:.2f}"] = {
            "truth_depth_mm": depth,
            "trial_majority_exact": int(sum(row["majority_exact"] for row in trials)),
            "trial_majority_exact_percent": float(100.0 * np.mean([row["majority_exact"] for row in trials])),
            "trial_majority_within_tool_radius": int(sum(row["majority_within_tool_radius"] for row in trials)),
            "held_frames": total,
            "detected_frames": detected_total,
            "detected_held_frame_percent": 100.0 * detected_total / total,
            "frame_exact_percent": 100.0 * exact_total / detected_total if detected_total else None,
            "frame_within_4mm_percent": 100.0 * within_total / detected_total if detected_total else None,
            "trial_median_prediction_mae_mm": float(np.mean(trial_depth_errors)),
            "trial_median_prediction_p95_absolute_error_mm": float(np.percentile(trial_depth_errors, 95)),
            "trial_median_prediction_maximum_absolute_error_mm": float(np.max(trial_depth_errors)),
        }
        details[f"{depth:.2f}"] = trials

    low = {row["point"]: row for row in details[f"{LOW_DEPTH:.2f}"]}
    high = {row["point"]: row for row in details[f"{HIGH_DEPTH:.2f}"]}
    ordered = [
        point for point in range(1, 18)
        if high[point]["median_predicted_depth_mm"] > low[point]["median_predicted_depth_mm"]
    ]
    low_summary = summaries[f"{LOW_DEPTH:.2f}"]
    high_summary = summaries[f"{HIGH_DEPTH:.2f}"]
    exact_total_trials = low_summary["trial_majority_exact"] + high_summary["trial_majority_exact"]
    position_pass = (
        low_summary["trial_majority_exact"] >= 14
        and high_summary["trial_majority_exact"] >= 17
        and exact_total_trials >= 31
        and all(
            item["trial_majority_within_tool_radius"] == 17
            and item["detected_held_frame_percent"] >= 100.0 - 1e-12
            and item["frame_within_4mm_percent"] >= 100.0 - 1e-12
            for item in summaries.values()
        )
    )
    depth_pass = all(
        item["trial_median_prediction_mae_mm"] <= 0.10 + 1e-12
        and item["trial_median_prediction_p95_absolute_error_mm"] <= 0.20 + 1e-12
        for item in summaries.values()
    ) and len(ordered) == 17
    result = {
        "schema": "f45_multidepth_causal_frame_models_validation_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lock": {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)},
        "validation_captured_after_lock": True,
        "random_frame_split_used": False,
        "training_frames": int(len(raw_train)),
        "activation": {
            "threshold_codes": threshold,
            "source_model": str(args.activation_model.resolve()),
            "source_model_sha256": sha256(args.activation_model),
        },
        "position_gate_passed": position_pass,
        "depth_proxy_gate_passed": depth_pass,
        "deployment_gate_passed": position_pass and depth_pass,
        "summaries": summaries,
        "trial_majority_exact_total": exact_total_trials,
        "per_point_depth_order_0p55_greater_than_0p25": len(ordered),
        "ordered_points": ordered,
        "per_point": details,
        "claims": {
            "force_measured": False,
            "physical_15hz_pressure_response_verified": False,
            "nominal_8mm_tool_face_area_mm2": float(np.pi * 4.0**2),
            "physical_contact_area_verified": False,
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "per_point"},
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if result["deployment_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
