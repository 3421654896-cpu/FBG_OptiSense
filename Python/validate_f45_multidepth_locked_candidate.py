"""Validate the locked F45 position and indentation candidates on new rounds."""

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

from evaluate_f45_cross_session_models import load_rounds
from select_f45_multidepth_position_candidate import load_depth_manifest, transform


FEATURE = "pair_transition_delta45"
SETTLE_NS = 200_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def inverse_depth(curve: dict, value: float) -> float:
    depths = np.asarray(curve["depth_knots_mm"], dtype=float)
    levels = np.asarray(curve["isotonic_response_knots"], dtype=float)
    unique = []
    for level in np.unique(levels):
        mask = np.isclose(levels, level, rtol=0.0, atol=1e-12)
        unique.append((float(level), float(np.mean(depths[mask]))))
    xp = np.asarray([item[0] for item in unique])
    fp = np.asarray([item[1] for item in unique])
    if len(xp) == 1:
        return float(fp[0])
    return float(np.interp(value, xp, fp, left=fp[0], right=fp[-1]))


def validate_manifest(
    path: Path, expected_depth: float, after: datetime, *, require_after: bool = True
) -> dict:
    manifest = json.loads(path.read_text("utf-8"))
    if (
        manifest.get("schema") != "f45_visible_depth_round_v1"
        or not manifest.get("complete")
        or set(manifest.get("completed_points") or ()) != set(range(1, 18))
        or set((manifest.get("captures") or {}).keys()) != {str(i) for i in range(1, 18)}
        or abs(float(manifest.get("depth_mm", -1)) - expected_depth) > 1e-9
    ):
        raise ValueError(f"{path}: incomplete or mismatched validation round")
    created = parse_time(manifest["created_utc"])
    if require_after and created <= after:
        raise ValueError(f"{path}: validation round was not captured after the lock")
    for point in range(1, 18):
        entry = manifest["captures"][str(point)]
        capture = Path(entry["capture"])
        if sha256(capture) != entry["capture_sha256"]:
            raise ValueError(f"{capture}: hash changed after guarded capture")
        if not all((entry.get("checks") or {}).values()):
            raise ValueError(f"{capture}: guarded capture checks are not all true")
    return manifest


def capture_arrays(path: Path) -> dict:
    data = json.loads(path.read_text("utf-8"))
    commissioning = data["commissioning"]
    frames = {int(frame["sequence"]): frame for frame in data["frames"]}
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commissioning["frame_receipts"]
    }
    events = {
        str(item["kind"]): int(item["host_monotonic_ns"])
        for item in commissioning["events"]
    }
    trigger = int(commissioning["trigger_sequence"])
    baseline_sequences = sorted(
        sequence for sequence in receipts if 16 <= sequence < trigger
    )[:64]
    held_sequences = sorted(
        sequence for sequence, received in receipts.items()
        if events["contact_reached_commanded_depth"] + SETTLE_NS
        <= received < events["release_started"]
    )
    if len(baseline_sequences) != 64 or len(held_sequences) < 25:
        raise ValueError(f"{path}: expected exactly 64 baseline and at least 25 held frames")

    def matrix(sequences, field):
        values = np.asarray([
            [record[field] for record in frames[sequence]["records"]]
            for sequence in sequences
        ], dtype=float)
        if values.shape[1:] != (45,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: malformed {field}")
        return values

    base_first_matrix = matrix(baseline_sequences, "first_code")
    base_second_matrix = matrix(baseline_sequences, "second_code")
    held_first = matrix(held_sequences, "first_code")
    held_second = matrix(held_sequences, "second_code")
    base_first = np.median(base_first_matrix, axis=0)
    base_second = np.median(base_second_matrix, axis=0)
    first_delta = held_first - base_first
    second_delta = held_second - base_second
    transition = (held_second - held_first) - (base_second - base_first)
    frame_features = np.concatenate((first_delta, second_delta, transition), axis=1)
    response_first = np.sqrt(np.mean(first_delta**2, axis=1))
    # This matches the selection implementation: median 45-D held vector,
    # followed by RMS.  It is not a median of per-frame scalar RMS values.
    contact_second_response = float(np.sqrt(np.mean(np.median(second_delta, axis=0) ** 2)))
    return {
        "frame_features": frame_features,
        "first_response": response_first,
        "contact_second_response": contact_second_response,
        "held_frames": len(held_sequences),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-lock", required=True, type=Path)
    parser.add_argument("--depth-lock", required=True, type=Path)
    parser.add_argument("--train-0p10", required=True, type=Path)
    parser.add_argument("--train-0p80", required=True, type=Path)
    parser.add_argument("--validate-0p20", required=True, type=Path)
    parser.add_argument("--validate-0p60", required=True, type=Path)
    parser.add_argument("--activation-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--development-prelock",
        action="store_true",
        help="score an earlier pilot, but force deployment eligibility false",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    position_lock = json.loads(args.position_lock.read_text("utf-8"))
    depth_lock = json.loads(args.depth_lock.read_text("utf-8"))
    if position_lock.get("schema") != "f45_multidepth_position_candidate_lock_v1":
        raise ValueError("unexpected position lock")
    if depth_lock.get("schema") != "f45_multidepth_depth_proxy_lock_v1":
        raise ValueError("unexpected depth lock")
    if depth_lock["sources"][1]["sha256"] != sha256(args.position_lock):
        raise ValueError("depth lock does not reference this position lock")
    if position_lock["sources"][1]["sha256"] != sha256(args.train_0p10):
        raise ValueError("0.10 mm training manifest hash differs from lock")
    if position_lock["sources"][2]["sha256"] != sha256(args.train_0p80):
        raise ValueError("0.80 mm training manifest hash differs from lock")
    lock_time = max(parse_time(position_lock["locked_utc"]), parse_time(depth_lock["locked_utc"]))
    validation_manifests = {
        0.20: validate_manifest(
            args.validate_0p20, 0.20, lock_time,
            require_after=not args.development_prelock,
        ),
        0.60: validate_manifest(
            args.validate_0p60, 0.60, lock_time,
            require_after=not args.development_prelock,
        ),
    }

    here = Path(__file__).resolve().parent
    old = load_rounds(
        (here.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve(),
        [
            (here.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
            here / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
            here / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
        ],
    )
    training_rounds = {
        "depth_0p10_r1": load_depth_manifest(args.train_0p10),
        **old,
        "depth_0p80_r1": load_depth_manifest(args.train_0p80),
    }
    training = [row for values in training_rounds.values() for row in values]
    if len(training) != int(position_lock["training_contacts"]):
        raise ValueError("training contact count differs from lock")
    preprocessing = str(position_lock["feature"]["preprocessing"])
    x_train = transform(
        np.stack([row["features"][FEATURE] for row in training]), preprocessing
    )
    y_train = np.asarray([row["point_index"] for row in training], dtype=int)
    classifier = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=1, max_features="sqrt",
        random_state=20260908, n_jobs=-1,
    ).fit(x_train, y_train)
    if int(np.sum(classifier.predict(x_train) == y_train)) != 119:
        raise ValueError("locked classifier no longer reproduces the training fit")
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in old["old_r1"]
    }
    depth_curves = {
        int(curve["point"]): curve
        for curve in depth_lock["calibration"]["curves"]
    }
    activation_payload = joblib.load(args.activation_model)
    threshold = float(activation_payload["activation"]["threshold_codes"])
    activation_contract = {
        "model": str(args.activation_model.resolve()),
        "sha256": sha256(args.activation_model),
        "quantity": activation_payload["activation"]["quantity"],
        "threshold_codes": threshold,
        "created_utc": activation_payload["created_utc"],
        "predates_validation_capture": parse_time(activation_payload["created_utc"]) < lock_time,
    }
    if not activation_contract["predates_validation_capture"]:
        raise ValueError("activation threshold was not fixed before validation")

    depth_results = {}
    point_results_by_depth = {}
    for expected_depth, manifest in validation_manifests.items():
        point_results = []
        total_frames = detected_frames = exact_frames = within_frames = 0
        for point in range(1, 18):
            path = Path(manifest["captures"][str(point)]["capture"])
            arrays = capture_arrays(path)
            detected = arrays["first_response"] >= threshold
            features = transform(arrays["frame_features"], preprocessing)
            predictions_all = np.asarray(classifier.predict(features), dtype=int)
            predictions = predictions_all[detected]
            counts = Counter(int(value) for value in predictions)
            majority = counts.most_common(1)[0][0] if counts else None
            errors = np.asarray([
                np.linalg.norm(coordinates[int(value)] - coordinates[point])
                for value in predictions
            ], dtype=float)
            predicted_depth = (
                inverse_depth(depth_curves[int(majority)], arrays["contact_second_response"])
                if majority is not None else None
            )
            oracle_depth = inverse_depth(depth_curves[point], arrays["contact_second_response"])
            point_results.append({
                "point": point,
                "capture": str(path.resolve()),
                "capture_sha256": sha256(path),
                "held_frames": int(arrays["held_frames"]),
                "detected_frames": int(np.sum(detected)),
                "detected_percent": float(100.0 * np.mean(detected)),
                "majority_prediction": majority,
                "majority_exact": majority == point,
                "exact_detected_frame_percent": float(100.0 * np.mean(predictions == point)) if len(predictions) else None,
                "within_4mm_detected_frame_percent": float(100.0 * np.mean(errors <= 4.0 + 1e-6)) if len(errors) else None,
                "second_response_codes": arrays["contact_second_response"],
                "pipeline_predicted_depth_mm": predicted_depth,
                "pipeline_depth_absolute_error_mm": abs(predicted_depth - expected_depth) if predicted_depth is not None else None,
                "oracle_position_predicted_depth_mm": oracle_depth,
            })
            total_frames += arrays["held_frames"]
            detected_frames += int(np.sum(detected))
            exact_frames += int(np.sum(predictions == point))
            within_frames += int(np.sum(errors <= 4.0 + 1e-6))
        depth_errors = np.asarray([
            row["pipeline_depth_absolute_error_mm"] for row in point_results
            if row["pipeline_depth_absolute_error_mm"] is not None
        ], dtype=float)
        summary = {
            "truth_depth_mm": expected_depth,
            "trial_majority_exact": int(sum(row["majority_exact"] for row in point_results)),
            "trial_majority_exact_percent": float(100.0 * np.mean([row["majority_exact"] for row in point_results])),
            "held_frames": total_frames,
            "detected_frames": detected_frames,
            "detected_held_frame_percent": float(100.0 * detected_frames / total_frames),
            "exact_detected_frame_percent": float(100.0 * exact_frames / detected_frames) if detected_frames else None,
            "within_4mm_detected_frame_percent": float(100.0 * within_frames / detected_frames) if detected_frames else None,
            "trial_depth_mae_mm": float(np.mean(depth_errors)) if len(depth_errors) == 17 else None,
            "trial_depth_p95_absolute_error_mm": float(np.percentile(depth_errors, 95)) if len(depth_errors) == 17 else None,
            "trial_depth_maximum_absolute_error_mm": float(np.max(depth_errors)) if len(depth_errors) == 17 else None,
        }
        depth_results[f"{expected_depth:.2f}"] = summary
        point_results_by_depth[f"{expected_depth:.2f}"] = point_results

    low = {row["point"]: row for row in point_results_by_depth["0.20"]}
    high = {row["point"]: row for row in point_results_by_depth["0.60"]}
    ordered = [
        point for point in range(1, 18)
        if low[point]["pipeline_predicted_depth_mm"] is not None
        and high[point]["pipeline_predicted_depth_mm"] is not None
        and high[point]["pipeline_predicted_depth_mm"] > low[point]["pipeline_predicted_depth_mm"]
    ]
    position_gate = all(
        row["trial_majority_exact"] == 17
        and row["within_4mm_detected_frame_percent"] >= 100.0 - 1e-12
        and row["detected_held_frame_percent"] >= 100.0 - 1e-12
        for row in depth_results.values()
    )
    depth_gate = all(
        row["trial_depth_mae_mm"] is not None
        and row["trial_depth_mae_mm"] <= 0.10 + 1e-12
        and row["trial_depth_p95_absolute_error_mm"] <= 0.20 + 1e-12
        for row in depth_results.values()
    ) and len(ordered) == 17
    result = {
        "schema": "f45_multidepth_locked_candidate_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locks": {
            "position": {"path": str(args.position_lock.resolve()), "sha256": sha256(args.position_lock)},
            "depth": {"path": str(args.depth_lock.resolve()), "sha256": sha256(args.depth_lock)},
        },
        "validation_captured_after_both_locks": not args.development_prelock,
        "development_prelock_scoring": bool(args.development_prelock),
        "random_frame_split_used": False,
        "activation_contract": activation_contract,
        "position_gate_passed": position_gate,
        "depth_proxy_gate_passed": depth_gate,
        "candidate_metrics_gate_passed": position_gate and depth_gate,
        "deployment_gate_passed": (
            position_gate and depth_gate and not args.development_prelock
        ),
        "depth_results": depth_results,
        "per_point_depth_order_0p60_greater_than_0p20": len(ordered),
        "ordered_points": ordered,
        "per_point": point_results_by_depth,
        "claims": {
            "force_measured": False,
            "physical_15hz_pressure_response_verified": False,
            "nominal_8mm_tool_face_area_mm2": float(np.pi * 4.0**2),
            "physical_contact_area_verified": False,
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_point"}, ensure_ascii=False, indent=2))
    return 0 if result["deployment_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
