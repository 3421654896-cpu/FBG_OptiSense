"""Validate the immutable F18 candidate on one untouched complete round."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f18_cross_session_models import load_contact


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_source(item: dict, *, depth_mm: float) -> dict:
    path = Path(item["path"])
    digest = sha256(path)
    if digest != item.get("sha256"):
        raise ValueError(f"locked source changed: {path}")
    return load_contact(path, int(item["point_index"]), expected_depth_mm=depth_mm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--validation-round", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    lock = json.loads(args.candidate_lock.read_text("utf-8"))
    if lock.get("schema") != "f18_multidepth_position_candidate_lock_v1":
        raise ValueError("unsupported candidate lock")
    feature = lock["feature"]["name"]
    if feature != "pair_flank_shape36" or lock["classifier"] != {
        "name": "Pipeline(StandardScaler, LinearDiscriminantAnalysis)",
        "standard_scaler": {"with_mean": True, "with_std": True},
        "lda": {"solver": "lsqr", "shrinkage": "auto"},
    }:
        raise ValueError("candidate algorithm changed")

    training = [
        verified_source(item, depth_mm=0.30)
        for item in lock["training"]["complete_0p30_capture_sources"]
    ] + [
        verified_source(item, depth_mm=0.10)
        for item in lock["training"]["complete_0p10_capture_sources"]
    ]
    if len(training) != 85:
        raise ValueError("locked training contact count changed")
    training_hashes = {
        item["sha256"]
        for key in ("complete_0p30_capture_sources", "complete_0p10_capture_sources")
        for item in lock["training"][key]
    }
    selection_hashes = {item["sha256"] for item in lock["selection_holdout"]["sources"]}

    manifest = json.loads(args.validation_round.read_text("utf-8"))
    if manifest.get("schema") != "f18_guarded_position_round_manifest_v1":
        raise ValueError("unsupported validation manifest")
    if abs(float(manifest.get("depth_mm", -1.0)) - 0.30) > 1e-9:
        raise ValueError("validation depth changed")
    validation_hashes = {item["sha256"] for item in manifest["captures"]}
    if validation_hashes & (training_hashes | selection_hashes):
        raise ValueError("validation capture was previously used")
    validation = [
        verified_source(item, depth_mm=0.30) for item in manifest["captures"]
    ]
    if [row["point_index"] for row in validation] != list(range(1, 18)):
        raise ValueError("validation positions are not exactly P1..P17")

    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ).fit(
        np.stack([row["median_features"][feature] for row in training]),
        np.asarray([row["point_index"] for row in training], dtype=int),
    )
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in training[:17]
    }
    total = exact = within = majority_exact = 0
    errors: list[float] = []
    per_point = []
    for row in validation:
        truth = int(row["point_index"])
        prediction = np.asarray(model.predict(row["frame_features"][feature]), dtype=int)
        counts = Counter(prediction.tolist())
        majority = min(counts, key=lambda value: (-counts[value], value))
        point_errors = np.asarray(
            [np.linalg.norm(coordinates[int(value)] - coordinates[truth]) for value in prediction]
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
    error = np.asarray(errors)
    metrics = {
        "trial_majority_exact": majority_exact,
        "trial_majority_exact_percent": 100.0 * majority_exact / 17.0,
        "frames": total,
        "frame_exact_percent": 100.0 * exact / total,
        "frame_within_4mm_percent": 100.0 * within / total,
        "median_error_mm": float(np.median(error)),
        "p95_error_mm": float(np.percentile(error, 95)),
        "maximum_error_mm": float(np.max(error)),
    }
    gate = lock["untouched_full_round_gate"]
    passed = (
        majority_exact == int(gate["trial_majority_exact_required"])
        and metrics["frame_exact_percent"] >= float(gate["minimum_frame_exact_percent"])
        and metrics["frame_within_4mm_percent"] >= float(gate["minimum_frame_within_4mm_percent"])
        and metrics["p95_error_mm"] <= float(gate["maximum_p95_error_mm"]) + 1e-6
        and float(manifest["minimum_rate_hz"]) >= float(gate["minimum_sampling_rate_hz"])
    )
    report = {
        "schema": "f18_multidepth_position_candidate_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_lock": str(args.candidate_lock.resolve()),
        "candidate_lock_sha256": sha256(args.candidate_lock),
        "validation_manifest": str(args.validation_round.resolve()),
        "validation_manifest_sha256": sha256(args.validation_round),
        "validation_was_untouched": True,
        "feature": feature,
        "metrics": metrics,
        "per_point": per_point,
        "gate": {**gate, "passed": passed},
        "physical_15hz_verified": False,
        "nominal_tool_area_mm2": float(lock["nominal_tool_area_mm2"]),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"metrics": metrics, "gate_passed": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
