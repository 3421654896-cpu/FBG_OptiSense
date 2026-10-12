"""Build the guarded F45 ExtraTrees runtime artifact from locked evidence.

The validation round is never added to training.  This builder verifies the
precommitted candidate and its untouched validation report, fits the exact
locked estimator on the five development rounds, and writes a new versioned
joblib artifact plus a readable JSON receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from evaluate_f45_cross_session_models import load_rounds


SCHEMA = "ch1_fast_fullmap_position_model_v2"
FEATURE = "pair_transition_delta45"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(owner: Path, relative: str) -> Path:
    return (owner.parent / relative).resolve()


def build(candidate_path: Path, validation_path: Path, old_model_path: Path) -> dict:
    candidate_path = candidate_path.resolve()
    validation_path = validation_path.resolve()
    old_model_path = old_model_path.resolve()
    candidate = json.loads(candidate_path.read_text("utf-8"))
    validation = json.loads(validation_path.read_text("utf-8"))
    old_model = json.loads(old_model_path.read_text("utf-8"))

    if candidate.get("schema") != "f45_cross_session_extratrees_candidate_lock_v1":
        raise ValueError("unsupported candidate lock")
    if candidate.get("deployment_status") != "candidate_only":
        raise ValueError("candidate lock has unexpected deployment state")
    if candidate.get("causal_feature", {}).get("name") != FEATURE:
        raise ValueError("candidate feature is not the locked 135-D feature")
    params = candidate.get("classifier") or {}
    expected = {
        "name": "ExtraTreesClassifier",
        "n_estimators": 400,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 20260908,
        "n_jobs": -1,
    }
    if params != expected:
        raise ValueError("candidate estimator parameters changed")

    candidate_digest = sha256(candidate_path)
    if validation.get("candidate_lock_sha256") != candidate_digest:
        raise ValueError("validation does not match the candidate lock")
    gate = validation.get("static_position_gate") or {}
    if not gate.get("passed"):
        raise ValueError("untouched static validation gate did not pass")
    if (
        int(validation.get("trial_majority_exact", -1)) != 17
        or float(validation.get("frame_within_4mm_percent", -1.0)) != 100.0
        or validation.get("feature") != FEATURE
        or validation.get("estimator") != params
    ):
        raise ValueError("validation metrics or estimator contract changed")

    verified_sources = []
    for source in candidate.get("training_sources") or []:
        path = resolve(candidate_path, source["path"])
        digest = sha256(path)
        if digest != source.get("sha256"):
            raise ValueError(f"locked training source changed: {path}")
        verified_sources.append({"path": str(path), "sha256": digest})
    if len(verified_sources) != 4:
        raise ValueError("unexpected locked training source count")

    training_manifest = Path(verified_sources[0]["path"])
    prior_validations = [Path(item["path"]) for item in verified_sources[1:]]
    rounds = load_rounds(training_manifest, prior_validations)
    rows = [row for records in rounds.values() for row in records]
    if len(rows) != 85:
        raise ValueError("expected exactly five 17-position development rounds")
    train_x = np.stack([row["features"][FEATURE] for row in rows])
    train_y = np.asarray([row["point_index"] for row in rows], dtype=int)
    if train_x.shape != (85, 135) or set(train_y.tolist()) != set(range(1, 18)):
        raise ValueError("malformed locked training matrix")

    classifier = ExtraTreesClassifier(
        n_estimators=int(params["n_estimators"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    ).fit(train_x, train_y)
    # The lock used all cores for offline fitting. A single F45 frame is only
    # one observation, so parallel predict_proba startup is slower and can
    # backlog the realtime UI. n_jobs does not change fitted trees or output.
    classifier.n_jobs = 1
    coordinates = {
        int(row["point_index"]): [float(value) for value in row["xy"]]
        for row in next(iter(rounds.values()))
    }
    if set(coordinates) != set(range(1, 18)):
        raise ValueError("missing calibrated grid coordinates")

    old_activation = old_model.get("activation") or {}
    threshold = float(old_activation.get("threshold_codes", np.nan))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("old qualified F45 activation threshold is invalid")

    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "F45",
            "route_rows": list(range(45)),
            "fresh_points_per_frame": 45,
            "points_per_grating": 5,
            "adc_samples": ["first_code", "second_code"],
            "spacing_us": 50,
            "feedback_selectors": [0, 2],
        },
        "baseline": {
            "minimum_laser_warmup_frames": 16,
            "no_contact_frames": 64,
            "freeze_after_ready": True,
            "separate_first_and_second_medians": True,
        },
        "activation": {
            "quantity": "RMS of 45 first-code deltas",
            "threshold_codes": threshold,
            "source_model": str(old_model_path),
            "source_model_sha256": sha256(old_model_path),
            "dynamic_15hz_detection_verified": False,
        },
        "feature": {"name": FEATURE, "dimensions": 135, "preprocessing": "none"},
        "classifier": classifier,
        "runtime_inference": {
            "n_jobs": 1,
            "reason": "avoid per-frame parallel-worker startup and UI backlog",
        },
        "coordinates_by_point": coordinates,
        "training": {
            "rounds": list(rounds),
            "contacts": len(rows),
            "validation_round_included": False,
            "verified_sources": verified_sources,
        },
        "candidate_lock": {
            "path": str(candidate_path),
            "sha256": candidate_digest,
        },
        "static_validation": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
            "trial_majority_exact_percent": float(
                validation["trial_majority_exact_percent"]
            ),
            "frame_exact_percent": float(validation["frame_exact_percent"]),
            "frame_within_4mm_percent": float(
                validation["frame_within_4mm_percent"]
            ),
            "p95_error_mm": float(validation["p95_error_mm"]),
            "maximum_error_mm": float(validation["maximum_error_mm"]),
            "gate": dict(gate),
        },
        "physical_15hz_verified": False,
        "nominal_tool_diameter_mm": 8.0,
        "nominal_tool_area_mm2": float(np.pi * 4.0**2),
        "area_convention": "fixed circular tool-face footprint, not deformed contact-area truth",
        "deployment_status": "guarded_static_position_runtime",
    }


def receipt(artifact: dict, model_path: Path) -> dict:
    return {
        "schema": "ch1_fast_fullmap_position_model_v2_receipt",
        "created_utc": artifact["created_utc"],
        "runtime_model": str(model_path.resolve()),
        "runtime_model_sha256": sha256(model_path),
        "protocol": artifact["protocol"],
        "feature": artifact["feature"],
        "training": artifact["training"],
        "candidate_lock": artifact["candidate_lock"],
        "static_validation": artifact["static_validation"],
        "physical_15hz_verified": False,
        "deployment_status": artifact["deployment_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_lock", type=Path)
    parser.add_argument("validation_report", type=Path)
    parser.add_argument("old_qualified_model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        parser.error("output and receipt paths must be new")
    artifact = build(args.candidate_lock, args.validation_report, args.old_qualified_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    summary = receipt(artifact, args.output)
    args.receipt.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
