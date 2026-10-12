"""Replay a precommitted F45 LDA candidate on one untouched physical round.

The lock file is the source of truth for the feature and estimator.  All
training-source hashes are verified before fitting.  Training uses one median
contact per point and round; validation predicts every causal held frame from
the untouched round.  This program is offline and never accesses hardware.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from validate_fast_fullmap_position_locked import load_validation_round


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(lock_path: Path, relative: str) -> Path:
    return (lock_path.parent / relative).resolve()


def validate(lock_path: Path, validation_manifest: Path) -> dict:
    lock = json.loads(lock_path.read_text("utf-8"))
    if lock.get("schema") != "f45_cross_session_lda_candidate_lock_v1":
        raise ValueError("unknown candidate-lock schema")
    if not lock.get("locked_before_next_physical_round"):
        raise ValueError("candidate was not marked locked before capture")
    if lock.get("causal_feature", {}).get("name") != "first_noise_scaled_delta45":
        raise ValueError("lock does not name the qualified causal feature")
    classifier = lock.get("classifier") or {}
    if classifier != {
        "name": "LinearDiscriminantAnalysis",
        "solver": "lsqr",
        "shrinkage": "auto",
    }:
        raise ValueError("lock does not name the qualified LDA estimator")

    verified_sources = []
    for source in lock.get("training_sources") or []:
        path = _resolved(lock_path, source["path"])
        actual = _sha256(path)
        expected = str(source["sha256"])
        if actual != expected:
            raise ValueError(f"training source changed after lock: {path}")
        verified_sources.append({"path": str(path), "sha256": actual})
    selection = lock.get("selection_report") or {}
    selection_path = _resolved(lock_path, selection["path"])
    selection_sha = _sha256(selection_path)
    if selection_sha != selection.get("sha256"):
        raise ValueError("selection report changed after lock")

    if len(verified_sources) != 3:
        raise ValueError("lock must contain the three precommitted training sources")
    training_manifest = Path(verified_sources[0]["path"])
    prior_validations = [Path(row["path"]) for row in verified_sources[1:]]
    rounds = load_rounds(training_manifest, prior_validations)
    training_records = [row for records in rounds.values() for row in records]
    feature = str(lock["causal_feature"]["name"])
    train_x = np.stack([row["features"][feature] for row in training_records])
    train_y = np.asarray([row["point_index"] for row in training_records], dtype=int)
    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ).fit(train_x, train_y)

    manifest = json.loads(validation_manifest.read_text("utf-8"))
    if manifest.get("schema") != "f45_position_validation_round_v1":
        raise ValueError("unknown validation manifest schema")
    if not manifest.get("algorithm_locked_before_first_capture"):
        raise ValueError("validation manifest is not marked post-lock")
    test_records = load_validation_round(validation_manifest)
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in training_records[:17]
    }

    total_frames = 0
    exact_frames = 0
    within_frames = 0
    all_errors: list[float] = []
    majority_exact = 0
    per_point = []
    for truth in test_records:
        point = int(truth["point_index"])
        vectors = causal_held_features(Path(truth["path"]))[feature]
        prediction = np.asarray(model.predict(vectors), dtype=int)
        counts = Counter(prediction.tolist())
        majority = min(counts, key=lambda label: (-counts[label], label))
        majority_exact += int(majority == point)
        errors = np.asarray(
            [
                np.linalg.norm(coordinates[int(label)] - coordinates[point])
                for label in prediction
            ],
            dtype=float,
        )
        exact = int(np.sum(prediction == point))
        within = int(np.sum(errors <= 4.0 + 1e-6))
        total_frames += len(prediction)
        exact_frames += exact
        within_frames += within
        all_errors.extend(errors.tolist())
        per_point.append(
            {
                "truth": point,
                "frames": int(len(prediction)),
                "exact_frames": exact,
                "exact_percent": float(100.0 * exact / len(prediction)),
                "majority": int(majority),
                "prediction_counts": {
                    str(label): int(count) for label, count in sorted(counts.items())
                },
                "p95_error_mm": float(np.percentile(errors, 95)),
                "maximum_error_mm": float(np.max(errors)),
            }
        )

    errors = np.asarray(all_errors, dtype=float)
    gate = bool(
        majority_exact == 17
        and 100.0 * within_frames / total_frames == 100.0
        and all(row["maximum_board_interval_us"] < 33334 for row in test_records)
    )
    return {
        "schema": "f45_locked_lda_untouched_round_validation_v1",
        "algorithm_was_locked_before_validation": True,
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": _sha256(lock_path),
        "verified_training_sources": verified_sources,
        "verified_selection_report": {
            "path": str(selection_path),
            "sha256": selection_sha,
        },
        "validation_manifest": str(validation_manifest.resolve()),
        "validation_manifest_sha256": _sha256(validation_manifest),
        "training_rounds": list(rounds),
        "training_contacts": len(training_records),
        "validation_contacts": 17,
        "feature": feature,
        "estimator": "StandardScaler + LDA(lsqr, shrinkage=auto)",
        "causal_baseline_frames": 64,
        "trial_majority_exact": majority_exact,
        "trial_majority_exact_percent": float(100.0 * majority_exact / 17.0),
        "frames": total_frames,
        "frame_exact_percent": float(100.0 * exact_frames / total_frames),
        "frame_within_4mm_percent": float(100.0 * within_frames / total_frames),
        "median_error_mm": float(np.median(errors)),
        "p95_error_mm": float(np.percentile(errors, 95)),
        "maximum_error_mm": float(np.max(errors)),
        "all_fresh_45_point_sampling_gate_passed": bool(
            all(row["maximum_board_interval_us"] < 33334 for row in test_records)
        ),
        "static_position_gate": {
            "required_trial_majority_exact": "17/17",
            "required_frame_within_4mm_percent": 100.0,
            "passed": gate,
        },
        "per_point": per_point,
        "nominal_tool_diameter_mm": 8.0,
        "nominal_tool_area_mm2": float(np.pi * 4.0**2),
        "area_convention": "fixed circular tool-face footprint, not deformed contact-area truth",
        "physical_15hz_verified": False,
        "force_unit": "indentation depth proxy only",
        "deployment_status": "validation_result_only; integration requires an explicit model artifact",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = validate(args.lock, args.validation_manifest)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "trial_majority_exact",
                    "trial_majority_exact_percent",
                    "frames",
                    "frame_exact_percent",
                    "frame_within_4mm_percent",
                    "p95_error_mm",
                    "maximum_error_mm",
                    "all_fresh_45_point_sampling_gate_passed",
                    "static_position_gate",
                    "per_point",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
