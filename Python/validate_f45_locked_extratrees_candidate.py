"""Replay the precommitted F45 ExtraTrees candidate on one untouched round.

The candidate and validation-protocol hashes are verified before fitting.
Training uses only the five rounds named in the candidate lock.  Validation
predicts every causal held frame in the new round.  This program is offline
and never accesses the optical board or CNC controller.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from validate_fast_fullmap_position_locked import load_validation_round


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(owner: Path, relative: str) -> Path:
    return (owner.parent / relative).resolve()


def replay(candidate_path: Path, protocol_path: Path, validation_path: Path) -> dict:
    candidate = json.loads(candidate_path.read_text("utf-8"))
    protocol = json.loads(protocol_path.read_text("utf-8"))
    validation = json.loads(validation_path.read_text("utf-8"))
    if candidate.get("schema") != "f45_cross_session_extratrees_candidate_lock_v1":
        raise ValueError("unknown candidate lock")
    if protocol.get("schema") != "f45_validation_protocol_lock_v1":
        raise ValueError("unknown validation-protocol lock")
    if validation.get("schema") != "f45_position_validation_round_v1":
        raise ValueError("unknown validation manifest")
    if not candidate.get("locked_before_next_physical_round"):
        raise ValueError("candidate was not locked before validation")
    if not protocol.get("locked_before_first_qualifying_capture"):
        raise ValueError("protocol was not locked before validation")
    if not validation.get("algorithm_locked_before_first_capture"):
        raise ValueError("validation manifest is not marked post-lock")

    candidate_ref = protocol.get("candidate_lock") or {}
    if resolve(protocol_path, candidate_ref["path"]) != candidate_path.resolve():
        raise ValueError("protocol references a different candidate lock")
    candidate_sha = sha256(candidate_path)
    if candidate_sha != candidate_ref.get("sha256"):
        raise ValueError("candidate lock changed after protocol lock")
    manifest_ref = resolve(protocol_path, protocol["next_validation"]["manifest"])
    if manifest_ref != validation_path.resolve():
        raise ValueError("protocol references a different validation manifest")
    locked_order = [int(value) for value in protocol["next_validation"]["acquisition_order"]]
    if locked_order != [int(value) for value in validation["acquisition_order"]]:
        raise ValueError("manifest acquisition order differs from protocol lock")

    verified_sources = []
    for source in candidate.get("training_sources") or []:
        path = resolve(candidate_path, source["path"])
        actual = sha256(path)
        if actual != source.get("sha256"):
            raise ValueError(f"training source changed after lock: {path}")
        verified_sources.append({"path": str(path), "sha256": actual})
    verified_reports = []
    for report in candidate.get("selection_reports") or []:
        path = resolve(candidate_path, report["path"])
        actual = sha256(path)
        if actual != report.get("sha256"):
            raise ValueError(f"selection report changed after lock: {path}")
        verified_reports.append({"path": str(path), "sha256": actual})
    if len(verified_sources) != 4 or len(verified_reports) != 2:
        raise ValueError("candidate lock has an unexpected source set")

    mapping = validation.get("round") or {}
    starts = []
    for point in range(1, 18):
        capture_path = (validation_path.parent / mapping[str(point)]).resolve()
        payload = json.loads(capture_path.read_text("utf-8"))
        starts.append(
            (
                int(payload["commissioning"]["events"][0]["host_monotonic_ns"]),
                point,
            )
        )
    actual_order = [point for _time, point in sorted(starts)]
    if actual_order != locked_order:
        raise ValueError(f"actual capture order {actual_order} differs from locked order")

    training_manifest = Path(verified_sources[0]["path"])
    prior_validations = [Path(item["path"]) for item in verified_sources[1:]]
    rounds = load_rounds(training_manifest, prior_validations)
    training = [row for records in rounds.values() for row in records]
    feature = str(candidate["causal_feature"]["name"])
    params = candidate["classifier"]
    model = ExtraTreesClassifier(
        n_estimators=int(params["n_estimators"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=int(params["random_state"]),
        n_jobs=int(params["n_jobs"]),
    )
    train_x = np.stack([row["features"][feature] for row in training])
    train_y = np.asarray([row["point_index"] for row in training], dtype=int)
    model.fit(train_x, train_y)

    test = load_validation_round(validation_path)
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in training[:17]
    }
    total = exact_total = within_total = majority_exact = 0
    all_errors: list[float] = []
    per_point = []
    for truth in test:
        point = int(truth["point_index"])
        vectors = causal_held_features(Path(truth["path"]))[feature]
        predictions = np.asarray(model.predict(vectors), dtype=int)
        probabilities = np.asarray(model.predict_proba(vectors), dtype=float)
        ordered = np.sort(probabilities, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
        counts = Counter(predictions.tolist())
        majority = min(counts, key=lambda label: (-counts[label], label))
        majority_exact += int(majority == point)
        errors = np.asarray(
            [
                np.linalg.norm(coordinates[int(label)] - coordinates[point])
                for label in predictions
            ],
            dtype=float,
        )
        exact = int(np.sum(predictions == point))
        within = int(np.sum(errors <= 4.0 + 1e-6))
        total += len(predictions)
        exact_total += exact
        within_total += within
        all_errors.extend(errors.tolist())
        per_point.append(
            {
                "truth": point,
                "frames": int(len(predictions)),
                "exact_frames": exact,
                "exact_percent": float(100.0 * exact / len(predictions)),
                "majority": int(majority),
                "prediction_counts": {
                    str(label): int(count) for label, count in sorted(counts.items())
                },
                "probability_margin_median": float(np.median(margins)),
                "probability_margin_p05": float(np.percentile(margins, 5)),
                "p95_error_mm": float(np.percentile(errors, 95)),
                "maximum_error_mm": float(np.max(errors)),
            }
        )

    errors = np.asarray(all_errors, dtype=float)
    cadence_pass = bool(all(row["maximum_board_interval_us"] < 33334 for row in test))
    frame_within = float(100.0 * within_total / total)
    gate = bool(majority_exact == 17 and frame_within == 100.0 and cadence_pass)
    return {
        "schema": "f45_locked_extratrees_untouched_round_validation_v1",
        "algorithm_was_locked_before_validation": True,
        "candidate_lock": str(candidate_path.resolve()),
        "candidate_lock_sha256": candidate_sha,
        "protocol_lock": str(protocol_path.resolve()),
        "protocol_lock_sha256": sha256(protocol_path),
        "validation_manifest": str(validation_path.resolve()),
        "validation_manifest_sha256_at_replay": sha256(validation_path),
        "locked_acquisition_order": locked_order,
        "actual_acquisition_order": actual_order,
        "verified_training_sources": verified_sources,
        "verified_selection_reports": verified_reports,
        "training_rounds": list(rounds),
        "training_contacts": len(training),
        "validation_contacts": 17,
        "feature": feature,
        "estimator": params,
        "causal_baseline_frames": 64,
        "trial_majority_exact": majority_exact,
        "trial_majority_exact_percent": float(100.0 * majority_exact / 17.0),
        "frames": total,
        "frame_exact_percent": float(100.0 * exact_total / total),
        "frame_within_4mm_percent": frame_within,
        "median_error_mm": float(np.median(errors)),
        "p95_error_mm": float(np.percentile(errors, 95)),
        "maximum_error_mm": float(np.max(errors)),
        "all_fresh_45_point_sampling_gate_passed": cadence_pass,
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
        "deployment_status": "eligible_for_guarded_static-position integration" if gate else "rejected",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_lock", type=Path)
    parser.add_argument("protocol_lock", type=Path)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = replay(args.candidate_lock, args.protocol_lock, args.validation_manifest)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "locked_acquisition_order",
                    "actual_acquisition_order",
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
