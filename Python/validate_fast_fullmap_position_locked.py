"""Validate the already-selected F45 position rule on one untouched round.

This program never commands the optical board or the machine.  Candidate
selection is recomputed from rounds r1/r2, the selected feature/scaling/metric
are then frozen, and every r3 capture is classified exactly once by 1-nearest
neighbour over the two stored training observations for each position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_coverage_position import _transform
from evaluate_fast_fullmap_position import (
    _capture_feature,
    load_manifest,
    metrics,
    select,
)


def load_validation_round(path: Path) -> list[dict]:
    raw = json.loads(path.read_text("utf-8"))
    if raw.get("schema") != "f45_position_validation_round_v1":
        raise ValueError("unknown validation manifest schema")
    mapping = raw.get("round") or {}
    records = [
        _capture_feature((path.parent / mapping[str(point)]).resolve())
        for point in range(1, 18)
    ]
    if [record["point_index"] for record in records] != list(range(1, 18)):
        raise ValueError("validation point indices are not exactly 1..17")
    return records


def validate_locked(
    training_manifest: Path,
    selection_path: Path,
    validation_manifest: Path,
) -> dict:
    rounds = load_manifest(training_manifest)
    saved_selection = json.loads(selection_path.read_text("utf-8"))
    if saved_selection.get("schema") != "f45_position_candidate_selection_v2":
        raise ValueError("unknown candidate-selection schema")

    recomputed = select(rounds)["best_candidate"]
    saved = saved_selection.get("best_candidate") or {}
    keys = ("feature", "scaling", "metric")
    if any(saved.get(key) != recomputed.get(key) for key in keys):
        raise ValueError("saved candidate no longer matches r1/r2 recomputation")

    feature = str(saved["feature"])
    scaling = str(saved["scaling"])
    metric = str(saved["metric"])
    train_records = rounds["r1"] + rounds["r2"]
    test_records = load_validation_round(validation_manifest)
    depths = {
        record["depth_mm"] for record in train_records + test_records
    }
    if len(depths) != 1:
        raise ValueError("training and validation indentation depths differ")

    train = np.stack([record["features"][feature] for record in train_records])
    test = np.stack([record["features"][feature] for record in test_records])
    train, test = _transform(train, test, scaling)
    if metric == "cosine":
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        test /= np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)

    predictions = []
    errors = []
    for vector, truth in zip(test, test_records, strict=True):
        if metric == "cosine":
            distances = 1.0 - train @ vector
        elif metric == "euclidean":
            distances = np.linalg.norm(train - vector, axis=1)
        else:
            raise ValueError(f"unsupported locked metric: {metric}")
        winner_index = int(np.argmin(distances))
        predicted = train_records[winner_index]
        error = float(np.linalg.norm(predicted["xy"] - truth["xy"]))
        errors.append(error)
        predictions.append(
            {
                "truth_point": truth["point_index"],
                "predicted_point": predicted["point_index"],
                "training_round": "r1" if winner_index < 17 else "r2",
                "distance": float(distances[winner_index]),
                "error_mm": error,
            }
        )

    result_metrics = metrics(errors, predictions)
    static_gate = bool(
        result_metrics["p95_error_mm"] <= 4.0
        and result_metrics["max_error_mm"] <= 4.0 + 1e-6
        and result_metrics["within_4mm_percent"] == 100.0
    )
    return {
        "schema": "f45_position_locked_validation_v2",
        "error_coordinate_frame": "mach3_work_xy_mm",
        "algorithm_was_locked_before_validation": True,
        "locked_candidate": {key: saved[key] for key in keys},
        "training_rounds": ["r1", "r2"],
        "validation_round": "r3",
        "training_observations": len(train_records),
        "validation_observations": len(test_records),
        "indentation_depth_mm": float(next(iter(depths))),
        "metrics": result_metrics,
        "predictions": predictions,
        "static_position_gate": {
            "p95_limit_mm": 4.0,
            "maximum_limit_mm": 4.0,
            "required_within_4mm_percent": 100.0,
            "passed": static_gate,
        },
        "all_fresh_45_point_sampling_gate_passed": bool(
            all(record["maximum_board_interval_us"] < 33334 for record in test_records)
        ),
        "deployment_eligible": False,
        "reason_not_deployable": (
            "Static position validation is only one subsystem gate. Physical 15 Hz "
            "pressure response and traceable contact-area labels remain unverified."
        ),
        "physical_15hz_response_verified": False,
        "physical_contact_area_verified": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    result = validate_locked(
        args.training_manifest, args.selection, args.validation_manifest
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({
        "locked_candidate": result["locked_candidate"],
        "metrics": result["metrics"],
        "static_position_gate": result["static_position_gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
