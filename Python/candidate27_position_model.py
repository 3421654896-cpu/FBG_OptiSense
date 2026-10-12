"""Train and validate the guarded C27 CH1 contact-position model.

The model uses only ADC changes relative to a no-contact baseline.  CNC point
labels are targets during training and scoring; they are never feature inputs.
Unknown/ambiguous contacts are rejected instead of forcing a position label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from capture_ch1_teacher_spectrum import now_text, save_checkpoint, sha256_file
from evaluate_candidate27_feature_variants import _capture


FEATURE_NAME = "delta_plus_energy"
READ_SOURCE = "second_code"
DEFAULT_MINIMUM_RELATIVE_MARGIN = 0.03


def _trial_vector(row: dict) -> np.ndarray:
    return np.median(row["fields"][READ_SOURCE][FEATURE_NAME], axis=0)


def _check_contract(rows: list[dict]) -> tuple:
    contracts = {
        (
            row["metadata"]["route_crc32"],
            row["metadata"]["first_delay_us"],
            row["metadata"]["spacing_us"],
            row["metadata"]["depth_mm"],
            row["metadata"]["stream_protocol"],
        )
        for row in rows
    }
    if len(contracts) != 1:
        raise ValueError("All captures must share route, timing, depth and protocol")
    contract = next(iter(contracts))
    if contract[-1] != "candidate27_guarded":
        raise ValueError("Only the path-guarded candidate27 protocol is accepted")
    return contract


def train(paths: list[Path], output: Path, minimum_relative_margin: float) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 0.0 < minimum_relative_margin < 1.0:
        raise ValueError("minimum-relative-margin must be between zero and one")
    rows = [_capture(path) for path in paths]
    contract = _check_contract(rows)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["point"])].append(row)
    if len(grouped) < 2:
        raise ValueError("At least two labelled physical positions are required")
    prototypes = []
    for point, captures in sorted(grouped.items()):
        trial_vectors = np.asarray([_trial_vector(row) for row in captures])
        prototypes.append({
            "point": point,
            "vector": np.median(trial_vectors, axis=0).tolist(),
            "captures": [
                {"source": row["path"], "sha256": row["metadata"]["sha256"]}
                for row in captures
            ],
            "capture_count": len(captures),
        })
    route_crc32, first_delay_us, spacing_us, depth_mm, protocol = contract
    report = {
        "schema": "candidate27_position_model_v2",
        "created": now_text(),
        "deployed": False,
        "stream_protocol": protocol,
        "route_crc32": route_crc32,
        "first_delay_us": first_delay_us,
        "spacing_us": spacing_us,
        "depth_mm": depth_mm,
        "read_source": READ_SOURCE,
        "feature": FEATURE_NAME,
        "minimum_relative_margin": float(minimum_relative_margin),
        "prototypes": prototypes,
        "method": (
            "Second direct CH1 ADC read; subtract capture-local no-contact baseline; "
            "concatenate global unit delta (27) and unit nine-FBG delta-energy "
            "distribution (9); nearest median class prototype with ambiguity reject."
        ),
        "physical_15hz_verified": False,
        "full_spatial_accuracy_verified": False,
        "scope": "Static 0.20 mm grid pilot; not a continuous-position or force model.",
    }
    save_checkpoint(output, report)
    return report


def validate(model_path: Path, paths: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    model = json.loads(model_path.read_text("utf-8"))
    if model.get("schema") != "candidate27_position_model_v2":
        raise ValueError("Unsupported model schema")
    rows = [_capture(path) for path in paths]
    _check_contract(rows)
    for row in rows:
        for key in ("route_crc32", "first_delay_us", "spacing_us", "depth_mm", "stream_protocol"):
            if row["metadata"][key] != model[key]:
                raise ValueError(f"Validation {key} does not match the model")
    training_hashes = {
        capture["sha256"]
        for prototype in model["prototypes"]
        for capture in prototype["captures"]
    }
    labels = [int(row["point"]) for row in model["prototypes"]]
    prototypes = np.asarray([row["vector"] for row in model["prototypes"]], dtype=float)
    minimum_margin = float(model["minimum_relative_margin"])
    results = []
    all_true = []
    all_predictions = []
    for row in rows:
        if row["metadata"]["sha256"] in training_hashes:
            raise ValueError("A training capture cannot be scored as independent validation")
        features = np.asarray(row["fields"][READ_SOURCE][FEATURE_NAME], dtype=float)
        distances = np.sqrt(np.mean((features[:, None] - prototypes[None, :]) ** 2, axis=2))
        order = np.argsort(distances, axis=1)
        nearest = distances[np.arange(len(features)), order[:, 0]]
        second = distances[np.arange(len(features)), order[:, 1]]
        relative_margin = (second - nearest) / np.maximum(nearest, 1e-12)
        predictions = [
            labels[index] if margin >= minimum_margin else None
            for index, margin in zip(order[:, 0], relative_margin, strict=True)
        ]
        median_feature = np.median(features, axis=0)
        trial_distances = np.sqrt(np.mean((prototypes - median_feature) ** 2, axis=1))
        trial_order = np.argsort(trial_distances)
        trial_margin = float(
            (trial_distances[trial_order[1]] - trial_distances[trial_order[0]])
            / max(trial_distances[trial_order[0]], 1e-12)
        )
        trial_prediction = labels[int(trial_order[0])] if trial_margin >= minimum_margin else None
        truth = int(row["point"])
        all_true.extend([truth] * len(predictions))
        all_predictions.extend(predictions)
        results.append({
            "source": row["path"],
            "sha256": row["metadata"]["sha256"],
            "truth": truth,
            "held_frames": len(predictions),
            "correct_frames": predictions.count(truth),
            "uncertain_frames": predictions.count(None),
            "frame_confusion": dict(Counter("uncertain" if value is None else str(value)
                                             for value in predictions)),
            "trial_prediction": trial_prediction,
            "trial_relative_margin": trial_margin,
            "distances_to_prototypes": dict(zip(map(str, labels), map(float, trial_distances))),
        })
    accepted = sum(value is not None for value in all_predictions)
    correct = sum(truth == prediction for truth, prediction in zip(all_true, all_predictions))
    report = {
        "schema": "candidate27_position_model_validation_v2",
        "created": now_text(),
        "model_sha256": sha256_file(model_path),
        "validation": results,
        "summary": {
            "capture_count": len(results),
            "trial_correct": sum(row["trial_prediction"] == row["truth"] for row in results),
            "trial_uncertain": sum(row["trial_prediction"] is None for row in results),
            "frame_count": len(all_predictions),
            "accepted_frames": accepted,
            "correct_frames": correct,
            "accepted_frame_accuracy": correct / accepted if accepted else None,
            "frame_coverage": accepted / len(all_predictions) if all_predictions else None,
        },
        "deployed": False,
        "physical_15hz_verified": False,
        "full_spatial_accuracy_verified": False,
        "scope": "Independent static contacts; adjacent held frames are correlated.",
    }
    save_checkpoint(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "validate"))
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-relative-margin", type=float,
                        default=DEFAULT_MINIMUM_RELATIVE_MARGIN)
    args = parser.parse_args()
    if args.mode == "train":
        report = train(args.captures, args.output, args.minimum_relative_margin)
        print("locked", sha256_file(args.output), "points",
              [row["point"] for row in report["prototypes"]])
    else:
        if args.model is None:
            parser.error("--model is required for validation")
        report = validate(args.model, args.captures, args.output)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
