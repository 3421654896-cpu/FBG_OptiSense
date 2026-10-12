"""Evaluate multi-depth C27 position inference in physical XY coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from capture_ch1_teacher_spectrum import save_checkpoint
from evaluate_candidate27_feature_variants import _capture


def _vector(row: dict, read_source: str, feature: str) -> np.ndarray:
    first = row["fields"]["first_code"][feature]
    second = row["fields"]["second_code"][feature]
    if read_source == "first_code":
        return first
    if read_source == "second_code":
        return second
    if read_source == "mean_reads":
        return (first + second) / 2.0
    if read_source == "concat_reads":
        return np.concatenate([first, second], axis=1)
    raise ValueError(read_source)


def _xy(row: dict) -> np.ndarray:
    # Use the calibrated physical work-plane coordinates, not normalized 3D
    # render coordinates.  The latter are unitless and would under-report mm
    # errors by the model scale factor.
    target = row["target"]
    if "calibration_work_xy_mm" in target:
        return np.asarray(target["calibration_work_xy_mm"], dtype=float)
    return np.asarray([target["work_x"], target["work_y"]], dtype=float)


def evaluate(train_paths: list[Path], validation_paths: list[Path]) -> dict:
    train = [_capture(path) for path in train_paths]
    validation = [_capture(path) for path in validation_paths]
    if {row["metadata"]["sha256"] for row in train} & {
        row["metadata"]["sha256"] for row in validation
    }:
        raise ValueError("Training and validation captures must be disjoint")
    labels = np.asarray([int(row["point"]) for row in train])
    positions = np.asarray([_xy(row) for row in train])
    methods = list(train[0]["fields"]["second_code"])
    results = []
    for read_source in ("first_code", "second_code", "mean_reads", "concat_reads"):
        for feature in methods:
            prototypes = np.asarray([
                np.median(_vector(row, read_source, feature), axis=0) for row in train
            ])
            for neighbours in (1, 2, 3, 4):
                trials = []
                frame_errors = []
                frame_exact = 0
                frame_count = 0
                for row in validation:
                    values = _vector(row, read_source, feature)
                    distances = np.sqrt(np.mean(
                        (values[:, None, :] - prototypes[None, :, :]) ** 2, axis=2))
                    order = np.argsort(distances, axis=1)[:, :neighbours]
                    selected_distances = np.take_along_axis(distances, order, axis=1)
                    weights = 1.0 / np.maximum(selected_distances, 1e-9)
                    predicted_xy = np.sum(positions[order] * weights[:, :, None], axis=1) / np.sum(
                        weights, axis=1, keepdims=True)
                    truth_xy = _xy(row)
                    errors = np.linalg.norm(predicted_xy - truth_xy, axis=1)
                    # Exact labels are scored using only the nearest capture;
                    # continuous XY uses distance-weighted neighbours.
                    predictions = labels[order[:, 0]]
                    frame_exact += int(np.count_nonzero(predictions == int(row["point"])))
                    frame_count += len(values)
                    frame_errors.extend(map(float, errors))
                    trial_value = np.median(values, axis=0)
                    trial_distances = np.sqrt(np.mean((prototypes - trial_value) ** 2, axis=1))
                    trial_order = np.argsort(trial_distances)[:neighbours]
                    trial_weights = 1.0 / np.maximum(trial_distances[trial_order], 1e-9)
                    trial_xy = np.sum(positions[trial_order] * trial_weights[:, None], axis=0) / np.sum(
                        trial_weights)
                    trials.append({
                        "truth": int(row["point"]),
                        "truth_xy": truth_xy.tolist(),
                        "nearest_capture_prediction": int(labels[trial_order[0]]),
                        "predicted_xy": trial_xy.tolist(),
                        "position_error_mm": float(np.linalg.norm(trial_xy - truth_xy)),
                    })
                results.append({
                    "read_source": read_source,
                    "feature": feature,
                    "neighbours": neighbours,
                    "trial_exact_accuracy": sum(
                        row["nearest_capture_prediction"] == row["truth"] for row in trials
                    ) / len(trials),
                    "trial_median_position_error_mm": float(np.median(
                        [row["position_error_mm"] for row in trials])),
                    "trial_p95_position_error_mm": float(np.percentile(
                        [row["position_error_mm"] for row in trials], 95)),
                    "frame_exact_accuracy": frame_exact / frame_count,
                    "frame_median_position_error_mm": float(np.median(frame_errors)),
                    "frame_p95_position_error_mm": float(np.percentile(frame_errors, 95)),
                    "trials": trials,
                })
    results.sort(key=lambda row: (
        row["trial_p95_position_error_mm"],
        row["trial_median_position_error_mm"],
        -row["trial_exact_accuracy"],
    ))
    return {
        "schema": "candidate27_multidepth_position_evaluation_v1",
        "training": [row["path"] for row in train],
        "validation": [row["path"] for row in validation],
        "results": results,
        "best": results[:12],
        "deployed": False,
        "physical_15hz_verified": False,
        "scope": "Exploratory static multi-depth contacts; validation must be repeated on untouched trials.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate(args.train, args.validation)
    save_checkpoint(args.output, report)
    print(json.dumps(report["best"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
