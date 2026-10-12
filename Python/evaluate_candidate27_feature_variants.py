"""Compare CH1-only contact features on locked training and held-out captures.

This is an offline diagnostic.  Point labels are used only to score predictions;
they never enter any feature vector or distance calculation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_candidate27_contact import load_contact
from capture_ch1_teacher_spectrum import save_checkpoint


EPS = 1e-9


def _unit(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=axis, keepdims=True), EPS)


def _windows(path: Path, field: str, expected_firmware: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    # load_contact performs the strict provenance, completion and CNC endpoint
    # validation before this routine consumes the already-decoded values.
    if expected_firmware is None:
        load_contact(path)
    else:
        load_contact(path, expected_firmware=expected_firmware)
    data = json.loads(path.read_text("utf-8"))
    frames = data["frames"]
    receipts = {row["sequence"]: row["host_monotonic_ns"]
                for row in data["commissioning"]["frame_receipts"]}
    events = {row["kind"]: row for row in data["commissioning"]["events"]}
    times = np.asarray([receipts[row["sequence"]] for row in frames], dtype=np.int64)
    baseline_mask = np.asarray([16 <= row["sequence"] < 80 for row in frames])
    held_mask = (
        (times >= events["contact_reached_commanded_depth"]["host_monotonic_ns"] + 200_000_000)
        & (times < events["release_started"]["host_monotonic_ns"])
    )
    values = np.asarray([[record[field] for record in row["records"]]
                         for row in frames], dtype=float)
    return values[baseline_mask], values[held_mask]


def _features(baseline_window: np.ndarray, samples: np.ndarray) -> dict[str, np.ndarray]:
    baseline = np.median(baseline_window, axis=0)
    delta = samples - baseline
    base_groups = baseline.reshape(9, 3)
    sample_groups = samples.reshape(-1, 9, 3)
    delta_groups = delta.reshape(-1, 9, 3)

    base_shape = _unit(base_groups, axis=1)
    sample_shape = _unit(sample_groups, axis=2)
    group_energy = np.linalg.norm(delta_groups, axis=2)
    # A signed flank coordinate estimates the direction of spectral movement
    # while rejecting common optical-power scaling within each three-point group.
    base_flank = (base_groups[:, 2] - base_groups[:, 0]) / np.maximum(
        np.sum(np.abs(base_groups), axis=1), EPS)
    sample_flank = (sample_groups[:, :, 2] - sample_groups[:, :, 0]) / np.maximum(
        np.sum(np.abs(sample_groups), axis=2), EPS)

    return {
        "legacy_shape_delta": (sample_shape - base_shape).reshape(len(samples), -1),
        "delta_unit_global": _unit(delta, axis=1),
        "delta_unit_per_fbg": _unit(delta_groups, axis=2).reshape(len(samples), -1),
        "group_energy_share": _unit(group_energy, axis=1),
        "flank_shift": sample_flank - base_flank,
        "delta_plus_energy": np.concatenate(
            [_unit(delta, axis=1), _unit(group_energy, axis=1)], axis=1),
    }


def _capture(path: Path, expected_firmware: int | None = None) -> dict:
    item, _ = (
        load_contact(path)
        if expected_firmware is None
        else load_contact(path, expected_firmware=expected_firmware)
    )
    raw = json.loads(path.read_text("utf-8"))
    fields = {}
    for field in ("first_code", "second_code"):
        baseline, held = _windows(path, field, expected_firmware=expected_firmware)
        fields[field] = _features(baseline, held)
    return {
        "path": str(path.resolve()),
        "point": item["point_index"],
        "metadata": item,
        "target": raw["commissioning"]["target"],
        "fields": fields,
    }


def evaluate(train_paths: list[Path], validation_paths: list[Path]) -> dict:
    train = [_capture(path) for path in train_paths]
    validation = [_capture(path) for path in validation_paths]
    if len({row["point"] for row in train}) != len(train):
        raise ValueError("Training captures need one unique point each")
    methods = list(train[0]["fields"]["second_code"])
    results = []
    for source in ("first_code", "second_code", "mean_reads", "concat_reads"):
        for method in methods:
            def vectors(row: dict) -> np.ndarray:
                first = row["fields"]["first_code"][method]
                second = row["fields"]["second_code"][method]
                if source == "first_code":
                    return first
                if source == "second_code":
                    return second
                if source == "mean_reads":
                    return (first + second) / 2.0
                return np.concatenate([first, second], axis=1)

            prototypes = np.asarray([np.median(vectors(row), axis=0) for row in train])
            labels = [row["point"] for row in train]
            trials = []
            total = correct = 0
            for row in validation:
                features = vectors(row)
                distances = np.sqrt(np.mean(
                    (features[:, None, :] - prototypes[None, :, :]) ** 2, axis=2))
                predictions = [labels[index] for index in np.argmin(distances, axis=1)]
                median_feature = np.median(features, axis=0)
                trial_distances = np.sqrt(np.mean((prototypes - median_feature) ** 2, axis=1))
                prediction = labels[int(np.argmin(trial_distances))]
                ordered = np.sort(trial_distances)
                total += len(predictions)
                correct += predictions.count(row["point"])
                trials.append({
                    "truth": row["point"],
                    "prediction": prediction,
                    "frame_confusion": dict(Counter(predictions)),
                    "distances": dict(zip(map(str, labels), map(float, trial_distances))),
                    "distance_margin": float(ordered[1] - ordered[0]),
                })
            results.append({
                "read_source": source,
                "feature": method,
                "frame_accuracy": correct / total,
                "trial_accuracy": sum(row["truth"] == row["prediction"] for row in trials) / len(trials),
                "trials": trials,
            })
    results.sort(key=lambda row: (row["trial_accuracy"], row["frame_accuracy"]), reverse=True)
    return {
        "schema": "candidate27_feature_variant_evaluation_v1",
        "training": [row["path"] for row in train],
        "validation": [row["path"] for row in validation],
        "results": results,
        "best": results[:8],
        "scope": "Offline three-position diagnostic only; no deployment and no full-silicone accuracy claim.",
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
