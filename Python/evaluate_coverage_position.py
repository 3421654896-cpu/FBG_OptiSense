"""Evaluate event-level CH1 coverage position repeatability without hardware.

One physical contact is one indivisible sample.  A point-local no-contact
baseline is built only from the first two seconds after that point's
``pre_contact`` event; the held response is then aggregated from later frames.
Sessions are never mixed within a fold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _aggregate(frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([frame["raw_adc_codes"]["1"] for frame in frames], dtype=float)
    median = np.median(values, axis=0)
    sigma = 1.4826 * np.median(np.abs(values - median), axis=0)
    return median, sigma


def _feature_variants(base: np.ndarray, held: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    delta = held - base
    base_segments = base.reshape(9, 5)
    held_segments = held.reshape(9, 5)
    delta_segments = delta.reshape(9, 5)
    segment_scale = np.maximum(np.ptp(base_segments, axis=1), 8.0)[:, None]
    normalized = delta_segments / segment_scale
    response = np.sqrt(np.mean(delta_segments**2, axis=1))
    signed = np.mean(delta_segments, axis=1)
    asymmetry = (delta_segments[:, 3] + delta_segments[:, 4]) - (
        delta_segments[:, 0] + delta_segments[:, 1]
    )
    centre = delta_segments[:, 2]
    summary = np.stack((response, signed, asymmetry, centre), axis=1).reshape(-1)
    return {
        "delta45": delta,
        "noise_scaled_delta45": delta / np.maximum(sigma, 1.0),
        "relative_delta45": delta / np.maximum(np.abs(base), 10.0),
        "segment_normalized_delta45": normalized.reshape(-1),
        "response9": response,
        "summary36": summary,
        "delta45_plus_summary36": np.concatenate((delta, summary)),
    }


def load_session(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    events = list(_read_jsonl(path / "events.jsonl"))
    contact_events = {
        str(event["metadata"]["press_event_id"]): event
        for event in events
        if event.get("kind") == "contact" and (event.get("metadata") or {}).get("press_event_id")
    }
    pre_events = {
        str(event["metadata"]["press_event_id"]): event
        for event in events
        if event.get("kind") == "pre_contact" and (event.get("metadata") or {}).get("press_event_id")
    }
    grouped = defaultdict(lambda: defaultdict(list))
    for frame in _read_jsonl(path / "frames.jsonl"):
        quality = frame.get("quality") or {}
        group = quality.get("press_event_id")
        phase = quality.get("coverage_phase")
        if group in contact_events and phase in {"pre_contact", "hold"}:
            grouped[str(group)][str(phase)].append(frame)
    records = []
    for group, contact in contact_events.items():
        pre_event = pre_events[group]
        baseline_end = int(pre_event["monotonic_ns"]) + 2_000_000_000
        baseline = [
            frame
            for frame in grouped[group]["pre_contact"]
            if int(frame.get("frame_received_monotonic_ns") or frame["monotonic_ns"])
            <= baseline_end
        ]
        held = grouped[group]["hold"]
        if len(baseline) < 10 or len(held) < 10:
            raise ValueError(f"{path.name} {group}: insufficient baseline/hold frames")
        base, sigma = _aggregate(baseline)
        contact_median, _ = _aggregate(held)
        features = _feature_variants(base, contact_median, sigma)
        metadata = contact["metadata"]
        xy = np.asarray(metadata["model_xy_mm"], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError(f"{path.name} {group}: invalid model XY")
        records.append(
            {
                "group": group,
                "point_index": int(metadata["point_index"]),
                "xy": xy,
                "features": features,
                "baseline_frames": len(baseline),
                "hold_frames": len(held),
                "response_rms_codes": float(np.sqrt(np.mean((contact_median - base) ** 2))),
                "baseline_sigma_rms_codes": float(np.sqrt(np.mean(sigma**2))),
            }
        )
    records.sort(key=lambda record: record["point_index"])
    if [record["point_index"] for record in records] != list(range(1, 18)):
        raise ValueError(f"{path.name}: point indices must be exactly 1..17")
    return {
        "session": path.name,
        "path": str(path.resolve()),
        "depth_mm": float(manifest["metadata"]["depth_mm"]),
        "records": records,
    }


def _transform(train: np.ndarray, test: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "standard":
        centre = np.mean(train, axis=0)
        scale = np.std(train, axis=0)
    elif method == "robust":
        centre = np.median(train, axis=0)
        scale = 1.4826 * np.median(np.abs(train - centre), axis=0)
    elif method == "none":
        centre = np.zeros(train.shape[1])
        scale = np.ones(train.shape[1])
    else:
        raise ValueError(method)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (train - centre) / scale, (test - centre) / scale


def evaluate_direction(train_session: dict, test_session: dict, variant: str, scaling: str, metric: str) -> dict:
    train_records = train_session["records"]
    test_records = test_session["records"]
    train = np.stack([record["features"][variant] for record in train_records])
    test = np.stack([record["features"][variant] for record in test_records])
    train, test = _transform(train, test, scaling)
    if metric == "cosine":
        train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
    errors = []
    predictions = []
    for test_vector, truth in zip(test, test_records):
        if metric == "cosine":
            distances = 1.0 - train @ test_vector
        else:
            distances = np.linalg.norm(train - test_vector, axis=1)
        predicted = train_records[int(np.argmin(distances))]
        error = float(np.linalg.norm(predicted["xy"] - truth["xy"]))
        errors.append(error)
        predictions.append(
            {
                "truth_point": truth["point_index"],
                "predicted_point": predicted["point_index"],
                "error_mm": error,
            }
        )
    return {
        "train_session": train_session["session"],
        "test_session": test_session["session"],
        "errors_mm": errors,
        "median_error_mm": float(np.median(errors)),
        "p95_error_mm": float(np.percentile(errors, 95)),
        "max_error_mm": float(np.max(errors)),
        "within_4mm_percent": float(100.0 * np.mean(np.asarray(errors) <= 4.0)),
        "exact_point_percent": float(
            100.0
            * np.mean(
                [
                    item["truth_point"] == item["predicted_point"]
                    for item in predictions
                ]
            )
        ),
        "predictions": predictions,
    }


def evaluate(paths: list[Path]) -> dict:
    if len(paths) != 2:
        raise ValueError("Exactly two independent sessions are required")
    sessions = [load_session(path) for path in paths]
    if len({session["depth_mm"] for session in sessions}) != 1:
        raise ValueError("Sessions must use the same indentation")
    variants = tuple(sessions[0]["records"][0]["features"])
    candidates = []
    for variant in variants:
        for scaling in ("none", "standard", "robust"):
            for metric in ("euclidean", "cosine"):
                directions = [
                    evaluate_direction(sessions[0], sessions[1], variant, scaling, metric),
                    evaluate_direction(sessions[1], sessions[0], variant, scaling, metric),
                ]
                errors = np.asarray(
                    [value for direction in directions for value in direction["errors_mm"]]
                )
                candidates.append(
                    {
                        "variant": variant,
                        "scaling": scaling,
                        "metric": metric,
                        "median_error_mm": float(np.median(errors)),
                        "p95_error_mm": float(np.percentile(errors, 95)),
                        "max_error_mm": float(np.max(errors)),
                        "within_4mm_percent": float(100.0 * np.mean(errors <= 4.0)),
                        "exact_point_percent": float(
                            np.mean([direction["exact_point_percent"] for direction in directions])
                        ),
                        "directions": directions,
                    }
                )
    candidates.sort(
        key=lambda item: (
            item["p95_error_mm"],
            item["median_error_mm"],
            item["max_error_mm"],
        )
    )
    session_quality = []
    for session in sessions:
        records = session["records"]
        session_quality.append(
            {
                "session": session["session"],
                "depth_mm": session["depth_mm"],
                "positions": len(records),
                "baseline_frames_min": min(record["baseline_frames"] for record in records),
                "hold_frames_min": min(record["hold_frames"] for record in records),
                "response_rms_codes_median": float(
                    np.median([record["response_rms_codes"] for record in records])
                ),
                "baseline_sigma_rms_codes_median": float(
                    np.median([record["baseline_sigma_rms_codes"] for record in records])
                ),
            }
        )
    return {
        "purpose": "exploratory bidirectional repeat evaluation; algorithm selection only",
        "deployment_eligible": False,
        "reason_not_deployable": "The same two repeats select the candidate; a third locked repeat is required.",
        "contact_area_evaluated": False,
        "physical_15hz_evaluated": False,
        "session_quality": session_quality,
        "best_candidate": candidates[0],
        "all_candidates": candidates,
    }


def evaluate_locked(train_paths: list[Path], test_path: Path) -> dict:
    if len(train_paths) != 2:
        raise ValueError("Locked evaluation requires exactly two development sessions")
    train_sessions = [load_session(path) for path in train_paths]
    test_session = load_session(test_path)
    if len({session["depth_mm"] for session in [*train_sessions, test_session]}) != 1:
        raise ValueError("All locked-evaluation sessions must use one indentation")
    variant = "summary36"
    train_by_point = {
        session["session"]: {
            record["point_index"]: record for record in session["records"]
        }
        for session in train_sessions
    }
    prototypes = []
    prototype_records = []
    for point_index in range(1, 18):
        records = [train_by_point[session["session"]][point_index] for session in train_sessions]
        vectors = np.stack([record["features"][variant] for record in records])
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        prototype = np.mean(vectors, axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
        prototype_records.append(records[0])
    prototypes = np.stack(prototypes)
    predictions = []
    errors = []
    margins = []
    for truth in test_session["records"]:
        vector = truth["features"][variant]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        distances = 1.0 - prototypes @ vector
        order = np.argsort(distances)
        predicted = prototype_records[int(order[0])]
        error = float(np.linalg.norm(predicted["xy"] - truth["xy"]))
        margin = float(distances[order[1]] - distances[order[0]])
        errors.append(error)
        margins.append(margin)
        predictions.append(
            {
                "truth_point": truth["point_index"],
                "predicted_point": predicted["point_index"],
                "error_mm": error,
                "nearest_margin": margin,
            }
        )
    errors_array = np.asarray(errors)
    return {
        "purpose": "locked third-repeat position evaluation",
        "algorithm_locked_before_test": True,
        "variant": variant,
        "scaling": "none",
        "metric": "cosine_to_mean_unit_prototype",
        "train_sessions": [session["session"] for session in train_sessions],
        "test_session": test_session["session"],
        "depth_mm": test_session["depth_mm"],
        "positions": len(test_session["records"]),
        "median_error_mm": float(np.median(errors_array)),
        "p95_error_mm": float(np.percentile(errors_array, 95)),
        "max_error_mm": float(np.max(errors_array)),
        "within_4mm_percent": float(100.0 * np.mean(errors_array <= 4.0)),
        "exact_point_percent": float(
            100.0
            * np.mean(
                [item["truth_point"] == item["predicted_point"] for item in predictions]
            )
        ),
        "nearest_margin_median": float(np.median(margins)),
        "predictions": predictions,
        "position_accuracy_evaluated": True,
        "contact_area_evaluated": False,
        "physical_15hz_evaluated": False,
        "deployment_eligible": False,
        "reason_not_deployable": (
            "Static 0.1 mm endpoint accuracy is measured, but 15 Hz physical response "
            "and independent contact-area truth are still absent."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs=2, type=Path)
    parser.add_argument("--locked-test", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = (
        evaluate_locked(arguments.sessions, arguments.locked_test)
        if arguments.locked_test is not None
        else evaluate(arguments.sessions)
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "all_candidates"}, ensure_ascii=False, indent=2))
